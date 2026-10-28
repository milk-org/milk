// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_views.c
 * @brief View geometry, N-D broadcasting, operand registration, and `fprog` lifecycle management.
 *
 * Role & Goal:
 *   Manages N-D iteration spaces (`ishape`), broadcast compatibility checks (`merge_shape`),
 *   image and slice view operand registration, dynamic slice refreshing, and allocation/reset
 *   of `fprog` child sub-programs and buffers.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_calc_fuse_internal.h"

static uint32_t anchor_shape[3];
static int      anchor_naxis;
static int      anchor_on;

/**
 * cli_calc_fuse_anchor_begin - Anchor subsequent program iteration shapes to destination view @dv
 * @dv:    Destination view
 * @naxis: Destination axis count
 *
 * Return: void
 */
void cli_calc_fuse_anchor_begin(const fview *dv, int naxis)
{
    memcpy(anchor_shape, dv->vshape, sizeof(anchor_shape));
    anchor_naxis = naxis;
    anchor_on    = 1;
}

/**
 * cli_calc_fuse_anchor_end - Clear destination shape anchoring
 *
 * Return: void
 */
void cli_calc_fuse_anchor_end(void)
{
    anchor_on = 0;
}

/**
 * cli_calc_fuse_anchor_save - Save and temporarily disable destination shape anchoring
 *
 * Return: Previous `anchor_on` state.
 */
int cli_calc_fuse_anchor_save(void)
{
    int s     = anchor_on;
    anchor_on = 0;
    return s;
}

/**
 * cli_calc_fuse_anchor_restore - Restore destination shape anchoring state @on
 * @on: Saved `anchor_on` state from `cli_calc_fuse_anchor_save()`
 *
 * Return: void
 */
void cli_calc_fuse_anchor_restore(int on)
{
    anchor_on = on;
}

/** merge_shape() - widen iteration shape to @vs where allowed (broadcast rule: equal or 1). */
static int merge_shape(fprog *p, const uint32_t vs[3], int naxis)
{
    for (int a = 0; a < 3; a++)
    {
        if (vs[a] == p->ishape[a] || vs[a] == 1)
        {
            continue;
        }
        if (p->ishape[a] == 1 && !p->anchored)
        {
            p->ishape[a] = vs[a];
            continue;
        }
        return -1;
    }
    if (!p->anchored && naxis > p->inaxis)
    {
        p->inaxis = naxis;
    }
    return 0;
}

/**
 * cli_calc_fuse_merge_shape - Widen iteration shape of @p to @vs under broadcast rules
 * @p:    Target fused program
 * @vs:   Operand or constructor extent per axis
 * @vnax: Operand or constructor axis count
 *
 * Return: 0 if compatible, -1 on shape mismatch.
 */
int cli_calc_fuse_merge_shape(fprog *p, const uint32_t vs[3], int vnax)
{
    return merge_shape(p, vs, vnax);
}

/**
 * cli_calc_fuse_restrict_diag - Restrict a 2-D elementwise program @sub to its 1-D diagonal @k
 * @sub: Compiled 2-D elementwise fused program
 * @k:   Diagonal offset (`0` = main diagonal, `> 0` above, `< 0` below)
 *
 * Return: 0 on success, -1 if @sub is not a restrictable 2-D elementwise program.
 */
int cli_calc_fuse_restrict_diag(fprog *sub, int64_t k)
{
    if (cli_calc_fuse_finalize_views(sub) != 0 || sub->inaxis != 2 || sub->nax > 0 ||
        sub->nmatmul > 0 || sub->nuser > 0)
    {
        return -1;
    }
    for (int i = 0; i < sub->n; i++)
    {
        fi_kind ik = sub->ins[i].k;
        if (ik == FI_SHIFT || ik == FI_MEDFILT || ik == FI_COORD || ik == FI_INTERP2 ||
            ik == FI_USER_STENCIL || ik == FI_EYE)
        {
            return -1;
        }
    }
    for (int i = 0; i < sub->nimg; i++)
    {
        if (sub->view[i].dyn || arith_is_complex_dt(sub->img[i]->md[0].datatype))
        {
            return -1;
        }
    }
    uint64_t M    = sub->ishape[0];
    uint64_t N    = sub->ishape[1];
    int64_t  absk = (k < 0) ? -k : k;
    uint64_t r0   = (k < 0) ? (uint64_t) absk : 0;
    uint64_t c0   = (k > 0) ? (uint64_t) absk : 0;
    if (r0 >= M || c0 >= N)
    {
        return -1;
    }
    uint64_t rm = M - r0;
    uint64_t cn = N - c0;
    uint64_t L  = (rm < cn) ? rm : cn;

    for (int i = 0; i < sub->nimg; i++)
    {
        fview *v = &sub->view[i];
        if (v->is_lut)
        {
            continue;
        }
        int64_t s0 = (v->vshape[0] == 1) ? 0 : v->stride[0];
        int64_t s1 = (v->vshape[1] == 1) ? 0 : v->stride[1];
        v->base += (int64_t) r0 * s0 + (int64_t) c0 * s1;
        v->stride[0] = s0 + s1;
        v->stride[1] = 0;
        v->stride[2] = 0;
        v->vshape[0] = (s0 + s1 == 0) ? 1 : (uint32_t) L;
        v->vshape[1] = 1;
        v->vshape[2] = 1;
        v->naxis     = 1;
        v->is_view   = 1;
        char old_spec[160];
        snprintf(old_spec, sizeof(old_spec), "%.150s", v->spec);
        snprintf(v->spec, sizeof(v->spec), "@D:%lld:%s", (long long) k, old_spec);
    }
    sub->ishape[0] = (uint32_t) L;
    sub->ishape[1] = 1;
    sub->ishape[2] = 1;
    sub->inaxis    = 1;
    sub->inel      = L;
    return cli_calc_fuse_finalize_views(sub);
}

/**
 * cli_calc_fuse_add_operand - Register image @im, or the slice @spec of it, as a leaf
 * @p:    Target fused program
 * @im:   Image pointer (ignored when @spec is non-NULL)
 * @spec: Optional slice expression string (e.g. `"im[0:64,:]"`)
 *
 * Return: Operand slot index in `p->img[]`, or -1 on shape/type mismatch.
 */
int cli_calc_fuse_add_operand(fprog *p, const IMAGE *im, const char *spec)
{
    fview v;
    if (spec == NULL)
    {
        whole_view(im, &v);
    }
    else
    {
        IMAGE *src = NULL;
        memset(&v, 0, sizeof(v));
        if (cli_calc_slice_view(spec, "Source image not found", &src, &v.base, v.stride, v.vshape,
                                &v.naxis, &v.dyn) != 0)
        {
            char why[256];
            snprintf(why, sizeof(why), "slice %.200s is not a view (binned, a pixel, or invalid)",
                     spec);
            cli_calc_fuse_reject(why);
            return -1;
        }
        im = src;
        snprintf(v.spec, sizeof(v.spec), "%s", spec);
        while (v.naxis > 1 && v.vshape[v.naxis - 1] == 1)
        {
            v.naxis--;
        }
    }
    if (!is_real_datatype(im->md[0].datatype))
    {
        cli_calc_fuse_reject("complex or half-precision operand");
        return -1;
    }
    for (int i = 0; i < p->nimg; i++)
    {
        if (!p->view[i].is_lut && p->img[i] == im && strcmp(p->view[i].spec, v.spec) == 0)
        {
            return i;
        }
    }
    if (p->nimg >= FUSE_MAXIMG)
    {
        cli_calc_fuse_reject("more than 16 image operands (FUSE_MAXIMG)");
        return -1;
    }
    if (merge_shape(p, v.vshape, v.naxis) != 0)
    {
        char why[256];
        snprintf(why, sizeof(why),
                 "shape of %.60s (%ux%ux%u) does not broadcast with the pass (%ux%ux%u)", im->name,
                 v.vshape[0], v.vshape[1], v.vshape[2], p->ishape[0], p->ishape[1], p->ishape[2]);
        cli_calc_fuse_reject(why);
        return -1;
    }
    if (im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    int k      = p->nimg++;
    p->img[k]  = im;
    p->view[k] = v;
    cli_calc_fuse_bind_slice_tape(p, &p->view[k]);
    return k;
}

/**
 * cli_calc_fuse_bind_slice_tape - Compile the expression bounds of dynamic slice view @v into @p's tape
 * @p: Program whose scalar tape evaluates the bounds each run
 * @v: Slice view of @p (an operand or `p->dstv`), already resolved once by the text path
 *
 * On any bound the fused compiler cannot take (an image, a string), the view keeps the text path;
 * when every bound folds to a constant the view is not dynamic after all.
 *
 * Return: 0 always.
 */
int cli_calc_fuse_bind_slice_tape(fprog *p, fview *v)
{
    char bare[200];
    char parts[9][64];
    int  np = 0;

    v->npart   = 0;
    v->tmpl[0] = '\0';
    if (!v->dyn || v->spec[0] == '\0' ||
        cli_calc_slice_template(v->spec, bare, sizeof(bare), v->tmpl, sizeof(v->tmpl), parts, 9,
                                &np) != 0)
    {
        v->tmpl[0] = '\0';
        return 0;
    }
    int all_const = 1;
    for (int i = 0; i < np; i++)
    {
        cli_token toks[CLI_CALC_MAX_TOKENS];
        char      text[80];
        snprintf(text, sizeof(text), "%s\n", parts[i]);
        int ntok = cli_tokenize(text, toks, CLI_CALC_MAX_TOKENS);
        int n0 = p->n, ns0 = p->ns, nimg0 = p->nimg, depth0 = p->depth;
        if (ntok <= 0)
        {
            v->tmpl[0] = '\0';
            return 0;
        }
        cli_calc_tok_state saved;
        fval               pv;
        cli_calc_tok_push(toks, ntok, &saved);
        int rc = cli_calc_fuse_c_expr(p, 0, &pv);
        int at_end =
            (cli_calc_cur_func()()->type == TOK_EOF || cli_calc_cur_func()()->type == TOK_NEWLINE);
        cli_calc_tok_pop(&saved);
        if (rc != 0 || pv.vec || !at_end)
        {
            p->n =
                n0; /* a bound the tape cannot hold: forget what it emitted, keep the text path */
            p->ns      = ns0;
            p->nimg    = nimg0;
            p->depth   = depth0;
            v->tmpl[0] = '\0';
            return 0;
        }
        v->part_s1[i] = (int16_t) pv.s1;
        v->part_c[i]  = pv.c;
        if (pv.s1 > 0)
        {
            all_const = 0;
        }
    }
    v->npart = np;
    if (all_const)
    {
        v->dyn = 0;
    }
    return 0;
}

/**
 * cli_calc_fuse_add_image - Register whole real image @im as an operand of @p
 * @p:  Target fused program
 * @im: Real image pointer
 *
 * Return: Operand slot index in `p->img[]`, or -1 on error.
 */
int cli_calc_fuse_add_image(fprog *p, const IMAGE *im)
{
    return cli_calc_fuse_add_operand(p, im, NULL);
}

/**
 * cli_calc_fuse_add_flat_image - Register real image @im as a flat 1-D vector of `nelement` pixels
 * @p:  Target fused program
 * @im: Real image pointer
 *
 * Return: Operand slot index in `p->img[]`, or -1 on error.
 */
int cli_calc_fuse_add_flat_image(fprog *p, const IMAGE *im)
{
    if (p->nimg >= FUSE_MAXIMG || !is_real_datatype(im->md[0].datatype))
    {
        return -1;
    }
    fview v;
    memset(&v, 0, sizeof(v));
    v.vshape[0] = (uint32_t) im->md[0].nelement;
    v.vshape[1] = v.vshape[2] = 1;
    v.naxis                   = 1;
    v.stride[0]               = 1;
    v.stride[1] = v.stride[2] = (int64_t) im->md[0].nelement;
    if (merge_shape(p, v.vshape, 1) != 0)
    {
        return -1;
    }
    if (im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    p->img[p->nimg]  = im;
    p->view[p->nimg] = v;
    return p->nimg++;
}

/**
 * cli_calc_fuse_add_complex_image - Register complex image @im as an operand of @p
 * @p:  Target fused program
 * @im: Complex image pointer
 *
 * Return: Operand slot index in `p->img[]`, or -1 on error.
 */
int cli_calc_fuse_add_complex_image(fprog *p, const IMAGE *im)
{
    if (!arith_is_complex_dt(im->md[0].datatype) || im->md[0].nelement <= 1)
    {
        return -1;
    }
    for (int i = 0; i < p->nimg; i++)
    {
        if (p->img[i] == im)
        {
            return i;
        }
    }
    fview v;
    whole_view(im, &v);
    if (p->nimg >= FUSE_MAXIMG || merge_shape(p, v.vshape, v.naxis) != 0)
    {
        return -1;
    }
    if (im->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        p->out_double = 1;
    }
    p->img[p->nimg]  = im;
    p->view[p->nimg] = v;
    return p->nimg++;
}

/**
 * cli_calc_fuse_add_lut_image - Register lookup/table image @im without merging its shape into @p
 * @p:  Target fused program
 * @im: Lookup/table image pointer
 *
 * Return: Operand slot index in `p->img[]`, or -1 on error.
 */
int cli_calc_fuse_add_lut_image(fprog *p, const IMAGE *im)
{
    if (im == NULL || im->md == NULL || !is_real_datatype(im->md[0].datatype) ||
        im->md[0].nelement < 1)
    {
        return -1;
    }
    for (int i = 0; i < p->nimg; i++)
    {
        if (p->view[i].is_lut && p->img[i] == im)
        {
            return i;
        }
    }
    if (p->nimg >= FUSE_MAXIMG)
    {
        return -1;
    }
    fview v;
    whole_view(im, &v);
    v.is_lut         = 1;
    p->img[p->nimg]  = im;
    p->view[p->nimg] = v;
    return p->nimg++;
}

/**
 * cli_calc_fuse_add_axis_operand - Register axis-reduction result @im of @p as an internal operand
 * @p:  Target fused program
 * @im: One of `p->ax_im[]`, the result of an `AX_KIND_AXIS` pre-stage
 *
 * Its shape is not merged into the iteration space here: `cli_calc_fuse_finalize_views()` places it
 * once the other operands are known (keep-dims when @p iterates the reduced image, else squeezed).
 *
 * Return: Operand slot index in `p->img[]`, or -1 when @p is full.
 */
int cli_calc_fuse_add_axis_operand(fprog *p, const IMAGE *im)
{
    if (p->nimg >= FUSE_MAXIMG)
    {
        return -1;
    }
    fview v;
    whole_view(im, &v);
    v.internal = 1;
    if (im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    p->img[p->nimg]  = im;
    p->view[p->nimg] = v;
    return p->nimg++;
}

/** natural_strides() - element strides of a contiguous array of shape @vs. */
static void natural_strides(const uint32_t vs[3], int64_t st[3])
{
    st[0] = 1;
    st[1] = (int64_t) vs[0];
    st[2] = (int64_t) vs[0] * vs[1];
}

/**
 * place_axis_results() - Give each axis-reduction operand of @p its view over the iteration space.
 *
 * `imean(im, 0)` of an nx*ny image is stored as ny values. Read by a program over im's shape it is
 * the keep-dims operand {1, ny} (broadcast along x); read by a program of no other shape, or of a
 * destination of its own shape, it is the squeezed image {ny}, which also sets the iteration space.
 */
static void place_axis_results(fprog *p)
{
    for (int i = 0; i < p->nimg; i++)
    {
        fview *v = &p->view[i];
        if (!v->internal || p->img[i] < p->ax_im || p->img[i] >= p->ax_im + FUSE_MAXAX)
        {
            continue;
        }
        int slot = (int) (p->img[i] - p->ax_im);
        if (p->ax_kind[slot] != AX_KIND_AXIS || p->ax_p[slot] == NULL)
        {
            continue;
        }
        const fprog *sub  = p->ax_p[slot];
        int          axis = p->ax_axis[slot];
        uint32_t     ks[3];
        uint32_t     sq[3] = { 1, 1, 1 };
        int          nsq   = 0;
        for (int a = 0; a < 3; a++)
        {
            ks[a] = (a == axis) ? 1 : sub->ishape[a];
            if (a != axis && a < sub->inaxis)
            {
                sq[nsq++] = sub->ishape[a];
            }
        }
        int trivial = !p->anchored && p->ishape[0] == 1 && p->ishape[1] == 1 && p->ishape[2] == 1;
        int ks_ok   = !trivial;
        for (int a = 0; ks_ok && a < 3; a++)
        {
            ks_ok = (ks[a] == 1 || ks[a] == p->ishape[a]);
        }
        v->base = 0;
        if (ks_ok)
        {
            memcpy(v->vshape, ks, sizeof(ks));
            v->naxis = sub->inaxis;
            natural_strides(ks, v->stride);
            if (sub->inaxis > p->inaxis)
            {
                p->inaxis = sub->inaxis;
            }
            continue;
        }
        memcpy(v->vshape, sq, sizeof(sq));
        v->naxis = (nsq > 0) ? nsq : 1;
        natural_strides(sq, v->stride);
        if (trivial)
        {
            memcpy(p->ishape, sq, sizeof(sq));
            p->inaxis = v->naxis;
        }
    }
}

/**
 * cli_calc_fuse_finalize_views - Fix the iteration space of @p and classify strided operands
 * @p: Compiled fused program
 *
 * Return: 0 on valid broadcast geometry, -1 on incompatible dimensions.
 */
int cli_calc_fuse_finalize_views(fprog *p)
{
    place_axis_results(p);
    const int64_t nat[3] = { 1, (int64_t) p->ishape[0], (int64_t) p->ishape[0] * p->ishape[1] };
    p->inel              = (uint64_t) p->ishape[0] * p->ishape[1] * p->ishape[2];
    if (p->inaxis < 1)
    {
        p->inaxis = 1;
    }
    for (int i = 0; i < p->nimg; i++)
    {
        fview *v = &p->view[i];
        if (v->is_lut)
        {
            continue;
        }
        int nat_ok = (v->base == 0);
        for (int a = 0; a < 3; a++)
        {
            if (v->vshape[a] != 1 && v->vshape[a] != p->ishape[a])
            {
                char why[200];
                snprintf(why, sizeof(why), "operand %.60s does not fit the pass shape (%ux%ux%u)",
                         p->img[i]->name, p->ishape[0], p->ishape[1], p->ishape[2]);
                cli_calc_fuse_reject(why);
                return -1;
            }
            if (v->vshape[a] == 1)
            {
                v->stride[a] = 0;
            }
            if (p->ishape[a] > 1 && (v->vshape[a] != p->ishape[a] || v->stride[a] != nat[a]))
            {
                nat_ok = 0;
            }
        }
        v->is_view = !nat_ok;
        if (v->is_view && (arith_is_complex_dt(p->img[i]->md[0].datatype) || v->is_shift))
        {
            /* stencils index the whole image by iteration coordinates: no slice, no broadcast */
            char why[200];
            snprintf(why, sizeof(why), "%s on a sliced or broadcast operand %.60s",
                     v->is_shift ? "stencil" : "complex read", p->img[i]->name);
            cli_calc_fuse_reject(why);
            return -1;
        }
    }
    if (p->dstv.spec[0] != '\0')
    {
        for (int a = 0; a < 3; a++)
        {
            if (p->dstv.vshape[a] != p->ishape[a])
            {
                cli_calc_fuse_reject(
                    "the sliced destination's shape differs from the expression's");
                return -1;
            }
        }
        p->dstv.is_view = 1;
    }
    return 0;
}

/**
 * cli_calc_fuse_reduce_shapes_match - Verify that @a and @b finalize to identical iteration shapes
 * @a: First fused program
 * @b: Second fused program
 *
 * Return: 1 if both finalize and share identical `ishape`, 0 otherwise.
 */
int cli_calc_fuse_reduce_shapes_match(fprog *a, fprog *b)
{
    return cli_calc_fuse_finalize_views(a) == 0 && cli_calc_fuse_finalize_views(b) == 0 &&
           memcmp(a->ishape, b->ishape, sizeof(a->ishape)) == 0;
}

/** format_slice_bounds() - @v's bracket text with each compiled bound's value from @sval; 0 or -1. */
static int format_slice_bounds(const fview *v, const double *sval, char *out, size_t outsz)
{
    size_t n  = 0;
    int    pi = 0;
    for (const char *c = v->tmpl; *c != '\0'; c++)
    {
        if (*c != '\001')
        {
            if (n + 1 >= outsz)
            {
                return -1;
            }
            out[n++] = *c;
            continue;
        }
        if (pi >= v->npart || (v->part_s1[pi] > 0 && sval == NULL))
        {
            return -1;
        }
        double val = (v->part_s1[pi] > 0) ? sval[v->part_s1[pi] - 1] : v->part_c[pi];
        pi++;
        int w = snprintf(out + n, outsz - n, "%lld", (long long) llround(val));
        if (w < 0 || n + (size_t) w >= outsz)
        {
            return -1;
        }
        n += (size_t) w;
    }
    out[n] = '\0';
    return 0;
}

/**
 * cli_calc_fuse_refresh_views - Re-evaluate dynamic slice bounds (`v->dyn`) before program execution
 * @p:    Compiled fused program
 * @sval: Its scalar tape values for this run (from `cli_calc_fuse_eval_scalars()`), or NULL
 * @dst:  The destination image when `p->dstv` is a slice of it, else NULL
 *
 * A view whose bounds were compiled into the tape is re-placed from @sval; the others re-parse
 * their slice text.
 *
 * Return: 0 on success, -1 if a slice bound became invalid or changed shape.
 */
int cli_calc_fuse_refresh_views(fprog *p, const double *sval, IMAGE *dst)
{
    for (int i = 0; i <= p->nimg; i++)
    {
        fview *v = (i < p->nimg) ? &p->view[i] : &p->dstv;
        if (!v->dyn)
        {
            continue;
        }
        const IMAGE *im = (i < p->nimg) ? p->img[i] : dst;
        fview        nv;
        memset(&nv, 0, sizeof(nv));
        int ok = 0;
        if (v->tmpl[0] != '\0' && im != NULL)
        {
            char btext[256];
            ok = (format_slice_bounds(v, sval, btext, sizeof(btext)) == 0 &&
                  cli_calc_slice_geometry(im, btext, &nv.base, nv.stride, nv.vshape, &nv.naxis) ==
                      0);
        }
        else
        {
            IMAGE *src = NULL;
            ok = (cli_calc_slice_view(v->spec, "Source image not found", &src, &nv.base, nv.stride,
                                      nv.vshape, &nv.naxis, &nv.dyn) == 0 &&
                  (i >= p->nimg || src == p->img[i]));
        }
        if (!ok || memcmp(nv.vshape, v->vshape, sizeof(nv.vshape)) != 0)
        {
            char msg[256];
            snprintf(msg, sizeof(msg), "slice %s changed shape", v->spec);
            parse_errmsg(msg);
            return -1;
        }
        v->base = nv.base;
        memcpy(v->stride, nv.stride, sizeof(v->stride));
    }
    return 0;
}

/**
 * cli_calc_fuse_free_fprog_children - Free all heap-allocated sub-programs and buffers of @p
 * @p: Fused program whose child programs and scratch buffers should be released
 *
 * Return: void
 */
void cli_calc_fuse_free_fprog_children(fprog *p)
{
    if (p == NULL)
    {
        return;
    }
    for (int r = 0; r < FUSE_MAXRED; r++)
    {
        if (p->red_p1[r] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->red_p1[r]);
            free(p->red_p1[r]);
            p->red_p1[r] = NULL;
        }
        if (p->red_p2[r] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->red_p2[r]);
            free(p->red_p2[r]);
            p->red_p2[r] = NULL;
        }
    }
    for (int a = 0; a < FUSE_MAXAX; a++)
    {
        if (p->ax_p[a] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->ax_p[a]);
            free(p->ax_p[a]);
            p->ax_p[a] = NULL;
        }
        if (p->ax_p2[a] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->ax_p2[a]);
            free(p->ax_p2[a]);
            p->ax_p2[a] = NULL;
        }
        arith_csr_cache_free(&p->ax_csr[a]);
        free(p->ax_data[a]);
        p->ax_data[a] = NULL;
        p->ax_cap[a]  = 0;
    }
    for (int m = 0; m < FUSE_MAXMATMUL; m++)
    {
        free(p->matmul[m].col_buf);
        p->matmul[m].col_buf = NULL;
        p->matmul[m].col_cap = 0;
    }
    free(p->alias_buf);
    p->alias_buf = NULL;
    p->alias_cap = 0;
}

/**
 * cli_calc_fuse_reset_fprog - Reset @p for a new compilation while preserving reusable buffers
 * @p: Fused program to reset
 *
 * Return: void
 */
void cli_calc_fuse_reset_fprog(fprog *p)
{
    for (int m = 0; m < FUSE_MAXMATMUL; m++)
    {
        free(p->matmul[m].col_buf);
        p->matmul[m].col_buf = NULL;
        p->matmul[m].col_cap = 0;
    }
    for (int r = 0; r < FUSE_MAXRED; r++)
    {
        if (p->red_p1[r] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->red_p1[r]);
            free(p->red_p1[r]);
            p->red_p1[r] = NULL;
        }
        if (p->red_p2[r] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->red_p2[r]);
            free(p->red_p2[r]);
            p->red_p2[r] = NULL;
        }
    }
    for (int a = 0; a < FUSE_MAXAX; a++)
    {
        if (p->ax_p[a] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->ax_p[a]);
            free(p->ax_p[a]);
            p->ax_p[a] = NULL;
        }
        if (p->ax_p2[a] != NULL)
        {
            cli_calc_fuse_free_fprog_children(p->ax_p2[a]);
            free(p->ax_p2[a]);
            p->ax_p2[a] = NULL;
        }
        p->ax_csr[a].valid     = 0;
        p->ax_csr[a].bin_valid = 0;
        p->ax_kind[a]          = 0;
        p->ax_s[a][0]          = 0;
        p->ax_s[a][1]          = 0;
    }
    p->n          = 0;
    p->ns         = 0;
    p->nimg       = 0;
    p->depth      = 0;
    p->maxdepth   = 0;
    p->nops       = 0;
    p->out_double = 0;
    p->out_dt     = 0;
    p->has_red    = 0;
    p->nred       = 0;
    p->has_cent   = 0;
    p->cent_th_s1 = 0;
    p->cent_th_c  = 0.0;
    p->inel       = 0;
    p->nax        = 0;
    p->nprop      = 0;
    p->nreg       = 0;
    p->max_nreg   = 0;
    p->nextra_dst = 0;
    p->nuser      = 0;
    p->nmatmul    = 0;
    memset(&p->dstv, 0, sizeof(p->dstv));
    if (anchor_on)
    {
        memcpy(p->ishape, anchor_shape, sizeof(p->ishape));
        p->inaxis   = anchor_naxis;
        p->anchored = 1;
    }
    else
    {
        p->ishape[0] = p->ishape[1] = p->ishape[2] = 1;
        p->inaxis                                  = 0;
        p->anchored                                = 0;
    }
}

/**
 * cli_calc_fuse_dst_view - Resolve slice @dst_tok as a view of its image, or whole image *@dst
 * @dst:     In/out destination image pointer
 * @dst_tok: Optional sliced destination token string
 * @dv:      Output destination view descriptor
 *
 * Return: 0 on success, non-zero on error.
 */
int cli_calc_fuse_dst_view(IMAGE **dst, const char *dst_tok, fview *dv)
{
    if (dst_tok != NULL && strchr(dst_tok, '[') != NULL)
    {
        IMAGE *src = NULL;
        memset(dv, 0, sizeof(*dv));
        int rc = cli_calc_slice_view(dst_tok, "Target image not found for slice assignment", &src,
                                     &dv->base, dv->stride, dv->vshape, &dv->naxis, &dv->dyn);
        if (rc != 0)
        {
            return rc;
        }
        *dst = src;
        snprintf(dv->spec, sizeof(dv->spec), "%s", dst_tok);
        return 0;
    }
    whole_view(*dst, dv);
    return 0;
}

/**
 * cli_calc_fuse_prepend_compound_lhs - Prepend LHS read + binary op for `+=`, `-=`, `*=`, `/=`
 * @p:       Target fused program
 * @dst:     Destination image
 * @dst_tok: Optional destination slice specification
 * @aop:     Compound assignment operator
 *
 * Return: 0 on success, -1 on capacity or stack overflow.
 */
int cli_calc_fuse_prepend_compound_lhs(fprog         *p,
                                       IMAGE         *dst,
                                       const char    *dst_tok,
                                       cli_token_type aop)
{
    if (aop == TOK_EQUAL)
    {
        return 0;
    }
    int k = (p->n + 2 <= FUSE_MAXINS && p->maxdepth + 1 + p->max_nreg <= FUSE_MAXSTK)
                ? cli_calc_fuse_add_operand(p, dst, dst_tok)
                : -1;
    if (k < 0)
    {
        return -1;
    }
    memmove(&p->ins[1], &p->ins[0], sizeof(fins) * (size_t) p->n);
    p->ins[0]        = (fins) { .k = FI_IMG, .img = k };
    p->ins[p->n + 1] = (fins) { .k    = FI_BIN,
                                .op   = (aop == TOK_OP_PLUS_EQ)    ? TOK_OP_PLUS
                                        : (aop == TOK_OP_MINUS_EQ) ? TOK_OP_MINUS
                                        : (aop == TOK_OP_STAR_EQ)  ? TOK_OP_STAR
                                                                   : TOK_OP_SLASH,
                                .cpos = 0 };
    p->n += 2;
    p->maxdepth++;
    return 0;
}

static int     post_defer = 0;
static imageID post_ids[64];
static int     post_n = 0;

/**
 * cli_calc_post_output - Publish updated image @im (or queue it when deferred posting is active)
 * @im: Updated output image
 *
 * Return: void
 */
void cli_calc_post_output(IMAGE *im)
{
    if (im == NULL || im->md == NULL || !im->md[0].shared)
    {
        if (im != NULL && im->md != NULL)
        {
            im->md[0].write = 0;
            im->md[0].cnt0++;
        }
        return;
    }
    if (!post_defer)
    {
        processinfo_update_output_stream(NULL, im, NULL);
        return;
    }
    imageID id = (imageID) (im - data.core.image);
    for (int i = 0; i < post_n; i++)
    {
        if (post_ids[i] == id)
        {
            return;
        }
    }
    if (post_n < (int) (sizeof(post_ids) / sizeof(post_ids[0])))
    {
        post_ids[post_n++] = id;
    }
    else
    {
        processinfo_update_output_stream(NULL, im, NULL);
    }
}

/**
 * cli_calc_post_defer - Enable or disable deferred semaphore/cnt0 posting for shared outputs
 * @on: 1 to defer output stream updates until `cli_calc_post_flush()`, 0 to post immediately
 *
 * Return: void
 */
void cli_calc_post_defer(int on)
{
    post_defer = on;
    post_n     = 0;
}

/**
 * cli_calc_post_flush - Flush all deferred output stream updates in a single batch
 * @processinfo: Optional PROCESSINFO pointer for loop telemetry
 *
 * Return: void
 */
void cli_calc_post_flush(void *processinfo)
{
    post_defer = 0;
    for (int i = 0; i < post_n; i++)
    {
        IMAGE *im = &data.core.image[post_ids[i]];
        if (im->used == 1 && im->md != NULL && im->md[0].shared)
        {
            processinfo_update_output_stream((PROCESSINFO *) processinfo, im, NULL);
        }
    }
    post_n = 0;
}

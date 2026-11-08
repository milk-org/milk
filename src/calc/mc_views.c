// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_views.c
 * @brief Slice view resolution, broadcast geometry, and pass fusion analysis for libmilkcalc.
 *
 * Provides two groups of routines used during planning (`mc_plan.c`), C code
 * emission (`mc_emit.c`), and pass execution (`mc_exec.c`):
 *
 * 1. **Runtime View & Broadcast Resolution**:
 *    - `mc_view_resolve()` evaluates dynamic slice bounds (`start:stop:step` or
 *      `::Nb` binned slices, supporting negative indices and the `end` keyword)
 *      against a live `IMAGE` and computes the resulting 3D view extents
 *      `(vnx, vny, vnz)`, base element offset (`offset_elem`), and signed
 *      per-axis element strides (`st[3]`). Binned RHS slices (`::Nb`) are
 *      materialized into temporary images via `imgid_slice_materialize()`.
 *    - `mc_view_merge_shape()` and `mc_view_apply_broadcast()` implement
 *      NumPy-style singleton-axis broadcasting, zeroing `st[a]` on axes where
 *      `vnx/vny/vnz == 1` and `pass_size[a] > 1`.
 * 2. **Pass Operand Lookup & Fusion/Elision Legality**:
 *    - `mc_find_slot_by_bare()`, `mc_find_im_op()`, and `mc_find_scalar_op()`
 *      locate planned slots and operands during C code generation.
 *    - `mc_can_merge_passes()`, `mc_has_bcast_stencil()`, and
 *      `mc_elide_unused_locals()` analyze data dependencies across statements
 *      to determine when consecutive elementwise passes can be fused and when
 *      `local` intermediate images can be kept purely in CPU registers.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"
#include "mc_ast.h"
#include "mc_plan.h"

/* ========================================================================== */
/* Slice Bound Evaluation & View Resolution                                   */
/* ========================================================================== */

/**
 * eval_bound_int - Evaluate a slice bound expression (`start`, `stop`, or `step`) to a long
 * @be:      Bound expression AST node
 * @out_val: Output integer bound value
 * @err:     Error message buffer
 * @errsz:   Size of @err in bytes
 *
 * Fast-paths literal integer nodes (`MC_EXPR_LONG`) and falls back to the
 * scalar evaluator (`mc_scalar_eval_expr`) for arbitrary arithmetic expressions
 * (including the `end` keyword set via `mc_scalar_set_slice_end()`).
 *
 * Return: 0 on success, -1 on error
 */
static int eval_bound_int(const mc_expr_t *be, long *out_val, char *err, size_t errsz)
{
    if (be->kind == MC_EXPR_LONG)
    {
        *out_val = be->u.lval;
        return 0;
    }
    mc_val_t v;
    if (mc_scalar_eval_expr(be, &v, err, errsz) != 0)
    {
        return -1;
    }
    if (v.type == MC_VAL_STRING || v.type == MC_VAL_IMAGE)
    {
        snprintf(err, errsz, "Invalid slice bound expression");
        return -1;
    }
    *out_val = (v.type == MC_VAL_LONG) ? v.lval : (long) llround(v.dval);
    return 0;
}

/**
 * eval_slice_direct - Populate an `IMGID_SLICE` descriptor from an `MC_EXPR_SLICE` AST node
 * @slice_expr: Slice AST node (`MC_EXPR_SLICE`)
 * @im:         Target `IMAGE` being sliced (used to bind `end` per axis)
 * @slc:        Output `IMGID_SLICE` structure
 * @err:        Error message buffer
 * @errsz:      Size of @err in bytes
 *
 * Evaluates each axis specification (`start`, `stop`, `step`, full-axis `:`,
 * reversed `::-1`, or binned `::Nb`), binding the axis-local `end` symbol to
 * `im->md[0].size[ax] - 1` during evaluation and resetting it to `-1` afterward.
 *
 * Return: 0 on success, -1 on error
 */
static int eval_slice_direct(const mc_expr_t *slice_expr,
                             const IMAGE     *im,
                             IMGID_SLICE     *slc,
                             char            *err,
                             size_t           errsz)
{
    memset(slc, 0, sizeof(*slc));
    int nax = slice_expr->u.slice.naxis;
    if (nax <= 0)
    {
        return 0;
    }
    if (nax > IMGID_SLICE_MAXAXIS)
    {
        snprintf(err, errsz, "more than %d axes in slice", IMGID_SLICE_MAXAXIS);
        return -1;
    }
    slc->has_slice = 1;
    slc->naxis     = (uint8_t) nax;
    for (int aa = 0; aa < IMGID_SLICE_MAXAXIS; aa++)
    {
        slc->start[aa] = 0;
        slc->end[aa]   = -1;
        slc->step[aa]  = 1;
        slc->bin[aa]   = 0;
    }

    int im_nax = (int) im->md[0].naxis;
    for (int ax = 0; ax < nax; ax++)
    {
        const mc_slice_axis_t *sa      = &slice_expr->u.slice.axes[ax];
        long                   end_val = (ax < im_nax) ? ((long) im->md[0].size[ax] - 1) : 0;
        mc_scalar_set_slice_end(end_val);

        if (sa->is_rev_full)
        {
            slc->step[ax] = -1;
            continue;
        }
        if (sa->is_full && !sa->has_step)
        {
            continue;
        }

        long bval = 0;
        if (sa->start != NULL)
        {
            if (eval_bound_int(sa->start, &bval, err, errsz) != 0)
            {
                mc_scalar_set_slice_end(-1);
                return -1;
            }
            slc->start[ax] = (int32_t) bval;
        }
        if (!sa->has_colon)
        {
            if (sa->start != NULL)
            {
                slc->end[ax] = slc->start[ax];
            }
            continue;
        }
        if (sa->stop != NULL)
        {
            if (eval_bound_int(sa->stop, &bval, err, errsz) != 0)
            {
                mc_scalar_set_slice_end(-1);
                return -1;
            }
            slc->end[ax] = (int32_t) bval;
        }
        if (sa->has_step)
        {
            if (sa->step == NULL || eval_bound_int(sa->step, &bval, err, errsz) != 0 || bval == 0)
            {
                mc_scalar_set_slice_end(-1);
                if (sa->step == NULL || bval == 0)
                {
                    snprintf(err, errsz, "invalid axis %d spec", ax);
                }
                return -1;
            }
            slc->step[ax] = (int32_t) bval;
            slc->bin[ax]  = sa->is_binned ? 1 : 0;
        }
    }
    mc_scalar_set_slice_end(-1);
    return 0;
}

/**
 * validate_slice_neg_bounds - Reject negative slice indices that underflow axis extents
 * @slc:   Evaluated `IMGID_SLICE` structure
 * @snax:  Number of axes in the source image
 * @ssz:   Source axis extents `[3]`
 * @err:   Error message buffer
 * @errsz: Size of @err in bytes
 *
 * Note: `end[aa] == -1` denotes "default end of axis" in `IMGID_SLICE` when
 * `sz >= 1`, whereas `sz + start < 0` or `sz + end < 0` is an out-of-bounds
 * negative index.
 *
 * Return: 0 if valid, -1 if out of bounds
 */
static int validate_slice_neg_bounds(const IMGID_SLICE *slc,
                                     int                snax,
                                     const uint32_t    *ssz,
                                     char              *err,
                                     size_t             errsz)
{
    for (int aa = 0; aa < slc->naxis && aa < snax; aa++)
    {
        int32_t sz = (int32_t) ssz[aa];
        if (slc->start[aa] < 0 && sz + slc->start[aa] < 0)
        {
            snprintf(err, errsz, "axis %d start %d out of bounds for size %d", aa, slc->start[aa],
                     sz);
            return -1;
        }
        if (slc->end[aa] < 0 && sz + slc->end[aa] < 0)
        {
            snprintf(err, errsz, "axis %d end %d out of bounds for size %d", aa, slc->end[aa], sz);
            return -1;
        }
    }
    return 0;
}

/**
 * mc_view_resolve - Resolve an image or slice expression into a concrete `mc_view_t`
 * @expr:       Optional `MC_EXPR_SLICE` expression (NULL for full-image view)
 * @base_im_id: Underlying `imageID` in `milk_data.image[]`
 * @is_lhs:     1 if resolving an assignment target view, 0 for an RHS input view
 * @out_v:      Output resolved view structure
 * @err:        Error message buffer
 * @errsz:      Size of @err in bytes
 *
 * For un-sliced images (or masked LHS assignments `dst[mask] = rhs`), returns
 * a natural contiguous view over @base_im_id. For strided slices, computes
 * `offset_elem`, signed strides `st[0..2]`, and squeezed view dimensions. For
 * binned RHS slices (`::Nb`), allocates a temporary image (`owned_tmp_id`) and
 * materializes the binned pixels via `imgid_slice_materialize()`.
 *
 * Return: 0 on success, -1 on error
 */
int mc_view_resolve(const mc_expr_t *expr,
                    imageID          base_im_id,
                    int              is_lhs,
                    mc_view_t       *out_v,
                    char            *err,
                    size_t           errsz)
{
    memset(out_v, 0, sizeof(*out_v));
    out_v->im_id        = base_im_id;
    out_v->owned_tmp_id = -1;

    if (base_im_id < 0 || base_im_id >= milk_data.NB_MAX_IMAGE ||
        milk_data.image[base_im_id].used == 0)
    {
        snprintf(err, errsz, "%s",
                 is_lhs ? "Target image not found for slice assignment" : "Source image not found");
        return -1;
    }

    IMAGE   *im     = &milk_data.image[base_im_id];
    int      snax   = (im->md[0].naxis < 1) ? 1 : (im->md[0].naxis > 3) ? 3 : (int) im->md[0].naxis;
    uint32_t ssz[3] = { 1, 1, 1 };
    for (int a = 0; a < snax; a++)
    {
        ssz[a] = im->md[0].size[a] ? im->md[0].size[a] : 1;
    }
    out_v->dt = im->md[0].datatype;

    if (expr == NULL || expr->kind != MC_EXPR_SLICE || (is_lhs && expr->u.slice.is_mask_index))
    {
        out_v->naxis       = (uint8_t) snax;
        out_v->vnx         = ssz[0];
        out_v->vny         = ssz[1];
        out_v->vnz         = ssz[2];
        out_v->offset_elem = 0;
        out_v->st[0]       = 1;
        out_v->st[1]       = (int64_t) ssz[0];
        out_v->st[2]       = (int64_t) ssz[0] * ssz[1];
        out_v->is_contig   = !(expr != NULL && expr->kind == MC_EXPR_SLICE);
        return 0;
    }

    IMGID_SLICE slc;
    if (eval_slice_direct(expr, im, &slc, err, errsz) != 0 ||
        validate_slice_neg_bounds(&slc, snax, ssz, err, errsz) != 0)
    {
        return -1;
    }

    uint32_t outsz[3] = { 0, 0, 0 };
    if (imgid_slice_output_size(&slc, snax, ssz, outsz) != 0)
    {
        snprintf(err, errsz, "%s", slc.errmsg[0] ? slc.errmsg : "Bad slice dims");
        return -1;
    }

    int has_bin = 0;
    for (int a = 0; a < slc.naxis && a < 3; a++)
    {
        has_bin |= (slc.bin[a] != 0);
    }

    if (has_bin)
    {
        if (is_lhs)
        {
            snprintf(err, errsz, "Cannot assign to a binned slice");
            return -1;
        }
        int      onax   = 0;
        uint32_t msz[3] = { 1, 1, 1 };
        for (int a = 0; a < 3; a++)
        {
            if (outsz[a] > 0)
            {
                msz[a] = outsz[a];
                onax   = a + 1;
            }
        }
        if (onax == 0)
        {
            onax = 1;
        }
        imageID tid = mc_tmp_image((uint8_t) onax, msz, im->md[0].datatype);
        if (tid == -1)
        {
            snprintf(err, errsz, "Failed to allocate slice image");
            return -1;
        }
        IMGID simg;
        memset(&simg, 0, sizeof(simg));
        simg.ID       = base_im_id;
        simg.im       = im;
        simg.md       = im->md;
        simg.slice    = slc;
        simg.slice_im = &milk_data.image[tid];
        imgid_slice_materialize(&simg);

        while (onax > 1 && msz[onax - 1] == 1)
        {
            onax--;
        }
        out_v->im_id        = tid;
        out_v->owned_tmp_id = tid;
        out_v->naxis        = (uint8_t) onax;
        out_v->vnx          = msz[0];
        out_v->vny          = msz[1];
        out_v->vnz          = msz[2];
        out_v->offset_elem  = 0;
        out_v->st[0]        = 1;
        out_v->st[1]        = (int64_t) msz[0];
        out_v->st[2]        = (int64_t) msz[0] * msz[1];
        out_v->is_contig    = 1;
        return 0;
    }

    const int64_t ax[3] = { 1, (int64_t) ssz[0], (int64_t) ssz[0] * ssz[1] };
    uint32_t      vs[3] = { 1, 1, 1 };
    int           onax  = 0;
    out_v->offset_elem  = 0;
    for (int a = 0; a < 3; a++)
    {
        int     in    = (a < slc.naxis && slc.has_slice);
        int64_t start = in ? slc.start[a] : 0;
        int64_t step  = in ? slc.step[a] : 1;
        out_v->offset_elem += start * ax[a];
        out_v->st[a] = step * ax[a];
        vs[a]        = (outsz[a] > 0) ? outsz[a] : ssz[a];
        if (outsz[a] > 0)
        {
            onax = a + 1;
        }
    }
    if (onax == 0)
    {
        onax = 1;
    }
    if (!is_lhs)
    {
        while (onax > 1 && vs[onax - 1] == 1)
        {
            onax--;
        }
    }
    out_v->naxis     = (uint8_t) onax;
    out_v->vnx       = vs[0];
    out_v->vny       = vs[1];
    out_v->vnz       = vs[2];
    out_v->is_contig = (!is_lhs && out_v->offset_elem == 0 && vs[0] == ssz[0] && vs[1] == ssz[1] &&
                        vs[2] == ssz[2] && out_v->st[0] == ax[0] && out_v->st[1] == ax[1] &&
                        out_v->st[2] == ax[2]);
    return 0;
}

/* ========================================================================== */
/* Shape Merging & Broadcast Stride Adjustment                                */
/* ========================================================================== */

/**
 * mc_view_merge_shape - Merge an operand view's extents into the pass iteration shape
 * @io_naxis:       In/out pass dimension count (`1..3`)
 * @io_size:        In/out pass iteration extents `[3]`
 * @has_lhs_anchor: 1 if an LHS slice or compound assignment has already fixed @io_size
 * @v:              Operand view to merge
 * @err:            Error message buffer
 * @errsz:          Size of @err in bytes
 *
 * Enforces standard broadcasting rules: along each axis `a`, `v`'s extent must
 * either match `io_size[a]`, be `1` (broadcast to `io_size[a]`), or expand
 * `io_size[a]` from `1` when `!has_lhs_anchor`.
 *
 * Return: 0 if compatible, -1 on dimension mismatch
 */
int mc_view_merge_shape(uint8_t         *io_naxis,
                        uint32_t         io_size[3],
                        int              has_lhs_anchor,
                        const mc_view_t *v,
                        char            *err,
                        size_t           errsz)
{
    uint32_t vs[3] = { v->vnx, v->vny, v->vnz };
    for (int a = 0; a < 3; a++)
    {
        if (vs[a] == io_size[a] || vs[a] == 1)
        {
            continue;
        }
        if (io_size[a] == 1 && !has_lhs_anchor)
        {
            io_size[a] = vs[a];
            continue;
        }
        snprintf(err, errsz,
                 "Dimension mismatch in image assignment: (%ux%ux%u) and (%ux%ux%u) do not match",
                 io_size[0], io_size[1], io_size[2], vs[0], vs[1], vs[2]);
        return -1;
    }
    if (!has_lhs_anchor && v->naxis > *io_naxis)
    {
        *io_naxis = v->naxis;
    }
    return 0;
}

/**
 * mc_view_apply_broadcast - Zero strides on singleton axes and update `is_contig`
 * @v:         View to update in place
 * @pass_size: Final 3D iteration extents of the pass
 *
 * Sets `v->st[a] = 0` on any axis where the view has extent 1, and recomputes
 * `v->is_contig` by checking whether @v spans the full @pass_size at natural
 * row-major strides from offset 0.
 *
 * Return: void
 */
void mc_view_apply_broadcast(mc_view_t *v, const uint32_t pass_size[3])
{
    uint32_t      vs[3]  = { v->vnx, v->vny, v->vnz };
    const int64_t nat[3] = { 1, (int64_t) pass_size[0], (int64_t) pass_size[0] * pass_size[1] };
    int           nat_ok = (v->offset_elem == 0 && v->mask_ptr == NULL);
    for (int a = 0; a < 3; a++)
    {
        if (vs[a] == 1)
        {
            v->st[a] = 0;
        }
        if (pass_size[a] > 1 && (vs[a] != pass_size[a] || v->st[a] != nat[a]))
        {
            nat_ok = 0;
        }
    }
    v->is_contig = nat_ok;
}

/* ========================================================================== */
/* Slot/Operand Lookup & Pass Fusion / Local Elision Analysis                 */
/* ========================================================================== */

/**
 * mc_find_slot_by_bare - Find a program slot index by bare or qualified image name
 * @prog: Planned program
 * @name: Qualified image name (fallback if @bare is empty)
 * @bare: Bare image name without `@S:`/`@L:` prefix
 *
 * Return: Slot index in `prog->slots[]`, or -1 if not found
 */
int mc_find_slot_by_bare(const mcalc_prog_t *prog, const char *name, const char *bare)
{
    const char *key = (bare != NULL && bare[0] != '\0') ? bare : name;
    for (int i = 0; i < prog->nslots; i++)
    {
        if (strcmp(prog->slots[i].bare, key) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * mc_find_im_op - Find an input image operand index in a planned pass
 * @pass:       Elementwise pass to search
 * @sidx:       Target slot index in `prog->slots[]` (when @is_eim is 0)
 * @has_slice:  1 if the operand is an `MC_EXPR_SLICE`
 * @slice_expr: Slice or pre-staged expression pointer
 * @tbl:        1 if searching for a table/source operand (`is_table_arg`)
 * @is_eim:     1 if searching by expression pointer (`slice_expr`) for a pre-staged
 *              or compound image operand
 *
 * Return: Operand index in `pass->ops[]`, or -1 if not found
 */
int mc_find_im_op(const mc_pass_t *pass,
                  int              sidx,
                  int              has_slice,
                  const mc_expr_t *slice_expr,
                  int              tbl,
                  int              is_eim)
{
    for (int i = 0; i < pass->nops; i++)
    {
        const mc_op_desc_t *o = &pass->ops[i];
        if (!o->is_out && !o->is_scalar && o->has_slice == has_slice &&
            o->slice_expr == slice_expr && o->is_table_arg == tbl &&
            (is_eim ? (o->is_expr_im || o->slot_idx >= 0)
                    : (o->slot_idx == sidx && !o->is_expr_im)))
        {
            return i;
        }
    }
    return -1;
}

/**
 * mc_find_scalar_op - Find a hoisted scalar or PRNG operand index in a planned pass
 * @pass:   Elementwise pass to search
 * @se:     Scalar expression AST node (when @prng_e is NULL)
 * @prng_e: PRNG call AST node (`rand`/`randn`), or NULL for regular scalars
 * @cg:     Comprehension generator bound index (`g * 3 + {0,1,2}`), or -1
 *
 * Return: Operand index in `pass->ops[]`, or -1 if not found
 */
int mc_find_scalar_op(const mc_pass_t *pass, const mc_expr_t *se, const mc_expr_t *prng_e, int cg)
{
    for (int i = 0; i < pass->nops; i++)
    {
        const mc_op_desc_t *o = &pass->ops[i];
        if (o->is_scalar && o->slice_expr == prng_e &&
            (prng_e != NULL || (o->scalar_expr == se && o->comp_gen_idx == cg)))
        {
            return i;
        }
    }
    return -1;
}

/**
 * mc_can_merge_passes - Check if two consecutive elementwise passes can be fused
 * @a: First (earlier) elementwise pass
 * @b: Second (later) elementwise pass
 *
 * Passes @a and @b can be fused into a single kernel invocation when:
 * - Neither is a comprehension or `no_dst_bcast` pass, both share the same
 *   complex mode and known static dimensions `(naxis, vnx, vny, vnz)`, and total
 *   outputs do not exceed 8.
 * - Neither pass uses LHS slice/mask assignments.
 * - Pass @b does not overwrite any slot written or read by pass @a (preventing
 *   WAR and WAW hazards across fused statements).
 * - If pass @b reads a slot written by pass @a, that access is purely pointwise
 *   and un-sliced (so `_v<out_idx>` can be forwarded in-register at the current
 *   pixel without requiring neighboring pixels of @a's output to be in memory).
 * - No hoisted scalar of pass @b (e.g. `imean(y)`) reads a slot written by pass
 *   @a: scalars are evaluated before the kernel runs, so they would see the
 *   previous value of the slot.
 *
 * Return: 1 if @a and @b can be safely fused, 0 otherwise
 */
int mc_can_merge_passes(const mcalc_prog_t *prog, const mc_pass_t *a, const mc_pass_t *b)
{
    int bad =
        (a->is_comp || b->is_comp || a->no_dst_bcast || b->no_dst_bcast ||
         a->is_complex != b->is_complex || a->nouts + b->nouts > 8 || a->vnx == 0 || b->vnx == 0 ||
         a->naxis != b->naxis || a->vnx != b->vnx || a->vny != b->vny || a->vnz != b->vnz);
    for (int i = 0; i < a->nouts; i++)
    {
        bad |= (a->outs[i].lhs_slice != NULL);
    }
    for (int i = 0; i < b->nouts; i++)
    {
        bad |= (b->outs[i].lhs_slice != NULL);
        for (int j = 0; j < a->nouts; j++)
        {
            bad |= (b->outs[i].slot_idx == a->outs[j].slot_idx);
        }
        for (int j = 0; j < a->nops; j++)
        {
            bad |= (!a->ops[j].is_out && a->ops[j].slot_idx == b->outs[i].slot_idx);
        }
    }
    for (int i = 0; i < b->nops; i++)
    {
        bad |= (!b->ops[i].is_out && b->ops[i].is_expr_im);
        for (int j = 0; b->ops[i].is_scalar && j < a->nouts; j++)
        {
            bad |= mc_expr_mentions_self(b->ops[i].scalar_expr,
                                         prog->slots[a->outs[j].slot_idx].bare, NULL);
        }
        for (int j = 0; !b->ops[i].is_out && j < a->nouts; j++)
        {
            bad |= (b->ops[i].slot_idx == a->outs[j].slot_idx &&
                    (b->ops[i].is_non_pointwise || b->ops[i].has_slice ||
                     a->outs[j].assign_op != MC_TOK_ASSIGN || a->outs[j].lhs_slice != NULL));
        }
    }
    return !bad;
}

/**
 * mc_has_bcast_stencil - Check if an expression applies `shift`/`roll` to a broadcast input
 * @e:   Expression AST node to inspect
 * @dsz: Destination 3D shape `[3]` of the enclosing multi-assignment
 *
 * `shift`/`roll` wrap or clamp coordinates against the input operand's own
 * view dimensions (`_vnx`, `_vny`, `_vnz`), so fusing a multi-assignment where
 * one RHS shifts a smaller broadcast image would use the wrong wrap bounds.
 *
 * Return: 1 if @e contains a `shift`/`roll` on an image whose shape != @dsz, 0 otherwise
 */
int mc_has_bcast_stencil(const mc_expr_t *e, const uint32_t dsz[3])
{
    int r = (e != NULL) &&
            ((e->kind == MC_EXPR_UNARY && mc_has_bcast_stencil(e->u.unary.operand, dsz)) ||
             (e->kind == MC_EXPR_BINARY && (mc_has_bcast_stencil(e->u.binary.left, dsz) ||
                                            mc_has_bcast_stencil(e->u.binary.right, dsz))) ||
             (e->kind == MC_EXPR_TERNARY && (mc_has_bcast_stencil(e->u.ternary.cond, dsz) ||
                                             mc_has_bcast_stencil(e->u.ternary.t_expr, dsz) ||
                                             mc_has_bcast_stencil(e->u.ternary.f_expr, dsz))) ||
             (e->kind == MC_EXPR_CALL && e->u.call.fn != NULL &&
              e->u.call.fn->kind == MC_FN_SHIFT && e->u.call.nargs >= 1 &&
              memcmp(e->u.call.args[0]->res_size, dsz, 3 * sizeof(uint32_t)) != 0));
    for (int i = 0; !r && e != NULL && e->kind == MC_EXPR_CALL && i < e->u.call.nargs; i++)
    {
        r |= mc_has_bcast_stencil(e->u.call.args[i], dsz);
    }
    return r;
}

/**
 * mc_elide_unused_locals - Mark dead `local` intermediate outputs as register-only
 * @prog: Planned program after pass fusion
 *
 * For every multi-output pass, checks whether each non-final output slot
 * declared with `local` (that did not exist prior to planning) is ever referenced
 * by any other step in @prog. If not, sets `pa->outs[i].is_elided = 1` and
 * `slot->is_elided = 1` so `mc_emit_pass()` emits `const float _v<i> = ...;`
 * without allocating or writing a backing `IMAGE` buffer.
 *
 * Return: void
 */
void mc_elide_unused_locals(mcalc_prog_t *prog)
{
    for (int st = 0; st < prog->nsteps; st++)
    {
        mc_pass_t *pa = (prog->steps[st].kind == MC_STEP_PASS && prog->steps[st].pass.nouts > 1)
                            ? &prog->steps[st].pass
                            : NULL;
        for (int i = 0; pa != NULL && i < pa->nouts - 1; i++)
        {
            int        sidx       = pa->outs[i].slot_idx;
            int        used_later = 0;
            mc_slot_t *slot       = &prog->slots[sidx];
            if (!slot->is_local || slot->existed_at_plan ||
                pa->outs[i].assign_op != MC_TOK_ASSIGN || pa->outs[i].lhs_slice != NULL)
            {
                continue;
            }
            for (int st2 = 0; st2 < prog->nsteps && !used_later; st2++)
            {
                const mc_step_t *os = &prog->steps[st2];
                if (st2 == st)
                {
                    continue;
                }
                if (os->kind == MC_STEP_PASS)
                {
                    for (int k = 0; !used_later && k < os->pass.nops; k++)
                    {
                        used_later |= (os->pass.ops[k].slot_idx == sidx);
                    }
                }
                else if (os->kind == MC_STEP_CTOR)
                {
                    used_later |= (os->ctor.slot_idx == sidx || os->ctor.multi_stmt != NULL ||
                                   mc_expr_mentions_self(os->ctor.rhs_expr, slot->bare, NULL));
                }
                else if (os->scalar_stmt != NULL && os->scalar_stmt->kind == MC_STMT_EXPR)
                {
                    used_later |= mc_expr_mentions_self(os->scalar_stmt->u.expr, slot->bare, NULL);
                }
                else if (os->scalar_stmt != NULL && os->scalar_stmt->kind == MC_STMT_MULTI_ASSIGN)
                {
                    for (int k = 0; !used_later && k < os->scalar_stmt->u.multi.nrhs; k++)
                    {
                        used_later |= mc_expr_mentions_self(os->scalar_stmt->u.multi.rhs[k],
                                                            slot->bare, NULL);
                    }
                }
                else
                {
                    used_later = 1;
                }
            }
            if (!used_later)
            {
                pa->outs[i].is_elided = 1;
                slot->is_elided       = 1;
            }
        }
    }
}

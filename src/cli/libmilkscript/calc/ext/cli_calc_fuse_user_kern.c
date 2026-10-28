// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_user_kern.c
 * @brief Compilation and execution of Tier 1 stateless user C kernels (`MILK_EXT_BLK1`,
 *        `MILK_EXT_BLK2`, `MILK_EXT_STENCIL`, `MILK_EXT_IMG`).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "fuse/cli_calc_fuse_internal.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "prog/cli_calc_prog_internal.h"

/**
 * cli_calc_fuse_user_blk_f - Execute a 1- or 2-input float32 user block kernel (`FI_USER_BLK`)
 */
void cli_calc_fuse_user_blk_f(const fprog *p,
                              int          u,
                              int          len,
                              float *restrict l,
                              const float *restrict r)
{
    const fuse_user_call_t      *uc = &p->ucall[u];
    const milk_calc_ext_entry_t *e  = cli_calc_ext_get(uc->ext_idx);
    MILK_ALIGNED(64) float       tmp[FUSE_BLK];

    if (e == NULL)
    {
        return;
    }
    if (r != NULL && e->blk2_fn != NULL)
    {
        e->blk2_fn(tmp, l, r, uc->pbuf, len);
        __builtin_memcpy(l, tmp, (size_t) len * sizeof(float));
    }
    else if (e->blk1_fn != NULL)
    {
        e->blk1_fn(tmp, l, uc->pbuf, len);
        __builtin_memcpy(l, tmp, (size_t) len * sizeof(float));
    }
}

/**
 * cli_calc_fuse_user_blk_d - Execute a 1- or 2-input user block kernel on a float64 stack
 */
void cli_calc_fuse_user_blk_d(const fprog *p,
                              int          u,
                              int          len,
                              double *restrict l,
                              const double *restrict r)
{
    const fuse_user_call_t      *uc = &p->ucall[u];
    const milk_calc_ext_entry_t *e  = cli_calc_ext_get(uc->ext_idx);
    MILK_ALIGNED(64) float       in0[FUSE_BLK];
    MILK_ALIGNED(64) float       in1[FUSE_BLK];
    MILK_ALIGNED(64) float       out[FUSE_BLK];

    if (e == NULL)
    {
        return;
    }
    _Pragma("omp simd") for (int j = 0; j < len; j++)
    {
        in0[j] = (float) l[j];
    }
    if (r != NULL && e->blk2_fn != NULL)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            in1[j] = (float) r[j];
        }
        e->blk2_fn(out, in0, in1, uc->pbuf, len);
    }
    else if (e->blk1_fn != NULL)
    {
        e->blk1_fn(out, in0, uc->pbuf, len);
    }
    else
    {
        return;
    }
    _Pragma("omp simd") for (int j = 0; j < len; j++)
    {
        l[j] = (double) out[j];
    }
}

/**
 * cli_calc_fuse_user_stencil_f - Execute a 2D row-strip user stencil on a float32 image
 */
void cli_calc_fuse_user_stencil_f(const fprog *p,
                                  int          k,
                                  int          u,
                                  uint64_t     off,
                                  int          len,
                                  float *restrict dst)
{
    const fuse_user_call_t      *uc = &p->ucall[u];
    const milk_calc_ext_entry_t *e  = cli_calc_ext_get(uc->ext_idx);
    if (e == NULL || e->stencil_fn == NULL)
    {
        return;
    }
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const int              rad = e->radius;
    const int              K   = 2 * rad + 1;
    const float           *a   = im->array.F;
    MILK_ALIGNED(64) float col[7][FUSE_BLK + 6];
    const float           *rows[7] = { col[0], col[1], col[2], col[3], col[4], col[5], col[6] };

    VIEW_WALK(p, v, off, len, {
        const int W = N + K - 1;
        for (int r = 0; r < K; r++)
        {
            int64_t sy       = (int64_t) y_ + (r - rad);
            sy               = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy);
            const float *row = a + ((int64_t) z_ * ny + sy) * nx;
            float       *cr  = col[r];
            for (int x = 0; x < W; x++)
            {
                int64_t sx = (int64_t) x_ + x - rad;
                sx         = (sx < 0) ? 0 : ((sx >= nx) ? nx - 1 : sx);
                cr[x]      = row[sx];
            }
        }
        e->stencil_fn(dst + J, rows, uc->pbuf, N);
    });
}

/**
 * cli_calc_fuse_user_stencil_d - Execute a 2D row-strip user stencil on any image datatype
 */
void cli_calc_fuse_user_stencil_d(const fprog *p,
                                  int          k,
                                  int          u,
                                  uint64_t     off,
                                  int          len,
                                  double *restrict dst)
{
    const fuse_user_call_t      *uc = &p->ucall[u];
    const milk_calc_ext_entry_t *e  = cli_calc_ext_get(uc->ext_idx);
    if (e == NULL || e->stencil_fn == NULL)
    {
        return;
    }
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const int              rad = e->radius;
    const int              K   = 2 * rad + 1;
    uint8_t                dt  = im->md[0].datatype;
    MILK_ALIGNED(64) float col[7][FUSE_BLK + 6];
    MILK_ALIGNED(64) float tmp_out[FUSE_BLK];
    const float           *rows[7] = { col[0], col[1], col[2], col[3], col[4], col[5], col[6] };

#define USER_STENCIL_LOAD_BODY(MBR)                                     \
    {                                                                   \
        const int W = N + K - 1;                                        \
        for (int r = 0; r < K; r++)                                     \
        {                                                               \
            int64_t sy = (int64_t) y_ + (r - rad);                      \
            sy         = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy);     \
            const __typeof__(im->array.MBR[0]) *row =                   \
                im->array.MBR + ((int64_t) z_ * ny + sy) * nx;          \
            float *cr = col[r];                                         \
            for (int x = 0; x < W; x++)                                 \
            {                                                           \
                int64_t sx = (int64_t) x_ + x - rad;                    \
                sx         = (sx < 0) ? 0 : ((sx >= nx) ? nx - 1 : sx); \
                cr[x]      = (float) row[sx];                           \
            }                                                           \
        }                                                               \
        e->stencil_fn(tmp_out, rows, uc->pbuf, N);                      \
        double *out_d = dst + J;                                        \
        _Pragma("omp simd") for (int i = 0; i < N; i++)                 \
        {                                                               \
            out_d[i] = (double) tmp_out[i];                             \
        }                                                               \
    }
    VIEW_WALK(p, v, off, len,
              { MILK_FOR_EACH_DATATYPE(dt, USER_STENCIL_LOAD_BODY, USER_STENCIL_LOAD_BODY(D)) });
#undef USER_STENCIL_LOAD_BODY
}

/**
 * cli_calc_fuse_emit_default_user_param - Bind an unbound kernel/operator parameter to
 * `<label>.<name>` or its default
 */
int cli_calc_fuse_emit_default_user_param(fprog                        *p,
                                          fuse_user_call_t             *uc,
                                          int                           j,
                                          const milk_calc_param_spec_t *sp)
{
    int vid      = -1;
    uc->pvals[j] = cli_calc_ext_resolve_param(uc->label, sp, &vid);
    uc->ps1[j]   = 0;

    if (cli_calc_fuse_get_keep_vars())
    {
        char vname[128];
        snprintf(vname, sizeof(vname), "%s.%s", uc->label, sp->name);
        if (vid < 0)
        {
            vid = (int) create_variable_ID(vname, sp->def_num);
        }
        if (vid >= 0)
        {
            fval sv;
            if (emit_scalar(p,
                            (fsins) { .k     = FI_SVAR,
                                      .vid   = vid,
                                      .vhash = name_hash32(data.core.variable[vid].name) },
                            &sv) != 0)
            {
                return -1;
            }
            uc->ps1[j] = (int16_t) sv.s1;
        }
    }
    return 0;
}

/**
 * compile_user_params - Parse optional positional or `kw = expr` scalar arguments after image args
 */
static int compile_user_params(fprog                       *p,
                               const cli_token             *ft,
                               const milk_calc_ext_entry_t *e,
                               int                         *out_u)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();

    if (p->nuser >= FUSE_MAXUSER)
    {
        cli_calc_fuse_reject("more than 8 user kernels (FUSE_MAXUSER)");
        return -1;
    }
    int               u  = p->nuser++;
    fuse_user_call_t *uc = &p->ucall[u];
    memset(uc, 0, sizeof(*uc));
    uc->ext_idx = (int) ft->val_l;

    char func_name[64];
    cli_calc_ext_split_call(ft->sval, func_name, sizeof(func_name), uc->label, sizeof(uc->label));

    const milk_calc_param_set_t *pset = (e->get_params != NULL) ? e->get_params() : NULL;
    uc->nparams = (pset != NULL && pset->nparams <= MILK_CALC_MAX_PARAMS) ? pset->nparams : 0;

    int bound[MILK_CALC_MAX_PARAMS] = { 0 };
    int seen_kw                     = 0;
    int pos_idx                     = 0;

    while (cur()->type == TOK_COMMA)
    {
        adv();
        int        save_pos = cli_calc_tok_pos();
        cli_token *kt       = cur();
        int        pidx     = -1;

        if (kt->type == TOK_VAR || kt->type == TOK_NVAR || kt->type == TOK_IMAGE)
        {
            char kw[64];
            snprintf(kw, sizeof(kw), "%s", kt->sval);
            adv();
            if (cur()->type == TOK_EQUAL)
            {
                adv();
                seen_kw = 1;
                if (pset != NULL)
                {
                    for (int j = 0; j < uc->nparams; j++)
                    {
                        if (strcmp(pset->specs[j].name, kw) == 0)
                        {
                            pidx = j;
                            break;
                        }
                    }
                }
                if (pidx < 0)
                {
                    return -1;
                }
            }
            else
            {
                cli_calc_tok_seek(save_pos);
            }
        }
        if (pidx < 0)
        {
            if (seen_kw || pos_idx >= uc->nparams)
            {
                return -1;
            }
            pidx = pos_idx++;
        }
        fval sv;
        if (cli_calc_fuse_c_expr(p, 0, &sv) != 0 || sv.vec)
        {
            return -1;
        }
        uc->pvals[pidx] = sv.c;
        uc->ps1[pidx]   = (int16_t) sv.s1;
        bound[pidx]     = 1;
    }

    if (cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();

    for (int j = 0; j < uc->nparams; j++)
    {
        if (!bound[j] && cli_calc_fuse_emit_default_user_param(p, uc, j, &pset->specs[j]) != 0)
        {
            return -1;
        }
    }
    cli_calc_ext_pack_params(pset, uc->pbuf, uc->pvals);
    *out_u = u;
    return 0;
}

/**
 * compile_user_img_prestage - Compile a `MILK_EXT_IMG` whole-image pre-stage kernel into @p
 */
static int compile_user_img_prestage(fprog                       *p,
                                     const cli_token             *ft,
                                     const milk_calc_ext_entry_t *e,
                                     fval                        *out)
{
    int slot = p->nax;
    if (slot >= FUSE_MAXAX)
    {
        return -1;
    }
    fprog *sub = calloc(1, sizeof(fprog));
    if (sub == NULL)
    {
        return -1;
    }
    int  saved_anchor = cli_calc_fuse_anchor_save();
    fval sv;
    int  rc = cli_calc_fuse_compile(sub, 0, &sv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    int u = -1;
    if (rc != 0 || !sv.vec || cli_calc_fuse_finalize_views(sub) != 0 ||
        compile_user_params(p, ft, e, &u) != 0)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->ax_p[slot]    = sub;
    p->ax_p2[slot]   = NULL;
    p->ax_kind[slot] = AX_KIND_USER;
    p->ax_axis[slot] = u;
    if (cli_calc_fuse_axred_setup(p, slot, sub->inaxis, sub->ishape, _DATATYPE_FLOAT) != 0)
    {
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->nax = slot + 1;
    int k  = cli_calc_fuse_add_operand(p, &p->ax_im[slot], NULL);
    if (k < 0)
    {
        p->nax        = slot;
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->view[k].internal = 1;
    p->out_dt           = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_IMG, .img = k }, 1);
}

/**
 * cli_calc_fuse_c_user - Compile a Tier 1 (`BLK1`, `BLK2`, `STENCIL`, `IMG`) call into @p
 */
int cli_calc_fuse_c_user(fprog *p, const cli_token *ft, fval *out)
{
    const milk_calc_ext_entry_t *e = cli_calc_ext_get((int) ft->val_l);
    cli_token *(*cur)(void)        = cli_calc_cur_func();
    cli_token *(*adv)(void)        = cli_calc_advance_func();

    if (e == NULL)
    {
        return -1;
    }
    if (e->kind == MILK_EXT_BLK1)
    {
        fval v0;
        int  u = -1;
        if (cli_calc_fuse_c_expr(p, 0, &v0) != 0 || !v0.vec ||
            compile_user_params(p, ft, e, &u) != 0)
        {
            return -1;
        }
        p->out_dt = 0;
        out->vec  = 1;
        out->s1   = 0;
        return emit(p, (fins) { .k = FI_USER_BLK, .img = u, .cpos = 1 }, 0);
    }
    if (e->kind == MILK_EXT_BLK2)
    {
        fval v0;
        fval v1;
        int  u = -1;
        if (cli_calc_fuse_c_expr(p, 0, &v0) != 0 || !v0.vec || cur()->type != TOK_COMMA)
        {
            return -1;
        }
        adv();
        if (cli_calc_fuse_c_expr(p, 0, &v1) != 0 || !v1.vec ||
            compile_user_params(p, ft, e, &u) != 0)
        {
            return -1;
        }
        p->out_dt = 0;
        out->vec  = 1;
        out->s1   = 0;
        return emit(p, (fins) { .k = FI_USER_BLK, .img = u, .cpos = 2 }, -1);
    }
    if (e->kind == MILK_EXT_STENCIL)
    {
        int k = -1;
        int u = -1;
        if (cli_calc_fuse_c_leaf_or_eval_image(p, 0, &k) != 0 ||
            compile_user_params(p, ft, e, &u) != 0)
        {
            return -1;
        }
        p->view[k].is_shift = 1;
        p->out_dt           = 0;
        p->nops++;
        out->vec = 1;
        out->s1  = 0;
        return emit(p, (fins) { .k = FI_USER_STENCIL, .img = k, .cpos = u }, 1);
    }
    if (e->kind == MILK_EXT_IMG)
    {
        return compile_user_img_prestage(p, ft, e, out);
    }
    return -1;
}

/**
 * cli_calc_fuse_run_user_prestage - Execute an `AX_KIND_USER` (`MILK_EXT_IMG`) pre-stage slot
 */
int cli_calc_fuse_run_user_prestage(fprog *p, int slot)
{
    fprog                       *sub = p->ax_p[slot];
    IMAGE                       *res = &p->ax_im[slot];
    int                          u   = p->ax_axis[slot];
    const fuse_user_call_t      *uc  = &p->ucall[u];
    const milk_calc_ext_entry_t *e   = cli_calc_ext_get(uc->ext_idx);

    if (sub == NULL || e == NULL || e->img_fn == NULL)
    {
        return -1;
    }

    /* Fast path when the input argument is a bare float32 image (zero intermediate copy) */
    if (sub->nops == 0 && sub->nimg == 1 && !sub->view[0].is_view &&
        sub->img[0]->md[0].datatype == _DATATYPE_FLOAT)
    {
        if (cli_calc_fuse_refresh_views(sub, NULL, NULL) != 0 ||
            cli_calc_fuse_finalize_views(sub) != 0)
        {
            return -1;
        }
        return (e->img_fn(res, sub->img[0], uc->pbuf) == 0) ? 0 : -1;
    }

    uint64_t nel  = res->md[0].nelement;
    size_t   need = (size_t) nel * sizeof(float);
    if (sub->alias_cap < need)
    {
        void *na = realloc(sub->alias_buf, need);
        if (na == NULL)
        {
            return -1;
        }
        sub->alias_buf = na;
        sub->alias_cap = need;
    }
    IMAGE          src_im;
    IMAGE_METADATA src_md;
    memset(&src_im, 0, sizeof(src_im));
    src_md           = res->md[0];
    src_md.datatype  = _DATATYPE_FLOAT;
    src_im.md        = &src_md;
    src_im.used      = 1;
    src_im.array.raw = sub->alias_buf;

    if (cli_calc_fuse_run_program(sub, &src_im) != 0)
    {
        return -1;
    }
    return (e->img_fn(res, &src_im, uc->pbuf) == 0) ? 0 : -1;
}

/**
 * cli_calc_ext_split_call - Split `"func("` or `"func:label("` into function name and FPS label
 */
void cli_calc_ext_split_call(const char *sval,
                             char       *func_out,
                             size_t      fsz,
                             char       *label_out,
                             size_t      lsz)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s", (sval != NULL) ? sval : "");
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '(')
    {
        buf[--len] = '\0';
    }
    char *colon = strchr(buf, ':');
    if (colon != NULL)
    {
        *colon = '\0';
        if (func_out != NULL && fsz > 0)
        {
            snprintf(func_out, fsz, "%s", buf);
        }
        if (label_out != NULL && lsz > 0)
        {
            snprintf(label_out, lsz, "%s", (colon[1] != '\0') ? (colon + 1) : buf);
        }
    }
    else
    {
        if (func_out != NULL && fsz > 0)
        {
            snprintf(func_out, fsz, "%s", buf);
        }
        if (label_out != NULL && lsz > 0)
        {
            snprintf(label_out, lsz, "%s", buf);
        }
    }
}

/**
 * cli_calc_ext_pack_params - Write scalar parameter values @vals into typed struct @pbuf
 */
void cli_calc_ext_pack_params(const milk_calc_param_set_t *pset, void *pbuf, const double *vals)
{
    if (pset == NULL || pbuf == NULL || vals == NULL)
    {
        return;
    }
    char *base = (char *) pbuf;
    int   n    = (pset->nparams < MILK_CALC_MAX_PARAMS) ? pset->nparams : MILK_CALC_MAX_PARAMS;
    for (int i = 0; i < n; i++)
    {
        const milk_calc_param_spec_t *sp  = &pset->specs[i];
        void                         *dst = base + sp->offset;
        double                        v   = vals[i];
        if (sp->size > 8 || sp->offset + sp->size > MILK_CALC_PARAM_BYTES)
        {
            continue; /* registration refuses such tables; never write past pbuf */
        }
        switch (sp->fptype)
        {
        case FPTYPE_FLOAT64:
            *(double *) dst = v;
            break;
        case FPTYPE_FLOAT32:
            *(float *) dst = (float) v;
            break;
        case FPTYPE_INT64:
            *(int64_t *) dst = (int64_t) llround(v);
            break;
        case FPTYPE_UINT64:
            *(uint64_t *) dst = (uint64_t) ((v > 0.0) ? llround(v) : 0);
            break;
        case FPTYPE_INT32:
            *(int32_t *) dst = (int32_t) llround(v);
            break;
        case FPTYPE_UINT32:
            *(uint32_t *) dst = (uint32_t) ((v > 0.0) ? llround(v) : 0);
            break;
        case FPTYPE_ONOFF:
            *(int *) dst = (v != 0.0) ? 1 : 0;
            break;
        default:
            break;
        }
    }
}

/**
 * cli_calc_ext_resolve_param - Look up calculator variable `<label>.<param>` or fall back to C default
 */
double cli_calc_ext_resolve_param(const char                   *label,
                                  const milk_calc_param_spec_t *spec,
                                  int                          *out_vid)
{
    char vname[128];
    snprintf(vname, sizeof(vname), "%s.%s", label, spec->name);
    long vid = variable_ID(vname);
    if (vid >= 0)
    {
        if (out_vid != NULL)
        {
            *out_vid = (int) vid;
        }
        const VARIABLE *v = &data.core.variable[vid];
        return (v->type == 1) ? (double) v->value.l : v->value.f;
    }
    if (out_vid != NULL)
    {
        *out_vid = -1;
    }
    return spec->def_num;
}

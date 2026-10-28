// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_ops_spatial.c
 * @brief Bytecode compilers for spatial stencils, median filters, LUTs, remap/unmap, and Gaussian.
 *
 * Role & Goal:
 *   Compiles spatial and table-driven operations (`shift`, `medfilt3/5/7`, `pixremap`,
 *   `pixunmap`, `rprof`, `gauss`, `interp2`, `lut`) into fused vector instructions or
 *   pre-stage evaluation slots.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "cli_calc_fuse_internal.h"

#define axis_result_dt cli_calc_fuse_axis_result_dt
#define axred_setup cli_calc_fuse_axred_setup

/**
 * cli_calc_fuse_c_shift - Compile `shift(im, dx[, dy])` 2D edge-clamped stencil load
 * @p:   Target fused program
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_shift(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    cli_token *t            = adv();
    int        k            = -1;

    if (t->type == TOK_VAR || t->type == TOK_IMAGE)
    {
        k = cli_calc_fuse_udef_param_img(t->sval);
    }
    if (k < 0 && t->type == TOK_IMAGE && strchr(t->sval, '[') == NULL)
    {
        imageID id = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
        k          = (id == -1) ? -1 : cli_calc_fuse_add_image(p, &data.core.image[id]);
    }
    if (k < 0 || cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    fval dx;
    fval dy = { 0, 0, 0.0, 1 };
    if (cli_calc_fuse_c_expr(p, 0, &dx) != 0 || dx.vec)
    {
        return -1;
    }
    if (cur()->type == TOK_COMMA)
    {
        adv();
        if (cli_calc_fuse_c_expr(p, 0, &dy) != 0 || dy.vec)
        {
            return -1;
        }
    }
    if (cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    p->view[k].is_shift = 1;
    p->out_dt           = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(
        p, (fins) { .k = FI_SHIFT, .img = k, .s1 = dx.s1, .c = dx.c, .s2 = dy.s1, .c2 = dy.c }, 1);
}

/** c_const_scalar_arg - Evaluate a compile-time constant scalar argument without mutating `p`. */
static int c_const_scalar_arg(double *out)
{
    int pos = cli_calc_tok_pos();
    if (cli_calc_eval_const_scalar(out) != 0)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }
    return 0;
}

/**
 * cli_calc_fuse_c_leaf_or_eval_image - Resolve leaf image or compile an `AX_KIND_EVAL` pre-stage
 * @p:      Target fused program
 * @as_lut: 1 to register operand as a 1-D lookup table (`is_lut`), 0 as a regular operand
 * @out_k:  Output operand index in `p->img[]`
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_leaf_or_eval_image(fprog *p, int as_lut, int *out_k)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int        pos          = cli_calc_tok_pos();
    cli_token *t            = cur();

    if ((t->type == TOK_VAR || t->type == TOK_IMAGE) && strchr(t->sval, '[') == NULL)
    {
        adv();
        if (cur()->type == TOK_COMMA || cur()->type == TOK_RPAREN)
        {
            int uimg = cli_calc_fuse_udef_param_img(t->sval);
            if (uimg >= 0)
            {
                const IMAGE *uim = p->img[uimg];
                *out_k           = as_lut ? cli_calc_fuse_add_lut_image(p, uim) : uimg;
                return (*out_k >= 0) ? 0 : -1;
            }
            if (t->type == TOK_IMAGE)
            {
                imageID id = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
                if (id != -1)
                {
                    *out_k = as_lut ? cli_calc_fuse_add_lut_image(p, &data.core.image[id])
                                    : cli_calc_fuse_add_image(p, &data.core.image[id]);
                    return (*out_k >= 0) ? 0 : -1;
                }
            }
        }
        cli_calc_tok_seek(pos);
    }

    int slot = p->nax;
    if (slot >= FUSE_MAXAX)
    {
        cli_calc_fuse_reject("more than 4 pre-stages (FUSE_MAXAX)");
        return -1;
    }
    fprog *sub = (fprog *) calloc(1, sizeof(fprog));
    if (sub == NULL)
    {
        return -1;
    }
    int  saved_anchor = cli_calc_fuse_anchor_save();
    fval sv;
    int  rc = cli_calc_fuse_compile(sub, 0, &sv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    if (rc != 0 || !sv.vec || cli_calc_fuse_finalize_views(sub) != 0)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->ax_p[slot]    = sub;
    p->ax_p2[slot]   = NULL;
    p->ax_kind[slot] = AX_KIND_EVAL;
    if (axred_setup(p, slot, sub->inaxis, sub->ishape, axis_result_dt(sub)) != 0)
    {
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->nax = slot + 1;
    int k  = as_lut ? cli_calc_fuse_add_lut_image(p, &p->ax_im[slot])
                    : cli_calc_fuse_add_image(p, &p->ax_im[slot]);
    if (k < 0)
    {
        p->nax        = slot;
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->view[k].internal = 1;
    *out_k              = k;
    return 0;
}
#define c_leaf_or_eval_image cli_calc_fuse_c_leaf_or_eval_image

/** c_pixunmap_map_args - Parse `, map[, nbins])` arguments of `pixunmap`. */
static int c_pixunmap_map_args(const fprog *sub,
                               int          saved_anchor,
                               fprog      **out_map_sub,
                               int         *out_nbins)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    if (cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    fprog *map_sub = (fprog *) calloc(1, sizeof(fprog));
    if (map_sub == NULL)
    {
        return -1;
    }
    fval mv;
    (void) cli_calc_fuse_anchor_save();
    int mrc = cli_calc_fuse_compile(map_sub, 0, &mv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    if (mrc != 0 || !mv.vec || cli_calc_fuse_finalize_views(map_sub) != 0 ||
        map_sub->inel != sub->inel)
    {
        cli_calc_fuse_free_fprog_children(map_sub);
        free(map_sub);
        return -1;
    }
    int nbins = -1;
    if (cur()->type == TOK_COMMA)
    {
        double nd = 0.0;
        adv();
        if (c_const_scalar_arg(&nd) != 0)
        {
            cli_calc_fuse_free_fprog_children(map_sub);
            free(map_sub);
            return -1;
        }
        nbins = (int) llround(nd);
    }
    else if (map_sub->nops == 0 && map_sub->nimg == 1)
    {
        nbins = cli_calc_fuse_map_nbins(map_sub->img[0]);
    }
    if (cur()->type != TOK_RPAREN)
    {
        cli_calc_fuse_free_fprog_children(map_sub);
        free(map_sub);
        return -1;
    }
    adv();
    *out_map_sub = map_sub;
    *out_nbins   = nbins;
    return 0;
}

static void free_pixunmap_subs(fprog *sub, fprog *map_sub)
{
    cli_calc_fuse_free_fprog_children(sub);
    free(sub);
    if (map_sub != NULL)
    {
        cli_calc_fuse_free_fprog_children(map_sub);
        free(map_sub);
    }
}

/** c_pixunmap_slot - Compile `pixunmap` or `rprof` into a pre-stage slot of @p. */
static int c_pixunmap_slot(fprog *p, const cli_token *ft, int *out_slot)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int is_rprof            = (strncmp(ft->sval, "rprof(", 6) == 0);
    int slot                = p->nax;

    if (slot >= FUSE_MAXAX)
    {
        return -1;
    }
    fprog *sub = (fprog *) calloc(1, sizeof(fprog));
    if (sub == NULL)
    {
        return -1;
    }
    int  saved_anchor = cli_calc_fuse_anchor_save();
    fval sv;
    int  rc = cli_calc_fuse_compile(sub, 0, &sv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    if (rc != 0 || !sv.vec || cli_calc_fuse_finalize_views(sub) != 0)
    {
        free_pixunmap_subs(sub, NULL);
        return -1;
    }

    fprog *map_sub = NULL;
    int    nbins   = -1;
    double cx      = 0.0;
    double cy      = 0.0;

    if (is_rprof)
    {
        cx = (double) (sub->ishape[0] / 2);
        cy = (double) ((sub->inaxis > 1) ? (sub->ishape[1] / 2) : 0);
        if (cur()->type == TOK_COMMA)
        {
            adv();
            if (c_const_scalar_arg(&cx) != 0 || cur()->type != TOK_COMMA)
            {
                free_pixunmap_subs(sub, NULL);
                return -1;
            }
            adv();
            if (c_const_scalar_arg(&cy) != 0)
            {
                free_pixunmap_subs(sub, NULL);
                return -1;
            }
        }
        if (cur()->type != TOK_RPAREN)
        {
            free_pixunmap_subs(sub, NULL);
            return -1;
        }
        adv();
        nbins = cli_calc_fuse_rprof_nbins(sub->ishape, cx, cy);
    }
    else if (c_pixunmap_map_args(sub, saved_anchor, &map_sub, &nbins) != 0)
    {
        free_pixunmap_subs(sub, NULL);
        return -1;
    }

    if (nbins <= 0 || nbins > 16777216)
    {
        free_pixunmap_subs(sub, map_sub);
        return -1;
    }

    uint32_t osz[3]       = { (uint32_t) nbins, 1, 1 };
    p->ax_p[slot]         = sub;
    p->ax_p2[slot]        = map_sub;
    p->ax_kind[slot]      = is_rprof ? AX_KIND_RPROF : AX_KIND_SEG;
    p->ax_c[slot][0]      = cx;
    p->ax_c[slot][1]      = cy;
    p->ax_s[slot][0]      = 0;
    p->ax_s[slot][1]      = 0;
    p->ax_csr[slot].valid = 0;
    if (axred_setup(p, slot, 2, osz, axis_result_dt(sub)) != 0)
    {
        p->ax_p[slot]  = NULL;
        p->ax_p2[slot] = NULL;
        free_pixunmap_subs(sub, map_sub);
        return -1;
    }
    p->nax    = slot + 1;
    *out_slot = slot;
    return 0;
}

/**
 * cli_calc_fuse_c_pixunmap - Compile `pixunmap(im, map[, n])` or `rprof(im[, cx, cy])` into @p
 * @p:   Target fused program
 * @ft:  Function token
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_pixunmap(fprog *p, const cli_token *ft, fval *out)
{
    int slot = -1;
    if (c_pixunmap_slot(p, ft, &slot) != 0)
    {
        return -1;
    }
    int k = cli_calc_fuse_add_operand(p, &p->ax_im[slot], NULL);
    if (k < 0)
    {
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
 * cli_calc_fuse_c_medfilt - Compile `medfilt3(im)`, `medfilt5(im)`, or `medfilt7(im)` into @p
 * @p:   Target fused program
 * @ft:  Median filter function token
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_medfilt(fprog *p, const cli_token *ft, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int K                   = (strncmp(ft->sval, "medfilt3(", 9) == 0)   ? 3
                              : (strncmp(ft->sval, "medfilt5(", 9) == 0) ? 5
                                                                         : 7;
    int k                   = -1;
    if (c_leaf_or_eval_image(p, 0, &k) != 0 || cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    p->view[k].is_shift = 1;
    p->out_dt           = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_MEDFILT, .img = k, .cpos = K }, 1);
}

/**
 * cli_calc_fuse_c_pixremap - Compile `pixremap(im, map[, fill])` index-map gather into @p
 * @p:   Target fused program
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_pixremap(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int k                   = -1;
    if (c_leaf_or_eval_image(p, 1, &k) != 0 || cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    fval mv;
    fval fv = { 0, 0, 0.0, 0 };
    if (cli_calc_fuse_c_expr(p, 0, &mv) != 0 || !mv.vec)
    {
        return -1;
    }
    if (cur()->type == TOK_COMMA)
    {
        adv();
        if (cli_calc_fuse_c_expr(p, 0, &fv) != 0 || fv.vec)
        {
            return -1;
        }
    }
    if (cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    if (p->img[k]->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_GATHER, .img = k, .s1 = fv.s1, .c = fv.c }, 0);
}

/**
 * cli_calc_fuse_c_gauss - Compile `gauss(im, sigma[, sigma_y])` separable Gaussian pre-stage
 * @p:   Target fused program
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_gauss(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int slot                = p->nax;
    if (slot >= FUSE_MAXAX)
    {
        return -1;
    }
    fprog *sub = (fprog *) calloc(1, sizeof(fprog));
    if (sub == NULL)
    {
        return -1;
    }
    int  saved_anchor = cli_calc_fuse_anchor_save();
    fval sv;
    int  rc = cli_calc_fuse_compile(sub, 0, &sv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    if (rc != 0 || !sv.vec || cli_calc_fuse_finalize_views(sub) != 0 || cur()->type != TOK_COMMA)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    adv();
    fval sx;
    if (cli_calc_fuse_c_expr(p, 0, &sx) != 0 || sx.vec || (sx.s1 == 0 && sx.c < 0.0))
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    fval sy = sx;
    if (cur()->type == TOK_COMMA)
    {
        adv();
        if (cli_calc_fuse_c_expr(p, 0, &sy) != 0 || sy.vec || (sy.s1 == 0 && sy.c < 0.0))
        {
            cli_calc_fuse_free_fprog_children(sub);
            free(sub);
            return -1;
        }
    }
    if (cur()->type != TOK_RPAREN)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    adv();
    p->ax_p[slot]    = sub;
    p->ax_p2[slot]   = NULL;
    p->ax_kind[slot] = AX_KIND_GAUSS;
    p->ax_c[slot][0] = sx.c;
    p->ax_c[slot][1] = sy.c;
    p->ax_s[slot][0] = sx.s1;
    p->ax_s[slot][1] = sy.s1;
    if (axred_setup(p, slot, sub->inaxis, sub->ishape, axis_result_dt(sub)) != 0)
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
 * cli_calc_fuse_c_cumsum - Compile `cumsum(expr[, axis])` as an `AX_KIND_CUMSUM` pre-stage of @p
 * @p:   Target fused program
 * @out: Output value descriptor
 *
 * The axis is a compile-time constant (0 when omitted); the result keeps the expression's shape.
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_cumsum(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int slot                = p->nax;
    if (slot >= FUSE_MAXAX)
    {
        cli_calc_fuse_reject("more than 4 pre-stages (FUSE_MAXAX)");
        return -1;
    }
    fprog *sub = (fprog *) calloc(1, sizeof(fprog));
    if (sub == NULL)
    {
        return -1;
    }
    int  saved_anchor = cli_calc_fuse_anchor_save();
    fval sv;
    int  rc = cli_calc_fuse_compile(sub, 0, &sv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    double axis = 0.0;
    if (rc == 0 && sv.vec && cli_calc_fuse_finalize_views(sub) == 0 && cur()->type == TOK_COMMA)
    {
        adv();
        if (cli_calc_eval_const_scalar(&axis) != 0)
        {
            rc = -1;
        }
    }
    if (rc != 0 || !sv.vec || cli_calc_fuse_finalize_views(sub) != 0 || cur()->type != TOK_RPAREN ||
        axis < 0.0 || axis >= (double) sub->inaxis)
    {
        cli_calc_fuse_reject("cumsum: the axis must be a constant below the expression's axes");
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    adv();
    p->ax_p[slot]    = sub;
    p->ax_p2[slot]   = NULL;
    p->ax_kind[slot] = AX_KIND_CUMSUM;
    p->ax_c[slot][0] = axis;
    p->ax_c[slot][1] = 0.0;
    p->ax_s[slot][0] = 0;
    p->ax_s[slot][1] = 0;
    if (axred_setup(p, slot, sub->inaxis, sub->ishape, axis_result_dt(sub)) != 0)
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
 * cli_calc_fuse_c_interp2 - Compile `interp2(im, xmap, ymap)` 2D bilinear sampling into @p
 * @p:   Target fused program
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_interp2(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int k                   = -1;
    if (c_leaf_or_eval_image(p, 1, &k) != 0 || cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    fval xv;
    fval yv;
    if (cli_calc_fuse_c_expr(p, 0, &xv) != 0 || !xv.vec || cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    if (cli_calc_fuse_c_expr(p, 0, &yv) != 0 || !yv.vec || cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    if (p->img[k]->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_INTERP2, .img = k }, -1);
}

/**
 * cli_calc_fuse_c_lut - Compile `lut(expr, tbl[, x0, dx])` 1D linear interpolation lookup into @p
 * @p:   Target fused program
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_c_lut(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    fval x;

    if (cli_calc_fuse_c_expr(p, 0, &x) != 0 || !x.vec || cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    cli_token *tt = adv();
    int        k  = -1;
    if (tt->type == TOK_FUNC_PIXUNMAP)
    {
        int slot = -1;
        if (c_pixunmap_slot(p, tt, &slot) == 0)
        {
            k = cli_calc_fuse_add_lut_image(p, &p->ax_im[slot]);
            if (k >= 0)
            {
                p->view[k].internal = 1;
            }
        }
    }
    else if (tt->type == TOK_IMAGE && strchr(tt->sval, '[') == NULL)
    {
        imageID id = image_ID(tt->sval, data.core.image, data.core.NB_MAX_IMAGE);
        k          = (id == -1) ? -1 : cli_calc_fuse_add_lut_image(p, &data.core.image[id]);
    }
    if (k < 0)
    {
        return -1;
    }
    fval c[2] = { { 0, 0, 0.0, 0 }, { 0, 0, 1.0, 0 } };
    if (cur()->type == TOK_COMMA)
    {
        adv();
        if (cli_calc_fuse_c_args(p, 2, c) != 0 || c[0].vec || c[1].vec ||
            (c[1].s1 == 0 && c[1].c == 0.0))
        {
            return -1;
        }
    }
    else if (cur()->type == TOK_RPAREN)
    {
        adv();
    }
    else
    {
        return -1;
    }
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(
        p,
        (fins) { .k = FI_LUT, .img = k, .s1 = c[0].s1, .c = c[0].c, .s2 = c[1].s1, .c2 = c[1].c },
        0);
}

static int add_custom_view_operand(fprog *p, const IMAGE *im, const fview *v)
{
    if (im == NULL || im->md == NULL || !is_real_datatype(im->md[0].datatype))
    {
        return -1;
    }
    for (int i = 0; i < p->nimg; i++)
    {
        if (!p->view[i].is_lut && p->img[i] == im && strcmp(p->view[i].spec, v->spec) == 0)
        {
            return i;
        }
    }
    if (p->nimg >= FUSE_MAXIMG || cli_calc_fuse_merge_shape(p, v->vshape, v->naxis) != 0)
    {
        return -1;
    }
    if (im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    int k      = p->nimg++;
    p->img[k]  = im;
    p->view[k] = *v;
    return k;
}

static int resolve_leaf_or_eval_im(fprog *p, const IMAGE **out_im, int *out_internal)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int        pos          = cli_calc_tok_pos();
    cli_token *t            = cur();

    if ((t->type == TOK_VAR || t->type == TOK_IMAGE) && strchr(t->sval, '[') == NULL)
    {
        adv();
        if (cur()->type == TOK_COMMA || cur()->type == TOK_RPAREN)
        {
            int uimg = cli_calc_fuse_udef_param_img(t->sval);
            if (uimg >= 0)
            {
                *out_im       = p->img[uimg];
                *out_internal = p->view[uimg].internal;
                return 0;
            }
            if (t->type == TOK_IMAGE)
            {
                imageID id = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
                if (id != -1 && is_real_datatype(data.core.image[id].md[0].datatype))
                {
                    *out_im       = &data.core.image[id];
                    *out_internal = 0;
                    return 0;
                }
            }
        }
        cli_calc_tok_seek(pos);
    }

    int slot = p->nax;
    if (slot >= FUSE_MAXAX)
    {
        cli_calc_fuse_reject("more than 4 pre-stages (FUSE_MAXAX)");
        return -1;
    }
    fprog *sub = (fprog *) calloc(1, sizeof(fprog));
    if (sub == NULL)
    {
        return -1;
    }
    int  saved_anchor = cli_calc_fuse_anchor_save();
    fval sv;
    int  rc = cli_calc_fuse_compile(sub, 0, &sv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    if (rc != 0 || !sv.vec || cli_calc_fuse_finalize_views(sub) != 0)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->ax_p[slot]    = sub;
    p->ax_p2[slot]   = NULL;
    p->ax_kind[slot] = AX_KIND_EVAL;
    if (axred_setup(p, slot, sub->inaxis, sub->ishape, axis_result_dt(sub)) != 0)
    {
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->nax        = slot + 1;
    *out_im       = &p->ax_im[slot];
    *out_internal = 1;
    return 0;
}

static int c_transpose(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    const IMAGE *im         = NULL;
    int          is_int     = 0;

    if (resolve_leaf_or_eval_im(p, &im, &is_int) != 0 || cur()->type != TOK_RPAREN ||
        (im->md[0].naxis != 1 && im->md[0].naxis != 2))
    {
        return -1;
    }
    adv();
    uint32_t M = im->md[0].size[0];
    uint32_t N = (im->md[0].naxis == 2) ? im->md[0].size[1] : 1;
    fview    v;
    memset(&v, 0, sizeof(v));
    v.vshape[0] = N;
    v.vshape[1] = M;
    v.vshape[2] = 1;
    v.naxis     = 2;
    v.stride[0] = (N == 1) ? 0 : (int64_t) M;
    v.stride[1] = (M == 1) ? 0 : 1;
    v.stride[2] = 0;
    v.is_view   = 1;
    v.internal  = is_int;
    snprintf(v.spec, sizeof(v.spec), "@T");
    int k = add_custom_view_operand(p, im, &v);
    if (k < 0)
    {
        return -1;
    }
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_IMG, .img = k }, 1);
}

static int c_outer(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    const IMAGE *imX        = NULL;
    const IMAGE *imY        = NULL;
    int          intX       = 0;
    int          intY       = 0;

    if (resolve_leaf_or_eval_im(p, &imX, &intX) != 0 || cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    if (resolve_leaf_or_eval_im(p, &imY, &intY) != 0 || cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    uint64_t M = 0;
    uint64_t N = 0;
    if (!arith_image_is_1d_vec(imX, &M) || !arith_image_is_1d_vec(imY, &N))
    {
        return -1;
    }
    fview vx;
    fview vy;
    memset(&vx, 0, sizeof(vx));
    memset(&vy, 0, sizeof(vy));
    vx.vshape[0] = (uint32_t) M;
    vx.vshape[1] = 1;
    vx.vshape[2] = 1;
    vx.naxis     = 2;
    vx.stride[0] = (M == 1) ? 0 : 1;
    vx.is_view   = 1;
    vx.internal  = intX;
    snprintf(vx.spec, sizeof(vx.spec), "@OX");

    vy.vshape[0] = 1;
    vy.vshape[1] = (uint32_t) N;
    vy.vshape[2] = 1;
    vy.naxis     = 2;
    vy.stride[1] = (N == 1) ? 0 : 1;
    vy.is_view   = 1;
    vy.internal  = intY;
    snprintf(vy.spec, sizeof(vy.spec), "@OY");

    int kx = add_custom_view_operand(p, imX, &vx);
    int ky = (kx >= 0) ? add_custom_view_operand(p, imY, &vy) : -1;
    if (ky < 0)
    {
        return -1;
    }
    if (emit(p, (fins) { .k = FI_IMG, .img = kx }, 1) != 0 ||
        emit(p, (fins) { .k = FI_IMG, .img = ky }, 1) != 0 ||
        cli_calc_fuse_emit_bin(p, (fins) { .k = FI_BIN, .op = TOK_OP_STAR, .cpos = 0 }, 1) != 0)
    {
        return -1;
    }
    p->out_dt = 0;
    out->vec  = 1;
    out->s1   = 0;
    return 0;
}

static int peek_top_transpose(void)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    if (cur()->type != TOK_FUNC_IMIM_IM || strcmp(cur()->sval, "transpose(") != 0)
    {
        return 0;
    }
    int saved = cli_calc_tok_pos();
    adv();
    int depth = 1;
    while (depth > 0 && cur()->type != TOK_EOF && cur()->type != TOK_NEWLINE)
    {
        cli_token_type t = cur()->type;
        if (t == TOK_LPAREN ||
            (t >= TOK_FUNC_D_D && t <= TOK_FUNC_COORD && strchr(cur()->sval, '(') != NULL))
        {
            depth++;
        }
        else if (t == TOK_RPAREN)
        {
            depth--;
        }
        adv();
    }
    int ok = (depth == 0 && (cur()->type == TOK_COMMA || cur()->type == TOK_RPAREN));
    cli_calc_tok_seek(saved);
    return ok;
}

static int parse_matmul_fuse_arg(fprog *p, int *out_k, int *out_trans)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int trans               = 0;

    if (peek_top_transpose())
    {
        adv();
        trans = 1;
    }
    if (c_leaf_or_eval_image(p, 1, out_k) != 0)
    {
        return -1;
    }
    if (trans)
    {
        if (cur()->type != TOK_RPAREN)
        {
            return -1;
        }
        adv();
    }
    *out_trans = trans;
    return 0;
}

static int c_matmul(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    if (p->nmatmul >= FUSE_MAXMATMUL)
    {
        cli_calc_fuse_reject("more than 4 matmul operations (FUSE_MAXMATMUL)");
        return -1;
    }
    int kA     = -1;
    int kB     = -1;
    int transA = 0;
    int transB = 0;
    if (parse_matmul_fuse_arg(p, &kA, &transA) != 0 || cur()->type != TOK_COMMA)
    {
        return -1;
    }
    adv();
    if (parse_matmul_fuse_arg(p, &kB, &transB) != 0 || cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();

    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (arith_image_matmul_dims(p->img[kA], transA, p->img[kB], transB, &tA, &tB, &M, &K, &N, &onax,
                                outsz) != 0 ||
        cli_calc_fuse_merge_shape(p, outsz, (int) onax) != 0)
    {
        return -1;
    }
    if (p->img[kA]->md[0].datatype == _DATATYPE_DOUBLE ||
        p->img[kB]->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    int m = p->nmatmul++;
    p->matmul[m] =
        (fuse_matmul_t) { .imgA = kA, .imgB = kB, .tA = tA, .tB = tB, .M = M, .K = K, .N = N };
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_MATMUL, .cpos = m }, 1);
}

static int parse_reshape_dims(uint64_t nel, uint32_t osz[3], int *out_ndims)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int      ndims          = 0;
    int      neg_idx        = -1;
    uint64_t prod           = 1;
    long     dims[3]        = { 1, 1, 1 };

    while (cur()->type == TOK_COMMA && ndims < 3)
    {
        double dv = 0.0;
        adv();
        if (c_const_scalar_arg(&dv) != 0)
        {
            return -1;
        }
        long d = (long) llround(dv);
        if (d == -1)
        {
            if (neg_idx != -1)
            {
                return -1;
            }
            neg_idx = ndims;
        }
        else if (d <= 0)
        {
            return -1;
        }
        else
        {
            dims[ndims] = d;
            prod *= (uint64_t) d;
        }
        ndims++;
    }
    if (ndims < 1 || cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    if (neg_idx != -1)
    {
        if (prod == 0 || nel % prod != 0)
        {
            return -1;
        }
        dims[neg_idx] = (long) (nel / prod);
        prod          = nel;
    }
    if (prod != nel)
    {
        return -1;
    }
    osz[0]     = (uint32_t) dims[0];
    osz[1]     = (uint32_t) dims[1];
    osz[2]     = (uint32_t) dims[2];
    *out_ndims = ndims;
    return 0;
}

static int c_reshape(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int pos                 = cli_calc_tok_pos();

    if (cur()->type == TOK_FUNC_IMIM_IM && strcmp(cur()->sval, "matmul(") == 0)
    {
        uint32_t saved_ishape[3];
        memcpy(saved_ishape, p->ishape, sizeof(saved_ishape));
        int saved_inaxis   = p->inaxis;
        int saved_anchored = p->anchored;
        int n0 = p->n, nimg0 = p->nimg, nax0 = p->nax, nmm0 = p->nmatmul, depth0 = p->depth;
        p->ishape[0] = p->ishape[1] = p->ishape[2] = 1;
        p->inaxis                                  = 0;
        p->anchored                                = 0;
        adv();
        if (c_matmul(p, out) == 0 && cur()->type == TOK_COMMA)
        {
            uint64_t nel    = (uint64_t) p->ishape[0] * p->ishape[1] * p->ishape[2];
            uint32_t osz[3] = { 1, 1, 1 };
            int      ndims  = 0;
            memcpy(p->ishape, saved_ishape, sizeof(saved_ishape));
            p->inaxis   = saved_inaxis;
            p->anchored = saved_anchored;
            if (parse_reshape_dims(nel, osz, &ndims) == 0 &&
                cli_calc_fuse_merge_shape(p, osz, ndims) == 0)
            {
                return 0;
            }
        }
        memcpy(p->ishape, saved_ishape, sizeof(saved_ishape));
        p->inaxis   = saved_inaxis;
        p->anchored = saved_anchored;
        p->n        = n0;
        p->nimg     = nimg0;
        p->nmatmul  = nmm0;
        p->depth    = depth0;
        for (int a = nax0; a < p->nax; a++)
        {
            cli_calc_fuse_free_fprog_children(p->ax_p[a]);
            free(p->ax_p[a]);
            p->ax_p[a] = NULL;
        }
        p->nax = nax0;
        cli_calc_tok_seek(pos);
    }

    const IMAGE *im     = NULL;
    int          is_int = 0;
    int          nax0   = p->nax;
    if (resolve_leaf_or_eval_im(p, &im, &is_int) != 0)
    {
        return -1;
    }
    uint32_t osz[3] = { 1, 1, 1 };
    int      ndims  = 0;
    if (parse_reshape_dims(im->md[0].nelement, osz, &ndims) != 0)
    {
        return -1;
    }
    if (is_int && p->nax == nax0 + 1 && im == &p->ax_im[nax0])
    {
        p->ax_md[nax0].naxis   = (uint8_t) ndims;
        p->ax_md[nax0].size[0] = osz[0];
        p->ax_md[nax0].size[1] = osz[1];
        p->ax_md[nax0].size[2] = osz[2];
        int k                  = cli_calc_fuse_add_image(p, &p->ax_im[nax0]);
        if (k < 0)
        {
            return -1;
        }
        p->view[k].internal = 1;
        p->out_dt           = 0;
        p->nops++;
        out->vec = 1;
        out->s1  = 0;
        return emit(p, (fins) { .k = FI_IMG, .img = k }, 1);
    }
    fview v;
    memset(&v, 0, sizeof(v));
    v.vshape[0] = osz[0];
    v.vshape[1] = osz[1];
    v.vshape[2] = osz[2];
    v.naxis     = ndims;
    v.stride[0] = 1;
    v.stride[1] = (int64_t) osz[0];
    v.stride[2] = (int64_t) osz[0] * osz[1];
    v.internal  = is_int;
    snprintf(v.spec, sizeof(v.spec), "@R:%ux%ux%u", osz[0], osz[1], osz[2]);
    int k = add_custom_view_operand(p, im, &v);
    if (k < 0)
    {
        return -1;
    }
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = FI_IMG, .img = k }, 1);
}

static int c_diag(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    fprog *sub              = (fprog *) calloc(1, sizeof(fprog));
    if (sub == NULL)
    {
        return -1;
    }
    int  saved_anchor = cli_calc_fuse_anchor_save();
    fval sv;
    int  rc = cli_calc_fuse_compile(sub, 0, &sv);
    cli_calc_fuse_anchor_restore(saved_anchor);
    int64_t kdiag = 0;
    if (rc == 0 && sv.vec && cur()->type == TOK_COMMA)
    {
        double kd = 0.0;
        adv();
        if (c_const_scalar_arg(&kd) != 0)
        {
            rc = -1;
        }
        kdiag = (int64_t) llround(kd);
    }
    if (rc != 0 || !sv.vec || cur()->type != TOK_RPAREN ||
        cli_calc_fuse_restrict_diag(sub, kdiag) != 0)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    adv();

    if (sub->ns == 0 && sub->nax == 0 && sub->nred == 0 && sub->nmatmul == 0 && sub->nuser == 0 &&
        sub->nreg == 0 && p->n + sub->n <= FUSE_MAXINS && p->nimg + sub->nimg <= FUSE_MAXIMG &&
        p->depth + sub->maxdepth + p->max_nreg <= FUSE_MAXSTK)
    {
        int map[FUSE_MAXIMG];
        for (int i = 0; i < sub->nimg; i++)
        {
            map[i] = sub->view[i].is_lut ? cli_calc_fuse_add_lut_image(p, sub->img[i])
                                         : add_custom_view_operand(p, sub->img[i], &sub->view[i]);
            if (map[i] < 0)
            {
                cli_calc_fuse_free_fprog_children(sub);
                free(sub);
                return -1;
            }
        }
        for (int i = 0; i < sub->n; i++)
        {
            fins in = sub->ins[i];
            if (in.k == FI_IMG || in.k == FI_IMGC || in.k == FI_VIMGC || in.k == FI_IMGIMG ||
                in.k == FI_LUT || in.k == FI_GATHER)
            {
                in.img = map[in.img];
                if (in.k == FI_IMGIMG)
                {
                    in.cpos = map[in.cpos];
                }
            }
            p->ins[p->n++] = in;
        }
        if (p->depth + sub->maxdepth > p->maxdepth)
        {
            p->maxdepth = p->depth + sub->maxdepth;
        }
        p->depth += 1;
        p->nops += sub->nops + 1;
        p->out_double = p->out_double || sub->out_double;
        p->out_dt     = 0;
        out->vec      = 1;
        out->s1       = 0;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return 0;
    }

    int slot = p->nax;
    if (slot >= FUSE_MAXAX)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->ax_p[slot]    = sub;
    p->ax_p2[slot]   = NULL;
    p->ax_kind[slot] = AX_KIND_EVAL;
    if (axred_setup(p, slot, sub->inaxis, sub->ishape, axis_result_dt(sub)) != 0)
    {
        p->ax_p[slot] = NULL;
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        return -1;
    }
    p->nax = slot + 1;
    int k  = cli_calc_fuse_add_image(p, &p->ax_im[slot]);
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

static int c_zeros_or_eye(fprog *p, int is_eye, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    uint32_t vs[3]          = { 1, 1, 1 };
    int      ndims          = 0;
    int      max_dims       = is_eye ? 2 : 3;

    while (ndims < max_dims)
    {
        if (ndims > 0)
        {
            if (cur()->type != TOK_COMMA)
            {
                break;
            }
            adv();
        }
        double dv = 0.0;
        if (c_const_scalar_arg(&dv) != 0)
        {
            return -1;
        }
        long d = (long) llround(dv);
        if (d <= 0)
        {
            return -1;
        }
        vs[ndims++] = (uint32_t) d;
    }
    if (ndims < 1 || cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    if (is_eye)
    {
        if (ndims == 1)
        {
            vs[1] = vs[0];
        }
        ndims = 2;
    }
    if (cli_calc_fuse_merge_shape(p, vs, ndims) != 0)
    {
        return -1;
    }
    p->out_dt = 0;
    p->nops++;
    out->vec = 1;
    out->s1  = 0;
    return emit(p, (fins) { .k = is_eye ? FI_EYE : FI_CONST, .c = 0.0 }, 1);
}

/**
 * cli_calc_fuse_c_linalg - Compile linear algebra and tensor shape functions into @p
 * @p:   Target fused program
 * @ft:  Function token (`TOK_FUNC_IMIM_IM`)
 * @out: Output value descriptor
 *
 * Return: 0 on success, -1 if call cannot be fused.
 */
int cli_calc_fuse_c_linalg(fprog *p, const cli_token *ft, fval *out)
{
    if (strcmp(ft->sval, "transpose(") == 0)
    {
        return c_transpose(p, out);
    }
    if (strcmp(ft->sval, "outer(") == 0)
    {
        return c_outer(p, out);
    }
    if (strcmp(ft->sval, "matmul(") == 0)
    {
        return c_matmul(p, out);
    }
    if (strcmp(ft->sval, "reshape(") == 0)
    {
        return c_reshape(p, out);
    }
    if (strcmp(ft->sval, "diag(") == 0)
    {
        return c_diag(p, out);
    }
    if (strcmp(ft->sval, "zeros(") == 0)
    {
        return c_zeros_or_eye(p, 0, out);
    }
    if (strcmp(ft->sval, "eye(") == 0)
    {
        return c_zeros_or_eye(p, 1, out);
    }
    char why[128];
    snprintf(why, sizeof(why), "function %.40s) has no fused form", ft->sval);
    cli_calc_fuse_reject(why);
    return -1;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_kept.c
 * @brief Persistent kept fused programs (`fkept`), torn-frame guard, and kept execution loop.
 *
 * Role & Goal:
 *   Compiles persistent kept statements (`cli_calc_fuse_keep`, `cli_calc_fuse_keep_tok`,
 *   `cli_calc_fuse_keep_var`), validates live image/variable identities across ticks, and
 *   executes kept programs with an optimistic torn-frame retry loop (`cli_calc_fused_run`).
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "CLIcore.h"
#include "libmilkdata/milkdata.h"
#include "cli_calc_fuse_internal.h"

static void record_fprog_images(fprog *fp)
{
    if (fp == NULL)
    {
        return;
    }
    for (int i = 0; i < fp->nimg; i++)
    {
        const IMAGE *im = fp->img[i];
        fview       *vw = &fp->view[i];
        if (vw->internal)
        {
            vw->kept_id = -1;
            continue;
        }
        vw->kept_id = (imageID) (im - data.core.image);
        vw->kept_dt = im->md[0].datatype;
        image_vshape(im, vw->kept_sz);
        snprintf(vw->kept_name, sizeof(vw->kept_name), "%s", im->name);
        vw->last_cnt0 = im->md[0].cnt0;
    }
    for (int r = 0; r < fp->nred; r++)
    {
        record_fprog_images(fp->red_p1[r]);
        record_fprog_images(fp->red_p2[r]);
    }
    for (int a = 0; a < fp->nax; a++)
    {
        record_fprog_images(fp->ax_p[a]);
        record_fprog_images(fp->ax_p2[a]);
    }
}

/**
 * cli_calc_fuse_record_kept_images - Snapshot image identities, shapes, and types into @k
 * @k:   Kept program handle
 * @dst: Primary destination image (or NULL for scalar-only kept statements)
 *
 * Return: void
 */
void cli_calc_fuse_record_kept_images(fkept *k, IMAGE *dst)
{
    k->dst = dst;
    if (dst != NULL)
    {
        k->dst_id  = (imageID) (dst - data.core.image);
        k->dst_dt  = dst->md[0].datatype;
        k->dst_vid = -1;
        image_vshape(dst, k->dst_sz);
        snprintf(k->dst_name, sizeof(k->dst_name), "%s", dst->name);
    }
    else
    {
        k->dst_id = -1;
        k->dst_dt = 0;
    }
    for (int d = 0; d < k->p.nextra_dst; d++)
    {
        IMAGE *ed          = k->p.extra_dst[d];
        k->extra_dst_id[d] = (imageID) (ed - data.core.image);
        k->extra_dst_dt[d] = ed->md[0].datatype;
        image_vshape(ed, k->extra_dst_sz[d]);
        snprintf(k->extra_dst_name[d], sizeof(k->extra_dst_name[d]), "%s", ed->name);
    }
    record_fprog_images(&k->p);
    k->last_gen = 0;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &k->last_check);
}

static int fprog_reads_image(const fprog *fp, const char *name)
{
    if (fp == NULL)
    {
        return 0;
    }
    for (int i = 0; i < fp->nimg; i++)
    {
        if (!fp->view[i].internal && fp->img[i] != NULL && strcmp(fp->img[i]->name, name) == 0)
        {
            return 1;
        }
    }
    for (int r = 0; r < fp->nred; r++)
    {
        if (fprog_reads_image(fp->red_p1[r], name) || fprog_reads_image(fp->red_p2[r], name))
        {
            return 1;
        }
    }
    for (int a = 0; a < fp->nax; a++)
    {
        if (fprog_reads_image(fp->ax_p[a], name) || fprog_reads_image(fp->ax_p2[a], name))
        {
            return 1;
        }
    }
    return 0;
}

/**
 * cli_calc_kept_reads_image - Return 1 if kept program @kept reads image @name (in any sub-program)
 * @kept: Kept program handle (NULL reads nothing)
 * @name: Image name
 *
 * Return: 1 if @name is an operand of the program, 0 otherwise.
 */
int cli_calc_kept_reads_image(const void *kept, const char *name)
{
    const fkept *k = (const fkept *) kept;
    return (k != NULL && name != NULL) ? fprog_reads_image(&k->p, name) : 0;
}

/**
 * cli_calc_kept_describe - Describe in one line how kept program @kept executes each tick
 * @kept: Kept program handle
 * @buf:  Output buffer
 * @sz:   Size of @buf
 *
 * Return: void
 */
void cli_calc_kept_describe(const void *kept, char *buf, size_t sz)
{
    const fkept *k = (const fkept *) kept;
    if (k == NULL)
    {
        snprintf(buf, sz, "evaluated as text");
        return;
    }
    if (k->uop != NULL)
    {
        snprintf(buf, sz, "user C operator, %d output port%s", k->uop->ndst,
                 (k->uop->ndst == 1) ? "" : "s");
        return;
    }
    if (k->m.kind != 0)
    {
        snprintf(buf, sz, "BLAS/LAPACK kernel on %d operand%s", k->m.nimg,
                 (k->m.nimg == 1) ? "" : "s");
        return;
    }
    const fprog *p = &k->p;
    if (k->dst == NULL)
    {
        snprintf(buf, sz, "scalar tape: %d op%s, %d reduction%s, %d pre-stage%s", p->ns,
                 (p->ns == 1) ? "" : "s", p->nred, (p->nred == 1) ? "" : "s", p->nax,
                 (p->nax == 1) ? "" : "s");
        return;
    }
    uint64_t nblk   = 0;
    uint64_t blk_sz = FUSE_BLK;
    int      nth    = 1;
    fprog_choose_grid(p->inel, fprog_effective_nel(p), &nth, &nblk, &blk_sz);
    int n = snprintf(buf, sz,
                     "fused pass over %ux%ux%u: %d vector op%s, %d operand%s, %d reduction%s, "
                     "%d pre-stage%s, %s, %d thread%s",
                     p->ishape[0], p->ishape[1], p->ishape[2], p->nops, (p->nops == 1) ? "" : "s",
                     p->nimg, (p->nimg == 1) ? "" : "s", p->nred, (p->nred == 1) ? "" : "s", p->nax,
                     (p->nax == 1) ? "" : "s",
                     cli_calc_fuse_all_float(p, k->dst) ? "float32" : "double", nth,
                     (nth == 1) ? "" : "s");
    for (int d = 0; d < p->nextra_dst && n > 0 && (size_t) n < sz; d++)
    {
        n += snprintf(buf + n, sz - (size_t) n, "%s%s", (d == 0) ? ", writes " : " ",
                      p->extra_dst[d]->name);
    }
    if (p->nextra_dst > 0 && n > 0 && (size_t) n < sz)
    {
        snprintf(buf + n, sz - (size_t) n, " %s", k->dst->name);
    }
}

/**
 * cli_calc_kept_free - Release a compiled kept program and all its child sub-programs
 * @kept: Kept program handle
 *
 * Return: void
 */
void cli_calc_kept_free(void *kept)
{
    fkept *k = (fkept *) kept;
    if (k == NULL)
    {
        return;
    }
    cli_calc_fuse_free_user_op(k->uop);
    cli_calc_fuse_free_fprog_children(&k->p);
    free(k);
}

/**
 * cli_calc_fuse_keep - Compile a persistent kept assignment to whole image @dst
 * @dst: Destination image
 * @aop: Assignment operator
 *
 * Return: Opaque `fkept *` handle, or NULL if not fusible.
 */
void *cli_calc_fuse_keep(IMAGE *dst, cli_token_type aop)
{
    return cli_calc_fuse_keep_tok(dst, NULL, aop);
}

/**
 * cli_calc_fuse_keep_tok - Compile a persistent kept assignment to @dst (or slice @dst_tok)
 * @dst:     Destination image
 * @dst_tok: Optional destination slice token string
 * @aop:     Assignment operator
 *
 * Return: Opaque `fkept *` handle, or NULL if not fusible.
 */
void *cli_calc_fuse_keep_tok(IMAGE *dst, const char *dst_tok, cli_token_type aop)
{
    fkept *k = (fkept *) calloc(1, sizeof(*k));
    fval   v;
    fview  dv;
    int    sliced = (dst_tok != NULL && strchr(dst_tok, '[') != NULL);
    int    pos    = cli_calc_tok_pos();

    if (k == NULL)
    {
        return NULL;
    }
    if (cli_calc_fuse_dst_view(&dst, dst_tok, &dv) != 0 || !is_real_datatype(dst->md[0].datatype))
    {
        cli_calc_fuse_reject("complex destination, or a destination slice that is not a view");
        free(k);
        return NULL;
    }
    if (!sliced && aop == TOK_EQUAL && cli_calc_cur_func()()->type == TOK_FUNC_USER)
    {
        const milk_calc_ext_entry_t *e = cli_calc_ext_get((int) cli_calc_cur_func()()->val_l);
        if (e != NULL && e->kind == MILK_EXT_OP)
        {
            char lhs[1][STRINGMAXLEN_IMAGE_NAME];
            int  loc[1] = { dst->md[0].shared != 1 };
            snprintf(lhs[0], sizeof(lhs[0]), "%s", dst->name);
            free(k);
            return cli_calc_fuse_keep_user_op(1, lhs, loc, dst->md[0].shared == 1);
        }
    }
    if (!sliced && cli_calc_compile_kept_matrix(&k->m, dst, aop, 1,
                                                aop == TOK_EQUAL && dst->md[0].shared != 1) == 0)
    {
        k->p.nimg = k->m.nimg;
        for (int i = 0; i < k->p.nimg; i++)
        {
            k->p.img[i] = k->m.img[i];
        }
        cli_calc_fuse_record_kept_images(k, dst);
        return k;
    }
    memset(k, 0, sizeof(*k));
    cli_calc_fuse_set_keep_vars(1);
    cli_calc_fuse_anchor_begin(&dv, dst->md[0].naxis);
    int rc = cli_calc_fuse_compile(&k->p, 0, &v);
    cli_calc_fuse_anchor_end();
    if (sliced)
    {
        k->p.dstv = dv;
        if (rc == 0)
        {
            /* still in keep-variables mode: the bounds read the variables each run, not their values now */
            cli_calc_fuse_bind_slice_tape(&k->p, &k->p.dstv);
        }
    }
    cli_calc_fuse_set_keep_vars(0);
    if (rc == 0 && v.vec && aop == TOK_OP_SLASH_EQ)
    {
        cli_calc_fuse_reject("/= by an image is evaluated as text (zero divisors are checked)");
        cli_calc_tok_seek(pos);
        cli_calc_kept_free(k);
        return NULL;
    }
    if (rc == 0 && !v.vec && !(aop == TOK_OP_SLASH_EQ && v.s1 == 0 && v.c == 0.0))
    {
        if ((!sliced && cli_calc_fuse_add_image(&k->p, dst) < 0) ||
            emit(&k->p, (fins) { .k = FI_CONST, .s1 = v.s1, .c = v.c }, 1) != 0)
        {
            cli_calc_tok_seek(pos);
            cli_calc_kept_free(k);
            return NULL;
        }
        v.vec = 1;
    }
    uint8_t out_dt =
        k->p.out_dt ? k->p.out_dt : (k->p.out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    if (rc == 0 && !sliced && aop == TOK_EQUAL && dst->md[0].shared != 1 &&
        dst->md[0].datatype != out_dt)
    {
        cli_calc_fuse_reject(
            "the local destination's type differs from the expression's result type");
    }
    if (rc != 0 || !v.vec || cli_calc_fuse_finalize_views(&k->p) != 0 ||
        (!sliced && k->p.nimg == 0 && k->p.nops == 0) ||
        (!sliced && aop == TOK_EQUAL && dst->md[0].shared != 1 && dst->md[0].datatype != out_dt) ||
        cli_calc_fuse_prepend_compound_lhs(&k->p, dst, sliced ? dst_tok : NULL, aop) != 0)
    {
        cli_calc_tok_seek(pos);
        cli_calc_kept_free(k);
        return NULL;
    }
    cli_calc_fuse_record_kept_images(k, dst);
    return k;
}

/**
 * cli_calc_fuse_keep_var - Compile a persistent kept assignment to scalar variable @vid
 * @vid: Target variable slot index
 * @aop: Assignment operator
 *
 * Return: Opaque `fkept *` handle, or NULL if not fusible.
 */
void *cli_calc_fuse_keep_var(int vid, cli_token_type aop)
{
    if (vid < 0 || vid >= data.core.NB_MAX_VARIABLE || !data.core.variable[vid].used ||
        data.core.variable[vid].type == 2)
    {
        return NULL;
    }
    fkept *k = (fkept *) calloc(1, sizeof(*k));
    fval   v;
    int    pos = cli_calc_tok_pos();
    if (k == NULL)
    {
        return NULL;
    }
    cli_calc_fuse_set_keep_vars(1);
    int rc = cli_calc_fuse_compile(&k->p, 0, &v);
    cli_calc_fuse_set_keep_vars(0);
    if (rc != 0 || v.vec || (aop == TOK_OP_SLASH_EQ && v.s1 == 0 && v.c == 0.0))
    {
        if (rc == 0)
        {
            cli_calc_fuse_reject(v.vec ? "an image expression assigned to a scalar variable"
                                       : "/= by a literal zero");
        }
        cli_calc_tok_seek(pos);
        cli_calc_kept_free(k);
        return NULL;
    }
    if (aop != TOK_EQUAL)
    {
        fval           lhs_v;
        cli_token_type bop = (aop == TOK_OP_PLUS_EQ)    ? TOK_OP_PLUS
                             : (aop == TOK_OP_MINUS_EQ) ? TOK_OP_MINUS
                             : (aop == TOK_OP_STAR_EQ)  ? TOK_OP_STAR
                                                        : TOK_OP_SLASH;
        if (emit_scalar(&k->p,
                        (fsins) { .k     = FI_SVAR,
                                  .vid   = vid,
                                  .vhash = name_hash32(data.core.variable[vid].name) },
                        &lhs_v) != 0 ||
            emit_scalar(&k->p,
                        (fsins) { .k  = FI_BIN,
                                  .op = bop,
                                  .s  = { lhs_v.s1, v.s1, 0 },
                                  .c  = { lhs_v.c, v.c, 0.0 },
                                  .cl = { 0, (uint8_t) v.lng, 0 } },
                        &v) != 0)
        {
            cli_calc_tok_seek(pos);
            cli_calc_kept_free(k);
            return NULL;
        }
    }
    k->dst_vid        = vid;
    k->scalar_s1      = v.s1;
    k->scalar_c       = v.c;
    k->scalar_is_long = v.lng;
    snprintf(k->dst_name, sizeof(k->dst_name), "%s", data.core.variable[vid].name);
    cli_calc_fuse_record_kept_images(k, NULL);
    return k;
}

/** check_fprog_svars() - each FI_SVAR variable still exists and is numeric; with @names (the tables
 * changed since the last run) also still bears the compiled name and is not shadowed by an image. */
static int check_fprog_svars(const fprog *fp, int names)
{
    if (fp == NULL)
    {
        return 0;
    }
    for (int i = 0; i < fp->ns; i++)
    {
        if (fp->sins[i].k == FI_SVAR)
        {
            int vid = fp->sins[i].vid;
            if (vid < 0 || vid >= data.core.NB_MAX_VARIABLE || !data.core.variable[vid].used ||
                data.core.variable[vid].type == 2 ||
                (names && (name_hash32(data.core.variable[vid].name) != fp->sins[i].vhash ||
                           image_ID(data.core.variable[vid].name, data.core.image,
                                    data.core.NB_MAX_IMAGE) != -1)))
            {
                return -1;
            }
        }
    }
    for (int r = 0; r < fp->nred; r++)
    {
        if (check_fprog_svars(fp->red_p1[r], names) != 0 ||
            check_fprog_svars(fp->red_p2[r], names) != 0)
        {
            return -1;
        }
    }
    for (int a = 0; a < fp->nax; a++)
    {
        if (check_fprog_svars(fp->ax_p[a], names) != 0 ||
            check_fprog_svars(fp->ax_p2[a], names) != 0)
        {
            return -1;
        }
    }
    return 0;
}

static int check_fprog_images(fprog                *fp,
                              int                   check_shm,
                              cli_calc_kept_matrix *m,
                              int                   allow_resize,
                              int                   names)
{
    if (fp == NULL)
    {
        return 0;
    }
    for (int i = 0; i < fp->nimg; i++)
    {
        fview *vw = &fp->view[i];
        if (vw->internal)
        {
            continue;
        }
        imageID id = vw->kept_id;
        if (id < 0 || id >= data.core.NB_MAX_IMAGE)
        {
            return -1;
        }
        IMAGE *im = &data.core.image[id];
        if (!im->used || im->md == NULL || (names && strcmp(im->name, vw->kept_name) != 0))
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                     vw->kept_name);
            return -1;
        }
        fp->img[i] = im;
        if (m != NULL && m->kind != 0 && i < m->nimg)
        {
            m->img[i] = im;
        }
        if (check_shm && im->md[0].shared == 1)
        {
            if (im->md[0].cnt0 == vw->last_cnt0)
            {
                cli_calc_refresh_shared_image(im);
            }
            vw->last_cnt0 = im->md[0].cnt0;
        }
        uint32_t vs[3];
        image_vshape(im, vs);
        if (im->md[0].datatype != vw->kept_dt || memcmp(vs, vw->kept_sz, sizeof(vs)) != 0)
        {
            if (allow_resize)
            {
                vw->kept_dt = im->md[0].datatype;
                memcpy(vw->kept_sz, vs, sizeof(vw->kept_sz));
            }
            else
            {
                snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                         im->name);
                return -1;
            }
        }
    }
    for (int r = 0; r < fp->nred; r++)
    {
        if (check_fprog_images(fp->red_p1[r], check_shm, NULL, 0, names) != 0 ||
            check_fprog_images(fp->red_p2[r], check_shm, NULL, 0, names) != 0)
        {
            return -1;
        }
    }
    for (int a = 0; a < fp->nax; a++)
    {
        if (check_fprog_images(fp->ax_p[a], check_shm, NULL, 0, names) != 0 ||
            check_fprog_images(fp->ax_p2[a], check_shm, NULL, 0, names) != 0)
        {
            return -1;
        }
    }
    return 0;
}

static void store_scalar_var(int vid, double res, int is_long, int64_t lres)
{
    VARIABLE *dv = &data.core.variable[vid];
    if (is_long)
    {
        dv->type    = 1;
        dv->value.l = (long) lres;
    }
    else
    {
        dv->type    = 0;
        dv->value.f = res;
    }
}

static int run_kept_once(fkept *k)
{
    if (k->uop != NULL)
    {
        return cli_calc_fuse_exec_user_op(k);
    }
    if (k->dst == NULL)
    {
        double sval[FUSE_MAXINS];
        if (cli_calc_fuse_eval_scalars(&k->p, sval) != 0)
        {
            return -1;
        }
        if (k->nsdst > 0)
        {
            for (int j = 0; j < k->nsdst; j++)
            {
                int     s1  = k->sdst_s1[j];
                double  res = (s1 > 0) ? sval[s1 - 1] : k->sdst_c[j];
                int     lng = (s1 > 0) ? k->p.slong[s1 - 1] : k->sdst_is_long[j];
                int64_t lr  = (s1 > 0) ? k->p.slval[s1 - 1] : (int64_t) k->sdst_c[j];
                store_scalar_var(k->sdst_vid[j], res, lng, lr);
            }
            return 0;
        }
        double  res     = (k->scalar_s1 > 0) ? sval[k->scalar_s1 - 1] : k->scalar_c;
        int     is_long = (k->scalar_s1 > 0) ? k->p.slong[k->scalar_s1 - 1] : k->scalar_is_long;
        int64_t lr      = (k->scalar_s1 > 0) ? k->p.slval[k->scalar_s1 - 1] : (int64_t) k->scalar_c;
        store_scalar_var(k->dst_vid, res, is_long, lr);
        return 0;
    }
    if (k->m.kind != 0)
    {
        return cli_calc_exec_kept_matrix(&k->m, k->dst);
    }
    return cli_calc_fuse_run_program(&k->p, k->dst);
}

#define TORN_WAIT_NS 100000
#define TORN_PASSES 3

static uint64_t torn_reads;

/**
 * cli_calc_torn_count - Return cumulative count of detected and retried torn SHM reads
 *
 * Return: Number of torn reads detected across the process lifetime.
 */
uint64_t cli_calc_torn_count(void)
{
    return torn_reads;
}

static int is_kept_output_img(const fkept *k, const IMAGE *im)
{
    if (im == k->dst)
    {
        return 1;
    }
    for (int d = 0; d < k->p.nextra_dst; d++)
    {
        if (im == k->p.extra_dst[d])
        {
            return 1;
        }
    }
    return 0;
}

static int fprog_inputs_snap(const fkept *k, fprog *fp)
{
    if (fp == NULL)
    {
        return 0;
    }
    int busy = 0;
    for (int i = 0; i < fp->nimg; i++)
    {
        const IMAGE *im = fp->img[i];
        if (fp->view[i].internal || is_kept_output_img(k, im) || im->md[0].shared != 1)
        {
            continue;
        }
        fp->view[i].snap_cnt0 = __atomic_load_n(&im->md[0].cnt0, __ATOMIC_ACQUIRE);
        busy |= (__atomic_load_n(&im->md[0].write, __ATOMIC_ACQUIRE) != 0);
    }
    for (int r = 0; r < fp->nred; r++)
    {
        busy |= fprog_inputs_snap(k, fp->red_p1[r]);
        busy |= fprog_inputs_snap(k, fp->red_p2[r]);
    }
    for (int a = 0; a < fp->nax; a++)
    {
        busy |= fprog_inputs_snap(k, fp->ax_p[a]);
        busy |= fprog_inputs_snap(k, fp->ax_p2[a]);
    }
    return busy;
}

static int fprog_inputs_moved(fkept *k, fprog *fp, int busy)
{
    if (fp == NULL)
    {
        return 0;
    }
    int moved = 0;
    for (int i = 0; i < fp->nimg; i++)
    {
        const IMAGE *im = fp->img[i];
        if (fp->view[i].internal || is_kept_output_img(k, im) || im->md[0].shared != 1)
        {
            continue;
        }
        if (__atomic_load_n(&im->md[0].cnt0, __ATOMIC_ACQUIRE) != fp->view[i].snap_cnt0)
        {
            k->writer_stuck = 0;
            return 1;
        }
        moved |= (!busy && __atomic_load_n(&im->md[0].write, __ATOMIC_ACQUIRE) != 0);
    }
    for (int r = 0; r < fp->nred; r++)
    {
        if (fprog_inputs_moved(k, fp->red_p1[r], busy) ||
            fprog_inputs_moved(k, fp->red_p2[r], busy))
        {
            return 1;
        }
    }
    for (int a = 0; a < fp->nax; a++)
    {
        if (fprog_inputs_moved(k, fp->ax_p[a], busy) || fprog_inputs_moved(k, fp->ax_p2[a], busy))
        {
            return 1;
        }
    }
    return moved;
}

/** check_kept_outputs() - the targets still exist with their shape and type; with @names (the tables
 * changed) also still bear the compiled names, unshadowed. @check_shm remaps recreated streams. */
static int check_kept_outputs(fkept *k, int check_shm, int names)
{
    if (k->dst_vid >= 0)
    {
        if (k->dst_vid >= data.core.NB_MAX_VARIABLE || !data.core.variable[k->dst_vid].used ||
            data.core.variable[k->dst_vid].type == 2 ||
            (names && (strcmp(data.core.variable[k->dst_vid].name, k->dst_name) != 0 ||
                       image_ID(k->dst_name, data.core.image, data.core.NB_MAX_IMAGE) != -1)))
        {
            return -1;
        }
    }
    for (int j = 0; j < k->nsdst; j++)
    {
        int vid = k->sdst_vid[j];
        if (vid < 0 || vid >= data.core.NB_MAX_VARIABLE || !data.core.variable[vid].used ||
            data.core.variable[vid].type == 2 ||
            (names && (strcmp(data.core.variable[vid].name, k->sdst_name[j]) != 0 ||
                       image_ID(k->sdst_name[j], data.core.image, data.core.NB_MAX_IMAGE) != -1)))
        {
            return -1;
        }
    }
    if (k->dst == NULL)
    {
        return 0;
    }
    if (k->dst_id < 0 || k->dst_id >= data.core.NB_MAX_IMAGE)
    {
        return -1;
    }
    IMAGE *im = &data.core.image[k->dst_id];
    if (!im->used || im->md == NULL || (names && strcmp(im->name, k->dst_name) != 0))
    {
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                 k->dst_name);
        return -1;
    }
    k->dst = im;
    if (check_shm && im->md[0].shared == 1)
    {
        cli_calc_refresh_shared_image(im);
    }
    uint32_t dvs[3];
    image_vshape(im, dvs);
    if (im->md[0].datatype != k->dst_dt || memcmp(dvs, k->dst_sz, sizeof(dvs)) != 0 ||
        (k->m.kind == 0 && !is_real_datatype(im->md[0].datatype)))
    {
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type", im->name);
        return -1;
    }
    for (int d = 0; d < k->p.nextra_dst; d++)
    {
        imageID eid = k->extra_dst_id[d];
        if (eid < 0 || eid >= data.core.NB_MAX_IMAGE)
        {
            return -1;
        }
        IMAGE *eim = &data.core.image[eid];
        if (!eim->used || eim->md == NULL ||
            (names && strcmp(eim->name, k->extra_dst_name[d]) != 0))
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                     k->extra_dst_name[d]);
            return -1;
        }
        k->p.extra_dst[d] = eim;
        if (check_shm && eim->md[0].shared == 1)
        {
            cli_calc_refresh_shared_image(eim);
        }
        uint32_t edvs[3];
        image_vshape(eim, edvs);
        if (eim->md[0].datatype != k->extra_dst_dt[d] ||
            memcmp(edvs, k->extra_dst_sz[d], sizeof(edvs)) != 0 ||
            !is_real_datatype(eim->md[0].datatype))
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                     eim->name);
            return -1;
        }
    }
    return 0;
}

/**
 * cli_calc_fused_run - Validate operands/targets and execute kept program @kept with torn-read guard
 * @kept: Kept program handle
 *
 * Return: 0 on success, -1 if an operand/target changed or execution failed.
 */
int cli_calc_fused_run(void *kept)
{
    fkept          *k = (fkept *) kept;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    int check_shm = (ts.tv_sec != k->last_check.tv_sec || ts.tv_nsec != k->last_check.tv_nsec);
    if (check_shm)
    {
        k->last_check = ts;
    }

    uint64_t gen   = milk_table_generation();
    int      names = (gen != k->last_gen); /* an image or variable came, went or was renamed */
    k->last_gen    = gen;
    if (check_fprog_svars(&k->p, names) != 0 || check_kept_outputs(k, check_shm, names) != 0 ||
        check_fprog_images(&k->p, check_shm, &k->m, k->uop != NULL, names) != 0)
    {
        return -1;
    }

    int rc = 0;
    if (k->dst != NULL && k->uop == NULL)
    {
        k->dst->md[0].write = 1;
        for (int d = 0; d < k->p.nextra_dst; d++)
        {
            k->p.extra_dst[d]->md[0].write = 1;
        }
    }
    for (int pass = 1;; pass++)
    {
        int busy = fprog_inputs_snap(k, &k->p);
        if (busy && !k->writer_stuck)
        {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            do
            {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                busy = fprog_inputs_snap(k, &k->p);
            } while (busy && (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec) <
                                 TORN_WAIT_NS);
            k->writer_stuck = busy;
            torn_reads += (uint64_t) busy;
        }
        rc = run_kept_once(k);
        if (rc != 0 || pass >= TORN_PASSES || !fprog_inputs_moved(k, &k->p, busy))
        {
            break;
        }
        torn_reads++;
    }
    if (k->dst != NULL && k->uop == NULL)
    {
        for (int d = 0; d < k->p.nextra_dst; d++)
        {
            k->p.extra_dst[d]->md[0].write = 0;
            if (rc == 0)
            {
                cli_calc_post_output(k->p.extra_dst[d]);
            }
        }
        k->dst->md[0].write = 0;
        if (rc == 0)
        {
            cli_calc_post_output(k->dst);
        }
    }
    return rc;
}

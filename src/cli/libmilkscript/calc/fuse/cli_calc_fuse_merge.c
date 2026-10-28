// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_merge.c
 * @brief Cross-statement fusion, multi-target kept compilation, and immediate in-place assignment.
 *
 * Role & Goal:
 *   Merges consecutive kept statements into a single multi-output fused pass
 *   (`cli_calc_fuse_try_merge_kept`), compiles multi-target kept assignments
 *   (`cli_calc_fuse_keep_multi`), and evaluates one-shot fused / in-place assignments.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_calc_fuse_internal.h"

static int unfused_extra(const fkept *k1, const fkept *k2);

/** map_merged_images - Build image index mapping from @k2 into @k1 without mutating @k1. */
static int map_merged_images(const fkept *k1, const fkept *k2, int img_map[FUSE_MAXIMG])
{
    int new_nimg = k1->p.nimg;
    for (int i = 0; i < k2->p.nimg; i++)
    {
        if (k2->p.img[i] == k1->dst)
        {
            img_map[i] = -1;
            continue;
        }
        int found = -1;
        for (int j = 0; j < k1->p.nimg; j++)
        {
            if (k1->p.img[j] == k2->p.img[i] && k1->p.view[j].is_lut == k2->p.view[i].is_lut &&
                strcmp(k1->p.view[j].spec, k2->p.view[i].spec) == 0)
            {
                found = j;
                break;
            }
        }
        if (found >= 0)
        {
            img_map[i] = found;
        }
        else
        {
            if (new_nimg >= FUSE_MAXIMG)
            {
                return -1;
            }
            img_map[i] = new_nimg++;
        }
    }
    return 0;
}

/** can_merge_kept_pair - Check shape, hazard, and capacity preconditions for merging @k2 into @k1. */
static int can_merge_kept_pair(const fkept *k1,
                               const fkept *k2,
                               int          elide_dst1,
                               int         *out_reads_dst1,
                               int          img_map[FUSE_MAXIMG])
{
    if (k1 == NULL || k2 == NULL || k1->dst == NULL || k2->dst == NULL || k1->m.kind != 0 ||
        k2->m.kind != 0 || k1->uop != NULL || k2->uop != NULL || k1->p.dstv.is_view ||
        k2->p.dstv.is_view || k1->dst == k2->dst || k1->p.inaxis != k2->p.inaxis ||
        memcmp(k1->p.ishape, k2->p.ishape, sizeof(k1->p.ishape)) != 0 || k2->p.nred > 0 ||
        k2->p.nax > 0 || k1->p.nuser + k2->p.nuser > FUSE_MAXUSER ||
        k1->p.nmatmul + k2->p.nmatmul > FUSE_MAXMATMUL)
    {
        return 0;
    }
    for (int d = 0; d < k1->p.nextra_dst; d++)
    {
        if (k1->p.extra_dst[d] == k2->dst)
        {
            return 0;
        }
    }
    for (int d2 = 0; d2 < k2->p.nextra_dst; d2++)
    {
        if (k2->p.extra_dst[d2] == k1->dst || k2->p.extra_dst[d2] == k2->dst)
        {
            return 0;
        }
        for (int d1 = 0; d1 < k1->p.nextra_dst; d1++)
        {
            if (k2->p.extra_dst[d2] == k1->p.extra_dst[d1])
            {
                return 0;
            }
        }
    }
    int reads_dst1 = 0;
    for (int i = 0; i < k2->p.nimg; i++)
    {
        const IMAGE *im = k2->p.img[i];
        if (im == k1->dst)
        {
            if (k2->p.view[i].is_view || k2->p.view[i].is_shift || k2->p.view[i].is_lut)
            {
                return 0;
            }
            reads_dst1 = 1;
        }
        for (int d = 0; d < k1->p.nextra_dst; d++)
        {
            if (im == k1->p.extra_dst[d])
            {
                return 0;
            }
        }
    }
    for (int i = 0; i < k1->p.nimg; i++)
    {
        int non_elem = k1->p.view[i].is_shift || k1->p.view[i].is_view || k1->p.view[i].is_lut;
        if (non_elem && k1->p.img[i] == k2->dst)
        {
            return 0;
        }
        for (int d2 = 0; non_elem && d2 < k2->p.nextra_dst; d2++)
        {
            if (k1->p.img[i] == k2->p.extra_dst[d2])
            {
                return 0;
            }
        }
    }

    int add_extra = elide_dst1 ? 0 : 1;
    int need_reg  = reads_dst1 ? 1 : 0;
    int reg_off   = k1->p.max_nreg + need_reg;
    int add_ins   = (reads_dst1 ? (elide_dst1 ? 1 : 2) : 1) + k2->p.n + unfused_extra(k1, k2);
    if (k1->p.nextra_dst + add_extra + k2->p.nextra_dst > FUSE_MAXEXTRA_DST ||
        reg_off + k2->p.max_nreg > FUSE_MAXREG ||
        ((k1->p.maxdepth > k2->p.maxdepth) ? k1->p.maxdepth : k2->p.maxdepth) + reg_off +
                k2->p.max_nreg >
            FUSE_MAXSTK ||
        k1->p.n + add_ins > FUSE_MAXINS || k1->p.ns + k2->p.ns > FUSE_MAXINS ||
        k1->p.nprop + k2->p.nprop > FUSE_MAXPROP || map_merged_images(k1, k2, img_map) != 0)
    {
        return 0;
    }
    *out_reads_dst1 = reads_dst1;
    return 1;
}

static void remap_merged_scalars_and_users(fkept       *k1,
                                           const fkept *k2,
                                           int         *out_ns_off,
                                           int         *out_u_off)
{
    int prop_off = k1->p.nprop;
    for (int i = 0; i < k2->p.nprop; i++)
    {
        snprintf(k1->p.prop_path[k1->p.nprop++], sizeof(k1->p.prop_path[0]), "%s",
                 k2->p.prop_path[i]);
    }
    int ns_off = k1->p.ns;
    for (int i = 0; i < k2->p.ns; i++)
    {
        fsins s = k2->p.sins[i];
        for (int c = 0; c < 3; c++)
        {
            if (s.s[c] > 0)
            {
                s.s[c] += ns_off;
            }
        }
        if (s.k == FI_PROP)
        {
            s.vid += prop_off;
        }
        k1->p.sins[k1->p.ns++] = s;
    }
    int u_off = k1->p.nuser;
    for (int u = 0; u < k2->p.nuser; u++)
    {
        fuse_user_call_t uc = k2->p.ucall[u];
        for (int j = 0; j < uc.nparams; j++)
        {
            if (uc.ps1[j] > 0)
            {
                uc.ps1[j] += (int16_t) ns_off;
            }
        }
        k1->p.ucall[k1->p.nuser++] = uc;
    }
    *out_ns_off = ns_off;
    *out_u_off  = u_off;
}

/** push_img_or_slot() - append a load of k2's image @k: k1's destination comes from its register slot. */
static void push_img_or_slot(fkept       *k1,
                             const fkept *k2,
                             int          k,
                             const int    img_map[FUSE_MAXIMG],
                             int          dst1_slot)
{
    if (k2->p.img[k] == k1->dst)
    {
        k1->p.ins[k1->p.n++] = (fins) { .k = FI_LOAD_SLOT, .cpos = dst1_slot };
    }
    else
    {
        k1->p.ins[k1->p.n++] = (fins) { .k = FI_IMG, .img = img_map[k] };
    }
}

/** append_unfused() - append superinstruction @in of @k2 as its plain loads and operator. */
static void append_unfused(fkept       *k1,
                           const fkept *k2,
                           const fins  *in,
                           const int    img_map[FUSE_MAXIMG],
                           int          dst1_slot)
{
    fins bin = { .k = FI_BIN, .op = in->op, .cpos = 0, .guarded = in->guarded };
    if (in->k == FI_IMGC)
    {
        push_img_or_slot(k1, k2, in->img, img_map, dst1_slot);
        bin.cpos = in->cpos;
        bin.s1   = in->s1;
        bin.c    = in->c;
    }
    else if (in->k == FI_IMGIMG)
    {
        push_img_or_slot(k1, k2, in->img, img_map, dst1_slot);
        push_img_or_slot(k1, k2, in->cpos, img_map, dst1_slot);
    }
    else /* FI_VIMGC: top op (image * c) */
    {
        push_img_or_slot(k1, k2, in->img, img_map, dst1_slot);
        if (in->s1 > 0 || in->c != 1.0)
        {
            k1->p.ins[k1->p.n++] = (fins) { .k       = FI_BIN,
                                            .op      = TOK_OP_STAR,
                                            .cpos    = 2,
                                            .s1      = in->s1,
                                            .c       = in->c,
                                            .guarded = in->guarded };
        }
    }
    k1->p.ins[k1->p.n++] = bin;
}

/** unfused_extra() - instructions the unfused form of @k2's fused reads of @k1's destination adds. */
static int unfused_extra(const fkept *k1, const fkept *k2)
{
    int extra = 0;
    for (int i = 0; i < k2->p.n; i++)
    {
        const fins *in = &k2->p.ins[i];
        if ((in->k == FI_IMGC || in->k == FI_VIMGC || in->k == FI_IMGIMG) &&
            (k2->p.img[in->img] == k1->dst ||
             (in->k == FI_IMGIMG && k2->p.img[in->cpos] == k1->dst)))
        {
            extra += 2;
        }
    }
    return extra;
}

/** remap_and_append_kept - Append remapped operands, scalars, and instructions of @k2 into @k1. */
static void remap_and_append_kept(fkept       *k1,
                                  const fkept *k2,
                                  int          elide_dst1,
                                  int          reads_dst1,
                                  const int    img_map[FUSE_MAXIMG])
{
    int reg_off  = k1->p.max_nreg + (reads_dst1 ? 1 : 0);
    int tape_off = k1->p.ns; /* k2's scalar tape is appended after k1's */
    for (int i = 0; i < k2->p.nimg; i++)
    {
        int m = img_map[i];
        if (m >= k1->p.nimg)
        {
            k1->p.img[m]  = k2->p.img[i];
            k1->p.view[m] = k2->p.view[i];
            k1->p.nimg    = m + 1;
            for (int j = 0; j < k1->p.view[m].npart; j++)
            {
                if (k1->p.view[m].part_s1[j] > 0)
                {
                    k1->p.view[m].part_s1[j] = (int16_t) (k1->p.view[m].part_s1[j] + tape_off);
                }
            }
        }
        else if (m >= 0)
        {
            k1->p.view[m].is_shift |= k2->p.view[i].is_shift;
        }
    }

    int ns_off = 0;
    int u_off  = 0;
    int mm_off = k1->p.nmatmul;
    for (int m = 0; m < k2->p.nmatmul; m++)
    {
        fuse_matmul_t mm              = k2->p.matmul[m];
        mm.imgA                       = img_map[mm.imgA];
        mm.imgB                       = img_map[mm.imgB];
        k1->p.matmul[k1->p.nmatmul++] = mm;
    }
    remap_merged_scalars_and_users(k1, k2, &ns_off, &u_off);

    int dst1_slot = k1->p.max_nreg;
    if (!elide_dst1)
    {
        int ed               = k1->p.nextra_dst++;
        k1->p.extra_dst[ed]  = k1->dst;
        k1->p.ins[k1->p.n++] = (fins) { .k = FI_STORE_DST, .img = ed, .cpos = reads_dst1 ? 1 : 0 };
    }
    if (reads_dst1)
    {
        k1->p.ins[k1->p.n++] = (fins) { .k = FI_STORE_SLOT, .cpos = dst1_slot };
    }
    else if (elide_dst1)
    {
        /* dead local nobody reads: drop it, or k2's tape would leave it in stk[0] for the store */
        k1->p.ins[k1->p.n++] = (fins) { .k = FI_POP };
    }

    for (int i = 0; i < k2->p.n; i++)
    {
        fins in = k2->p.ins[i];
        if (in.s1 > 0)
        {
            in.s1 += ns_off;
        }
        if (in.s2 > 0)
        {
            in.s2 += ns_off;
        }
        if (in.k == FI_STORE_SLOT || in.k == FI_LOAD_SLOT)
        {
            in.cpos += reg_off;
        }
        else if (in.k == FI_IMG && k2->p.img[in.img] == k1->dst)
        {
            in = (fins) { .k = FI_LOAD_SLOT, .cpos = dst1_slot };
        }
        else if (in.k == FI_IMGC || in.k == FI_VIMGC || in.k == FI_IMGIMG)
        {
            /* a fused read of k1's destination comes from the register slot, so it is unfused */
            if (k2->p.img[in.img] == k1->dst ||
                (in.k == FI_IMGIMG && k2->p.img[in.cpos] == k1->dst))
            {
                append_unfused(k1, k2, &in, img_map, dst1_slot);
                continue;
            }
            in.img = img_map[in.img];
            if (in.k == FI_IMGIMG)
            {
                in.cpos = img_map[in.cpos];
            }
        }
        else if (in.k == FI_IMG || in.k == FI_CPART || in.k == FI_COORD || in.k == FI_SHIFT ||
                 in.k == FI_LUT || in.k == FI_MEDFILT || in.k == FI_GATHER || in.k == FI_INTERP2)
        {
            in.img = img_map[in.img];
        }
        else if (in.k == FI_USER_BLK)
        {
            in.img += u_off;
        }
        else if (in.k == FI_USER_STENCIL)
        {
            in.img = img_map[in.img];
            in.cpos += u_off;
        }
        else if (in.k == FI_MATMUL)
        {
            in.cpos += mm_off;
        }
        else if (in.k == FI_STORE_DST)
        {
            int ed              = k1->p.nextra_dst++;
            k1->p.extra_dst[ed] = k2->p.extra_dst[in.img];
            in.img              = ed;
        }
        k1->p.ins[k1->p.n++] = in;
    }

    if (k2->p.maxdepth > k1->p.maxdepth)
    {
        k1->p.maxdepth = k2->p.maxdepth;
    }
    k1->p.max_nreg = reg_off + k2->p.max_nreg;
    k1->p.nops += k2->p.nops + 1;
    k1->p.out_double = k1->p.out_double || k2->p.out_double;
    k1->p.out_dt     = k2->p.out_dt;
    cli_calc_fuse_record_kept_images(k1, k2->dst);
}

/**
 * cli_calc_fuse_try_merge_kept - Merge @kept2 into @kept1 for single-pass block execution
 * @kept1:      First compiled kept program (modified in place on success)
 * @kept2:      Second compiled kept program
 * @elide_dst1: 1 to elide writing `kept1->dst` to memory when it is a dead temporary
 *
 * Return: 1 if merged, 0 if not eligible for single-pass fusion.
 */
int cli_calc_fuse_try_merge_kept(void *kept1, void *kept2, int elide_dst1)
{
    fkept *k1         = (fkept *) kept1;
    fkept *k2         = (fkept *) kept2;
    int    reads_dst1 = 0;
    int    img_map[FUSE_MAXIMG];

    if (!can_merge_kept_pair(k1, k2, elide_dst1, &reads_dst1, img_map))
    {
        return 0;
    }
    remap_and_append_kept(k1, k2, elide_dst1, reads_dst1, img_map);
    return 1;
}

static void *keep_multi_images(int ndst, IMAGE *const *dst_im, int pos)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    fkept *k                = (fkept *) cli_calc_fuse_keep_tok(dst_im[0], NULL, TOK_EQUAL);
    if (k == NULL)
    {
        cli_calc_tok_seek(pos);
        return NULL;
    }
    for (int j = 1; j < ndst; j++)
    {
        if (dst_im[j] == NULL || cur()->type != TOK_COMMA)
        {
            cli_calc_kept_free(k);
            cli_calc_tok_seek(pos);
            return NULL;
        }
        adv();
        fkept *kj = (fkept *) cli_calc_fuse_keep_tok(dst_im[j], NULL, TOK_EQUAL);
        if (kj == NULL || !cli_calc_fuse_try_merge_kept(k, kj, 0))
        {
            cli_calc_kept_free(kj);
            cli_calc_kept_free(k);
            cli_calc_tok_seek(pos);
            return NULL;
        }
        cli_calc_kept_free(kj);
    }
    if (cur()->type != TOK_EOF && cur()->type != TOK_NEWLINE)
    {
        cli_calc_kept_free(k);
        cli_calc_tok_seek(pos);
        return NULL;
    }
    return k;
}

static void *keep_multi_istats(fkept *k, int ndst, const int *dst_vid, int pos)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int had_lparen          = (cur()->type == TOK_FUNC_IM_D);
    adv();
    if (!had_lparen)
    {
        if (cur()->type != TOK_LPAREN)
        {
            cli_calc_fuse_set_keep_vars(0);
            cli_calc_kept_free(k);
            cli_calc_tok_seek(pos);
            return NULL;
        }
        adv();
    }
    fprog *sub = (fprog *) calloc(1, sizeof(fprog));
    fval   sv;
    if (sub == NULL || cli_calc_fuse_compile(sub, 0, &sv) != 0 || !sv.vec ||
        cur()->type != TOK_RPAREN || cli_calc_fuse_finalize_views(sub) != 0)
    {
        if (sub != NULL)
        {
            cli_calc_fuse_free_fprog_children(sub);
            free(sub);
        }
        cli_calc_fuse_set_keep_vars(0);
        cli_calc_kept_free(k);
        cli_calc_tok_seek(pos);
        return NULL;
    }
    adv();
    if (cur()->type != TOK_EOF && cur()->type != TOK_NEWLINE)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        cli_calc_fuse_set_keep_vars(0);
        cli_calc_kept_free(k);
        cli_calc_tok_seek(pos);
        return NULL;
    }
    cli_calc_fuse_set_keep_vars(0);
    k->p.red_p1[0]        = sub;
    k->p.red_op[0]        = ARITH_RED_ALL;
    k->p.nred             = 1;
    k->p.has_red          = 1;
    const reduce_op ops[] = { RED_MEAN, RED_STD, RED_MIN, RED_MAX };
    k->nsdst              = ndst;
    for (int j = 0; j < ndst; j++)
    {
        fval rv;
        emit_scalar(
            &k->p,
            (fsins) {
                .k = FI_RED, .op = (cli_token_type) ops[j], .vid = 0, .c = { 0.0, 0.0, 0.0 } },
            &rv);
        k->sdst_vid[j]     = dst_vid[j];
        k->sdst_s1[j]      = rv.s1;
        k->sdst_c[j]       = 0.0;
        k->sdst_is_long[j] = 0;
        snprintf(k->sdst_name[j], sizeof(k->sdst_name[j]), "%s",
                 data.core.variable[dst_vid[j]].name);
    }
    k->dst_vid = -1;
    cli_calc_fuse_record_kept_images(k, NULL);
    return k;
}

static void *keep_multi_icentroid(fkept *k, int ndst, const int *dst_vid, int pos)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int had_lparen          = (cur()->type == TOK_FUNC_IM_D);
    adv();
    if (!had_lparen)
    {
        if (cur()->type != TOK_LPAREN)
        {
            cli_calc_fuse_set_keep_vars(0);
            cli_calc_kept_free(k);
            cli_calc_tok_seek(pos);
            return NULL;
        }
        adv();
    }
    fprog *sub = (fprog *) calloc(1, sizeof(fprog));
    fval   sv;
    if (sub == NULL || cli_calc_fuse_compile(sub, 0, &sv) != 0 || !sv.vec ||
        cli_calc_fuse_finalize_views(sub) != 0)
    {
        if (sub != NULL)
        {
            cli_calc_fuse_free_fprog_children(sub);
            free(sub);
        }
        cli_calc_fuse_set_keep_vars(0);
        cli_calc_kept_free(k);
        cli_calc_tok_seek(pos);
        return NULL;
    }
    fval thv = { 0, 0, 0.0, 0 };
    if (cur()->type == TOK_COMMA)
    {
        adv();
        if (cli_calc_fuse_compile_more(&k->p, 0, &thv) != 0 || thv.vec)
        {
            cli_calc_fuse_free_fprog_children(sub);
            free(sub);
            cli_calc_fuse_set_keep_vars(0);
            cli_calc_kept_free(k);
            cli_calc_tok_seek(pos);
            return NULL;
        }
    }
    if (cur()->type != TOK_RPAREN)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        cli_calc_fuse_set_keep_vars(0);
        cli_calc_kept_free(k);
        cli_calc_tok_seek(pos);
        return NULL;
    }
    adv();
    if (cur()->type != TOK_EOF && cur()->type != TOK_NEWLINE)
    {
        cli_calc_fuse_free_fprog_children(sub);
        free(sub);
        cli_calc_fuse_set_keep_vars(0);
        cli_calc_kept_free(k);
        cli_calc_tok_seek(pos);
        return NULL;
    }
    cli_calc_fuse_set_keep_vars(0);
    k->p.red_p1[0]  = sub;
    k->p.nred       = 1;
    k->p.has_cent   = 1;
    k->p.cent_th_s1 = thv.s1;
    k->p.cent_th_c  = thv.c;
    k->nsdst        = ndst;
    for (int j = 0; j < ndst; j++)
    {
        fval rv;
        emit_scalar(
            &k->p,
            (fsins) { .k = FI_CENT, .op = (cli_token_type) j, .vid = 0, .c = { 0.0, 0.0, 0.0 } },
            &rv);
        k->sdst_vid[j]     = dst_vid[j];
        k->sdst_s1[j]      = rv.s1;
        k->sdst_c[j]       = 0.0;
        k->sdst_is_long[j] = 0;
        snprintf(k->sdst_name[j], sizeof(k->sdst_name[j]), "%s",
                 data.core.variable[dst_vid[j]].name);
    }
    k->dst_vid = -1;
    cli_calc_fuse_record_kept_images(k, NULL);
    return k;
}

/**
 * cli_calc_fuse_keep_multi - Compile `a, b = expr1, expr2` or `m, s = istats(im)` / `icentroid(im)`
 * @ndst:    Number of LHS targets (`2..FUSE_MAXDST`)
 * @dst_im:  Target image pointers (or NULL for scalar targets)
 * @dst_vid: Target scalar variable indices (or NULL for image targets)
 *
 * Return: Opaque `fkept *` handle, or NULL if not fusible.
 */
void *cli_calc_fuse_keep_multi(int ndst, IMAGE *const *dst_im, const int *dst_vid)
{
    if (ndst < 2 || ndst > FUSE_MAXDST)
    {
        cli_calc_fuse_reject("more than 4 targets (FUSE_MAXDST)");
        return NULL;
    }
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int pos                 = cli_calc_tok_pos();

    if (dst_im != NULL && dst_im[0] != NULL)
    {
        return keep_multi_images(ndst, dst_im, pos);
    }
    if (dst_vid == NULL)
    {
        return NULL;
    }
    for (int j = 0; j < ndst; j++)
    {
        int vid = dst_vid[j];
        if (vid < 0 || vid >= data.core.NB_MAX_VARIABLE || !data.core.variable[vid].used ||
            data.core.variable[vid].type == 2)
        {
            return NULL;
        }
    }

    fkept *k = (fkept *) calloc(1, sizeof(*k));
    if (k == NULL)
    {
        return NULL;
    }
    cli_calc_fuse_reset_fprog(&k->p);
    cli_calc_fuse_set_keep_vars(1);

    if ((cur()->type == TOK_FUNC_IM_D && strcmp(cur()->sval, "istats(") == 0) ||
        (cur()->type == TOK_VAR && strcmp(cur()->sval, "istats") == 0))
    {
        return keep_multi_istats(k, ndst, dst_vid, pos);
    }
    if (ndst <= 3 && ((cur()->type == TOK_FUNC_IM_D && strcmp(cur()->sval, "icentroid(") == 0) ||
                      (cur()->type == TOK_VAR && strcmp(cur()->sval, "icentroid") == 0)))
    {
        return keep_multi_icentroid(k, ndst, dst_vid, pos);
    }

    k->nsdst   = ndst;
    k->dst_vid = -1;
    for (int j = 0; j < ndst; j++)
    {
        if (j > 0)
        {
            if (cur()->type != TOK_COMMA)
            {
                cli_calc_fuse_set_keep_vars(0);
                cli_calc_kept_free(k);
                cli_calc_tok_seek(pos);
                return NULL;
            }
            adv();
        }
        fval vj;
        if (cli_calc_fuse_compile_more(&k->p, 0, &vj) != 0 || vj.vec)
        {
            cli_calc_fuse_set_keep_vars(0);
            cli_calc_kept_free(k);
            cli_calc_tok_seek(pos);
            return NULL;
        }
        k->sdst_vid[j]     = dst_vid[j];
        k->sdst_s1[j]      = vj.s1;
        k->sdst_c[j]       = vj.c;
        k->sdst_is_long[j] = vj.lng;
        snprintf(k->sdst_name[j], sizeof(k->sdst_name[j]), "%s",
                 data.core.variable[dst_vid[j]].name);
    }
    cli_calc_fuse_set_keep_vars(0);
    if (cur()->type != TOK_EOF && cur()->type != TOK_NEWLINE)
    {
        cli_calc_kept_free(k);
        cli_calc_tok_seek(pos);
        return NULL;
    }
    cli_calc_fuse_record_kept_images(k, NULL);
    return k;
}

/**
 * cli_calc_fuse_assign - Evaluate a one-shot fused assignment into whole image @dst
 * @dst: Destination image
 * @aop: Assignment operator
 *
 * Return: 1 on fused success, 0 if not fusible, -1 on runtime error.
 */
int cli_calc_fuse_assign(IMAGE *dst, cli_token_type aop)
{
    return cli_calc_fuse_assign_tok(dst, NULL, aop);
}

static void free_fprog_and_seek(fprog *p, int pos)
{
    cli_calc_fuse_free_fprog_children(p);
    free(p);
    cli_calc_tok_seek(pos);
}

/**
 * cli_calc_fuse_assign_tok - Evaluate a one-shot fused assignment into @dst (or slice @dst_tok)
 * @dst:     Destination image
 * @dst_tok: Optional destination slice specification
 * @aop:     Assignment operator
 *
 * Return: 1 on fused success, 0 if not fusible, -1 on runtime error.
 */
int cli_calc_fuse_assign_tok(IMAGE *dst, const char *dst_tok, cli_token_type aop)
{
    fview dv;
    int   sliced = (dst_tok != NULL && strchr(dst_tok, '[') != NULL);

    if (parse_mode == 0)
    {
        return 0;
    }
    int vrc = cli_calc_fuse_dst_view(&dst, dst_tok, &dv);
    if (vrc != 0 || !is_real_datatype(dst->md[0].datatype))
    {
        return (vrc < 0) ? -1 : 0;
    }
    int pos = cli_calc_tok_pos();
    if (!sliced)
    {
        cli_calc_kept_matrix km;
        if (cli_calc_compile_kept_matrix(&km, dst, aop, 0,
                                         aop == TOK_EQUAL && dst->md[0].shared != 1) == 0)
        {
            if (cli_calc_exec_kept_matrix(&km, dst) == 0)
            {
                cli_calc_post_output(dst);
                return 1;
            }
            cli_calc_tok_seek(pos);
            return 0;
        }
    }
    fprog *p = (fprog *) calloc(1, sizeof(*p));
    if (p == NULL)
    {
        return 0;
    }
    fval v;
    cli_calc_fuse_anchor_begin(&dv, dst->md[0].naxis);
    int crc = cli_calc_fuse_compile(p, 0, &v);
    cli_calc_fuse_anchor_end();
    if (crc != 0 || (aop == TOK_OP_SLASH_EQ && v.vec))
    {
        free_fprog_and_seek(p, pos);
        return 0;
    }
    if (sliced)
    {
        p->dstv = dv;
    }
    if (!v.vec && !sliced)
    {
        cli_calc_fuse_free_fprog_children(p);
        free(p);
        dst->md[0].write = 1;
        if (cli_calc_inplace(dst, aop, NULL, v.c) != 0)
        {
            dst->md[0].write = 0;
            return -1;
        }
        cli_calc_post_output(dst);
        return 1;
    }
    if (!v.vec)
    {
        if ((aop == TOK_OP_SLASH_EQ && v.s1 == 0 && v.c == 0.0) ||
            emit(p, (fins) { .k = FI_CONST, .s1 = v.s1, .c = v.c }, 1) != 0)
        {
            free_fprog_and_seek(p, pos);
            return 0;
        }
        v.vec = 1;
    }
    uint8_t out_dt = p->out_dt ? p->out_dt : (p->out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    if (cli_calc_fuse_finalize_views(p) != 0 ||
        (!sliced && p->nops == 0 && aop == TOK_EQUAL && !(p->nimg == 1 && p->view[0].is_view)) ||
        (!sliced && aop == TOK_EQUAL && dst->md[0].shared != 1 && dst->md[0].datatype != out_dt) ||
        cli_calc_fuse_prepend_compound_lhs(p, dst, sliced ? dst_tok : NULL, aop) != 0)
    {
        free_fprog_and_seek(p, pos);
        return 0;
    }
    dst->md[0].write = 1;
    int rrc          = cli_calc_fuse_run_program(p, dst);
    cli_calc_fuse_free_fprog_children(p);
    free(p);
    if (rrc != 0)
    {
        dst->md[0].write = 0;
        return -1;
    }
    cli_calc_post_output(dst);
    return 1;
}

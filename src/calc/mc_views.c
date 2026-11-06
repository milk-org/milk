// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

static void append_str(char *buf, size_t bufsz, size_t *len, const char *s)
{
    if (*len + 1 < bufsz)
    {
        int w = snprintf(buf + *len, bufsz - *len, "%s", s);
        if (w > 0)
        {
            *len = (*len + (size_t) w < bufsz) ? (*len + (size_t) w) : (bufsz - 1);
        }
    }
}

static int eval_bound_int(const mc_expr_t *be, long *out_val, char *err, size_t errsz)
{
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

static int eval_slice_btext(const mc_expr_t *slice_expr,
                            const IMAGE     *im,
                            char            *btext,
                            size_t           bsz,
                            char            *err,
                            size_t           errsz)
{
    size_t len    = 0;
    int    im_nax = (int) im->md[0].naxis;
    btext[0]      = '\0';

    for (int ax = 0; ax < slice_expr->u.slice.naxis; ax++)
    {
        const mc_slice_axis_t *sa      = &slice_expr->u.slice.axes[ax];
        long                   end_val = (ax < im_nax) ? ((long) im->md[0].size[ax] - 1) : 0;
        mc_scalar_set_slice_end(end_val);

        if (ax > 0)
        {
            append_str(btext, bsz, &len, ",");
        }
        if (sa->is_rev_full)
        {
            append_str(btext, bsz, &len, "-*");
            continue;
        }
        if (sa->is_full && !sa->has_step)
        {
            append_str(btext, bsz, &len, "*");
            continue;
        }

        char nbuf[64];
        long bval = 0;
        if (sa->start != NULL)
        {
            if (eval_bound_int(sa->start, &bval, err, errsz) != 0)
            {
                mc_scalar_set_slice_end(-1);
                return -1;
            }
            snprintf(nbuf, sizeof(nbuf), "%ld", bval);
            append_str(btext, bsz, &len, nbuf);
        }
        if (sa->has_colon)
        {
            append_str(btext, bsz, &len, ":");
            if (sa->stop != NULL)
            {
                if (eval_bound_int(sa->stop, &bval, err, errsz) != 0)
                {
                    mc_scalar_set_slice_end(-1);
                    return -1;
                }
                snprintf(nbuf, sizeof(nbuf), "%ld", bval);
                append_str(btext, bsz, &len, nbuf);
            }
            if (sa->has_step)
            {
                append_str(btext, bsz, &len, ":");
                if (sa->step != NULL)
                {
                    if (eval_bound_int(sa->step, &bval, err, errsz) != 0)
                    {
                        mc_scalar_set_slice_end(-1);
                        return -1;
                    }
                    snprintf(nbuf, sizeof(nbuf), "%ld%s", bval, sa->is_binned ? "b" : "");
                    append_str(btext, bsz, &len, nbuf);
                }
            }
        }
    }
    mc_scalar_set_slice_end(-1);
    return 0;
}

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

    char btext[256];
    if (eval_slice_btext(expr, im, btext, sizeof(btext), err, errsz) != 0)
    {
        return -1;
    }

    IMGID_SLICE slc = imgid_slice_parse(btext);
    if (slc.error)
    {
        snprintf(err, errsz, "%s", slc.errmsg);
        return -1;
    }
    if (validate_slice_neg_bounds(&slc, snax, ssz, err, errsz) != 0)
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
        if (!o->is_out && !o->is_scalar && o->slot_idx == sidx && o->has_slice == has_slice &&
            o->slice_expr == slice_expr && o->is_table_arg == tbl && o->is_expr_im == is_eim)
        {
            return i;
        }
    }
    return -1;
}

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

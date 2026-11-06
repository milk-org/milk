// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libfps/fps.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/create_variable.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/variable_ID.h"
#include "linalg/mc_linalg.h"
#include "milk_calc_ext.h"

int mc_exec_multi_svd_or_eigh(mcalc_prog_t    *prog,
                              const mc_stmt_t *s,
                              int              in_unit,
                              char            *err,
                              size_t           errsz);

__attribute__((format(printf, 3, 4))) static int ext_err(char       *err,
                                                         size_t      errsz,
                                                         const char *fmt,
                                                         ...)
{
    if (err != NULL && errsz > 0 && err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errsz, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static const char *strip_dot(const char *s)
{
    return (s != NULL && s[0] == '.') ? (s + 1) : (s != NULL ? s : "");
}

int mc_ext_resolve_call(mc_expr_t *e, int nlhs, char *err, size_t errsz)
{
    const milk_calc_ext_entry_t *ext = milk_calc_ext_get(e->u.call.ext_id);
    if (ext == NULL)
    {
        return ext_err(err, errsz, "unknown extension '%s'", e->u.call.name);
    }
    const milk_calc_param_set_t *pset = ext->get_params ? ext->get_params() : NULL;
    const milk_calc_op_ports_t  *pts  = ext->get_ports ? ext->get_ports() : NULL;
    int                          npos = 0;

    for (int i = 0; i < e->u.call.nargs; i++)
    {
        if (e->u.call.kw_names[i][0] == '\0')
        {
            npos++;
            continue;
        }
        const char *kw    = strip_dot(e->u.call.kw_names[i]);
        int         found = 0;
        for (int p = 0; pset != NULL && p < pset->nparams; p++)
        {
            if (strcmp(strip_dot(pset->specs[p].name), kw) == 0)
            {
                found = 1;
                break;
            }
        }
        for (int p = 0; !found && pts != NULL && p < pts->nin; p++)
        {
            if (milk_calc_port_match_(ext->name, pts->in_ports[p].name, kw))
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            return ext_err(err, errsz, "%s: unknown keyword argument '%s'", ext->name, kw);
        }
    }

    int min_img = (ext->kind == MILK_EXT_BLK2) ? 2 : 1;
    if (ext->kind == MILK_EXT_OP && pts != NULL)
    {
        min_img = 0;
        for (int p = 0; p < pts->nin; p++)
        {
            if (!(pts->in_ports[p].flags & MILK_PORT_OPT))
            {
                min_img++;
            }
        }
        if (nlhs > pts->nout)
        {
            return ext_err(err, errsz, "%s: too many LHS targets (%d > %d)", ext->name, nlhs,
                           pts->nout);
        }
    }
    else if (nlhs > 1)
    {
        return ext_err(err, errsz, "%s: single-output kernel cannot unpack into %d targets",
                       ext->name, nlhs);
    }

    if (npos < min_img)
    {
        return ext_err(err, errsz, "%s: expected at least %d positional argument(s)", ext->name,
                       min_img);
    }
    for (int i = 0; i < min_img && i < e->u.call.nargs; i++)
    {
        if (e->u.call.kw_names[i][0] == '\0' && e->u.call.args[i]->res_kind != MC_RES_IMAGE &&
            !(ext->kind == MILK_EXT_OP && pts != NULL && pts->in_ports[i].kind == MILK_PORT_SCALAR))
        {
            return ext_err(err, errsz, "%s: argument %d must be an image", ext->name, i + 1);
        }
    }

    e->needs_image = 1;
    e->res_kind    = MC_RES_IMAGE;
    e->res_dt      = (strcmp(ext->name, "rotate2d") == 0 && e->u.call.nargs > 0 &&
                      e->u.call.args[0]->res_dt == _DATATYPE_DOUBLE)
                         ? _DATATYPE_DOUBLE
                         : _DATATYPE_FLOAT;
    if (ext->kind != MILK_EXT_OP && e->u.call.nargs > 0 && e->u.call.args[0]->res_naxis > 0)
    {
        e->res_naxis = e->u.call.args[0]->res_naxis;
        memcpy(e->res_size, e->u.call.args[0]->res_size, sizeof(e->res_size));
    }
    return 0;
}

static void write_param_field(uint8_t *pbuf, const milk_calc_param_spec_t *sp, double val)
{
    if (sp->offset + sp->size > MILK_CALC_PARAM_BYTES)
    {
        return;
    }
    void *dst = pbuf + sp->offset;
    switch (sp->fptype)
    {
    case FPTYPE_INT64:
    {
        int64_t v = (int64_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_UINT64:
    {
        uint64_t v = (val <= 0.0) ? 0 : (uint64_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_INT32:
    case FPTYPE_ONOFF:
    {
        int32_t v = (int32_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_UINT32:
    {
        uint32_t v = (val <= 0.0) ? 0 : (uint32_t) llround(val);
        memcpy(dst, &v, sizeof(v));
        break;
    }
    case FPTYPE_FLOAT32:
    {
        float v = (float) val;
        memcpy(dst, &v, sizeof(v));
        break;
    }
    default:
        memcpy(dst, &val, sizeof(val));
        break;
    }
}

static int lookup_calc_var_d(const char *name, double *out)
{
    long vid = variable_ID(name);
    if (vid < 0 || !milk_data.variable[vid].used)
    {
        return 0;
    }
    *out = (milk_data.variable[vid].type == 1)   ? (double) milk_data.variable[vid].value.l
           : (milk_data.variable[vid].type == 0) ? milk_data.variable[vid].value.f
                                                 : 0.0;
    return milk_data.variable[vid].type == 0 || milk_data.variable[vid].type == 1;
}

static int pack_ext_params(const milk_calc_ext_entry_t *ext,
                           const mc_expr_t             *rhs,
                           const mc_expr_t             *call_site,
                           int                          n_img_pos,
                           uint8_t                     *pbuf,
                           char                        *err,
                           size_t                       errsz)
{
    memset(pbuf, 0, MILK_CALC_PARAM_BYTES);
    const milk_calc_param_set_t *pset = ext->get_params ? ext->get_params() : NULL;
    if (pset == NULL || pset->nparams <= 0)
    {
        return 0;
    }

    const char *colon = strchr(rhs->u.call.name, ':');
    const char *label = (colon != NULL && colon[1] != '\0') ? (colon + 1) : ext->name;

    int pos_args[MC_MAX_ARGS];
    int npos = 0;
    for (int i = 0; i < rhs->u.call.nargs; i++)
    {
        if (rhs->u.call.kw_names[i][0] == '\0')
        {
            pos_args[npos++] = i;
        }
    }

    for (int p = 0; p < pset->nparams; p++)
    {
        const milk_calc_param_spec_t *sp    = &pset->specs[p];
        const char                   *pname = strip_dot(sp->name);
        double                        val   = sp->def_num;
        int                           bound = 0;

        for (int i = 0; i < rhs->u.call.nargs; i++)
        {
            if (rhs->u.call.kw_names[i][0] != '\0' &&
                strcmp(strip_dot(rhs->u.call.kw_names[i]), pname) == 0)
            {
                if (mc_eval_scalar_d(rhs->u.call.args[i], call_site, &val, err, errsz) != 0)
                {
                    return -1;
                }
                bound = 1;
                break;
            }
        }
        if (!bound && n_img_pos + p < npos)
        {
            int ai = pos_args[n_img_pos + p];
            if (rhs->u.call.args[ai]->res_kind != MC_RES_IMAGE)
            {
                if (mc_eval_scalar_d(rhs->u.call.args[ai], call_site, &val, err, errsz) != 0)
                {
                    return -1;
                }
                bound = 1;
            }
        }
        if (!bound)
        {
            char vname[128], fbname[128];
            snprintf(vname, sizeof(vname), "%s.%s", label, pname);
            if (!lookup_calc_var_d(vname, &val))
            {
                if (strcmp(label, ext->name) == 0 ||
                    (snprintf(fbname, sizeof(fbname), "%s.%s", ext->name, pname),
                     !lookup_calc_var_d(fbname, &val)))
                {
                    create_variable_ID(vname, sp->def_num);
                }
            }
        }
        write_param_field(pbuf, sp, val);
    }
    return 0;
}

static imageID ensure_f32_im(imageID src_id, int *out_tmp)
{
    const IMAGE *src = &milk_data.image[src_id];
    if (src->md[0].datatype == _DATATYPE_FLOAT)
    {
        *out_tmp = 0;
        return src_id;
    }
    imageID tid = mc_tmp_image(src->md[0].naxis, src->md[0].size, _DATATYPE_FLOAT);
    if (tid >= 0)
    {
        IMAGE   *dst = &milk_data.image[tid];
        uint8_t  sdt = src->md[0].datatype;
        uint64_t nel = src->md[0].nelement;
        for (uint64_t i = 0; i < nel; i++)
        {
            dst->array.F[i] = (float) arith_read_im_pix(src, sdt, i);
        }
        *out_tmp = 1;
    }
    return tid;
}

typedef struct
{
    const milk_calc_ext_entry_t *ext;
    const float                 *in0;
    const float                 *in1;
    float                       *out;
    const uint8_t               *pbuf;
    uint32_t                     nx, ny;
} ext_team_job_t;

static void blk_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    const ext_team_job_t *j = (const ext_team_job_t *) ctx;
    if (hi <= lo)
    {
        return;
    }
    if (j->ext->kind == MILK_EXT_BLK1)
    {
        j->ext->blk1_fn(j->out + lo, j->in0 + lo, j->pbuf, (int) (hi - lo));
    }
    else
    {
        j->ext->blk2_fn(j->out + lo, j->in0 + lo, j->in1 + lo, j->pbuf, (int) (hi - lo));
    }
}

static void stencil_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) (tid + nth);
    const ext_team_job_t *j  = (const ext_team_job_t *) ctx;
    int                   R  = j->ext->radius;
    int                   nr = 2 * R + 1;
    uint32_t              nx = j->nx, ny = j->ny;
    uint32_t              pw  = nx + 2 * (uint32_t) R;
    float                *pad = malloc(sizeof(float) * (size_t) nr * (size_t) pw);
    const float          *rows[15];
    if (pad == NULL)
    {
        return;
    }
    for (int k = 0; k < nr; k++)
    {
        rows[k] = pad + (size_t) k * pw;
    }
    for (uint64_t y = lo; y < hi; y++)
    {
        for (int dy = -R; dy <= R; dy++)
        {
            int          sy  = (int) y + dy;
            uint32_t     cy  = (sy < 0) ? 0 : ((uint32_t) sy >= ny) ? (ny - 1) : (uint32_t) sy;
            const float *src = j->in0 + (uint64_t) cy * nx;
            float       *dst = pad + (size_t) (dy + R) * pw;
            for (int l = 0; l < R; l++)
            {
                dst[l] = src[0];
            }
            memcpy(dst + R, src, sizeof(float) * nx);
            for (int r = 0; r < R; r++)
            {
                dst[(uint32_t) R + nx + (uint32_t) r] = src[nx - 1];
            }
        }
        j->ext->stencil_fn(j->out + y * nx, rows, j->pbuf, (int) nx);
    }
    free(pad);
}

typedef struct
{
    const milk_calc_ext_entry_t *ext;
    void                        *op_state;
    int                          state_valid;
    uint8_t                      last_params[MILK_CALC_PARAM_BYTES];
    uint32_t                     last_in_sz[MILK_CALC_MAX_PORTS][4];
    uint8_t                      out_naxis[MILK_CALC_MAX_PORTS];
    uint32_t                     out_size[MILK_CALC_MAX_PORTS][3];
    uint8_t                      out_dtype[MILK_CALC_MAX_PORTS];
} op_cache_state_t;

static void free_op_cache_state(void *ptr)
{
    op_cache_state_t *st = (op_cache_state_t *) ptr;
    if (st == NULL)
    {
        return;
    }
    if (st->state_valid && st->ext != NULL && st->ext->op_free != NULL)
    {
        st->ext->op_free(st->op_state);
    }
    free(st);
}

static int exec_ext_op(mcalc_prog_t                *prog,
                       mc_ctor_step_t              *cs,
                       const milk_calc_ext_entry_t *ext,
                       const mc_expr_t             *rhs,
                       const mc_expr_t             *call_site,
                       const mc_stmt_t             *multi_s,
                       imageID                      avoid_id,
                       int                          in_unit,
                       mc_val_t                    *out_val,
                       char                        *err,
                       size_t                       errsz)
{
    const milk_calc_op_ports_t  *pts  = ext->get_ports ? ext->get_ports() : NULL;
    const milk_calc_param_set_t *pset = ext->get_params ? ext->get_params() : NULL;
    if (pts == NULL)
    {
        return ext_err(err, errsz, "%s: missing operator port table", ext->name);
    }

    milk_calc_op_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.op_name = ext->name;
    ctx.ports   = pts;

    imageID in_ids[MILK_CALC_MAX_PORTS];
    int     in_tmp[MILK_CALC_MAX_PORTS];
    int     f32_tmp[MILK_CALC_MAX_PORTS];
    imageID f32_ids[MILK_CALC_MAX_PORTS];
    int     rc      = 0;
    int     pos_cur = 0;

    for (int p = 0; p < MILK_CALC_MAX_PORTS; p++)
    {
        in_ids[p] = f32_ids[p] = -1;
        in_tmp[p] = f32_tmp[p] = 0;
    }

    for (int p = 0; p < pts->nin; p++)
    {
        const mc_expr_t *arg = NULL;
        for (int i = 0; i < rhs->u.call.nargs; i++)
        {
            if (rhs->u.call.kw_names[i][0] != '\0' &&
                milk_calc_port_match_(ext->name, pts->in_ports[p].name,
                                      strip_dot(rhs->u.call.kw_names[i])))
            {
                arg = rhs->u.call.args[i];
                break;
            }
        }
        while (arg == NULL && pos_cur < rhs->u.call.nargs)
        {
            int idx = pos_cur++;
            if (rhs->u.call.kw_names[idx][0] == '\0')
            {
                arg = rhs->u.call.args[idx];
                break;
            }
        }
        if (arg == NULL)
        {
            if (pts->in_ports[p].flags & MILK_PORT_OPT)
            {
                continue;
            }
            rc = ext_err(err, errsz, "%s: missing required input port '%s'", ext->name,
                         pts->in_ports[p].name);
            goto done;
        }
        if (pts->in_ports[p].kind == MILK_PORT_SCALAR)
        {
            if (mc_eval_scalar_d(arg, call_site, &ctx.in_scalar[p], err, errsz) != 0)
            {
                rc = -1;
                goto done;
            }
            ctx.in_bound[p] = 1;
        }
        else
        {
            if (mc_resolve_or_eval_im(arg, call_site, avoid_id, in_unit, &in_ids[p], &in_tmp[p],
                                      err, errsz) != 0)
            {
                rc = -1;
                goto done;
            }
            f32_ids[p] = ensure_f32_im(in_ids[p], &f32_tmp[p]);
            if (f32_ids[p] < 0)
            {
                rc = ext_err(err, errsz, "%s: failed to convert input port '%s'", ext->name,
                             pts->in_ports[p].name);
                goto done;
            }
            ctx.in_img[p]   = &milk_data.image[f32_ids[p]];
            ctx.in_bound[p] = 1;
        }
    }

    uint8_t pbuf[MILK_CALC_PARAM_BYTES];
    if (pack_ext_params(ext, rhs, call_site, pos_cur, pbuf, err, errsz) != 0)
    {
        rc = -1;
        goto done;
    }

    op_cache_state_t  local_st = { .ext = ext };
    op_cache_state_t *st       = &local_st;
    if (cs != NULL)
    {
        if (cs->ext_state == NULL)
        {
            cs->ext_state      = calloc(1, sizeof(op_cache_state_t));
            cs->ext_state_free = free_op_cache_state;
        }
        if (cs->ext_state != NULL)
        {
            st      = (op_cache_state_t *) cs->ext_state;
            st->ext = ext;
        }
    }

    int need_reinit = !st->state_valid;
    for (int p = 0; !need_reinit && p < pts->nin; p++)
    {
        const IMAGE *im     = ctx.in_img[p];
        uint32_t     cur[4] = {
            im ? im->md[0].naxis : 0,
            im ? im->md[0].size[0] : 0,
            (im && im->md[0].naxis > 1) ? im->md[0].size[1] : 0,
            (im && im->md[0].naxis > 2) ? im->md[0].size[2] : 0,
        };
        if (memcmp(cur, st->last_in_sz[p], sizeof(cur)) != 0)
        {
            need_reinit = 1;
        }
    }
    for (int p = 0; !need_reinit && pset != NULL && p < pset->nparams; p++)
    {
        const milk_calc_param_spec_t *sp = &pset->specs[p];
        if (!(sp->fpflags & FPFLAG_WRITERUN) &&
            memcmp(pbuf + sp->offset, st->last_params + sp->offset, sp->size) != 0)
        {
            need_reinit = 1;
        }
    }

    if (need_reinit)
    {
        if (st->state_valid && ext->op_free != NULL)
        {
            ext->op_free(st->op_state);
            st->op_state    = NULL;
            st->state_valid = 0;
        }
        const IMAGE *im0 = ctx.in_img[0];
        for (int p = 0; p < pts->nout; p++)
        {
            ctx.out_naxis[p]   = im0 ? im0->md[0].naxis : 2;
            ctx.out_size[p][0] = im0 ? im0->md[0].size[0] : 1;
            ctx.out_size[p][1] = (im0 && im0->md[0].naxis > 1) ? im0->md[0].size[1] : 1;
            ctx.out_size[p][2] = (im0 && im0->md[0].naxis > 2) ? im0->md[0].size[2] : 1;
            ctx.out_dtype[p]   = _DATATYPE_FLOAT;
        }
        if (ext->op_init != NULL && ext->op_init(&ctx, pbuf, &st->op_state) != 0)
        {
            rc = ext_err(err, errsz, "user operator '%s' init failed", ext->name);
            goto done;
        }
        st->state_valid = 1;
        memcpy(st->last_params, pbuf, sizeof(pbuf));
        for (int p = 0; p < pts->nin; p++)
        {
            const IMAGE *im      = ctx.in_img[p];
            st->last_in_sz[p][0] = im ? im->md[0].naxis : 0;
            st->last_in_sz[p][1] = im ? im->md[0].size[0] : 0;
            st->last_in_sz[p][2] = (im && im->md[0].naxis > 1) ? im->md[0].size[1] : 0;
            st->last_in_sz[p][3] = (im && im->md[0].naxis > 2) ? im->md[0].size[2] : 0;
        }
        memcpy(st->out_naxis, ctx.out_naxis, sizeof(st->out_naxis));
        memcpy(st->out_size, ctx.out_size, sizeof(st->out_size));
        memcpy(st->out_dtype, ctx.out_dtype, sizeof(st->out_dtype));
    }
    else
    {
        memcpy(ctx.out_naxis, st->out_naxis, sizeof(ctx.out_naxis));
        memcpy(ctx.out_size, st->out_size, sizeof(ctx.out_size));
        memcpy(ctx.out_dtype, st->out_dtype, sizeof(ctx.out_dtype));
    }

    int     nlhs = (multi_s != NULL) ? multi_s->u.multi.nlhs : 1;
    imageID out_ids[MILK_CALC_MAX_PORTS];
    int     out_is_tmp[MILK_CALC_MAX_PORTS];
    for (int p = 0; p < pts->nout; p++)
    {
        out_ids[p]    = -1;
        out_is_tmp[p] = 0;
        if (pts->out_ports[p].kind != MILK_PORT_IMAGE)
        {
            continue;
        }
        uint8_t dt = ctx.out_dtype[p] ? ctx.out_dtype[p] : _DATATYPE_FLOAT;
        if (p < nlhs)
        {
            mc_slot_t  tmp_slot = { 0 };
            mc_slot_t *sl       = NULL;
            if (multi_s == NULL && cs != NULL && prog != NULL)
            {
                sl = &prog->slots[cs->slot_idx];
            }
            else if (multi_s != NULL)
            {
                const mc_expr_t *lhs = multi_s->u.multi.lhs[p];
                for (int k = 0; prog != NULL && k < prog->nslots; k++)
                {
                    if (strcmp(prog->slots[k].bare, lhs->u.ident.bare) == 0)
                    {
                        sl = &prog->slots[k];
                        break;
                    }
                }
                if (sl == NULL)
                {
                    snprintf(tmp_slot.name, sizeof(tmp_slot.name), "%s", lhs->u.ident.name);
                    snprintf(tmp_slot.bare, sizeof(tmp_slot.bare), "%s", lhs->u.ident.bare);
                    tmp_slot.loc   = lhs->u.ident.loc;
                    tmp_slot.im_id = mc_resolve_or_autoconnect_image(lhs->u.ident.name);
                    sl             = &tmp_slot;
                }
            }
            out_ids[p] =
                mc_out_prepare(sl, ctx.out_naxis[p], ctx.out_size[p], dt, in_unit, err, errsz);
        }
        else
        {
            out_ids[p]    = mc_tmp_image(ctx.out_naxis[p], ctx.out_size[p], dt);
            out_is_tmp[p] = (out_ids[p] >= 0);
        }
        if (out_ids[p] < 0)
        {
            rc = -1;
            goto done_out;
        }
        ctx.out_img[p] = &milk_data.image[out_ids[p]];
    }

    if (ext->op_exec != NULL && ext->op_exec(&ctx, pbuf, st->op_state) != 0)
    {
        rc = ext_err(err, errsz, "user operator '%s' exec failed", ext->name);
        goto done_out;
    }

    for (int p = 0; p < pts->nout; p++)
    {
        if (pts->out_ports[p].kind == MILK_PORT_IMAGE)
        {
            if (!out_is_tmp[p] && out_ids[p] >= 0)
            {
                mc_out_post(out_ids[p]);
            }
        }
        else if (multi_s != NULL && p < nlhs)
        {
            mc_expr_t *lhs = multi_s->u.multi.lhs[p];
            mc_val_t   sv  = { .type = MC_VAL_DOUBLE, .dval = ctx.out_scalar[p] };
            mc_store_scalar_var(lhs->u.ident.bare[0] != '\0' ? lhs->u.ident.bare
                                                             : lhs->u.ident.name,
                                &lhs->u.ident.var_id, &sv);
            if (out_val != NULL)
            {
                *out_val = sv;
            }
        }
    }

done_out:
    for (int p = 0; p < pts->nout; p++)
    {
        (void) (out_is_tmp[p] && out_ids[p] >= 0 && (mc_tmp_release(out_ids[p]), 1));
    }
    if (st == &local_st && local_st.state_valid && ext->op_free != NULL)
    {
        ext->op_free(local_st.op_state);
    }

done:
    for (int p = 0; p < MILK_CALC_MAX_PORTS; p++)
    {
        (void) (f32_tmp[p] && f32_ids[p] >= 0 && (mc_tmp_release(f32_ids[p]), 1));
        (void) (in_tmp[p] && in_ids[p] >= 0 && (mc_tmp_release(in_ids[p]), 1));
    }
    return rc;
}

int mc_exec_ext_ctor(mcalc_prog_t    *prog,
                     mc_ctor_step_t  *cs,
                     imageID          avoid_id,
                     const mc_expr_t *call_site,
                     int              in_unit,
                     char            *err,
                     size_t           errsz)
{
    mc_slot_t                   *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t             *rhs  = call_site ? cs->rhs_expr->u.call.udef_body : cs->rhs_expr;
    const milk_calc_ext_entry_t *ext  = milk_calc_ext_get(rhs->u.call.ext_id);
    if (ext == NULL)
    {
        return ext_err(err, errsz, "unknown extension '%s'", rhs->u.call.name);
    }
    if (ext->kind == MILK_EXT_OP)
    {
        return exec_ext_op(prog, cs, ext, rhs, call_site, NULL, avoid_id, in_unit, NULL, err,
                           errsz);
    }

    int     n_img = (ext->kind == MILK_EXT_BLK2) ? 2 : 1;
    imageID id0 = -1, id1 = -1, f0 = -1, f1 = -1;
    int     tmp0 = 0, tmp1 = 0, ftmp0 = 0, ftmp1 = 0, rc = 0;

    if (mc_resolve_or_eval_im(rhs->u.call.args[0], call_site, avoid_id, in_unit, &id0, &tmp0, err,
                              errsz) != 0 ||
        (n_img == 2 && mc_resolve_or_eval_im(rhs->u.call.args[1], call_site, avoid_id, in_unit,
                                             &id1, &tmp1, err, errsz) != 0))
    {
        (void) (tmp0 && (mc_tmp_release(id0), 1));
        return -1;
    }

    uint8_t pbuf[MILK_CALC_PARAM_BYTES];
    if (pack_ext_params(ext, rhs, call_site, n_img, pbuf, err, errsz) != 0)
    {
        rc = -1;
        goto done;
    }

    const IMAGE *im0 = &milk_data.image[id0];
    if (n_img == 2 && milk_data.image[id1].md[0].nelement != im0->md[0].nelement)
    {
        rc = ext_err(err, errsz, "%s: input image element count mismatch", ext->name);
        goto done;
    }
    if (ext->kind == MILK_EXT_STENCIL && im0->md[0].naxis != 2)
    {
        rc = ext_err(err, errsz, "%s: stencil kernel requires a 2D image", ext->name);
        goto done;
    }

    int keep_f64 = (ext->kind == MILK_EXT_IMG && strcmp(ext->name, "rotate2d") == 0 &&
                    im0->md[0].datatype == _DATATYPE_DOUBLE);
    f0           = keep_f64 ? id0 : ensure_f32_im(id0, &ftmp0);
    f1           = (n_img == 2) ? ensure_f32_im(id1, &ftmp1) : -1;
    if (f0 < 0 || (n_img == 2 && f1 < 0))
    {
        rc = ext_err(err, errsz, "%s: failed to prepare float32 input buffer", ext->name);
        goto done;
    }

    uint32_t sz[3] = {
        im0->md[0].size[0],
        (im0->md[0].naxis > 1) ? im0->md[0].size[1] : 1,
        (im0->md[0].naxis > 2) ? im0->md[0].size[2] : 1,
    };
    uint8_t out_dt = keep_f64 ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    imageID oid    = mc_out_prepare(slot, im0->md[0].naxis, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        rc = -1;
        goto done;
    }
    IMAGE *out = &milk_data.image[oid];

    if (ext->kind == MILK_EXT_IMG)
    {
        if (ext->img_fn(out, &milk_data.image[f0], pbuf) != 0)
        {
            rc = ext_err(err, errsz, "user image kernel '%s' failed", ext->name);
            goto done;
        }
    }
    else if (ext->kind == MILK_EXT_STENCIL)
    {
        ext_team_job_t job = {
            .ext  = ext,
            .in0  = milk_data.image[f0].array.F,
            .out  = out->array.F,
            .pbuf = pbuf,
            .nx   = sz[0],
            .ny   = sz[1],
        };
        int nth = milk_omp_team_size(
            im0->md[0].nelement * (uint64_t) (ext->cost > 0 ? ext->cost : 4), 262144);
        (void) ((nth <= 1) ? (stencil_team_cb(&job, 0, 1, 0, sz[1]), 1)
                           : (arith_team_run(nth, sz[1], stencil_team_cb, &job), 1));
    }
    else
    {
        ext_team_job_t job = {
            .ext  = ext,
            .in0  = milk_data.image[f0].array.F,
            .in1  = (n_img == 2) ? milk_data.image[f1].array.F : NULL,
            .out  = out->array.F,
            .pbuf = pbuf,
        };
        uint64_t nel = im0->md[0].nelement;
        int      nth = milk_omp_team_size(nel * (uint64_t) (ext->cost > 0 ? ext->cost : 2), 262144);
        (void) ((nth <= 1) ? (blk_team_cb(&job, 0, 1, 0, nel), 1)
                           : (arith_team_run(nth, nel, blk_team_cb, &job), 1));
    }
    mc_out_post(oid);

done:
    (void) (ftmp0 && f0 >= 0 && (mc_tmp_release(f0), 1));
    (void) (ftmp1 && f1 >= 0 && (mc_tmp_release(f1), 1));
    (void) (tmp0 && id0 >= 0 && (mc_tmp_release(id0), 1));
    (void) (tmp1 && id1 >= 0 && (mc_tmp_release(id1), 1));
    return rc;
}

int mc_exec_multi_linalg_or_ext(mcalc_prog_t    *prog,
                                mc_ctor_step_t  *cs,
                                const mc_stmt_t *s,
                                int              in_unit,
                                mc_val_t        *out,
                                char            *err,
                                size_t           errsz)
{
    const mc_expr_t *rhs = s->u.multi.rhs[0];
    if (rhs->kind == MC_EXPR_CALL && rhs->u.call.ext_id >= 0)
    {
        const milk_calc_ext_entry_t *ext = milk_calc_ext_get(rhs->u.call.ext_id);
        return (ext != NULL && ext->kind == MILK_EXT_OP)
                   ? exec_ext_op(prog, cs, ext, rhs, NULL, s, -1, in_unit, out, err, errsz)
                   : ext_err(err, errsz, "unknown multi-output extension '%s'", rhs->u.call.name);
    }
    if (rhs->kind == MC_EXPR_CALL && rhs->u.call.fn != NULL &&
        (strcmp(rhs->u.call.fn->name, "svd") == 0 || strcmp(rhs->u.call.fn->name, "eigh") == 0))
    {
        return mc_exec_multi_svd_or_eigh(prog, s, in_unit, err, errsz);
    }
    return ext_err(err, errsz, "not implemented in the new engine");
}

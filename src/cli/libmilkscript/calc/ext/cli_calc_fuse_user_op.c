// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_user_op.c
 * @brief Compilation and execution of Tier 2 stateful multi-port user C operators (`MILK_EXT_OP`)
 *        and non-kept user extension evaluation.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "fuse/cli_calc_fuse_internal.h"
#include "prog/cli_calc_prog_internal.h"

/**
 * ensure_out_image - Create or resize output image `uop->out_name[j]` to match `ctx` dimensions
 */
static int ensure_out_image(fuse_user_op_t *uop, const milk_calc_op_ctx_t *ctx, int j)
{
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(uop->out_name[j]);
    const char           *bare =
        (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : uop->out_name[j];
    int shared = (sp.loc == 'S')        ? 1
                 : (sp.loc == 'L')      ? 0
                 : uop->out_is_local[j] ? 0
                                        : uop->default_shared;

    uint8_t         naxis = (ctx->out_naxis[j] > 0) ? ctx->out_naxis[j] : 2;
    const uint32_t *sz    = ctx->out_size[j];
    uint8_t         dtype = (ctx->out_dtype[j] > 0) ? ctx->out_dtype[j] : _DATATYPE_FLOAT;

    imageID id = cli_calc_prog_connect_stream(bare);
    if (id != -1)
    {
        IMAGE   *im = &data.core.image[id];
        uint32_t cur_sz[3];
        image_vshape(im, cur_sz);
        if (im->md[0].naxis == naxis && im->md[0].datatype == dtype && cur_sz[0] == sz[0] &&
            cur_sz[1] == sz[1] && cur_sz[2] == sz[2])
        {
            uop->out_id[j]    = id;
            uop->out_naxis[j] = naxis;
            memcpy(uop->out_size[j], sz, sizeof(uop->out_size[j]));
            uop->out_dtype[j] = dtype;
            return 0;
        }
        delete_image_ID(bare, DELETE_IMAGE_ERRMODE_IGNORE);
        id = -1;
    }
    uint32_t csz[3] = { sz[0] > 0 ? sz[0] : 1, sz[1] > 0 ? sz[1] : 1, sz[2] > 0 ? sz[2] : 1 };
    create_image_ID(bare, naxis, csz, dtype, shared, 10, 0, &id);
    if (id == -1)
    {
        return -1;
    }
    uop->out_id[j]    = id;
    uop->out_naxis[j] = naxis;
    memcpy(uop->out_size[j], csz, sizeof(uop->out_size[j]));
    uop->out_dtype[j] = dtype;
    return 0;
}

/**
 * populate_op_ctx_inputs - Fill `ctx->in_img[]` and `ctx->in_scalar[]` from `k->p`
 */
static void populate_op_ctx_inputs(fkept                       *k,
                                   const milk_calc_ext_entry_t *e,
                                   const milk_calc_op_ports_t  *ports,
                                   const double                *sval,
                                   milk_calc_op_ctx_t          *ctx)
{
    fuse_user_op_t *uop = k->uop;
    memset(ctx, 0, sizeof(*ctx));
    ctx->op_name = e->name;
    ctx->ports   = ports;

    const IMAGE *first_im = NULL;
    for (int i = 0; i < ports->nin && i < MILK_CALC_MAX_PORTS; i++)
    {
        ctx->in_bound[i] = uop->in_bound[i];
        if (!uop->in_bound[i])
        {
            continue;
        }
        if (ports->in_ports[i].kind == MILK_PORT_IMAGE)
        {
            int k_idx      = uop->in_img_k[i];
            ctx->in_img[i] = (k_idx >= 0 && k_idx < k->p.nimg) ? k->p.img[k_idx] : NULL;
            if (first_im == NULL && ctx->in_img[i] != NULL)
            {
                first_im = ctx->in_img[i];
            }
        }
        else
        {
            int s1            = uop->in_ss1[i];
            ctx->in_scalar[i] = (s1 > 0 && sval != NULL) ? sval[s1 - 1] : uop->in_sval[i];
        }
    }

    for (int j = 0; j < ports->nout && j < MILK_CALC_MAX_PORTS; j++)
    {
        if (uop->inited && uop->out_naxis[j] > 0)
        {
            ctx->out_naxis[j] = uop->out_naxis[j];
            memcpy(ctx->out_size[j], uop->out_size[j], sizeof(ctx->out_size[j]));
            ctx->out_dtype[j] = uop->out_dtype[j];
        }
        else if (first_im != NULL)
        {
            ctx->out_naxis[j] = first_im->md[0].naxis;
            image_vshape(first_im, ctx->out_size[j]);
            ctx->out_dtype[j] = _DATATYPE_FLOAT;
        }
        else
        {
            ctx->out_naxis[j]   = 2;
            ctx->out_size[j][0] = 1;
            ctx->out_size[j][1] = 1;
            ctx->out_size[j][2] = 1;
            ctx->out_dtype[j]   = _DATATYPE_FLOAT;
        }
    }
}

/**
 * init_or_reinit_user_op - Run `op_init` and create/resize output streams and scalar variables
 */
static int init_or_reinit_user_op(fkept                       *k,
                                  const milk_calc_ext_entry_t *e,
                                  const milk_calc_op_ports_t  *ports,
                                  const milk_calc_param_set_t *pset,
                                  const double                *sval)
{
    fuse_user_op_t   *uop = k->uop;
    fuse_user_call_t *uc  = &k->p.ucall[0];

    if (uop->inited && uop->state != NULL && e->op_free != NULL)
    {
        e->op_free(uop->state);
        uop->state = NULL;
    }
    uop->inited = 0;

    milk_calc_op_ctx_t ctx;
    populate_op_ctx_inputs(k, e, ports, sval, &ctx);

    if (e->op_init != NULL && e->op_init(&ctx, uc->pbuf, &uop->state) != 0)
    {
        parse_errmsg("User operator init() failed");
        return -1;
    }

    for (int j = 0; j < uop->ndst && j < ports->nout; j++)
    {
        if (ports->out_ports[j].kind == MILK_PORT_IMAGE)
        {
            if (ensure_out_image(uop, &ctx, j) != 0)
            {
                parse_errmsg("Failed to allocate user operator output image");
                return -1;
            }
            uop->out_vid[j] = -1;
        }
        else
        {
            FPS_STREAMNAME_PARSED sp = fps_streamname_parse(uop->out_name[j]);
            const char           *bare =
                (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : uop->out_name[j];
            long vid = variable_ID(bare);
            if (vid < 0)
            {
                vid = create_variable_ID(bare, 0.0);
            }
            if (vid < 0)
            {
                return -1;
            }
            uop->out_vid[j] = (int) vid;
            uop->out_id[j]  = -1;
        }
    }

    for (int i = 0; i < ports->nin && i < MILK_CALC_MAX_PORTS; i++)
    {
        if (uop->in_bound[i] && ports->in_ports[i].kind == MILK_PORT_IMAGE && ctx.in_img[i] != NULL)
        {
            image_vshape(ctx.in_img[i], uop->in_init_sz[i]);
            uop->in_init_dt[i] = ctx.in_img[i]->md[0].datatype;
        }
    }
    if (pset != NULL)
    {
        for (int p = 0; p < uc->nparams; p++)
        {
            uop->init_pvals[p] = uc->pvals[p];
        }
    }
    uop->inited = 1;
    return 0;
}

/**
 * needs_reinit - Check whether input image shapes or CONF-time parameters changed since last init
 */
static int needs_reinit(const fkept                 *k,
                        const milk_calc_op_ports_t  *ports,
                        const milk_calc_param_set_t *pset)
{
    const fuse_user_op_t   *uop = k->uop;
    const fuse_user_call_t *uc  = &k->p.ucall[0];

    if (!uop->inited)
    {
        return 1;
    }
    for (int i = 0; i < ports->nin && i < MILK_CALC_MAX_PORTS; i++)
    {
        if (uop->in_bound[i] && ports->in_ports[i].kind == MILK_PORT_IMAGE)
        {
            int          k_idx = uop->in_img_k[i];
            const IMAGE *im    = (k_idx >= 0 && k_idx < k->p.nimg) ? k->p.img[k_idx] : NULL;
            if (im == NULL)
            {
                return 1;
            }
            uint32_t vs[3];
            image_vshape(im, vs);
            if (im->md[0].datatype != uop->in_init_dt[i] ||
                memcmp(vs, uop->in_init_sz[i], sizeof(vs)) != 0)
            {
                return 1;
            }
        }
    }
    if (pset != NULL)
    {
        for (int p = 0; p < uc->nparams; p++)
        {
            if ((pset->specs[p].fpflags & FPFLAG_WRITERUN) == 0 &&
                uc->pvals[p] != uop->init_pvals[p])
            {
                return 1;
            }
        }
    }
    return 0;
}

/**
 * parse_op_arg_image - Resolve or compile an image argument for an operator input port
 */
static int parse_op_arg_image(fprog *p, int *out_k)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *t            = cur();
    if ((t->type == TOK_VAR || t->type == TOK_NVAR || t->type == TOK_IMAGE) &&
        strchr(t->sval, '[') == NULL)
    {
        imageID id = cli_calc_prog_connect_stream(t->sval);
        if (id != -1)
        {
            t->type = TOK_IMAGE;
        }
    }
    return cli_calc_fuse_c_leaf_or_eval_image(p, 1, out_k);
}

/**
 * parse_one_user_op_arg - Parse one positional or `kw = val` argument of a Tier 2 operator call
 */
static int parse_one_user_op_arg(fkept                       *k,
                                 const milk_calc_op_ports_t  *ports,
                                 const milk_calc_param_set_t *pset,
                                 int                         *pos_in,
                                 int                         *seen_kw,
                                 int                          param_bound[MILK_CALC_MAX_PARAMS])
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    fuse_user_op_t   *uop   = k->uop;
    fuse_user_call_t *uc    = &k->p.ucall[0];

    int        arg_pos  = cli_calc_tok_pos();
    cli_token *kt       = cur();
    int        port_idx = -1;
    int        par_idx  = -1;

    if (kt->type == TOK_VAR || kt->type == TOK_NVAR || kt->type == TOK_IMAGE)
    {
        char kw[64];
        snprintf(kw, sizeof(kw), "%s", kt->sval);
        adv();
        if (cur()->type == TOK_EQUAL)
        {
            adv();
            *seen_kw = 1;
            for (int i = 0; i < ports->nin; i++)
            {
                if (strcmp(ports->in_ports[i].name, kw) == 0)
                {
                    port_idx = i;
                    break;
                }
            }
            if (port_idx < 0 && pset != NULL)
            {
                for (int p = 0; p < uc->nparams; p++)
                {
                    if (strcmp(pset->specs[p].name, kw) == 0)
                    {
                        par_idx = p;
                        break;
                    }
                }
            }
            if (port_idx < 0 && par_idx < 0)
            {
                return -1;
            }
        }
        else
        {
            cli_calc_tok_seek(arg_pos);
        }
    }

    if (port_idx < 0 && par_idx < 0)
    {
        if (*seen_kw || *pos_in >= ports->nin)
        {
            return -1;
        }
        port_idx = (*pos_in)++;
    }

    if (port_idx >= 0)
    {
        if (ports->in_ports[port_idx].kind == MILK_PORT_IMAGE)
        {
            int k_idx = -1;
            if (parse_op_arg_image(&k->p, &k_idx) != 0)
            {
                if ((ports->in_ports[port_idx].flags & MILK_PORT_OPT) != 0 &&
                    (cur()->type == TOK_VAR || cur()->type == TOK_NVAR))
                {
                    adv();
                    return 0;
                }
                return -1;
            }
            uop->in_img_k[port_idx] = k_idx;
            uop->in_bound[port_idx] = 1;
            return 0;
        }
        fval sv;
        if (cli_calc_fuse_c_expr(&k->p, 0, &sv) != 0 || sv.vec)
        {
            return -1;
        }
        uop->in_sval[port_idx]  = sv.c;
        uop->in_ss1[port_idx]   = sv.s1;
        uop->in_bound[port_idx] = 1;
        return 0;
    }

    fval sv;
    if (cli_calc_fuse_c_expr(&k->p, 0, &sv) != 0 || sv.vec)
    {
        return -1;
    }
    uc->pvals[par_idx]   = sv.c;
    uc->ps1[par_idx]     = (int16_t) sv.s1;
    param_bound[par_idx] = 1;
    return 0;
}

static fkept *alloc_user_op_kept(const cli_token             *ft,
                                 const milk_calc_param_set_t *pset,
                                 int                          ndst,
                                 const char                   lhs[][STRINGMAXLEN_IMAGE_NAME],
                                 const int                   *is_local,
                                 int                          default_shared)
{
    fkept          *k   = calloc(1, sizeof(*k));
    fuse_user_op_t *uop = calloc(1, sizeof(*uop));
    if (k == NULL || uop == NULL)
    {
        free(k);
        free(uop);
        return NULL;
    }
    cli_calc_fuse_reset_fprog(&k->p);
    cli_calc_fuse_set_keep_vars(1);

    k->uop              = uop;
    k->dst_vid          = -1;
    uop->ext_idx        = (int) ft->val_l;
    uop->ndst           = ndst;
    uop->default_shared = default_shared;
    for (int i = 0; i < MILK_CALC_MAX_PORTS; i++)
    {
        uop->in_img_k[i] = -1;
        uop->out_id[i]   = -1;
        uop->out_vid[i]  = -1;
    }
    for (int j = 0; j < ndst; j++)
    {
        snprintf(uop->out_name[j], sizeof(uop->out_name[j]), "%s", lhs[j]);
        uop->out_is_local[j] = (is_local != NULL) ? is_local[j] : 0;
    }

    k->p.nuser           = 1;
    fuse_user_call_t *uc = &k->p.ucall[0];
    uc->ext_idx          = (int) ft->val_l;
    char func_name[64];
    cli_calc_ext_split_call(ft->sval, func_name, sizeof(func_name), uc->label, sizeof(uc->label));
    uc->nparams = (pset != NULL && pset->nparams <= MILK_CALC_MAX_PARAMS) ? pset->nparams : 0;
    return k;
}

/**
 * cli_calc_fuse_keep_user_op - Compile a top-level Tier 2 operator call (`lhs... = op(...)`)
 */
void *cli_calc_fuse_keep_user_op(int        ndst,
                                 const char lhs[][STRINGMAXLEN_IMAGE_NAME],
                                 const int *is_local,
                                 int        default_shared)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    int start_pos           = cli_calc_tok_pos();

    if (cur()->type != TOK_FUNC_USER)
    {
        return NULL;
    }
    const cli_token             *ft = adv();
    const milk_calc_ext_entry_t *e  = cli_calc_ext_get((int) ft->val_l);
    if (e == NULL || e->kind != MILK_EXT_OP || e->get_ports == NULL)
    {
        cli_calc_tok_seek(start_pos);
        return NULL;
    }
    const milk_calc_op_ports_t  *ports = e->get_ports();
    const milk_calc_param_set_t *pset  = (e->get_params != NULL) ? e->get_params() : NULL;
    if (ndst < 1 || ndst > ports->nout || ndst > MILK_CALC_MAX_PORTS)
    {
        cli_calc_tok_seek(start_pos);
        return NULL;
    }

    fkept *k = alloc_user_op_kept(ft, pset, ndst, lhs, is_local, default_shared);
    if (k == NULL)
    {
        cli_calc_tok_seek(start_pos);
        return NULL;
    }
    fuse_user_op_t   *uop                               = k->uop;
    fuse_user_call_t *uc                                = &k->p.ucall[0];
    int               param_bound[MILK_CALC_MAX_PARAMS] = { 0 };
    int               pos_in                            = 0;
    int               seen_kw                           = 0;

    while (cur()->type != TOK_RPAREN)
    {
        if (parse_one_user_op_arg(k, ports, pset, &pos_in, &seen_kw, param_bound) != 0)
        {
            goto fail;
        }
        if (cur()->type == TOK_COMMA)
        {
            adv();
            continue;
        }
        break;
    }
    if (cur()->type != TOK_RPAREN)
    {
        goto fail;
    }
    adv();
    if (cur()->type != TOK_EOF && cur()->type != TOK_NEWLINE)
    {
        goto fail;
    }
    for (int i = 0; i < ports->nin; i++)
    {
        if (!uop->in_bound[i] && (ports->in_ports[i].flags & MILK_PORT_OPT) == 0)
        {
            goto fail;
        }
    }
    for (int p = 0; p < uc->nparams; p++)
    {
        if (!param_bound[p] &&
            cli_calc_fuse_emit_default_user_param(&k->p, uc, p, &pset->specs[p]) != 0)
        {
            goto fail;
        }
    }
    cli_calc_fuse_set_keep_vars(0);

    double sval[FUSE_MAXINS];
    if (cli_calc_fuse_eval_scalars(&k->p, sval) != 0 || cli_calc_fuse_run_axes(&k->p) != 0 ||
        init_or_reinit_user_op(k, e, ports, pset, sval) != 0)
    {
        cli_calc_kept_free(k);
        cli_calc_tok_seek(start_pos);
        return NULL;
    }

    cli_calc_fuse_record_kept_images(k, NULL);
    return k;

fail:
    cli_calc_fuse_set_keep_vars(0);
    cli_calc_kept_free(k);
    cli_calc_tok_seek(start_pos);
    return NULL;
}

/**
 * cli_calc_fuse_exec_user_op - Execute one frame of a kept Tier 2 operator @k
 */
int cli_calc_fuse_exec_user_op(fkept *k)
{
    fuse_user_op_t              *uop = k->uop;
    const fuse_user_call_t      *uc  = &k->p.ucall[0];
    const milk_calc_ext_entry_t *e   = cli_calc_ext_get(uop->ext_idx);

    if (e == NULL || e->get_ports == NULL || e->op_exec == NULL)
    {
        return -1;
    }
    const milk_calc_op_ports_t  *ports = e->get_ports();
    const milk_calc_param_set_t *pset  = (e->get_params != NULL) ? e->get_params() : NULL;

    double sval[FUSE_MAXINS];
    if (cli_calc_fuse_eval_scalars(&k->p, sval) != 0 || cli_calc_fuse_run_axes(&k->p) != 0)
    {
        return -1;
    }

    if (needs_reinit(k, ports, pset) && init_or_reinit_user_op(k, e, ports, pset, sval) != 0)
    {
        return -1;
    }

    milk_calc_op_ctx_t ctx;
    populate_op_ctx_inputs(k, e, ports, sval, &ctx);

    for (int j = 0; j < uop->ndst && j < ports->nout; j++)
    {
        if (ports->out_ports[j].kind == MILK_PORT_IMAGE)
        {
            if (ensure_out_image(uop, &ctx, j) != 0)
            {
                return -1;
            }
            ctx.out_img[j]              = &data.core.image[uop->out_id[j]];
            ctx.out_img[j]->md[0].write = 1;
        }
    }

    int rc = e->op_exec(&ctx, uc->pbuf, uop->state);

    for (int j = 0; j < uop->ndst && j < ports->nout; j++)
    {
        if (ports->out_ports[j].kind == MILK_PORT_IMAGE && ctx.out_img[j] != NULL)
        {
            ctx.out_img[j]->md[0].write = 0;
            if (rc == 0)
            {
                cli_calc_post_output(ctx.out_img[j]);
            }
        }
        else if (ports->out_ports[j].kind == MILK_PORT_SCALAR && rc == 0 && uop->out_vid[j] >= 0)
        {
            VARIABLE *dv = &data.core.variable[uop->out_vid[j]];
            dv->type     = 0;
            dv->value.f  = ctx.out_scalar[j];
        }
    }
    return (rc == 0) ? 0 : -1;
}

/**
 * cli_calc_fuse_free_user_op - Release state and descriptor of a Tier 2 operator
 */
void cli_calc_fuse_free_user_op(fuse_user_op_t *uop)
{
    if (uop == NULL)
    {
        return;
    }
    const milk_calc_ext_entry_t *e = cli_calc_ext_get(uop->ext_idx);
    if (uop->state != NULL && e != NULL && e->op_free != NULL)
    {
        e->op_free(uop->state);
        uop->state = NULL;
    }
    free(uop);
}

/**
 * cli_calc_eval_user_call - Evaluate a `TOK_FUNC_USER` call in non-kept expression mode
 */
val_t cli_calc_eval_user_call(const cli_token *ftok)
{
    int save_pos = cli_calc_tok_pos() - 1;
    if (save_pos >= 0)
    {
        cli_calc_tok_seek(save_pos);
        val_t fused;
        if (cli_calc_fuse_expr(100, &fused) > 0)
        {
            return fused;
        }
        const milk_calc_ext_entry_t *e = cli_calc_ext_get((int) ftok->val_l);
        if (e != NULL && e->kind == MILK_EXT_OP && e->get_ports != NULL &&
            e->get_ports()->nout == 1 && e->get_ports()->out_ports[0].kind == MILK_PORT_IMAGE)
        {
            cli_calc_tok_seek(save_pos);
            char        lhs[1][STRINGMAXLEN_IMAGE_NAME];
            int         loc[1] = { 1 };
            const char *tmpn   = alloc_tmpname();
            snprintf(lhs[0], sizeof(lhs[0]), "%s", tmpn);
            fkept *k = cli_calc_fuse_keep_user_op(1, lhs, loc, 0);
            if (k != NULL)
            {
                int rc = cli_calc_fused_run(k);
                cli_calc_kept_free(k);
                if (rc == 0)
                {
                    return mk_string(tmpn);
                }
            }
        }
    }
    parse_errmsg("Invalid arguments to user extension function");
    return mk_double(0);
}

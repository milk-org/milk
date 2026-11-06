// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "libmilkdata/milkdata.h"
#include "libprocessinfo/processinfo.h"
#include "COREMOD_arith/image_coord.h"
#include "COREMOD_memory/create_variable.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/stream_sem.h"
#include "COREMOD_memory/variable_ID.h"
#include "mc_ast.h"
#include "mc_plan.h"
#include "mcalc_host.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

#define MC_UDEF_MAX_DEPTH 4
#define MC_FOR_MAX_ITER 100000

typedef struct
{
    const mcalc_udef_t *u;
    mc_val_t            args[MCALC_UDEF_MAXARGS];
} mc_udef_frame_t;

static mc_udef_frame_t g_udef_stk[MC_UDEF_MAX_DEPTH];
static int             g_udef_depth    = 0;
static long            g_slice_end_val = -1;
static int             g_sync_host_cnt = 0;

void mc_scalar_set_slice_end(long end_val)
{
    g_slice_end_val = end_val;
}

int mc_scalar_sync_host(int delta)
{
    return (g_sync_host_cnt += delta);
}

__attribute__((format(printf, 3, 4))) static int sc_err(char       *err,
                                                        size_t      errsz,
                                                        const char *fmt,
                                                        ...)
{
    if (err != NULL && errsz > 0 && err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt), vsnprintf(err, errsz, fmt, ap), va_end(ap);
    }
    return -1;
}

static inline double to_double(const mc_val_t *v)
{
    return (v->type == MC_VAL_LONG) ? (double) v->lval : v->dval;
}

static inline int is_int_dt(uint8_t dt)
{
    return dt == _DATATYPE_UINT8 || dt == _DATATYPE_INT8 || dt == _DATATYPE_UINT16 ||
           dt == _DATATYPE_INT16 || dt == _DATATYPE_UINT32 || dt == _DATATYPE_INT32 ||
           dt == _DATATYPE_UINT64 || dt == _DATATYPE_INT64;
}

static void read_im_pix_val(const IMAGE *im, uint64_t ii, mc_val_t *out)
{
    uint8_t dt = im->md[0].datatype;
    *out       = (mc_val_t) { .type = is_int_dt(dt) ? MC_VAL_LONG : MC_VAL_DOUBLE };
#define READ_V_BODY(MBR) \
    out->lval = (long) im->array.MBR[ii], out->dval = (double) im->array.MBR[ii];
    MILK_FOR_EACH_DATATYPE(dt, READ_V_BODY, READ_V_BODY(D))
#undef READ_V_BODY
}

static inline double read_im_pix_d(const IMAGE *im, uint64_t ii)
{
    mc_val_t v;
    read_im_pix_val(im, ii, &v);
    return v.dval;
}

static void write_im_pix_val(IMAGE *im, uint64_t ii, double dval, long lval, int use_long)
{
    uint8_t dt = im->md[0].datatype;
#define WRITE_P_BODY(MBR) \
    im->array.MBR[ii] =   \
        use_long ? (__typeof__(im->array.MBR[0])) lval : (__typeof__(im->array.MBR[0])) dval;
    MILK_FOR_EACH_DATATYPE(dt, WRITE_P_BODY, WRITE_P_BODY(D))
#undef WRITE_P_BODY
}

void mc_store_scalar_var(const char *bare, long *io_vid, const mc_val_t *v)
{
    if (v->type == MC_VAL_STRING)
    {
        mc_host_str_set(bare, v->sval);
        return;
    }
    long vid =
        (io_vid != NULL && *io_vid >= 0 && *io_vid < milk_data.NB_MAX_VARIABLE &&
         milk_data.variable[*io_vid].used && strcmp(milk_data.variable[*io_vid].name, bare) == 0)
            ? *io_vid
            : -1;
    double dv = to_double(v);
    vid       = (v->type == MC_VAL_LONG)
                    ? ((vid >= 0) ? ((milk_data.variable[vid].type = 1),
                                     (milk_data.variable[vid].value.l = v->lval), vid)
                                  : create_variable_long_ID(bare, v->lval))
                    : ((vid >= 0) ? ((milk_data.variable[vid].type = 0),
                                     (milk_data.variable[vid].value.f = dv), vid)
                                  : create_variable_ID(bare, dv));
    (void) (io_vid != NULL && (*io_vid = vid, 1));
    if (g_sync_host_cnt > 0 && mc_host_has_str_sink())
    {
        char numv[64];
        if (v->type == MC_VAL_LONG)
        {
            snprintf(numv, sizeof(numv), "%ld", v->lval);
        }
        else
        {
            snprintf(numv, sizeof(numv), "%.*g", mcalc_float_digits, dv);
        }
        mc_host_str_set(bare, numv);
    }
}

static int eval_ident(const mc_expr_t *e, mc_val_t *out, char *err, size_t errsz)
{
    const char *bare = e->u.ident.bare[0] != '\0' ? e->u.ident.bare : e->u.ident.name;
    if (e->u.ident.loc == 0 && g_udef_depth > 0)
    {
        const mc_udef_frame_t *fr = &g_udef_stk[g_udef_depth - 1];
        for (int j = 0; j < fr->u->nargs; j++)
        {
            if ((e->u.ident.id_kind == MC_ID_DEF_PARAM && e->u.ident.def_param_idx == j) ||
                strcmp(fr->u->arg_names[j], bare) == 0)
            {
                *out = fr->args[j];
                return 0;
            }
        }
    }
    if (e->u.ident.id_kind == MC_ID_CONST)
    {
        *out = (mc_val_t) { .type = MC_VAL_DOUBLE, .dval = e->u.ident.const_val };
        return 0;
    }
    if (e->u.ident.id_kind == MC_ID_PROP)
    {
        double dv = 0.0;
        long   lv = 0;
        int    il = 0;
        if (milk_shm_prop_get_num(e->u.ident.name, &dv, &lv, &il) == 0)
        {
            *out = (mc_val_t) { .type = il ? MC_VAL_LONG : MC_VAL_DOUBLE, .lval = lv, .dval = dv };
            return 0;
        }
        return (milk_shm_prop_get(e->u.ident.name, out->sval, sizeof(out->sval)) == 0)
                   ? ((out->type = MC_VAL_STRING), 0)
                   : sc_err(err, errsz, "property not found");
    }

    long vid = (e->u.ident.var_id >= 0 && e->u.ident.var_id < milk_data.NB_MAX_VARIABLE &&
                milk_data.variable[e->u.ident.var_id].used &&
                strcmp(milk_data.variable[e->u.ident.var_id].name, bare) == 0)
                   ? e->u.ident.var_id
                   : variable_ID(bare);
    if (vid != -1 && milk_data.variable[vid].type != 3)
    {
        ((mc_expr_t *) e)->u.ident.var_id = vid;
        const VARIABLE *vp                = &milk_data.variable[vid];
        out->type = (vp->type == 1) ? MC_VAL_LONG : (vp->type == 2) ? MC_VAL_STRING : MC_VAL_DOUBLE;
        out->lval = vp->value.l;
        out->dval = vp->value.f;
        (void) (vp->type == 2 && snprintf(out->sval, sizeof(out->sval), "%s", vp->value.s));
        return 0;
    }

    const char *cval = mc_host_str_get(bare);
    if (cval == NULL && mc_is_numeric_env_var(bare))
    {
        cval = getenv(bare);
    }
    if (cval != NULL)
    {
        char  *ep1 = NULL, *ep2 = NULL;
        long   lv = strtol(cval, &ep1, 0);
        double dv = strtod(cval, &ep2);
        if (cval[0] != '\0' && (*ep1 == '\0' || *ep2 == '\0'))
        {
            *out = (*ep1 == '\0') ? (mc_val_t) { .type = MC_VAL_LONG, .lval = lv }
                                  : (mc_val_t) { .type = MC_VAL_DOUBLE, .dval = dv };
            return 0;
        }
        out->type = MC_VAL_STRING;
        snprintf(out->sval, sizeof(out->sval), "%s", cval);
        return 0;
    }

    if (e->u.ident.loc == 0 &&
        (strcmp(bare, "pi") == 0 || strcmp(bare, "nan") == 0 || strcmp(bare, "inf") == 0))
    {
        out->type = MC_VAL_DOUBLE;
        out->dval = (bare[0] == 'p') ? M_PI : (bare[0] == 'n') ? NAN : INFINITY;
        return 0;
    }

    const char *gone = milk_entity_ns_removed(e->u.ident.name);
    return sc_err(err, errsz, "undefined identifier '%s'%s%s", e->u.ident.name, gone ? ": " : "",
                  gone ? gone : "");
}

static int resolve_single_pix_idx(const mc_expr_t *s_expr,
                                  IMAGE          **out_im,
                                  uint64_t        *out_ii,
                                  char            *err,
                                  size_t           errsz)
{
    long imid = s_expr->u.slice.im_id;
    if (imid < 0 || imid >= milk_data.NB_MAX_IMAGE || !milk_data.image[imid].used)
    {
        imid = image_ID(s_expr->u.slice.bare, milk_data.image, milk_data.NB_MAX_IMAGE);
    }
    if (imid < 0)
    {
        return sc_err(err, errsz, "Source image not found");
    }
    IMAGE   *im     = &milk_data.image[imid];
    int      nax    = im->md[0].naxis;
    uint32_t ssz[3] = { (nax > 0) ? im->md[0].size[0] : 1, (nax > 1) ? im->md[0].size[1] : 1,
                        (nax > 2) ? im->md[0].size[2] : 1 };
    int32_t  idx[3] = { 0, 0, 0 };
    for (int ax = 0; ax < s_expr->u.slice.naxis && ax < 3; ax++)
    {
        mc_val_t v;
        long     saved_end = g_slice_end_val;
        g_slice_end_val    = (long) ssz[ax] - 1;
        int rc             = mc_scalar_eval_expr(s_expr->u.slice.axes[ax].start, &v, err, errsz);
        g_slice_end_val    = saved_end;
        if (rc != 0)
        {
            return -1;
        }
        int32_t raw = (v.type == MC_VAL_LONG) ? (int32_t) v.lval : (int32_t) llround(v.dval);
        int32_t sz  = (int32_t) ssz[ax];
        int32_t pos = (raw < 0) ? (sz + raw) : raw;
        if (pos < 0 || pos >= sz)
        {
            return sc_err(err, errsz, "axis %d start %d out of bounds for size %d", ax, raw, sz);
        }
        idx[ax] = pos;
    }
    *out_im = im;
    *out_ii = (uint64_t) idx[0] + (nax >= 2 ? (uint64_t) idx[1] * ssz[0] : 0) +
              (nax >= 3 ? (uint64_t) idx[2] * ssz[0] * ssz[1] : 0);
    return 0;
}

static int eval_call(const mc_expr_t *e, mc_val_t *out, char *err, size_t errsz)
{
    if (e->u.call.udef_id >= 0)
    {
        const mcalc_udef_t *u = mcalc_udef_get(e->u.call.udef_id);
        if (u == NULL || e->u.call.nargs != u->nargs || e->u.call.udef_body == NULL)
        {
            return sc_err(err, errsz, "invalid call to '%s'", e->u.call.name);
        }
        if (g_udef_depth >= MC_UDEF_MAX_DEPTH)
        {
            return sc_err(err, errsz, "User function call depth exceeded");
        }
        mc_udef_frame_t next_fr = { .u = u };
        for (int i = 0; i < u->nargs; i++)
        {
            if (mc_scalar_eval_expr(e->u.call.args[i], &next_fr.args[i], err, errsz) != 0)
            {
                return -1;
            }
        }
        g_udef_stk[g_udef_depth++] = next_fr;
        int rc                     = mc_scalar_eval_expr(e->u.call.udef_body, out, err, errsz);
        g_udef_depth--;
        return rc;
    }

    const mc_func_t *fn = e->u.call.fn;
    if (fn == NULL)
    {
        return sc_err(err, errsz, "not implemented in the new engine");
    }

    int nargs = e->u.call.nargs;
    if (fn->kind == MC_FN_LUT)
    {
        mc_val_t xval, x0val, dxval;
        if (mc_scalar_eval_expr(e->u.call.args[0], &xval, err, errsz) != 0)
        {
            return -1;
        }
        double x0 = 0.0, dx = 1.0;
        if (nargs == 4)
        {
            if (mc_scalar_eval_expr(e->u.call.args[2], &x0val, err, errsz) != 0 ||
                mc_scalar_eval_expr(e->u.call.args[3], &dxval, err, errsz) != 0)
            {
                return -1;
            }
            x0 = to_double(&x0val);
            dx = to_double(&dxval);
        }
        if (dx == 0.0)
        {
            return sc_err(err, errsz, "lut: dx must be non-zero");
        }
        const mc_expr_t *tble = e->u.call.args[1];
        long             tid  = (tble->kind == MC_EXPR_IDENT) ? tble->u.ident.im_id : -1;
        if (tid < 0 && tble->kind == MC_EXPR_IDENT)
        {
            tid = image_ID(tble->u.ident.bare, milk_data.image, milk_data.NB_MAX_IMAGE);
        }
        if (tid < 0)
        {
            return sc_err(err, errsz, "lut: expected lut(x, tbl[, x0, dx])");
        }
        const IMAGE *tbl = &milk_data.image[tid];
        int64_t      N   = (int64_t) tbl->md[0].nelement;
        out->type        = MC_VAL_DOUBLE;
        double u         = (to_double(&xval) - x0) / dx;
        if (N < 1 || isnan(u) || u <= 0.0 || u >= (double) (N - 1))
        {
            out->dval = (N < 1)    ? 0.0
                        : isnan(u) ? u
                                   : read_im_pix_d(tbl, (u <= 0.0) ? 0 : (uint64_t) (N - 1));
            return 0;
        }
        int64_t idx = (int64_t) u;
        double  v0  = read_im_pix_d(tbl, (uint64_t) idx);
        out->dval   = v0 + (u - (double) idx) * (read_im_pix_d(tbl, (uint64_t) (idx + 1)) - v0);
        return 0;
    }

    if ((fn->kind == MC_FN_REDUCE_D || fn->kind == MC_FN_REDUCE_L || fn->kind == MC_FN_DOT ||
         fn->kind == MC_FN_EAGER_RED) &&
        e->res_kind != MC_RES_IMAGE)
    {
        return mc_exec_reduce_scalar(e, out, err, errsz);
    }

    mc_val_t args[MC_MAX_ARGS];
    for (int i = 0; i < nargs; i++)
    {
        if (mc_scalar_eval_expr(e->u.call.args[i], &args[i], err, errsz) != 0)
        {
            return -1;
        }
    }

    switch (fn->kind)
    {
    case MC_FN_SCALAR:
        for (int i = 0; i < nargs; i++)
        {
            if (args[i].type == MC_VAL_STRING)
            {
                return sc_err(err, errsz, "String cannot be converted to number");
            }
        }
        if (strcmp(fn->name, "pow") == 0)
        {
            mc_scalar_pow(&args[0], &args[1], out);
            return 0;
        }
        out->type = MC_VAL_DOUBLE;
        out->dval = (nargs == 1 && fn->fn1 != NULL) ? fn->fn1(to_double(&args[0]))
                    : (nargs == 2 && fn->fn2 != NULL)
                        ? fn->fn2(to_double(&args[0]), to_double(&args[1]))
                        : fn->fn3(to_double(&args[0]), to_double(&args[1]), to_double(&args[2]));
        return 0;

    case MC_FN_WHERE:
        if (args[0].type == MC_VAL_STRING)
        {
            return sc_err(err, errsz, "String cannot be converted to number");
        }
        *out = (to_double(&args[0]) != 0.0) ? args[1] : args[2];
        return 0;

    case MC_FN_PRNG:
    {
        int      kind = (strcmp(fn->name, "randn") == 0 || strcmp(fn->name, "grand") == 0)
                            ? ARITH_COORD_RANDN
                            : ARITH_COORD_RAND;
        uint32_t seed = (nargs == 1) ? (uint32_t) (int64_t) llround(to_double(&args[0]))
                                     : arith_next_rng_seed();
        out->type     = MC_VAL_DOUBLE;
        out->dval     = arith_rng_draw(kind, seed);
        return 0;
    }

    case MC_FN_STR:
    {
        const char *serr = mc_scalar_str_fn(fn, args, out);
        return (serr != NULL) ? sc_err(err, errsz, "%s", serr) : 0;
    }

    case MC_FN_CPLX:
        if (args[0].type == MC_VAL_STRING)
        {
            return sc_err(err, errsz, "String cannot be converted to number");
        }
        *out = (strcmp(fn->name, "real") == 0 || strcmp(fn->name, "conj") == 0)
                   ? args[0]
                   : (mc_val_t) { .type = MC_VAL_DOUBLE,
                                  .dval = (strcmp(fn->name, "imag") == 0)
                                              ? 0.0
                                              : atan2(0.0, to_double(&args[0])) };
        return 0;

    default:
        return sc_err(err, errsz, "not implemented in the new engine");
    }
}

static int apply_compound_op(mc_tok_type_t   op,
                             const mc_val_t *oldv,
                             const mc_val_t *rhsv,
                             mc_val_t       *out,
                             char           *err,
                             size_t          errsz)
{
    if (op == MC_TOK_DIV_ASSIGN)
    {
        double rv = to_double(rhsv);
        out->type = MC_VAL_DOUBLE;
        out->dval = (rv == 0.0) ? 0.0 : to_double(oldv) / rv;
        return (rv == 0.0) ? sc_err(err, errsz, "Division by zero") : 0;
    }
    mc_tok_type_t bop  = (op == MC_TOK_ADD_ASSIGN)   ? MC_TOK_PLUS
                         : (op == MC_TOK_SUB_ASSIGN) ? MC_TOK_MINUS
                                                     : MC_TOK_MUL;
    const char   *berr = mc_scalar_binop(bop, oldv, rhsv, out);
    return (berr != NULL) ? sc_err(err, errsz, "%s", berr) : 0;
}

static int eval_assign(const mc_expr_t *e, mc_val_t *out, char *err, size_t errsz)
{
    const mc_expr_t *target = e->u.assign.target;
    mc_tok_type_t    op     = e->u.assign.op;
    mc_val_t         rv;

    if (mc_scalar_eval_expr(e->u.assign.rhs, &rv, err, errsz) != 0)
    {
        return -1;
    }

    if (target->kind == MC_EXPR_SLICE)
    {
        if (!target->u.slice.is_single_pixel || rv.type == MC_VAL_STRING)
        {
            return sc_err(err, errsz, "not implemented in the new engine");
        }
        IMAGE   *im = NULL;
        uint64_t ii = 0;
        if (resolve_single_pix_idx(target, &im, &ii, err, errsz) != 0)
        {
            return -1;
        }
        double nv = to_double(&rv);
        if (op != MC_TOK_ASSIGN)
        {
            mc_val_t oldv = { .type = MC_VAL_DOUBLE, .dval = read_im_pix_d(im, ii) }, comb;
            if (apply_compound_op(op, &oldv, &rv, &comb, err, errsz) != 0)
            {
                return -1;
            }
            nv = to_double(&comb);
        }
        im->md[0].write = 1;
        write_im_pix_val(im, ii, nv, rv.lval,
                         rv.type == MC_VAL_LONG && op == MC_TOK_ASSIGN &&
                             is_int_dt(im->md[0].datatype));
        if (!im->md[0].shared)
        {
            im->md[0].write = 0;
            im->md[0].cnt0++;
        }
        else
        {
            processinfo_update_output_stream(NULL, im, NULL);
        }
        out->type = (rv.type == MC_VAL_LONG && op == MC_TOK_ASSIGN) ? MC_VAL_LONG : MC_VAL_DOUBLE;
        out->lval = rv.lval;
        out->dval = nv;
        return 0;
    }

    if (target->kind != MC_EXPR_IDENT)
    {
        return sc_err(err, errsz, "invalid assignment target");
    }

    const char *bare =
        target->u.ident.bare[0] != '\0' ? target->u.ident.bare : target->u.ident.name;
    mc_val_t final_v = rv;
    if (op != MC_TOK_ASSIGN)
    {
        mc_val_t oldv;
        if (eval_ident(target, &oldv, err, errsz) != 0)
        {
            return (target->u.ident.id_kind == MC_ID_PROP)
                       ? -1
                       : sc_err(err, errsz, "Variable not found for compound assign");
        }
        if (apply_compound_op(op, &oldv, &rv, &final_v, err, errsz) != 0)
        {
            return -1;
        }
    }

    if (target->u.ident.id_kind == MC_ID_PROP)
    {
        char sbuf[MC_TOK_STR_MAX];
        mc_val_to_str(&final_v, sbuf, sizeof(sbuf), 1);
        if (milk_shm_prop_set(target->u.ident.name, sbuf, NULL) != 0)
        {
            return sc_err(err, errsz, "property write failed");
        }
    }
    else
    {
        mc_store_scalar_var(bare, &((mc_expr_t *) target)->u.ident.var_id, &final_v);
    }
    *out = final_v;
    return 0;
}

int mc_scalar_eval_expr(const mc_expr_t *e, mc_val_t *out, char *err, size_t errsz)
{
    if (e == NULL || e->needs_image)
    {
        return sc_err(err, errsz,
                      (e == NULL) ? "syntax error" : "not implemented in the new engine");
    }

    switch (e->kind)
    {
    case MC_EXPR_LONG:
    case MC_EXPR_DOUBLE:
    case MC_EXPR_STRING:
    case MC_EXPR_END:
        out->type = (e->kind == MC_EXPR_LONG || e->kind == MC_EXPR_END) ? MC_VAL_LONG
                    : (e->kind == MC_EXPR_DOUBLE)                       ? MC_VAL_DOUBLE
                                                                        : MC_VAL_STRING;
        out->lval = (e->kind == MC_EXPR_END && g_slice_end_val >= 0) ? g_slice_end_val : e->u.lval;
        out->dval = e->u.dval;
        if (e->kind == MC_EXPR_STRING)
        {
            snprintf(out->sval, sizeof(out->sval), "%s", e->u.sval);
        }
        return 0;

    case MC_EXPR_IDENT:
        return eval_ident(e, out, err, errsz);

    case MC_EXPR_SLICE:
    {
        IMAGE   *im = NULL;
        uint64_t ii = 0;
        if (!e->u.slice.is_single_pixel)
        {
            return sc_err(err, errsz, "not implemented in the new engine");
        }
        if (resolve_single_pix_idx(e, &im, &ii, err, errsz) != 0)
        {
            return -1;
        }
        read_im_pix_val(im, ii, out);
        return 0;
    }

    case MC_EXPR_UNARY:
    {
        mc_val_t v;
        if (mc_scalar_eval_expr(e->u.unary.operand, &v, err, errsz) != 0)
        {
            return -1;
        }
        if (v.type == MC_VAL_STRING)
        {
            return sc_err(err, errsz,
                          (e->u.unary.op == MC_TOK_MINUS) ? "Cannot negate expression"
                                                          : "Invalid unary op on string");
        }
        if (e->u.unary.op == MC_TOK_MINUS)
        {
            out->type = v.type;
            out->lval = (long) (-(uint64_t) v.lval);
            out->dval = -v.dval;
            return 0;
        }
        out->type = MC_VAL_LONG;
        out->lval = (e->u.unary.op == MC_TOK_BNOT)
                        ? ~((v.type == MC_VAL_LONG) ? v.lval : (long) v.dval)
                        : (to_double(&v) == 0.0);
        return 0;
    }

    case MC_EXPR_BINARY:
    {
        mc_tok_type_t op = e->u.binary.op;
        mc_val_t      l, r;
        if (mc_scalar_eval_expr(e->u.binary.left, &l, err, errsz) != 0)
        {
            return -1;
        }
        if (op == MC_TOK_AND || op == MC_TOK_OR)
        {
            if (l.type == MC_VAL_STRING)
            {
                return sc_err(err, errsz, "Logical op on string");
            }
            int lt = (to_double(&l) != 0.0);
            if ((op == MC_TOK_AND && !lt) || (op == MC_TOK_OR && lt))
            {
                *out = (mc_val_t) { .type = MC_VAL_LONG, .lval = (op == MC_TOK_OR) };
                return 0;
            }
            if (mc_scalar_eval_expr(e->u.binary.right, &r, err, errsz) != 0 ||
                r.type == MC_VAL_STRING)
            {
                return (r.type == MC_VAL_STRING) ? sc_err(err, errsz, "Logical op on string") : -1;
            }
            *out = (mc_val_t) { .type = MC_VAL_LONG, .lval = (to_double(&r) != 0.0) };
            return 0;
        }

        if (mc_scalar_eval_expr(e->u.binary.right, &r, err, errsz) != 0)
        {
            return -1;
        }

        if (l.type == MC_VAL_STRING || r.type == MC_VAL_STRING)
        {
            char s1[MC_TOK_STR_MAX], s2[MC_TOK_STR_MAX];
            mc_val_to_str(&l, s1, sizeof(s1), 1);
            mc_val_to_str(&r, s2, sizeof(s2), 1);
            if (op == MC_TOK_PLUS)
            {
                if (strlen(s1) + strlen(s2) >= sizeof(out->sval))
                {
                    return sc_err(err, errsz, "String concatenation overflow");
                }
                snprintf(out->sval, sizeof(out->sval), "%s%s", s1, s2);
                out->type = MC_VAL_STRING;
                return 0;
            }
            if ((op == MC_TOK_EQ || op == MC_TOK_NE) && l.type == MC_VAL_STRING &&
                r.type == MC_VAL_STRING)
            {
                int eq = (strcmp(s1, s2) == 0);
                *out   = (mc_val_t) { .type = MC_VAL_LONG, .lval = (op == MC_TOK_EQ) ? eq : !eq };
                return 0;
            }
            return sc_err(err, errsz, "Cannot apply operator to string");
        }

        const char *berr = mc_scalar_binop(op, &l, &r, out);
        return (berr != NULL) ? sc_err(err, errsz, "%s", berr) : 0;
    }

    case MC_EXPR_CHAIN:
    {
        mc_val_t prev;
        if (mc_scalar_eval_expr(e->u.chain.operands[0], &prev, err, errsz) != 0)
        {
            return -1;
        }
        for (int i = 0; i < e->u.chain.nops; i++)
        {
            mc_val_t cur, cmp;
            if (mc_scalar_eval_expr(e->u.chain.operands[i + 1], &cur, err, errsz) != 0)
            {
                return -1;
            }
            const char *berr = mc_scalar_binop(e->u.chain.ops[i], &prev, &cur, &cmp);
            if (berr != NULL)
            {
                return sc_err(err, errsz, "%s", berr);
            }
            if (cmp.lval == 0)
            {
                *out = (mc_val_t) { .type = MC_VAL_LONG, .lval = 0 };
                return 0;
            }
            prev = cur;
        }
        *out = (mc_val_t) { .type = MC_VAL_LONG, .lval = 1 };
        return 0;
    }

    case MC_EXPR_TERNARY:
    {
        mc_val_t cv;
        if (mc_scalar_eval_expr(e->u.ternary.cond, &cv, err, errsz) != 0)
        {
            return -1;
        }
        if (cv.type == MC_VAL_STRING)
        {
            return sc_err(err, errsz, "Condition cannot be string");
        }
        return mc_scalar_eval_expr(
            (to_double(&cv) != 0.0) ? e->u.ternary.t_expr : e->u.ternary.f_expr, out, err, errsz);
    }

    case MC_EXPR_CALL:
        return eval_call(e, out, err, errsz);

    case MC_EXPR_ASSIGN:
        return eval_assign(e, out, err, errsz);

    default:
        return sc_err(err, errsz, "not implemented in the new engine");
    }
}

int mc_scalar_exec_stmt(const mc_stmt_t *s, mc_val_t *out, char *err, size_t errsz)
{
    if (s == NULL || (s->kind == MC_STMT_EXPR && s->u.expr->res_kind == MC_RES_IMAGE))
    {
        return sc_err(err, errsz,
                      (s == NULL) ? "syntax error" : "not implemented in the new engine");
    }

    switch (s->kind)
    {
    case MC_STMT_EMPTY:
    case MC_STMT_DEF:
    case MC_STMT_SKIP:
        (void) (out != NULL && (*out = (mc_val_t) { .type = MC_VAL_LONG, .lval = 0 }, 1));
        return 0;

    case MC_STMT_EXPR:
    {
        mc_val_t v;
        if (mc_scalar_eval_expr(s->u.expr, &v, err, errsz) != 0)
        {
            return -1;
        }
        (void) (out != NULL && (*out = v, 1));
        return 0;
    }

    case MC_STMT_MULTI_ASSIGN:
    {
        if (s->u.multi.nrhs == 1 && s->u.multi.nlhs >= 2)
        {
            return mc_exec_multi_reduce(s, out, err, errsz);
        }
        if (s->u.multi.nrhs != s->u.multi.nlhs)
        {
            return sc_err(err, errsz, "not implemented in the new engine");
        }
        mc_val_t vals[MC_MAX_LHS] = { 0 };
        for (int j = 0; j < s->u.multi.nrhs; j++)
        {
            const mc_expr_t *lhs = s->u.multi.lhs[j], *rhs = s->u.multi.rhs[j];
            if (lhs->u.ident.id_kind != MC_ID_IMAGE && lhs->u.ident.id_kind != MC_ID_NEW &&
                rhs->res_kind != MC_RES_IMAGE &&
                mc_scalar_eval_expr(rhs, &vals[j], err, errsz) != 0)
            {
                return -1;
            }
        }
        for (int j = 0; j < s->u.multi.nlhs; j++)
        {
            mc_expr_t       *lhs = s->u.multi.lhs[j];
            const mc_expr_t *rhs = s->u.multi.rhs[j];
            if (lhs->u.ident.id_kind != MC_ID_IMAGE && lhs->u.ident.id_kind != MC_ID_NEW &&
                rhs->res_kind != MC_RES_IMAGE)
            {
                mc_store_scalar_var(lhs->u.ident.bare[0] != '\0' ? lhs->u.ident.bare
                                                                 : lhs->u.ident.name,
                                    &lhs->u.ident.var_id, &vals[j]);
            }
        }
        (void) (out != NULL && (*out = vals[s->u.multi.nlhs - 1], 1));
        return 0;
    }

    case MC_STMT_IF:
        for (int b = 0; b < s->u.if_stmt.nbranches; b++)
        {
            const mc_if_branch_t *br = &s->u.if_stmt.branches[b];
            mc_val_t              cv = { .type = MC_VAL_LONG, .lval = 1 };
            if (br->cond != NULL && mc_scalar_eval_expr(br->cond, &cv, err, errsz) != 0)
            {
                return -1;
            }
            if (to_double(&cv) != 0.0)
            {
                return mc_scalar_exec_prog(br->body, err, errsz);
            }
        }
        return 0;

    case MC_STMT_FOR:
    {
        mc_val_t vstart;
        if (mc_scalar_eval_expr(s->u.for_stmt.start, &vstart, err, errsz) != 0)
        {
            return -1;
        }
        long vid = -1;
        mc_store_scalar_var(s->u.for_stmt.var, &vid, &vstart);
        int iters = 0;
        while (1)
        {
            mc_val_t vstop, vstep = { .type = MC_VAL_LONG, .lval = 1 };
            if (mc_scalar_eval_expr(s->u.for_stmt.stop, &vstop, err, errsz) != 0 ||
                (s->u.for_stmt.step != NULL &&
                 mc_scalar_eval_expr(s->u.for_stmt.step, &vstep, err, errsz) != 0))
            {
                return -1;
            }
            double stpv = to_double(&vstep);
            if (stpv == 0.0)
            {
                return sc_err(err, errsz, "for loop step cannot be 0");
            }
            vid                  = (vid < 0) ? variable_ID(s->u.for_stmt.var) : vid;
            const VARIABLE *vv   = &milk_data.variable[vid];
            double          cur  = (vv->type == 1) ? (double) vv->value.l : vv->value.f;
            double          endv = to_double(&vstop);
            if (!((stpv > 0.0) ? (cur <= endv + 1e-12) : (cur >= endv - 1e-12)))
            {
                break;
            }
            if (iters++ >= MC_FOR_MAX_ITER ||
                mc_scalar_exec_prog(s->u.for_stmt.body, err, errsz) != 0)
            {
                return (iters > MC_FOR_MAX_ITER)
                           ? sc_err(err, errsz, "for loop exceeded %d iterations", MC_FOR_MAX_ITER)
                           : -1;
            }
            VARIABLE *vmut = &milk_data.variable[vid];
            int       use_l =
                (vmut->type == 1 && (vstep.type == MC_VAL_LONG || stpv == (double) (long) stpv));
            mc_val_t nextv =
                use_l ? (mc_val_t) { .type = MC_VAL_LONG,
                                     .lval =
                                         vmut->value.l +
                                         ((vstep.type == MC_VAL_LONG) ? vstep.lval : (long) stpv) }
                      : (mc_val_t) { .type = MC_VAL_DOUBLE, .dval = cur + stpv };
            mc_store_scalar_var(s->u.for_stmt.var, &vid, &nextv);
        }
        return 0;
    }
    }
    return -1;
}

int mc_scalar_exec_prog(const mc_stmt_t *prog, char *err, size_t errsz)
{
    for (const mc_stmt_t *cur = prog; cur != NULL; cur = cur->next)
    {
        if (cur->needs_image && cur->kind != MC_STMT_IF && cur->kind != MC_STMT_FOR)
        {
            return mc_exec_stmt_list((mc_stmt_t *) prog, err, errsz);
        }
    }
    for (const mc_stmt_t *cur = prog; cur != NULL; cur = cur->next)
    {
        if (cur->kind == MC_STMT_SKIP)
        {
            return 0;
        }
        mc_val_t v;
        if (mc_scalar_exec_stmt(cur, &v, err, errsz) != 0)
        {
            return -1;
        }
    }
    return 0;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse.c
 * @brief User-function inlining and expression-to-bytecode compiler for fused programs.
 *
 * Role & Goal:
 *   Compiles calculator token streams into `fprog` vector and scalar bytecode tapes,
 *   including user-defined function inlining via L1d register slots (`FI_STORE_SLOT` /
 *   `FI_LOAD_SLOT`) and live shared-memory property reads (`FI_PROP`).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_calc_fuse_internal.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

static int keep_vars;
/* > 0 while compiling a branch of `?:` or where(): a division by zero there is selected away */
static int guard_depth;
static char
    reject_msg[200]; /* why the current keep attempt is not fusible: the first reason noted */

/**
 * cli_calc_fuse_reject - Note why the expression being compiled cannot be fused
 * @msg: Reason, kept only if none was noted since the last `cli_calc_fuse_reject_reset()`
 *
 * Return: void
 */
void cli_calc_fuse_reject(const char *msg)
{
    if (reject_msg[0] == '\0' && msg != NULL)
    {
        snprintf(reject_msg, sizeof(reject_msg), "%s", msg);
    }
}

/**
 * cli_calc_fuse_reject_reset - Forget the noted reason before a new keep attempt
 *
 * Return: void
 */
void cli_calc_fuse_reject_reset(void)
{
    reject_msg[0] = '\0';
}

/**
 * cli_calc_fuse_reject_reason - Why the last keep attempt was not fused
 *
 * Return: The noted reason, or "not fusible" when none was noted.
 */
const char *cli_calc_fuse_reject_reason(void)
{
    return (reject_msg[0] != '\0') ? reject_msg : "not fusible";
}

/** mark_guarded - Flag the instructions emitted since (@n0, @ns0) as living inside a select branch. */
static void mark_guarded(fprog *p, int n0, int ns0)
{
    for (int i = n0; i < p->n; i++)
    {
        p->ins[i].guarded = 1;
    }
    for (int i = ns0; i < p->ns; i++)
    {
        p->sins[i].guarded = 1;
    }
}

static int c_select_scalar_cond(fprog *p, int n0, fval cond, fval t, fval f, fval *out);

/**
 * cli_calc_fuse_set_keep_vars - Toggle whether scalar variables compile to live `FI_SVAR` slots
 * @on: 1 to emit `FI_SVAR`/`FI_PROP` instructions, 0 to fold scalar variables at compile time
 *
 * Return: void
 */
void cli_calc_fuse_set_keep_vars(int on)
{
    keep_vars = on;
}

/**
 * cli_calc_fuse_get_keep_vars - Return current `keep_vars` compilation mode
 *
 * Return: 1 if live scalar variable emission is active, 0 otherwise.
 */
int cli_calc_fuse_get_keep_vars(void)
{
    return keep_vars;
}

/** fusible_op - Return 1 if binary operator @op is supported by the fused vector/scalar engines. */
static int fusible_op(cli_token_type op)
{
    return (op >= TOK_OP_PLUS && op <= TOK_OP_NEQ) || op == TOK_OP_AND || op == TOK_OP_OR ||
           (op >= TOK_OP_BAND && op <= TOK_OP_BXOR) || op == TOK_OP_LSHIFT || op == TOK_OP_RSHIFT;
}

/** fusible_arith() - 1 for the operators the superinstructions implement. */
static inline int fusible_arith(cli_token_type op)
{
    return op == TOK_OP_PLUS || op == TOK_OP_MINUS || op == TOK_OP_STAR || op == TOK_OP_SLASH;
}

int cli_calc_fuse_emit_bin(fprog *p, fins bi, int both)
{
    const cli_token_type op   = bi.op;
    const fins          *last = (p->n >= 1) ? &p->ins[p->n - 1] : NULL;
    const fins          *prev = (p->n >= 2) ? &p->ins[p->n - 2] : NULL;

    if (!both && fusible_arith(op) && last != NULL && last->k == FI_IMG)
    {
        /* image op c, c op image: one pass loads, converts and applies */
        fins f = { .k       = FI_IMGC,
                   .op      = op,
                   .img     = last->img,
                   .cpos    = bi.cpos,
                   .s1      = bi.s1,
                   .c       = bi.c,
                   .guarded = bi.guarded };
        p->n--;
        p->depth--;
        return emit(p, f, 1);
    }
    if (both && fusible_arith(op) && last != NULL && prev != NULL && last->k == FI_IMG &&
        prev->k == FI_IMG)
    {
        /* image op image: two loads and the operator in one pass */
        fins f = {
            .k = FI_IMGIMG, .op = op, .img = prev->img, .cpos = last->img, .guarded = bi.guarded
        };
        p->n -= 2;
        p->depth -= 2;
        return emit(p, f, 1);
    }
    if (both && fusible_arith(op) && last != NULL && last->k == FI_IMGC && last->op == TOK_OP_STAR)
    {
        /* top op (image * c): the right operand is read and scaled in the operator's pass */
        fins f = { .k       = FI_VIMGC,
                   .op      = op,
                   .img     = last->img,
                   .s1      = last->s1,
                   .c       = last->c,
                   .guarded = bi.guarded };
        p->n--;
        p->depth--;
        p->nops--; /* the folded FI_IMGC was counted as an operation */
        return emit(p, f, 0);
    }
    if (both && fusible_arith(op) && last != NULL && last->k == FI_IMG)
    {
        /* top op image: the right operand is read in the operator's pass */
        fins f = { .k = FI_VIMGC, .op = op, .img = last->img, .c = 1.0, .guarded = bi.guarded };
        p->n--;
        p->depth--;
        return emit(p, f, 0);
    }
    return emit(p, bi, both ? -1 : 0);
}

uint8_t cli_calc_fuse_fn1_kind(double (*fn)())
{
    double (*f)(double) = (double (*)(double)) fn;
    static const struct
    {
        double (*f)(double);
        uint8_t k;
    } table[] = {
        { fabs, FN1_FABS },
        { sqrt, FN1_SQRT },
        { floor, FN1_FLOOR },
        { ceil, FN1_CEIL },
        { round, FN1_ROUND },
        { sin, FN1_SIN },
        { cos, FN1_COS },
        { tan, FN1_TAN },
        { sinh, FN1_SINH },
        { cosh, FN1_COSH },
        { tanh, FN1_TANH },
        { exp, FN1_EXP },
        { exp2, FN1_EXP2 },
        { log, FN1_LOG },
        { log10, FN1_LOG10 },
        { log2, FN1_LOG2 },
        { erf, FN1_ERF },
        { erfc, FN1_ERFC },
        { atan, FN1_ATAN },
        { asin, FN1_ASIN },
        { acos, FN1_ACOS },
        { cbrt, FN1_CBRT },
        { cli_calc_posi, FN1_POSI },
        { cli_calc_sign, FN1_SIGN },
    };
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
    {
        if (table[i].f == f)
        {
            return table[i].k;
        }
    }
    return FN1_OTHER;
}

/**
 * cli_calc_fuse_c_args - Compile @n comma-separated arguments and consume the closing `')'`
 * @p: Target fused program
 * @n: Number of arguments expected
 * @a: Output array of @n compiled `fval` descriptors
 *
 * Return: 0 on success, -1 on syntax/compilation error.
 */
int cli_calc_fuse_c_args(fprog *p, int n, fval *a)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    for (int i = 0; i < n; i++)
    {
        if (i > 0)
        {
            if (cur()->type != TOK_COMMA)
            {
                return -1;
            }
            adv();
        }
        if (cli_calc_fuse_c_expr(p, 0, &a[i]) != 0)
        {
            return -1;
        }
    }
    if (cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    return 0;
}

#define UDEF_MAX_DEPTH 4
typedef struct
{
    const cli_calc_udef_t *u;
    fval                   arg[CALC_UDEF_MAXARGS];
    int                    slot[CALC_UDEF_MAXARGS];
    int                    img[CALC_UDEF_MAXARGS];
} udef_frame;

static udef_frame udef_stk[UDEF_MAX_DEPTH];
static int        udef_depth = 0;

/**
 * cli_calc_fuse_udef_depth - Return current user-function inlining call depth
 *
 * Return: Active user-function nesting depth (`0..UDEF_MAX_DEPTH`).
 */
int cli_calc_fuse_udef_depth(void)
{
    return udef_depth;
}

/** udef_lookup_param - Find @name in the innermost active user function frame. */
static int udef_lookup_param(const char *name, const udef_frame **out_uf)
{
    if (udef_depth <= 0 || strchr(name, '[') != NULL)
    {
        return -1;
    }
    const udef_frame *uf = &udef_stk[udef_depth - 1];
    for (int j = 0; j < uf->u->nargs; j++)
    {
        if (strcmp(name, uf->u->arg_names[j]) == 0)
        {
            *out_uf = uf;
            return j;
        }
    }
    return -1;
}

/**
 * cli_calc_fuse_udef_param_img - Return leaf image slot index bound to user-function parameter @name
 * @name: Parameter identifier
 *
 * Return: Image index in `p->img[]` (`>= 0`) if @name is a leaf image parameter, or -1 otherwise.
 */
int cli_calc_fuse_udef_param_img(const char *name)
{
    const udef_frame *uf = NULL;
    int               pj = udef_lookup_param(name, &uf);
    return (pj >= 0) ? uf->img[pj] : -1;
}

/** c_udef_bind_vec_arg - Allocate a register slot for vector argument @j of @frame. */
static int c_udef_bind_vec_arg(fprog *p, udef_frame *frame, int j, int n_before)
{
    if (p->n == n_before + 1 && p->ins[n_before].k == FI_IMG &&
        !p->view[p->ins[n_before].img].is_view)
    {
        frame->img[j] = p->ins[n_before].img;
    }
    else if (p->n == n_before + 1 && p->ins[n_before].k == FI_LOAD_SLOT && udef_depth > 0)
    {
        const udef_frame *outer = &udef_stk[udef_depth - 1];
        for (int oj = 0; oj < outer->u->nargs; oj++)
        {
            if (outer->slot[oj] == p->ins[n_before].cpos)
            {
                frame->img[j] = outer->img[oj];
                break;
            }
        }
    }
    if (p->nreg >= FUSE_MAXREG)
    {
        cli_calc_fuse_reject("more than 4 def image arguments live at once (FUSE_MAXREG)");
        return -1;
    }
    int slot       = p->nreg++;
    frame->slot[j] = slot;
    if (p->nreg > p->max_nreg)
    {
        p->max_nreg = p->nreg;
    }
    return emit(p, (fins) { .k = FI_STORE_SLOT, .cpos = slot }, -1);
}

/** c_udef - Compile a user-defined inline function call via L1d register slots. */
static int c_udef(fprog *p, const cli_token *ft, fval *out)
{
    cli_token *(*cur)(void)  = cli_calc_cur_func();
    cli_token *(*adv)(void)  = cli_calc_advance_func();
    const cli_calc_udef_t *u = cli_calc_udef_get((int) ft->val_l);

    if (u == NULL || udef_depth >= UDEF_MAX_DEPTH)
    {
        cli_calc_fuse_reject((u == NULL) ? "unknown def" : "def calls nested deeper than 4");
        return -1;
    }
    udef_frame frame;
    memset(&frame, 0, sizeof(frame));
    frame.u        = u;
    int saved_nreg = p->nreg;

    for (int j = 0; j < u->nargs; j++)
    {
        if (j > 0)
        {
            if (cur()->type != TOK_COMMA)
            {
                p->nreg = saved_nreg;
                return -1;
            }
            adv();
        }
        int n_before  = p->n;
        frame.img[j]  = -1;
        frame.slot[j] = -1;
        if (cli_calc_fuse_c_expr(p, 0, &frame.arg[j]) != 0 ||
            (frame.arg[j].vec && c_udef_bind_vec_arg(p, &frame, j, n_before) != 0))
        {
            p->nreg = saved_nreg;
            return -1;
        }
    }
    if (cur()->type != TOK_RPAREN)
    {
        p->nreg = saved_nreg;
        return -1;
    }
    adv();

    cli_token body_toks[CLI_CALC_MAX_TOKENS];
    int       ntok = cli_tokenize(u->body, body_toks, CLI_CALC_MAX_TOKENS);
    if (ntok <= 0)
    {
        p->nreg = saved_nreg;
        return -1;
    }
    cli_calc_tok_state saved_tok;
    udef_stk[udef_depth++] = frame;
    cli_calc_tok_push(body_toks, ntok, &saved_tok);
    int rc = cli_calc_fuse_c_expr(p, 0, out);
    if (rc == 0 && cli_calc_cur_func()()->type != TOK_EOF &&
        cli_calc_cur_func()()->type != TOK_NEWLINE)
    {
        rc = -1;
    }
    cli_calc_tok_pop(&saved_tok);
    udef_depth--;
    p->nreg = saved_nreg;
    return rc;
}

/** c_primary_var_or_img - Compile a variable, user-function parameter, or image operand token. */
static int c_primary_var_or_img(fprog *p, const cli_token *t, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    if (cur()->type >= TOK_EQUAL && cur()->type <= TOK_OP_SLASH_EQ)
    {
        return -1;
    }
    const udef_frame *uf = NULL;
    int               pj = udef_lookup_param(t->sval, &uf);
    if (pj >= 0)
    {
        if (uf->arg[pj].vec)
        {
            out->vec = 1;
            out->s1  = 0;
            return emit(p, (fins) { .k = FI_LOAD_SLOT, .cpos = uf->slot[pj] }, 1);
        }
        *out = uf->arg[pj];
        return 0;
    }
    if (t->type == TOK_IMAGE)
    {
        int k;
        if (strchr(t->sval, '[') != NULL)
        {
            k = cli_calc_fuse_add_operand(p, NULL, t->sval);
        }
        else
        {
            imageID id = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
            k          = (id == -1) ? -1 : cli_calc_fuse_add_image(p, &data.core.image[id]);
        }
        out->vec = 1;
        out->s1  = 0;
        return (k < 0) ? -1 : emit(p, (fins) { .k = FI_IMG, .img = k }, 1);
    }
    long vid = keep_vars ? variable_ID(t->sval) : -1;
    if (vid >= 0 && data.core.variable[vid].type != 2)
    {
        return emit_scalar(p,
                           (fsins) { .k     = FI_SVAR,
                                     .vid   = (int) vid,
                                     .vhash = name_hash32(data.core.variable[vid].name) },
                           out);
    }
    if (keep_vars)
    {
        char why[128];
        snprintf(why, sizeof(why), "'%.60s' is not a numeric variable or an image", t->sval);
        cli_calc_fuse_reject(why);
        return -1;
    }
    val_t v = eval_existing_var_ref(t->sval);
    if (v.type != VAL_LONG && v.type != VAL_DOUBLE)
    {
        cli_calc_fuse_reject("string operand");
        return -1;
    }
    *out = (fval) { 0, 0, to_double(v), v.type == VAL_LONG };
    return 0;
}

/** c_primary_nvar - Compile a `TOK_NVAR` token (either SHM property or new variable). */
static int c_primary_nvar(fprog *p, const cli_token *t, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    const char *dot1        = strchr(t->sval, '.');
    if (dot1 != NULL && strchr(dot1 + 1, '.') != NULL &&
        milk_entity_ns_kind(t->sval, NULL) != MILK_NS_NONE)
    {
        if (cur()->type >= TOK_EQUAL && cur()->type <= TOK_OP_SLASH_EQ)
        {
            return -1;
        }
        double val     = 0.0;
        long   lval    = 0;
        int    is_long = 0;
        if (milk_shm_prop_get_num(t->sval, &val, &lval, &is_long) != 0)
        {
            return -1;
        }
        if (is_long)
        {
            val = (double) lval;
        }
        if (!keep_vars)
        {
            *out = (fval) { 0, 0, val, is_long };
            return 0;
        }
        int idx = -1;
        for (int i = 0; i < p->nprop; i++)
        {
            if (strcmp(p->prop_path[i], t->sval) == 0)
            {
                idx = i;
                break;
            }
        }
        if (idx < 0)
        {
            if (p->nprop >= FUSE_MAXPROP)
            {
                cli_calc_fuse_reject("more than 8 live properties (FUSE_MAXPROP)");
                return -1;
            }
            idx = p->nprop++;
            snprintf(p->prop_path[idx], sizeof(p->prop_path[idx]), "%s", t->sval);
        }
        return emit_scalar(
            p, (fsins) { .k = FI_PROP, .vid = idx, .cl = { (uint8_t) is_long, 0, 0 } }, out);
    }
    return c_primary_var_or_img(p, t, out);
}

/** c_primary_func - Dispatch function-call tokens in `c_primary`. */
static int c_primary_func(fprog *p, const cli_token *t, fval *out)
{
    switch (t->type)
    {
    case TOK_FUNC_D_D:
    case TOK_FUNC_DD_D:
    case TOK_FUNC_DDD_D:
        return cli_calc_fuse_c_func(p, t, out);
    case TOK_FUNC_WHERE:
    {
        cli_token *(*cur)(void) = cli_calc_cur_func();
        cli_token *(*adv)(void) = cli_calc_advance_func();
        fval a[3];
        if (cli_calc_fuse_c_expr(p, 0, &a[0]) != 0 || cur()->type != TOK_COMMA)
        {
            return -1;
        }
        adv();
        int n0  = p->n;
        int ns0 = p->ns;
        guard_depth++;
        int rc = cli_calc_fuse_c_expr(p, 0, &a[1]);
        if (rc == 0 && cur()->type == TOK_COMMA)
        {
            adv();
            rc = cli_calc_fuse_c_expr(p, 0, &a[2]);
        }
        else
        {
            rc = -1;
        }
        guard_depth--;
        if (rc != 0 || cur()->type != TOK_RPAREN)
        {
            return -1;
        }
        adv();
        mark_guarded(p, n0, ns0);
        if (!a[0].vec)
        {
            if (a[1].vec || a[2].vec)
            {
                return c_select_scalar_cond(p, n0, a[0], a[1], a[2], out);
            }
            if (!keep_vars)
            {
                return -1;
            }
            return emit_scalar(
                p,
                (fsins) { .k  = FI_SEL,
                          .s  = { a[0].s1, a[1].s1, a[2].s1 },
                          .c  = { a[0].c, a[1].c, a[2].c },
                          .cl = { (uint8_t) a[0].lng, (uint8_t) a[1].lng, (uint8_t) a[2].lng } },
                out);
        }
        return cli_calc_fuse_c_select(p, a[1], a[2], out);
    }
    case TOK_FUNC_COORD:
        return cli_calc_fuse_c_coord(p, t, out);
    case TOK_FUNC_CPLX:
    {
        int which = (strncmp(t->sval, "real(", 5) == 0)   ? ARITH_CPLX_REAL
                    : (strncmp(t->sval, "imag(", 5) == 0) ? ARITH_CPLX_IMAG
                    : (strncmp(t->sval, "arg(", 4) == 0)  ? ARITH_CPLX_ARG
                                                          : -1;
        return (which < 0) ? -1 : cli_calc_fuse_c_cplx_part(p, which, out);
    }
    case TOK_FUNC_IM_D:
    case TOK_FUNC_IM_L:
    case TOK_FUNC_IMIM_D:
        return cli_calc_fuse_c_reduce(p, t, out);
    case TOK_FUNC_SHIFT:
        return cli_calc_fuse_c_shift(p, out);
    case TOK_FUNC_LUT:
        return cli_calc_fuse_c_lut(p, out);
    case TOK_FUNC_MEDFILT:
        return cli_calc_fuse_c_medfilt(p, t, out);
    case TOK_FUNC_PIXREMAP:
        return cli_calc_fuse_c_pixremap(p, out);
    case TOK_FUNC_PIXUNMAP:
        return cli_calc_fuse_c_pixunmap(p, t, out);
    case TOK_FUNC_GAUSS:
        return cli_calc_fuse_c_gauss(p, out);
    case TOK_FUNC_CUMSUM:
        return cli_calc_fuse_c_cumsum(p, out);
    case TOK_FUNC_INTERP2:
        return cli_calc_fuse_c_interp2(p, out);
    case TOK_FUNC_UDEF:
        return c_udef(p, t, out);
    case TOK_FUNC_USER:
        return cli_calc_fuse_c_user(p, t, out);
    case TOK_FUNC_IMIM_IM:
        return cli_calc_fuse_c_linalg(p, t, out);
    default:
    {
        char why[128];
        snprintf(why, sizeof(why), "function %.40s) has no fused form", t->sval);
        cli_calc_fuse_reject(why);
        return -1;
    }
    } /* switch (t->type) */
}

/** c_primary - Compile a primary expression atom into a fused program. */
static int c_primary(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    cli_token *t            = adv();

    switch (t->type)
    {
    case TOK_LONG:
        *out = (fval) { 0, 0, (double) t->val_l, 1 };
        return 0;
    case TOK_DOUBLE:
        *out = (fval) { 0, 0, t->val_d, 0 };
        return 0;
    case TOK_OP_MINUS:
    case TOK_OP_NOT:
        if (c_primary(p, out) != 0)
        {
            return -1;
        }
        if (!out->vec)
        {
            if (out->s1 == 0)
            {
                out->c   = (t->type == TOK_OP_MINUS) ? -out->c : (out->c == 0.0);
                out->lng = (t->type == TOK_OP_MINUS) ? out->lng : 1;
                return 0;
            }
            return emit_scalar(p,
                               (fsins) { .k = (t->type == TOK_OP_MINUS) ? FI_NEG : FI_NOT,
                                         .s = { out->s1, 0, 0 } },
                               out);
        }
        p->out_dt = 0;
        return emit(p, (fins) { .k = (t->type == TOK_OP_MINUS) ? FI_NEG : FI_NOT }, 0);
    case TOK_OP_BNOT:
        if (c_primary(p, out) != 0)
        {
            return -1;
        }
        if (!out->vec)
        {
            if (out->s1 == 0)
            {
                out->c   = (double) (~(int64_t) out->c);
                out->lng = 1;
                return 0;
            }
            return emit_scalar(
                p, (fsins) { .k = FI_BNOT, .s = { out->s1, 0, 0 }, .cl = { 1, 0, 0 } }, out);
        }
        p->out_dt = 0;
        return emit(p, (fins) { .k = FI_BNOT }, 0);
    case TOK_LPAREN:
        if (cli_calc_fuse_c_expr(p, 0, out) != 0 || cur()->type != TOK_RPAREN)
        {
            return -1;
        }
        adv();
        return 0;
    case TOK_NVAR:
        return c_primary_nvar(p, t, out);
    case TOK_VAR:
    case TOK_IMAGE:
        return c_primary_var_or_img(p, t, out);
    default:
        return c_primary_func(p, t, out);
    } /* switch (t->type) */
}

/**
 * c_select_scalar_cond - Select between branches on scalar condition @cond when a branch is a vector
 * @p:    Target fused program
 * @n0:   Instruction index where the branches' code starts
 * @cond: Scalar condition (constant or scalar-tape slot)
 * @t:    True branch
 * @f:    False branch
 * @out:  Output value descriptor
 *
 * The condition is pushed as a constant block below the branches, so the per-pixel select applies.
 *
 * Return: 0 on success, -1 on capacity or stack overflow.
 */
static int c_select_scalar_cond(fprog *p, int n0, fval cond, fval t, fval f, fval *out)
{
    if (p->n >= FUSE_MAXINS || p->maxdepth + 1 + p->max_nreg > FUSE_MAXSTK)
    {
        return -1;
    }
    memmove(&p->ins[n0 + 1], &p->ins[n0], sizeof(fins) * (size_t) (p->n - n0));
    p->ins[n0] = (fins) { .k = FI_CONST, .s1 = cond.s1, .c = cond.c };
    p->n++;
    p->depth++;
    p->maxdepth++;
    return cli_calc_fuse_c_select(p, t, f, out);
}

static int c_expr_ternary(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    fval t;
    fval f;
    adv();
    int n0  = p->n;
    int ns0 = p->ns;
    guard_depth++;
    int rc = cli_calc_fuse_c_expr(p, 0, &t);
    if (rc == 0 && cur()->type == TOK_OP_COLON)
    {
        adv();
        rc = cli_calc_fuse_c_expr(p, 0, &f);
    }
    else
    {
        rc = -1;
    }
    guard_depth--;
    if (rc != 0)
    {
        return -1;
    }
    mark_guarded(p, n0, ns0);
    if (!out->vec)
    {
        if (t.vec || f.vec)
        {
            return c_select_scalar_cond(p, n0, *out, t, f, out);
        }
        fsins si = { .k  = FI_SEL,
                     .s  = { out->s1, t.s1, f.s1 },
                     .c  = { out->c, t.c, f.c },
                     .cl = { (uint8_t) out->lng, (uint8_t) t.lng, (uint8_t) f.lng } };
        return (t.vec || f.vec || emit_scalar(p, si, out) != 0) ? -1 : 0;
    }
    return cli_calc_fuse_c_select(p, t, f, out);
}

/**
 * cli_calc_fuse_c_expr - Compile an expression with minimum precedence @min_prec into @p
 * @p:        Target fused program
 * @min_prec: Minimum operator precedence to consume
 * @out:      Output value descriptor
 *
 * Return: 0 on success, -1 if expression cannot be fused.
 */
int cli_calc_fuse_c_expr(fprog *p, int min_prec, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();

    if (c_primary(p, out) != 0)
    {
        return -1;
    }
    for (;;)
    {
        cli_token_type op   = cur()->type;
        int            prec = cli_calc_op_prec(op);
        if (prec < min_prec)
        {
            return 0;
        }
        if (op == TOK_OP_QUESTION && (out->vec || keep_vars))
        {
            if (c_expr_ternary(p, out) != 0)
            {
                return -1;
            }
            continue;
        }
        if (!fusible_op(op) || (!out->vec && !keep_vars && (op == TOK_OP_AND || op == TOK_OP_OR) &&
                                (out->s1 != 0 || (out->c != 0.0) == (op == TOK_OP_OR))))
        {
            cli_calc_fuse_reject(fusible_op(op) ? "scalar && / || with a known result"
                                                : "operator has no fused form");
            return -1;
        }
        adv();
        fval r;
        if (cli_calc_fuse_c_expr(p, (op == TOK_OP_CARET) ? prec : prec + 1, &r) != 0)
        {
            return -1;
        }
        p->out_dt = 0;
        if (!out->vec && !r.vec)
        {
            if (guard_depth == 0 && (op == TOK_OP_SLASH || op == TOK_OP_MOD) && r.s1 == 0 &&
                r.c == 0.0)
            {
                return -1;
            }
            if (out->s1 == 0 && r.s1 == 0)
            {
                val_t       lr;
                const char *err = NULL;
                if (out->lng && r.lng && op != TOK_OP_SLASH &&
                    cli_calc_long_binop(op, (long) out->c, (long) r.c, &lr, &err))
                {
                    out->lng = (lr.type == VAL_LONG); /* folded as the text engine would */
                    out->c   = out->lng ? (double) lr.lval : lr.dval;
                }
                else
                {
                    out->c   = scalar_bin(op, out->c, r.c);
                    out->lng = bin_is_long(op, out->lng, r.lng);
                }
            }
            else if (emit_scalar(p,
                                 (fsins) { .k  = FI_BIN,
                                           .op = op,
                                           .s  = { out->s1, r.s1, 0 },
                                           .c  = { out->c, r.c, 0.0 },
                                           .cl = { (uint8_t) out->lng, (uint8_t) r.lng, 0 } },
                                 out) != 0)
            {
                return -1;
            }
            continue;
        }
        if ((guard_depth == 0 && !r.vec && r.s1 == 0 && (op == TOK_OP_SLASH || op == TOK_OP_MOD) &&
             r.c == 0.0) ||
            (!out->vec && (op == TOK_OP_MOD || op == TOK_OP_CARET)))
        {
            return -1;
        }
        int  both = out->vec && r.vec;
        fins bi   = { .k    = FI_BIN,
                      .op   = op,
                      .cpos = both ? 0 : (out->vec ? 2 : 1),
                      .s1   = out->vec ? r.s1 : out->s1,
                      .c    = out->vec ? r.c : out->c };
        if (cli_calc_fuse_emit_bin(p, bi, both) != 0)
        {
            return -1;
        }
        out->vec = 1;
        out->s1  = 0;
    } /* for (;;) */
}

/**
 * cli_calc_fuse_compile - Reset @p and compile an expression from the current token cursor
 * @p:        Target fused program
 * @min_prec: Minimum operator precedence
 * @v:        Output value descriptor
 *
 * Return: 0 on success, -1 on error (restoring token position).
 */
int cli_calc_fuse_compile(fprog *p, int min_prec, fval *v)
{
    int pos = cli_calc_tok_pos();
    cli_calc_fuse_reset_fprog(p);
    if (cli_calc_fuse_c_expr(p, min_prec, v) != 0)
    {
        cli_calc_fuse_free_fprog_children(p);
        cli_calc_tok_seek(pos);
        return -1;
    }
    return 0;
}

/**
 * cli_calc_fuse_compile_more - Compile an additional expression into an already-initialized @p
 * @p:        Target fused program
 * @min_prec: Minimum operator precedence
 * @v:        Output value descriptor
 *
 * Return: 0 on success, -1 on error (restoring token position).
 */
int cli_calc_fuse_compile_more(fprog *p, int min_prec, fval *v)
{
    int pos = cli_calc_tok_pos();
    if (cli_calc_fuse_c_expr(p, min_prec, v) != 0)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }
    return 0;
}

/**
 * cli_calc_coord_kind - Map coordinate/PRNG function token @t to its `CALC_COORD_*` enum value
 * @t: Function token
 *
 * Return: `CALC_COORD_*` constant.
 */
int cli_calc_coord_kind(const cli_token *t)
{
    if (strncmp(t->sval, "rand(", 5) == 0 || strncmp(t->sval, "urand(", 6) == 0)
    {
        return CALC_COORD_RAND;
    }
    if (strncmp(t->sval, "randn(", 6) == 0 || strncmp(t->sval, "grand(", 6) == 0)
    {
        return CALC_COORD_RANDN;
    }
    return (strncmp(t->sval, "ix(", 3) == 0)   ? CALC_COORD_X
           : (strncmp(t->sval, "iy(", 3) == 0) ? CALC_COORD_Y
           : (strncmp(t->sval, "ir(", 3) == 0) ? CALC_COORD_R
                                               : CALC_COORD_THETA;
}

/**
 * cli_calc_eval_const_scalar - Evaluate a compile-time scalar expression at the token cursor
 * @out: Output double value
 *
 * Return: 0 on success, -1 if expression is non-scalar or fails to evaluate.
 */
int cli_calc_eval_const_scalar(double *out)
{
    fprog *tp = (fprog *) calloc(1, sizeof(*tp));
    if (tp == NULL)
    {
        return -1;
    }
    fval dv;
    int  save_keep = keep_vars;
    int  save_anc  = cli_calc_fuse_anchor_save();
    char save_msg[sizeof(reject_msg)];
    memcpy(save_msg, reject_msg,
           sizeof(save_msg)); /* a speculative compile: its reasons are not kept */
    cli_calc_fuse_reset_fprog(tp);
    keep_vars = 0;
    int rc    = cli_calc_fuse_c_expr(tp, 0, &dv);
    keep_vars = save_keep;
    cli_calc_fuse_anchor_restore(save_anc);
    memcpy(reject_msg, save_msg, sizeof(reject_msg));
    cli_calc_fuse_free_fprog_children(tp);
    free(tp);
    if (rc != 0 || dv.vec || dv.s1 != 0)
    {
        return -1;
    }
    *out = dv.c;
    return 0;
}

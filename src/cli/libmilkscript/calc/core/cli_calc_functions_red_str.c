// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_functions_red_str.c
 * @brief Calculator image-reduction fallbacks, string/format built-ins, and user inline functions.
 *
 * Role & Goal:
 *   Evaluates full-image, single-axis, and masked image reductions (`sum`, `mean`,
 *   `min`, `max`, `std`, `var`, `norm`, `asum`, `count`, `dot`, `perc`, `median`),
 *   string and base-formatting built-ins (`strlen`, `toupper`, `tolower`, `substr`,
 *   `replace`, `hex`, `oct`, `bin`), and user-defined inline functions (`def f(x) = ...`).
 */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "../fuse/cli_calc_fuse_internal.h" /* RED_ARGMAX */

/**
 * reduce_axis - Reduce image @name along @axis into a temporary image with that axis removed
 * @name: Source image name
 * @axis: Axis index to reduce along (`0 .. naxis - 1`)
 * @op:   Reduction operator (`RED_*`)
 *
 * Return: `VAL_DOUBLE` for 1D inputs, else `VAL_STRING` naming the temporary output image
 */
static val_t reduce_axis(const char *name, long axis, reduce_op op)
{
    IMAGE  *im  = &data.core.image[image_ID(name, data.core.image, data.core.NB_MAX_IMAGE)];
    int     nax = im->md[0].naxis;
    uint8_t dt  = im->md[0].datatype;
    if (axis < 0 || axis >= nax)
    {
        parse_errmsg("Axis out of range");
        return mk_double(0);
    }
    if (dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        parse_errmsg("Axis reduction of a complex image is not supported");
        return mk_double(0);
    }

    uint32_t osz[3];
    int      onax = 0;
    for (int a = 0; a < nax; a++)
    {
        if (a != axis)
        {
            osz[onax++] = im->md[0].size[a];
        }
    }

    if (onax == 0)
    {
        return mk_double(arith_image_reduce_lane(im, dt, 0, im->md[0].size[axis], 1, op));
    }

    int         out_d = !(dt == _DATATYPE_FLOAT || dt == _DATATYPE_HALF || dt == _DATATYPE_UINT8 ||
                          dt == _DATATYPE_INT8 || dt == _DATATYPE_UINT16 || dt == _DATATYPE_INT16);
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    cli_calc_tmp_image(tmpn, onax, osz, out_d ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return mk_string("");
    }
    arith_image_reduce_axis_im(im, axis, op, &data.core.image[tid]);
    return mk_string(tmpn);
}

/**
 * reduce_masked - Reduce pixels of image @name where @mask is non-zero
 * @name: Source image name
 * @mask: Mask image name
 * @op:   Reduction operator
 * @f:    Percentile fraction for `RED_PERC`
 *
 * Return: Reduced scalar value as `VAL_DOUBLE`
 */
static val_t reduce_masked(const char *name, const char *mask, reduce_op op, double f)
{
    const IMAGE *im = &data.core.image[image_ID(name, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *mk = &data.core.image[image_ID(mask, data.core.image, data.core.NB_MAX_IMAGE)];

    int same = (mk->md[0].naxis == im->md[0].naxis);
    for (int a = 0; same && a < im->md[0].naxis; a++)
    {
        same = (mk->md[0].size[a] == im->md[0].size[a]);
    }
    if (!same)
    {
        parse_errmsg("The mask must have the image's size");
        return mk_double(0);
    }
    return mk_double(arith_image_reduce_masked_im(im, mk, op, f));
}

/**
 * axis_reduce_op - Return the `reduce_op` enum for reduction function @fn, or -1
 * @fn: Reduction C function pointer
 *
 * Return: `RED_*` enum value, or -1 if @fn does not support axis/masked reduction
 */
static int axis_reduce_op(double (*fn)())
{
    if (fn == (double (*)()) arith_image_total)
    {
        return RED_SUM;
    }
    if (fn == (double (*)()) arith_image_mean)
    {
        return RED_MEAN;
    }
    if (fn == (double (*)()) arith_image_min)
    {
        return RED_MIN;
    }
    if (fn == (double (*)()) arith_image_max)
    {
        return RED_MAX;
    }
    if (fn == (double (*)()) arith_image_rms)
    {
        return RED_STD;
    }
    if (fn == (double (*)()) arith_image_var)
    {
        return RED_VAR;
    }
    if (fn == (double (*)()) arith_image_norm)
    {
        return RED_NORM;
    }
    if (fn == (double (*)()) arith_image_asum)
    {
        return RED_ASUM;
    }
    return -1;
}

/**
 * cli_calc_fused_reduce_op - Return the `reduce_op` of @fn for fused block reduction, or -1
 * @fn: Reduction C function pointer
 *
 * Return: `RED_*` enum value, or -1
 */
int cli_calc_fused_reduce_op(double (*fn)())
{
    int op = axis_reduce_op(fn);
    if (op >= 0)
    {
        return op;
    }
    if (fn == (double (*)()) arith_image_count)
    {
        return RED_COUNT;
    }
    if (fn == (double (*)()) arith_image_dot)
    {
        return RED_DOT;
    }
    if (fn == (double (*)()) arith_image_argmax)
    {
        return RED_ARGMAX;
    }
    if (fn == (double (*)()) arith_image_argmin)
    {
        return RED_ARGMIN;
    }
    if (fn == (double (*)()) arith_image_iamax)
    {
        return RED_AMAX;
    }
    if (fn == (double (*)()) arith_image_trace)
    {
        return RED_TRACE;
    }
    return -1;
}

/**
 * eval_unary_image_reduction - Evaluate `TOK_FUNC_IM_D` or `TOK_FUNC_IM_L` with optional axis/mask
 * @ftok: Reduction function token
 *
 * Return: Reduced scalar or axis-reduced image `val_t`
 */
static val_t eval_unary_image_reduction(const cli_token *ftok)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();

    val_t arg      = parse_expr(0);
    val_t axisv    = mk_long(-1);
    int   has_axis = 0;
    if (!parse_error && !eval_error && cur_func()->type == TOK_COMMA)
    {
        advance_func();
        axisv    = parse_expr(0);
        has_axis = 1;
    }
    if (parse_error || eval_error)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_double(0);
    }
    advance_func();
    if (arg.type != VAL_STRING)
    {
        parse_errmsg("Expected image argument");
        return mk_double(0);
    }
    if (!check_image(arg.sval) || cli_calc_reject_complex(&arg, 1))
    {
        return mk_double(0);
    }
    if (has_axis)
    {
        int op = (ftok->type == TOK_FUNC_IM_D) ? axis_reduce_op(ftok->fnctptr) : -1;
        if (axisv.type == VAL_STRING)
        {
            op = (ftok->fnctptr == (double (*)()) arith_image_median) ? RED_PERC : op;
            if (op < 0)
            {
                parse_errmsg("This function takes no mask argument");
                return mk_double(0);
            }
            if (!check_image(axisv.sval) || cli_calc_reject_complex(&axisv, 1))
            {
                return mk_double(0);
            }
            return reduce_masked(arg.sval, axisv.sval, (reduce_op) op, 0.5);
        }
        if (op < 0)
        {
            parse_errmsg("This function takes no axis argument");
            return mk_double(0);
        }
        return reduce_axis(arg.sval, (long) to_double(axisv), (reduce_op) op);
    }
    if (ftok->fnctptr == NULL)
    {
        parse_errmsg("Multi-return reduction must be unpacked into variables");
        return mk_double(0);
    }
    double res = ((double (*)(const char *)) ftok->fnctptr)(arg.sval);
    return (ftok->type == TOK_FUNC_IM_L) ? mk_long((long) res) : mk_double(res);
}

/**
 * cli_calc_func_eval_image_reduction - Evaluate image-to-scalar or axis reduction function @ftok
 * @ftok: Reduction function token
 *
 * Return: Evaluated `val_t`
 */
val_t cli_calc_func_eval_image_reduction(const cli_token *ftok)
{
    int fop = cli_calc_fused_reduce_op(ftok->fnctptr);
    if (fop >= 0)
    {
        val_t fout;
        if (cli_calc_fuse_reduce((reduce_op) fop, ftok->type == TOK_FUNC_IM_L, &fout))
        {
            return fout;
        }
    }

    if (ftok->type == TOK_FUNC_IM_D || ftok->type == TOK_FUNC_IM_L)
    {
        return eval_unary_image_reduction(ftok);
    }

    val_t args[3];
    int   nargs =
        cli_calc_parse_func_argv(2, (ftok->type == TOK_FUNC_IMD_D) ? 3 : 2, args, "Expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, nargs))
    {
        return mk_double(0);
    }
    if (ftok->type == TOK_FUNC_IMD_D)
    {
        if (args[0].type != VAL_STRING || (nargs == 3 && args[2].type != VAL_STRING))
        {
            parse_errmsg(nargs == 3 ? "Expected perc(image, fraction, mask)"
                                    : "Expected image argument");
            return mk_double(0);
        }
        if (!check_image(args[0].sval) || (nargs == 3 && !check_image(args[2].sval)))
        {
            return mk_double(0);
        }
        if (nargs == 3)
        {
            return reduce_masked(args[0].sval, args[2].sval, RED_PERC, to_double(args[1]));
        }
        double res =
            ((double (*)(const char *, double)) ftok->fnctptr)(args[0].sval, to_double(args[1]));
        return mk_double(res);
    }

    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING)
    {
        parse_errmsg("Expected two image arguments");
        return mk_double(0);
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_double(0);
    }
    double res =
        ((double (*)(const char *, const char *)) ftok->fnctptr)(args[0].sval, args[1].sval);
    return mk_double(res);
}

/**
 * cli_calc_func_replace - Evaluate `replace(s, old, new)` returning a modified string
 *
 * Return: Modified `VAL_STRING` value
 */
val_t cli_calc_func_replace(void)
{
    val_t args[3];
    if (cli_calc_parse_func_argv(3, 3, args, "replace: need 3 args") < 0)
    {
        return mk_double(0);
    }

    const char *s[3] = { NULL, NULL, NULL };
    for (int i = 0; i < 3; i++)
    {
        if (args[i].type == VAL_STRING)
        {
            const char *cv = cli_var_get(args[i].sval);
            s[i]           = cv ? cv : args[i].sval;
        }
    }
    if (!s[0] || !s[1] || !s[2])
    {
        parse_errmsg("replace: all args must be strings");
        return mk_double(0);
    }

    char result[CLI_CALC_TOKEN_MAXLEN];
    result[0] = '\0';
    int s2len = (int) strlen(s[1]);
    if (s2len == 0)
    {
        strncpy(result, s[0], sizeof(result) - 1);
        result[sizeof(result) - 1] = '\0';
    }
    else
    {
        const char *p   = s[0];
        char       *w   = result;
        char       *end = result + sizeof(result) - 1;
        while (*p && w < end)
        {
            if (strncmp(p, s[1], (size_t) s2len) == 0)
            {
                int rlen = (int) strlen(s[2]);
                if (w + rlen < end)
                {
                    memcpy(w, s[2], (size_t) rlen);
                    w += rlen;
                }
                p += s2len;
            }
            else
            {
                *w++ = *p++;
            }
        }
        *w = '\0';
    }

    return mk_string(result);
}

/**
 * eval_substr_or_base_format - Evaluate `substr(s, off, len)` or `hex/oct/bin(n)`
 * @ftok: Function token (`TOK_FUNC_SDD_S` or `TOK_FUNC_D_S`)
 *
 * Return: Formatted `VAL_STRING` value
 */
static val_t eval_substr_or_base_format(const cli_token *ftok)
{
    if (ftok->type == TOK_FUNC_SDD_S)
    {
        val_t args[3];
        if (cli_calc_parse_func_argv(3, 3, args, "substr: need 3 args") < 0)
        {
            return mk_double(0);
        }
        if (args[0].type != VAL_STRING)
        {
            parse_errmsg("substr: first arg must be string");
            return mk_double(0);
        }
        const char *cv   = cli_var_get(args[0].sval);
        const char *sv   = cv ? cv : args[0].sval;
        int         slen = (int) strlen(sv);
        int         off  = (int) to_double(args[1]);
        int         len  = (int) to_double(args[2]);
        off              = (off < 0) ? 0 : (off > slen) ? slen : off;
        len              = (len < 0) ? 0 : (off + len > slen) ? (slen - off) : len;
        char result[CLI_CALC_TOKEN_MAXLEN];
        memcpy(result, sv + off, (size_t) len);
        result[len] = '\0';
        return mk_string(result);
    }

    val_t arg;
    if (cli_calc_parse_func_argv(1, 1, &arg, NULL) < 0)
    {
        return mk_double(0);
    }
    long        iv = (arg.type == VAL_LONG) ? arg.lval : (long) to_double(arg);
    char        result[CLI_CALC_TOKEN_MAXLEN];
    const char *fn = ftok->sval;
    if (strncmp(fn, "hex(", 4) == 0)
    {
        snprintf(result, sizeof(result), "0x%lx", iv);
    }
    else if (strncmp(fn, "oct(", 4) == 0)
    {
        snprintf(result, sizeof(result), "0o%lo", iv);
    }
    else
    {
        char *w = result;
        *w++    = '0';
        *w++    = 'b';
        if (iv == 0)
        {
            *w++ = '0';
        }
        else
        {
            unsigned long v = (unsigned long) iv;
            char          bits[65];
            int           bi = 0;
            while (v > 0 && bi < 64)
            {
                bits[bi++] = (v & 1) ? '1' : '0';
                v >>= 1;
            }
            for (int i = bi - 1; i >= 0; i--)
            {
                *w++ = bits[i];
            }
        }
        *w = '\0';
    }
    return mk_string(result);
}

/**
 * cli_calc_func_eval_string_or_format - Evaluate `strlen`, `toupper`, `tolower`, `substr`, `hex`, `oct`, `bin`
 * @ftok: Function token
 *
 * Return: Evaluated `val_t`
 */
val_t cli_calc_func_eval_string_or_format(const cli_token *ftok)
{
    if (ftok->type == TOK_FUNC_S_D)
    {
        val_t arg;
        if (cli_calc_parse_func_argv(1, 1, &arg, NULL) < 0)
        {
            return mk_double(0);
        }
        if (arg.type == VAL_STRING)
        {
            const char *cv = cli_var_get(arg.sval);
            return mk_long((long) strlen(cv != NULL ? cv : arg.sval));
        }
        char numstr[64];
        if (arg.type == VAL_LONG)
        {
            snprintf(numstr, sizeof(numstr), "%ld", arg.lval);
        }
        else
        {
            snprintf(numstr, sizeof(numstr), "%g", arg.dval);
        }
        return mk_long((long) strlen(numstr));
    }

    if (ftok->type == TOK_FUNC_S_S)
    {
        val_t arg;
        if (cli_calc_parse_func_argv(1, 1, &arg, NULL) < 0)
        {
            return mk_double(0);
        }
        const char *sv = NULL;
        char        numbuf[64];
        if (arg.type == VAL_STRING)
        {
            const char *cv = cli_var_get(arg.sval);
            sv             = cv ? cv : arg.sval;
        }
        else if (arg.type == VAL_LONG)
        {
            snprintf(numbuf, sizeof(numbuf), "%ld", arg.lval);
            sv = numbuf;
        }
        else
        {
            snprintf(numbuf, sizeof(numbuf), "%g", arg.dval);
            sv = numbuf;
        }

        char result[CLI_CALC_TOKEN_MAXLEN];
        strncpy(result, sv, sizeof(result) - 1);
        result[sizeof(result) - 1] = '\0';

        int is_upper = (strncmp(ftok->sval, "toupper(", 8) == 0);
        for (int i = 0; result[i]; i++)
        {
            result[i] = is_upper ? (char) toupper((unsigned char) result[i])
                                 : (char) tolower((unsigned char) result[i]);
        }
        return mk_string(result);
    }

    return eval_substr_or_base_format(ftok);
}

#define UDEF_EVAL_MAX_DEPTH 4
typedef struct
{
    const cli_calc_udef_t *u;
    val_t                  arg[CALC_UDEF_MAXARGS];
} udef_eval_frame;

static udef_eval_frame udef_eval_stk[UDEF_EVAL_MAX_DEPTH];
static int             udef_eval_depth = 0;

/**
 * cli_calc_udef_eval_lookup - Look up a bound parameter value in the active user-function frame
 * @name: Identifier name to resolve
 * @out:  Output parameter value when matched
 *
 * Return: 1 if @name matched a parameter of the active user-defined function, 0 otherwise
 */
int cli_calc_udef_eval_lookup(const char *name, val_t *out)
{
    if (udef_eval_depth <= 0 || strchr(name, '[') != NULL)
    {
        return 0;
    }
    const udef_eval_frame *uf = &udef_eval_stk[udef_eval_depth - 1];
    for (int j = 0; j < uf->u->nargs; j++)
    {
        if (strcmp(name, uf->u->arg_names[j]) == 0)
        {
            *out = uf->arg[j];
            return 1;
        }
    }
    return 0;
}

/**
 * cli_calc_func_udef - Evaluate a user-defined inline function call (`TOK_FUNC_UDEF`)
 * @ftok: User-defined function token
 *
 * Return: Evaluated `val_t`
 */
val_t cli_calc_func_udef(const cli_token *ftok)
{
    const cli_calc_udef_t *u = cli_calc_udef_get((int) ftok->val_l);
    if (u == NULL || udef_eval_depth >= UDEF_EVAL_MAX_DEPTH)
    {
        parse_errmsg("User function call depth exceeded");
        return mk_double(0);
    }
    udef_eval_frame frame;
    frame.u = u;
    if (cli_calc_parse_func_argv(u->nargs, u->nargs, frame.arg, "Expected ','") < 0)
    {
        return mk_double(0);
    }
    cli_token body_toks[CLI_CALC_MAX_TOKENS];
    int       ntok = cli_tokenize(u->body, body_toks, CLI_CALC_MAX_TOKENS);
    if (ntok <= 0)
    {
        return mk_double(0);
    }
    cli_calc_tok_state saved_tok;
    udef_eval_stk[udef_eval_depth++] = frame;
    cli_calc_tok_push(body_toks, ntok, &saved_tok);
    val_t res = parse_expr(0);
    cli_calc_tok_pop(&saved_tok);
    udef_eval_depth--;
    return res;
}

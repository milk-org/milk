// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "libmilkdata/milk_type_dispatch.h"

/**
 * @brief Read pixel @p ii of @p im as a double.
 */
static double read_image_pixel_as_double(const IMAGE *im, uint64_t ii)
{
    uint8_t dt = im->md[0].datatype;
#define READ_PIX_BODY(MBR) return (double) im->array.MBR[ii];
    MILK_FOR_EACH_DATATYPE(dt, READ_PIX_BODY, READ_PIX_BODY(D))
#undef READ_PIX_BODY
    return 0.0;
}

/**
 * @brief Write @p val to pixel @p ii of @p im.
 */
static void write_image_pixel_from_double(IMAGE *im, uint64_t ii, double val)
{
    uint8_t dt = im->md[0].datatype;
#define WRITE_PIX_BODY(MBR)                                     \
    {                                                           \
        im->array.MBR[ii] = (__typeof__(im->array.MBR[0])) val; \
        return;                                                 \
    }
    MILK_FOR_EACH_DATATYPE(dt, WRITE_PIX_BODY, WRITE_PIX_BODY(D))
#undef WRITE_PIX_BODY
}

/**
 * @brief Apply scalar assignment/compound op (=, +=, -=, *=, /=) to all pixels of @p im in-place.
 */
static int apply_scalar_to_image_inplace(IMAGE *im, cli_token_type aop, double val)
{
    if (aop == TOK_OP_SLASH_EQ && val == 0.0)
    {
        parse_errmsg("Division by zero");
        return -1;
    }
    uint64_t nelement = im->md[0].nelement;
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        double v = val;
        if (aop != TOK_EQUAL)
        {
            double old = read_image_pixel_as_double(im, ii);
            if (aop == TOK_OP_PLUS_EQ)
            {
                v = old + val;
            }
            else if (aop == TOK_OP_MINUS_EQ)
            {
                v = old - val;
            }
            else if (aop == TOK_OP_STAR_EQ)
            {
                v = old * val;
            }
            else if (aop == TOK_OP_SLASH_EQ)
            {
                v = old / val;
            }
        }
        write_image_pixel_from_double(im, ii, v);
    }
    return 0;
}

/**
 * @brief Apply image assignment/compound op (=, +=, -=, *=, /=) from @p srcim into @p dstim in-place.
 */
static int apply_image_to_image_inplace(IMAGE *dstim, const IMAGE *srcim, cli_token_type aop)
{
    uint64_t nelement = dstim->md[0].nelement;
    if (aop == TOK_EQUAL && dstim->md[0].datatype == srcim->md[0].datatype)
    {
        int esize = ImageStreamIO_typesize(dstim->md[0].datatype);
        if (esize > 0)
        {
            __builtin_memcpy(dstim->array.raw, srcim->array.raw, (size_t) esize * nelement);
            return 0;
        }
    }
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        double sv = read_image_pixel_as_double(srcim, ii);
        double v  = sv;
        if (aop != TOK_EQUAL)
        {
            double old = read_image_pixel_as_double(dstim, ii);
            if (aop == TOK_OP_PLUS_EQ)
            {
                v = old + sv;
            }
            else if (aop == TOK_OP_MINUS_EQ)
            {
                v = old - sv;
            }
            else if (aop == TOK_OP_STAR_EQ)
            {
                v = old * sv;
            }
            else if (aop == TOK_OP_SLASH_EQ)
            {
                if (sv == 0.0)
                {
                    parse_errmsg("Division by zero");
                    return -1;
                }
                v = old / sv;
            }
        }
        write_image_pixel_from_double(dstim, ii, v);
    }
    return 0;
}

/**
 * @brief Evaluate variable or arithmetic expression bounds inside a slice string.
 *
 * Converts e.g. "x0:x1, cx-r:cx+r" into resolved integer bounds "32:95,48:80".
 * Sets @p *out_is_single_pixel to 1 if all axes are single indices without ':' or '*'.
 *
 * @return 0 on success, -1 on error
 */
static int eval_slice_bounds(const char *btext,
                             char       *out_btext,
                             size_t      out_sz,
                             int        *out_is_single_pixel)
{
    *out_is_single_pixel = 1;
    out_btext[0]         = '\0';
    size_t out_len       = 0;

    char buf[256];
    strncpy(buf, btext, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *p    = buf;
    int   axis = 0;
    while (*p != '\0')
    {
        /* Find next comma at parenthesis depth 0 */
        char *axis_start = p;
        int   pdepth     = 0;
        while (*p != '\0' && !(*p == ',' && pdepth == 0))
        {
            if (*p == '(')
            {
                pdepth++;
            }
            else if (*p == ')')
            {
                pdepth--;
            }
            p++;
        }
        if (*p == ',')
        {
            *p = '\0';
            p++;
        }

        if (axis > 0)
        {
            out_len += (size_t) snprintf(out_btext + out_len, out_sz - out_len, ",");
        }
        axis++;

        /* Split axis_start on ':' */
        char *ap     = axis_start;
        int   part_i = 0;
        while (1)
        {
            char *part_start = ap;
            int   sub_depth  = 0;
            while (*ap != '\0' && !(*ap == ':' && sub_depth == 0))
            {
                if (*ap == '(')
                {
                    sub_depth++;
                }
                else if (*ap == ')')
                {
                    sub_depth--;
                }
                ap++;
            }
            int has_more_colon = (*ap == ':');
            if (has_more_colon)
            {
                *ap = '\0';
                ap++;
                *out_is_single_pixel = 0;
            }

            if (part_i > 0)
            {
                out_len += (size_t) snprintf(out_btext + out_len, out_sz - out_len, ":");
            }

            /* Trim whitespace */
            while (*part_start == ' ' || *part_start == '\t')
            {
                part_start++;
            }
            char *part_end = part_start + strlen(part_start);
            while (part_end > part_start && (*(part_end - 1) == ' ' || *(part_end - 1) == '\t'))
            {
                *(--part_end) = '\0';
            }

            if (part_start[0] == '\0' || strcmp(part_start, "*") == 0 ||
                strcmp(part_start, "-*") == 0)
            {
                if (part_i == 0 && !has_more_colon)
                {
                    *out_is_single_pixel = 0;
                }
                out_len +=
                    (size_t) snprintf(out_btext + out_len, out_sz - out_len, "%s", part_start);
            }
            else
            {
                /* Check for trailing 'b'/'B' binning suffix on 3rd part (step) */
                int    has_bin = 0;
                size_t plen    = strlen(part_start);
                if (part_i >= 2 && plen > 1 &&
                    (part_start[plen - 1] == 'b' || part_start[plen - 1] == 'B') &&
                    variable_ID(part_start) == -1 && cli_var_get(part_start) == NULL)
                {
                    has_bin              = 1;
                    part_start[plen - 1] = '\0';
                }

                /* Check if pure integer literal */
                char *endp = NULL;
                long  ival = strtol(part_start, &endp, 10);
                if (endp != part_start && *endp == '\0')
                {
                    out_len += (size_t) snprintf(out_btext + out_len, out_sz - out_len, "%ld%s",
                                                 ival, has_bin ? "b" : "");
                }
                else
                {
                    int    mtype = 0;
                    long   mlval = 0;
                    double mdval = 0.0;
                    if (!cli_calc_eval_math_to_val(part_start, &mtype, &mlval, &mdval))
                    {
                        char msg[256];
                        snprintf(msg, sizeof(msg), "Invalid slice bound expression '%s'",
                                 part_start);
                        parse_errmsg(msg);
                        return -1;
                    }
                    long resolved = (mtype == 1) ? mlval : (long) llround(mdval);
                    out_len += (size_t) snprintf(out_btext + out_len, out_sz - out_len, "%ld%s",
                                                 resolved, has_bin ? "b" : "");
                }
            }

            part_i++;
            if (!has_more_colon)
            {
                break;
            }
        }
    }

    return 0;
}


/* --------------------------------------------------------
 * Operator precedence table
 *
 * Higher number = tighter binding.
 * Right-associative operators use prec - 1 for the
 * right-hand parse.
 *
 *   + -    : 1
 *   * /    : 2
 *   ^      : 3 (right-assoc)
 * -------------------------------------------------------- */

static inline int get_prec(cli_token_type t)
{
    switch (t)
    {
    case TOK_OP_QUESTION:
        return -1;
    case TOK_OP_OR:
        return 1;
    case TOK_OP_AND:
        return 2;
    case TOK_OP_BOR:
        return 3;
    case TOK_OP_BXOR:
        return 4;
    case TOK_OP_BAND:
        return 5;
    case TOK_OP_EQ:
    case TOK_OP_NEQ:
        return 6;
    case TOK_OP_LT:
    case TOK_OP_LE:
    case TOK_OP_GT:
    case TOK_OP_GE:
        return 7;
    case TOK_OP_LSHIFT:
    case TOK_OP_RSHIFT:
        return 8;
    case TOK_OP_PLUS:
    case TOK_OP_MINUS:
        return 9;
    case TOK_OP_STAR:
    case TOK_OP_SLASH:
    case TOK_OP_MOD:
        return 10;
    case TOK_OP_CARET:
        return 11;
    case TOK_EQUAL:
    case TOK_OP_PLUS_EQ:
    case TOK_OP_MINUS_EQ:
    case TOK_OP_STAR_EQ:
    case TOK_OP_SLASH_EQ:
        return 0;
    default:
        return -1;
    }
}

/**
 * @brief Check if an operator is right-associative.
 *
 * Used by the expression parser for precedence.
 */
static inline int is_right_assoc(cli_token_type t)
{
    return (t == TOK_OP_CARET) || (t == TOK_EQUAL) || (t == TOK_OP_PLUS_EQ) ||
           (t == TOK_OP_MINUS_EQ) || (t == TOK_OP_STAR_EQ) || (t == TOK_OP_SLASH_EQ) ||
           (t == TOK_OP_QUESTION);
}
/**
 * @brief Evaluate a binary operation
 *
 * Implements type-aware arithmetic matching the
 * original bison grammar's precedence and type rules.
 */
val_t parse_primary(void)
{
    cli_token *t;
    cli_token *(*cur_func)(void);
    cli_token *(*advance_func)(void);

    if (parse_error || eval_error) // Check both error flags
    {
        return mk_long(0);
    }

    // Determine which token stream to use based on which error flag is active
    if (parse_mode == 0) // cli_parse is active
    {
        cur_func     = cur_parse;
        advance_func = advance_parse;
    }
    else // cli_calc_eval_line is active
    {
        cur_func     = cur_eval;
        advance_func = advance_eval;
    }

    t = cur_func();

    /* number literals */
    if (t->type == TOK_LONG)
    {
        advance_func();
        if (data.core.Debug > 0)
        {
            printf("this is a long\n");
        }
        return mk_long(t->val_l);
    }

    if (t->type == TOK_DOUBLE)
    {
        advance_func();
        if (data.core.Debug > 0)
        {
            printf("this is a double\n");
        }
        return mk_double(t->val_d);
    }

    /* unary minus */
    if (t->type == TOK_OP_MINUS)
    {
        advance_func();
        val_t v = parse_primary();
        if (parse_error || eval_error)
        {
            return mk_long(0);
        }
        if (v.type == VAL_LONG)
        {
            if (data.core.Debug > 0)
            {
                printf("-long\n");
            }
            return mk_long(-v.lval);
        }
        if (v.type == VAL_DOUBLE)
        {
            if (data.core.Debug > 0)
            {
                printf("-double\n");
            }
            return mk_double(-v.dval);
        }
        if (v.type == VAL_STRING)
        {
            if (data.core.Debug > 0)
            {
                printf("-image\n");
            }
            if (!check_image(v.sval))
            {
                return mk_double(0);
            }
            const char *tmpn = alloc_tmpname();
            arith_image_cstsubm(v.sval, 0.0, tmpn);
            return mk_string(tmpn);
        }
        parse_errmsg("Cannot negate expression");
        return mk_double(0);
    }

    /* unary bitwise NOT */
    if (t->type == TOK_OP_BNOT)
    {
        advance_func();
        val_t v = parse_primary();
        if (parse_error || eval_error)
        {
            return mk_long(0);
        }
        long iv = (v.type == VAL_LONG) ? v.lval : (long) v.dval;
        return mk_long(~iv);
    }

    /* parenthesized expression */
    if (t->type == TOK_LPAREN)
    {
        advance_func();
        val_t v = parse_expr(0);
        if (parse_error || eval_error)
        {
            return v;
        }
        if (cur_func()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_long(0);
        }
        advance_func();
        return v;
    }

    /* function calls */
    if (t->type == TOK_FUNC_D_D || t->type == TOK_FUNC_DD_D || t->type == TOK_FUNC_DDD_D ||
        t->type == TOK_FUNC_IM_D || t->type == TOK_FUNC_IMD_D || t->type == TOK_FUNC_WHERE ||
        t->type == TOK_FUNC_IMIM_D || t->type == TOK_FUNC_IMIM_IM || t->type == TOK_FUNC_S_D ||
        t->type == TOK_FUNC_S_S || t->type == TOK_FUNC_SDD_S || t->type == TOK_FUNC_SSS_S ||
        t->type == TOK_FUNC_D_S)
    {
        advance_func();
        return parse_funccall(t);
    }

    /* existing variable: might be assignment */
    if (t->type == TOK_VAR)
    {
        advance_func();
        cli_token_type aop = cur_func()->type;
        if (aop == TOK_EQUAL || aop == TOK_OP_PLUS_EQ || aop == TOK_OP_MINUS_EQ ||
            aop == TOK_OP_STAR_EQ || aop == TOK_OP_SLASH_EQ)
        {
            advance_func();
            val_t v = parse_expr(0);
            if (parse_error || eval_error)
            {
                return v;
            }

            /* compound: fetch old, apply op */
            if (aop != TOK_EQUAL)
            {
                long   vid = variable_ID(t->sval);
                double old = 0.0;
                if (vid == -1)
                {
                    const char *cval = cli_var_get(t->sval);
                    if (cval == NULL)
                    {
                        cval = getenv(t->sval);
                    }
                    if (cval != NULL)
                    {
                        old = strtod(cval, NULL);
                    }
                    else
                    {
                        parse_errmsg("Variable not found"
                                     " for compound assign");
                        return mk_double(0);
                    }
                }
                else if (data.core.variable[vid].type == 1)
                {
                    old = (double) data.core.variable[vid].value.l;
                }
                else
                {
                    old = data.core.variable[vid].value.f;
                }
                double nv = to_double(v);
                switch (aop)
                {
                case TOK_OP_PLUS_EQ:
                    nv = old + nv;
                    break;
                case TOK_OP_MINUS_EQ:
                    nv = old - nv;
                    break;
                case TOK_OP_STAR_EQ:
                    nv = old * nv;
                    break;
                case TOK_OP_SLASH_EQ:
                    if (nv == 0.0)
                    {
                        parse_errmsg("Division by zero");
                        return mk_double(0);
                    }
                    nv = old / nv;
                    break;
                default:
                    break;
                }
                /* check if result is integer */
                if (v.type == VAL_LONG &&
                    ((vid != -1 && data.core.variable[vid].type == 1) || vid == -1) &&
                    aop != TOK_OP_SLASH_EQ)
                {
                    v = mk_long((long) nv);
                }
                else
                {
                    v = mk_double(nv);
                }
            }

            if (v.type == VAL_STRING)
            {
                if (image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) != -1)
                {
                    if (strncmp(v.sval, "_tmpcalc", 8) == 0)
                    {
                        chname_image_ID(v.sval, t->sval);
                    }
                    else
                    {
                        copy_image_ID(v.sval, t->sval, 0);
                    }
                    if (data.core.Debug > 0)
                    {
                        printf("changing name\n");
                    }
                    return mk_string(t->sval);
                }
                const char *strval = cli_var_get(v.sval);
                if (strval == NULL)
                {
                    strval = v.sval;
                }
                cli_var_set(t->sval, strval);
                return mk_string(t->sval);
            }
            if (v.type == VAL_LONG)
            {
                create_variable_long_ID(t->sval, v.lval);
                char numv[64];
                snprintf(numv, 64, "%ld", v.lval);
                cli_var_set(t->sval, numv);
                return mk_long(v.lval);
            }
            {
                create_variable_ID(t->sval, to_double(v));
                char numv[64];
                snprintf(numv, 64, "%.*g", cli_float_digits, to_double(v));
                cli_var_set(t->sval, numv);
                return mk_double(to_double(v));
            }
        }
        /* just a variable reference */
        long vID = variable_ID(t->sval);
        if (vID == -1)
        {
            const char *cval = cli_var_get(t->sval);
            if (cval == NULL)
            {
                cval = getenv(t->sval);
            }
            if (cval != NULL)
            {
                char *endp = NULL;
                long  lval = strtol(cval, &endp, 0);
                if (*endp == '\0' && endp != cval)
                {
                    return mk_long(lval);
                }
                double dval = strtod(cval, &endp);
                if (*endp == '\0' && endp != cval)
                {
                    return mk_double(dval);
                }
                return mk_string(t->sval);
            }
            char msg[2048];
            snprintf(msg, sizeof(msg), "Variable '%s' not found", t->sval);
            parse_errmsg(msg);
            return mk_double(0);
        }
        if (data.core.variable[vID].type == 1)
        {
            return mk_long(data.core.variable[vID].value.l);
        }
        if (data.core.variable[vID].type == 2)
        {
            return mk_string(data.core.variable[vID].value.s);
        }
        return mk_double(data.core.variable[vID].value.f);
    }

    /* new variable: must be assignment, or it's
     * a new string/image name */
    if (t->type == TOK_NVAR)
    {
        advance_func();
        if (cur_func()->type == TOK_EQUAL)
        {
            advance_func();
            val_t v = parse_expr(0);
            if (parse_error || eval_error)
            {
                return v;
            }
            if (v.type == VAL_STRING)
            {
                if (image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) != -1)
                {
                    if (strncmp(v.sval, "_tmpcalc", 8) == 0)
                    {
                        chname_image_ID(v.sval, t->sval);
                    }
                    else
                    {
                        copy_image_ID(v.sval, t->sval, 0);
                    }
                    if (data.core.Debug > 0)
                    {
                        printf("changing name\n");
                    }
                    return mk_string(t->sval);
                }
                const char *strval = cli_var_get(v.sval);
                if (strval == NULL)
                {
                    strval = v.sval;
                }
                cli_var_set(t->sval, strval);
                return mk_string(t->sval);
            }
            if (v.type == VAL_LONG)
            {
                if (data.core.Debug > 0)
                {
                    printf("creating long\n");
                }
            }
            else
            {
                if (data.core.Debug > 0)
                {
                    printf("creating double\n");
                }
            }
            if (v.type == VAL_LONG)
            {
                create_variable_long_ID(t->sval, v.lval);
                char numv[64];
                snprintf(numv, 64, "%ld", v.lval);
                if (parse_mode == 1)
                {
                    cli_var_set(t->sval, numv);
                }
                return mk_long(v.lval);
            }
            create_variable_ID(t->sval, to_double(v));
            {
                char numv[64];
                snprintf(numv, 64, "%.*g", cli_float_digits, to_double(v));
                if (parse_mode == 1)
                {
                    cli_var_set(t->sval, numv);
                }
            }
            return mk_double(to_double(v));
        }
        /* standalone new variable/image name */
        // This path is only for cli_parse, not cli_calc_eval_line
        if (parse_mode == 0) // If not in eval_line context
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_STRING;
            if (data.core.Debug > 0)
            {
                printf("this is a string "
                       "(new variable/image)\n");
            }
        }
        else
        {
            if (strchr(t->sval, '[') != NULL)
            {
                char bare[200];
                char btext[200];
                int  has_brk = imgid_slice_split_name(t->sval, bare, (int) sizeof(bare), btext,
                                                      (int) sizeof(btext));
                if (!has_brk)
                {
                    parse_errmsg("Malformed slice: missing closing ']'");
                    return mk_double(0);
                }
                else
                {
                    parse_errmsg("Source image not found for slice");
                    return mk_double(0);
                }
            }
        }
        return mk_string(t->sval);
    }

    /* existing image: might be assignment */
    if (t->type == TOK_IMAGE)
    {
        advance_func();
        cli_token_type aop = cur_func()->type;
        if (aop == TOK_EQUAL || aop == TOK_OP_PLUS_EQ || aop == TOK_OP_MINUS_EQ ||
            aop == TOK_OP_STAR_EQ || aop == TOK_OP_SLASH_EQ)
        {
            advance_func();
            val_t v = parse_expr(0);
            if (parse_error || eval_error)
            {
                return v;
            }

            /* Case A: LHS slice/single-pixel assignment im[...] = rhs */
            if (strchr(t->sval, '[') != NULL)
            {
                char bare[200];
                char btext[200];
                char resolved_btext[256];
                int  is_single_pixel = 0;

                if (!imgid_slice_split_name(t->sval, bare, (int) sizeof(bare), btext,
                                            (int) sizeof(btext)))
                {
                    parse_errmsg("Malformed slice: missing closing ']'");
                    return mk_double(0);
                }
                imageID srcid = image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE);
                if (srcid == -1)
                {
                    parse_errmsg("Target image not found for slice assignment");
                    return mk_double(0);
                }
                if (eval_slice_bounds(btext, resolved_btext, sizeof(resolved_btext),
                                      &is_single_pixel) != 0)
                {
                    return mk_double(0);
                }
                IMAGE      *srcim = &data.core.image[srcid];
                IMGID_SLICE slc   = imgid_slice_parse(resolved_btext);
                if (slc.error)
                {
                    parse_errmsg(slc.errmsg);
                    return mk_double(0);
                }
                uint32_t outsz[3] = { 0 };
                int      snax     = srcim->md[0].naxis;
                uint32_t ssz[3]   = { 1, 1, 1 };
                for (int a = 0; a < snax && a < 3; a++)
                {
                    ssz[a] = srcim->md[0].size[a];
                }
                if (imgid_slice_output_size(&slc, snax, ssz, outsz) != 0)
                {
                    parse_errmsg(slc.errmsg[0] ? slc.errmsg : "Bad slice dims");
                    return mk_double(0);
                }

                /* Fast path: single-pixel scalar assignment im[x, y] = val */
                if (is_single_pixel && slc.naxis == snax && v.type != VAL_STRING)
                {
                    uint64_t ii = (uint64_t) slc.start[0];
                    if (snax >= 2)
                    {
                        ii += (uint64_t) slc.start[1] * ssz[0];
                    }
                    if (snax >= 3)
                    {
                        ii += (uint64_t) slc.start[2] * ssz[0] * ssz[1];
                    }
                    double nv = to_double(v);
                    if (aop != TOK_EQUAL)
                    {
                        double old = read_image_pixel_as_double(srcim, ii);
                        if (aop == TOK_OP_PLUS_EQ)
                        {
                            nv = old + nv;
                        }
                        else if (aop == TOK_OP_MINUS_EQ)
                        {
                            nv = old - nv;
                        }
                        else if (aop == TOK_OP_STAR_EQ)
                        {
                            nv = old * nv;
                        }
                        else if (aop == TOK_OP_SLASH_EQ)
                        {
                            if (nv == 0.0)
                            {
                                parse_errmsg("Division by zero");
                                return mk_double(0);
                            }
                            nv = old / nv;
                        }
                    }
                    srcim->md[0].write = 1;
                    write_image_pixel_from_double(srcim, ii, nv);
                    processinfo_update_output_stream(NULL, srcim, NULL);
                    return (v.type == VAL_LONG && aop == TOK_EQUAL) ? mk_long(v.lval)
                                                                    : mk_double(nv);
                }

                /* General slice region assignment */
                int onax = 0;
                for (int a = 0; a < 3; a++)
                {
                    if (outsz[a] > 0)
                    {
                        onax = a + 1;
                    }
                }
                if (onax == 0)
                {
                    onax = 1;
                }
                const char *tmpn = alloc_tmpname();
                imageID     tid  = -1;
                create_image_ID(tmpn, onax, outsz, srcim->md[0].datatype, 0, 10, 0, &tid);
                if (tid == -1)
                {
                    parse_errmsg("Failed to allocate slice buffer");
                    return mk_double(0);
                }
                IMGID simg;
                memset(&simg, 0, sizeof(simg));
                simg.ID       = srcid;
                simg.im       = srcim;
                simg.md       = srcim->md;
                simg.slice    = slc;
                simg.slice_im = &data.core.image[tid];
                if (aop != TOK_EQUAL)
                {
                    imgid_slice_materialize(&simg);
                }

                if (v.type != VAL_STRING)
                {
                    if (apply_scalar_to_image_inplace(simg.slice_im, aop, to_double(v)) != 0)
                    {
                        return mk_double(0);
                    }
                }
                else
                {
                    imageID rhsid = image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE);
                    if (rhsid == -1)
                    {
                        parse_errmsg("Source image does not exist");
                        return mk_double(0);
                    }
                    IMAGE *rhsim = &data.core.image[rhsid];
                    if (rhsim->md[0].nelement != simg.slice_im->md[0].nelement)
                    {
                        parse_errmsg("Dimension mismatch in slice assignment");
                        return mk_double(0);
                    }
                    if (apply_image_to_image_inplace(simg.slice_im, rhsim, aop) != 0)
                    {
                        return mk_double(0);
                    }
                }

                srcim->md[0].write = 1;
                imgid_slice_writeback(&simg);
                processinfo_update_output_stream(NULL, srcim, NULL);
                return mk_string(bare);
            }

            /* Case B: Whole-image assignment im = rhs, +=, -=, *=, /= */
            imageID dstid = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
            if (dstid == -1)
            {
                parse_errmsg("Target image not found");
                return mk_double(0);
            }
            IMAGE *dstim = &data.core.image[dstid];

            if (v.type != VAL_STRING)
            {
                /* Scalar fill or compound scalar operation on whole image */
                dstim->md[0].write = 1;
                if (apply_scalar_to_image_inplace(dstim, aop, to_double(v)) != 0)
                {
                    dstim->md[0].write = 0;
                    return mk_double(0);
                }
                processinfo_update_output_stream(NULL, dstim, NULL);
                return mk_string(t->sval);
            }

            /* Image RHS */
            imageID srcid = image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE);
            if (srcid == -1)
            {
                parse_errmsg("Source image does not exist");
                return mk_double(0);
            }
            IMAGE *srcim = &data.core.image[srcid];

            if (dstim->md[0].nelement == srcim->md[0].nelement &&
                (dstim->md[0].shared == 1 || aop != TOK_EQUAL ||
                 (dstim->md[0].naxis == srcim->md[0].naxis &&
                  dstim->md[0].datatype == srcim->md[0].datatype)))
            {
                /* In-place SHM-safe copy/compound update */
                if (dstid != srcid || aop != TOK_EQUAL)
                {
                    dstim->md[0].write = 1;
                    if (apply_image_to_image_inplace(dstim, srcim, aop) != 0)
                    {
                        dstim->md[0].write = 0;
                        return mk_double(0);
                    }
                    processinfo_update_output_stream(NULL, dstim, NULL);
                }
                return mk_string(t->sval);
            }

            if (dstim->md[0].shared == 1 || aop != TOK_EQUAL)
            {
                parse_errmsg("Dimension mismatch in image assignment");
                return mk_double(0);
            }

            /* Local image with different dimensions/datatype and aop == TOK_EQUAL */
            delete_image_ID(t->sval, 1);
            if (strncmp(v.sval, "_tmpcalc", 8) == 0)
            {
                chname_image_ID(v.sval, t->sval);
            }
            else
            {
                copy_image_ID(v.sval, t->sval, 0);
            }
            if (data.core.Debug > 0)
            {
                printf("changing name\n");
            }
            return mk_string(t->sval);
        }
        // This path is only for cli_parse, not cli_calc_eval_line
        if (!eval_error) // If not in eval_line context
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_EXISTINGIMAGE;
            if (data.core.Debug > 0)
            {
                printf("this is a string "
                       "(existing image)\n");
            }
        }
        /* Check for slice bracket syntax.
         * Materialize the sliced region into
         * a temporary image and return that
         * name instead (or scalar for 1x1 single-pixel index). */
        if (strchr(t->sval, '[') != NULL)
        {
            char bare[200];
            char btext[200];
            char resolved_btext[256];
            int  is_single_pixel = 0;
            int  has_brk         = imgid_slice_split_name(t->sval, bare, (int) sizeof(bare), btext,
                                                          (int) sizeof(btext));
            if (has_brk)
            {
                imageID srcid = image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE);
                if (srcid == -1)
                {
                    parse_errmsg("Source image "
                                 "not found");
                    return mk_double(0);
                }
                if (eval_slice_bounds(btext, resolved_btext, sizeof(resolved_btext),
                                      &is_single_pixel) != 0)
                {
                    return mk_double(0);
                }
                IMAGE      *srcim = &data.core.image[srcid];
                IMGID_SLICE slc   = imgid_slice_parse(resolved_btext);
                if (slc.error)
                {
                    parse_errmsg(slc.errmsg);
                    return mk_double(0);
                }
                uint32_t outsz[3] = { 0 };
                int      snax     = srcim->md[0].naxis;
                uint32_t ssz[3]   = { 1, 1, 1 };
                for (int a = 0; a < snax && a < 3; a++)
                {
                    ssz[a] = srcim->md[0].size[a];
                }
                if (imgid_slice_output_size(&slc, snax, ssz, outsz) != 0)
                {
                    parse_errmsg(slc.errmsg[0] ? slc.errmsg : "Bad slice dims");
                    return mk_double(0);
                }
                /* Single-pixel index read in math expression: return scalar pixel value */
                if (parse_mode != 0 && is_single_pixel && slc.naxis == snax)
                {
                    uint64_t ii = (uint64_t) slc.start[0];
                    if (snax >= 2)
                    {
                        ii += (uint64_t) slc.start[1] * ssz[0];
                    }
                    if (snax >= 3)
                    {
                        ii += (uint64_t) slc.start[2] * ssz[0] * ssz[1];
                    }
                    return mk_double(read_image_pixel_as_double(srcim, ii));
                }
                /* Count output axes */
                int onax = 0;
                for (int a = 0; a < 3; a++)
                {
                    if (outsz[a] > 0)
                    {
                        onax = a + 1;
                    }
                }
                if (onax == 0)
                {
                    onax = 1;
                }
                const char *tmpn = alloc_tmpname();
                imageID     tid  = -1;
                create_image_ID(tmpn, onax, outsz, srcim->md[0].datatype, 0, 10, 0, &tid);
                if (tid != -1)
                {
                    IMGID simg;
                    memset(&simg, 0, sizeof(simg));
                    simg.ID       = srcid;
                    simg.im       = srcim;
                    simg.md       = srcim->md;
                    simg.slice    = slc;
                    simg.slice_im = &data.core.image[tid];
                    imgid_slice_materialize(&simg);
                }
                return mk_string(tmpn);
            }
            else
            {
                parse_errmsg("Malformed slice: missing closing ']'");
                return mk_double(0);
            }
        }
        return mk_string(t->sval);
    }


    /* command */
    if (t->type == TOK_COMMAND)
    {
        advance_func();
        // This path is only for cli_parse, not cli_calc_eval_line
        if (parse_mode == 0) // If not in eval_line context
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_COMMAND;
            if (data.core.Debug > 0)
            {
                printf("this is a string (command)\n");
            }
        }
        return mk_string(t->sval);
    }

    parse_errmsg("Unexpected token");
    return mk_long(0);
}

/* --------------------------------------------------------
 * Expression parser (precedence climbing)
 * -------------------------------------------------------- */

/**
 * @brief Parse an expression with minimum precedence
 *
 * Implements Pratt / precedence-climbing:
 *  1. Parse a primary (atom)
 *  2. While the next token is a binary operator with
 *     precedence >= min_prec, consume it and parse
 *     the right-hand side with appropriate precedence
 */
val_t parse_expr(int min_prec)
{
    cli_token *(*cur_func)(void);
    cli_token *(*advance_func)(void);

    if (parse_error || eval_error) // Check both error flags
    {
        return mk_long(0);
    }

    // Determine which token stream to use based on which error flag is active
    if (parse_mode == 0) // cli_parse is active
    {
        cur_func     = cur_parse;
        advance_func = advance_parse;
    }
    else // cli_calc_eval_line is active
    {
        cur_func     = cur_eval;
        advance_func = advance_eval;
    }


    val_t left = parse_primary();
    if (parse_error || eval_error)
    {
        return left;
    }

    for (;;)
    {
        cli_token_type op   = cur_func()->type;
        int            prec = get_prec(op);

        if (prec < min_prec)
        {
            break;
        }

        advance_func();

        /* Ternary ? : operator */
        if (op == TOK_OP_QUESTION)
        {
            val_t true_val = parse_expr(-1);
            if (parse_error || eval_error)
            {
                return left;
            }
            if (cur_func()->type != TOK_OP_COLON)
            {
                parse_errmsg("Expected ':' in ternary");
                return mk_double(0);
            }
            advance_func();
            val_t false_val = parse_expr(-1);
            if (parse_error || eval_error)
            {
                return left;
            }
            /* Scalar ternary */
            if (left.type != VAL_STRING)
            {
                double c = to_double(left);
                left     = (c != 0.0) ? true_val : false_val;
            }
            else
            {
                /* Image ternary -> where() */
                if (!check_image(left.sval))
                {
                    return mk_string("");
                }
                const char *tmask  = alloc_tmpname();
                const char *timask = alloc_tmpname();
                const char *tpart  = alloc_tmpname();
                const char *fpart  = alloc_tmpname();
                const char *tmpn   = alloc_tmpname();
                arith_image_csttestne(left.sval, 0.0, tmask);
                arith_image_cstteste(left.sval, 0.0, timask);
                if (true_val.type == VAL_STRING)
                {
                    if (!check_image(true_val.sval))
                    {
                        return mk_string("");
                    }
                    arith_image_mult(true_val.sval, tmask, tpart);
                }
                else
                {
                    arith_image_cstmult(tmask, to_double(true_val), tpart);
                }
                if (false_val.type == VAL_STRING)
                {
                    if (!check_image(false_val.sval))
                    {
                        return mk_string("");
                    }
                    arith_image_mult(false_val.sval, timask, fpart);
                }
                else
                {
                    arith_image_cstmult(timask, to_double(false_val), fpart);
                }
                arith_image_add(tpart, fpart, tmpn);
                left = mk_string(tmpn);
            }
            continue;
        }

        int   next_min = is_right_assoc(op) ? prec : prec + 1;
        val_t right    = parse_expr(next_min);
        if (parse_error || eval_error)
        {
            return left;
        }

        left = eval_binop(op, left, right);
        if (parse_error || eval_error)
        {
            return left;
        }
    }

    return left;
}

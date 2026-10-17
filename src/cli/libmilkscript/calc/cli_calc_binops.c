// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_binops.c
 *
 * @brief Binary operations for CLI calculator
 *
 * Implements binary arithmetic dispatching between images, floats, and ints.
 * Extracted from cli_calc_functions.c to reduce file size.
 */

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"

#include "CLIcore_script.h"
#include "cli_calc_internal.h"

/**
 * @brief Check if a token refers to an image.
 *
 * Returns 1 if the name matches an active image.
 */
int check_image(const char *name);

/**
 * @brief Image OP image binary operation.
 *
 * Dispatches to the appropriate arith_image_*
 * function for two image operands.
 *
 * @param op    Token operator type
 * @param lname Left image name
 * @param rname Right image name
 * @param tmpn  Temporary output image name
 * @return 1 on success, 0 on unsupported op
 */
static int binop_img_img(cli_token_type op, const char *lname, const char *rname, const char *tmpn)
{
    switch (op)
    {
    case TOK_OP_PLUS:
        arith_image_add(lname, rname, tmpn);
        break;
    case TOK_OP_MINUS:
        arith_image_sub(lname, rname, tmpn);
        break;
    case TOK_OP_STAR:
        arith_image_mult(lname, rname, tmpn);
        break;
    case TOK_OP_SLASH:
        arith_image_div(lname, rname, tmpn);
        break;
    case TOK_OP_MOD:
        arith_image_fmod(lname, rname, tmpn);
        break;
    case TOK_OP_CARET:
        arith_image_pow(lname, rname, tmpn);
        break;
    case TOK_OP_LT:
        arith_image_testlt(lname, rname, tmpn);
        break;
    case TOK_OP_LE:
        arith_image_testle(lname, rname, tmpn);
        break;
    case TOK_OP_GT:
        arith_image_testgt(lname, rname, tmpn);
        break;
    case TOK_OP_GE:
        arith_image_testge(lname, rname, tmpn);
        break;
    case TOK_OP_EQ:
        arith_image_teste(lname, rname, tmpn);
        break;
    case TOK_OP_NEQ:
        arith_image_testne(lname, rname, tmpn);
        break;
    case TOK_OP_AND:
        arith_image_and(lname, rname, tmpn);
        break;
    case TOK_OP_OR:
        arith_image_or(lname, rname, tmpn);
        break;
    default:
        return 0;
    }
    return 1;
}


/**
 * @brief Image OP scalar binary operation.
 *
 * @param op    Token operator type
 * @param iname Image name (left operand)
 * @param rv    Scalar value (right operand)
 * @param tmpn  Output image name
 * @return 1 on success, 0 on unsupported op
 */
static int binop_img_scalar(cli_token_type op, const char *iname, double rv, const char *tmpn)
{
    switch (op)
    {
    case TOK_OP_PLUS:
        arith_image_cstadd(iname, rv, tmpn);
        break;
    case TOK_OP_MINUS:
        arith_image_cstadd(iname, -rv, tmpn);
        break;
    case TOK_OP_STAR:
        arith_image_cstmult(iname, rv, tmpn);
        break;
    case TOK_OP_SLASH:
        if (rv == 0.0)
        {
            parse_errmsg("Division by zero");
            return 0;
        }
        arith_image_cstdiv(iname, rv, tmpn);
        break;
    case TOK_OP_MOD:
        if (rv == 0.0)
        {
            parse_errmsg("Modulo by zero");
            return 0;
        }
        arith_image_cstfmod(iname, rv, tmpn);
        break;
    case TOK_OP_CARET:
        arith_image_cstpow(iname, rv, tmpn);
        break;
    case TOK_OP_LT:
        arith_image_csttestlt(iname, rv, tmpn);
        break;
    case TOK_OP_LE:
        arith_image_csttestle(iname, rv, tmpn);
        break;
    case TOK_OP_GT:
        arith_image_csttestgt(iname, rv, tmpn);
        break;
    case TOK_OP_GE:
        arith_image_csttestge(iname, rv, tmpn);
        break;
    case TOK_OP_EQ:
        arith_image_cstteste(iname, rv, tmpn);
        break;
    case TOK_OP_NEQ:
        arith_image_csttestne(iname, rv, tmpn);
        break;
    case TOK_OP_AND:
        arith_image_cstand(iname, rv, tmpn);
        break;
    case TOK_OP_OR:
        arith_image_cstor(iname, rv, tmpn);
        break;
    default:
        return 0;
    }
    return 1;
}


/**
 * @brief Scalar OP image binary operation.
 *
 * @param op    Token operator type
 * @param lv    Scalar value (left operand)
 * @param iname Image name (right operand)
 * @param tmpn  Output image name
 * @return 1 on success, 0 on unsupported op
 */
static int binop_scalar_img(cli_token_type op, double lv, const char *iname, const char *tmpn)
{
    switch (op)
    {
    case TOK_OP_PLUS:
        arith_image_cstadd(iname, lv, tmpn);
        break;
    case TOK_OP_MINUS:
        arith_image_cstsubm(iname, lv, tmpn);
        break;
    case TOK_OP_STAR:
        arith_image_cstmult(iname, lv, tmpn);
        break;
    case TOK_OP_SLASH:
        arith_image_cstdiv1(iname, lv, tmpn);
        break;
    case TOK_OP_LT:
        arith_image_csttestgt(iname, lv, tmpn);
        break;
    case TOK_OP_LE:
        arith_image_csttestge(iname, lv, tmpn);
        break;
    case TOK_OP_GT:
        arith_image_csttestlt(iname, lv, tmpn);
        break;
    case TOK_OP_GE:
        arith_image_csttestle(iname, lv, tmpn);
        break;
    case TOK_OP_EQ:
        arith_image_cstteste(iname, lv, tmpn);
        break;
    case TOK_OP_NEQ:
        arith_image_csttestne(iname, lv, tmpn);
        break;
    case TOK_OP_AND:
        arith_image_cstand(iname, lv, tmpn);
        break;
    case TOK_OP_OR:
        arith_image_cstor(iname, lv, tmpn);
        break;
    default:
        return 0;
    }
    return 1;
}


/**
 * ipow_checked() - @b^@e for @e >= 0; returns 0 and sets *@out, or -1 on overflow.
 */
static int ipow_checked(long b, long e, long *out)
{
    long r = 1;
    while (e > 0)
    {
        if ((e & 1) && __builtin_mul_overflow(r, b, &r))
        {
            return -1;
        }
        e >>= 1;
        if (e > 0 && __builtin_mul_overflow(b, b, &b))
        {
            return -1;
        }
    }
    *out = r;
    return 0;
}

/**
 * images_broadcastable() - 1 if images @a and @b can be combined pixel by pixel: every axis the
 * same size or 1 in one of them (the rule of arith_image_function_2_1_IMGID(), which aborts otherwise).
 */
static int images_broadcastable(const char *a, const char *b)
{
    const IMAGE *ia = &data.core.image[image_ID(a, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *ib = &data.core.image[image_ID(b, data.core.image, data.core.NB_MAX_IMAGE)];
    for (int ax = 0; ax < 3; ax++)
    {
        uint32_t sa = (ax < ia->md[0].naxis) ? ia->md[0].size[ax] : 1;
        uint32_t sb = (ax < ib->md[0].naxis) ? ib->md[0].size[ax] : 1;
        if (sa != sb && sa != 1 && sb != 1)
        {
            char msg[160];
            snprintf(msg, sizeof(msg), "Image sizes do not match on axis %d (%u vs %u)", ax, sa,
                     sb);
            parse_errmsg(msg);
            return 0;
        }
    }
    return 1;
}

/** is_text() - 1 if @v is a string that names no image. */
static int is_text(const val_t *v)
{
    return v->type == VAL_STRING &&
           image_ID(v->sval, data.core.image, data.core.NB_MAX_IMAGE) == -1;
}

/** eval_concat() - @left + @right as one string, numbers formatted as @calc prints them. */
static val_t eval_concat(const val_t *left, const val_t *right)
{
    char         part[2][CLI_CALC_TOKEN_MAXLEN];
    const val_t *v[2] = { left, right };
    for (int i = 0; i < 2; i++)
    {
        if (v[i]->type == VAL_STRING)
        {
            snprintf(part[i], sizeof(part[i]), "%s", v[i]->sval);
        }
        else if (v[i]->type == VAL_LONG)
        {
            snprintf(part[i], sizeof(part[i]), "%ld", v[i]->lval);
        }
        else
        {
            snprintf(part[i], sizeof(part[i]), "%.*g", cli_float_digits, v[i]->dval);
        }
    }
    char out[2 * CLI_CALC_TOKEN_MAXLEN];
    if (snprintf(out, sizeof(out), "%s%s", part[0], part[1]) >= CLI_CALC_TOKEN_MAXLEN)
    {
        parse_errmsg("String too long");
        return mk_string("");
    }
    return mk_string(out);
}

/** eval_binop() - apply binary operator @op to scalars, images (element-wise) or strings (`+` concatenates). */
val_t eval_binop(cli_token_type op, val_t left, val_t right)
{
    /* string (image) operands */
    if (left.type == VAL_STRING || right.type == VAL_STRING)
    {
        if (op == TOK_OP_PLUS && (is_text(&left) || is_text(&right)) &&
            (left.type != VAL_STRING || is_text(&left)) &&
            (right.type != VAL_STRING || is_text(&right)))
        {
            return eval_concat(&left, &right);
        }
        if (cli_calc_complex_image(&left) != NULL || cli_calc_complex_image(&right) != NULL)
        {
            return cli_calc_complex_binop(op, left, right);
        }

        if (left.type == VAL_STRING && right.type == VAL_STRING &&
            (op == TOK_OP_EQ || op == TOK_OP_NEQ) &&
            image_ID(left.sval, data.core.image, data.core.NB_MAX_IMAGE) == -1 &&
            image_ID(right.sval, data.core.image, data.core.NB_MAX_IMAGE) == -1)
        {
            int eq = (strcmp(left.sval, right.sval) == 0);
            return mk_long((op == TOK_OP_EQ) ? (eq ? 1 : 0) : (eq ? 0 : 1));
        }

        const char *tmpn = alloc_tmpname();

        /* image OP image */
        if (left.type == VAL_STRING && right.type == VAL_STRING)
        {
            if (!check_image(left.sval) || !check_image(right.sval))
            {
                return mk_string("");
            }
            if (!images_broadcastable(left.sval, right.sval))
            {
                return mk_string("");
            }
            if (!binop_img_img(op, left.sval, right.sval, tmpn))
            {
                if (!parse_error)
                {
                    parse_errmsg("Unsupported image op");
                }
                return mk_string("");
            }
            return mk_string(tmpn);
        }

        /* image OP scalar */
        if (left.type == VAL_STRING)
        {
            if (!check_image(left.sval))
            {
                return mk_string("");
            }
            double rv = to_double(right);
            if (!binop_img_scalar(op, left.sval, rv, tmpn))
            {
                if (!parse_error)
                {
                    parse_errmsg("Unsupported image op");
                }
                return mk_string("");
            }
            return mk_string(tmpn);
        }

        /* scalar OP image */
        if (right.type == VAL_STRING)
        {
            if (!check_image(right.sval))
            {
                return mk_string("");
            }
            double lv = to_double(left);
            if (!binop_scalar_img(op, lv, right.sval, tmpn))
            {
                if (!parse_error)
                {
                    parse_errmsg("Unsupported image op");
                }
                return mk_string("");
            }
            return mk_string(tmpn);
        }
    }

    /* Both numeric (no string) */

    /* long OP long -> long (except / -> double) */
    if (left.type == VAL_LONG && right.type == VAL_LONG && op != TOK_OP_SLASH)
    {
        long r;
        switch (op) // overflowing integer results fall back to double
        {
        case TOK_OP_PLUS:
            return __builtin_add_overflow(left.lval, right.lval, &r)
                       ? mk_double((double) left.lval + (double) right.lval)
                       : mk_long(r);
        case TOK_OP_MINUS:
            return __builtin_sub_overflow(left.lval, right.lval, &r)
                       ? mk_double((double) left.lval - (double) right.lval)
                       : mk_long(r);
        case TOK_OP_STAR:
            return __builtin_mul_overflow(left.lval, right.lval, &r)
                       ? mk_double((double) left.lval * (double) right.lval)
                       : mk_long(r);
        case TOK_OP_MOD:
            if (right.lval == 0)
            {
                parse_errmsg("Modulo by zero");
                return mk_long(0);
            }
            if (left.lval == LONG_MIN && right.lval == -1)
            {
                return mk_long(0);
            }
            return mk_long(left.lval % right.lval);
        case TOK_OP_CARET:
            if (right.lval < 0 || ipow_checked(left.lval, right.lval, &r) != 0)
            {
                return mk_double(pow((double) left.lval, (double) right.lval));
            }
            return mk_long(r);
        case TOK_OP_LT:
            return mk_long(left.lval < right.lval ? 1 : 0);
        case TOK_OP_LE:
            return mk_long(left.lval <= right.lval ? 1 : 0);
        case TOK_OP_GT:
            return mk_long(left.lval > right.lval ? 1 : 0);
        case TOK_OP_GE:
            return mk_long(left.lval >= right.lval ? 1 : 0);
        case TOK_OP_EQ:
            return mk_long(left.lval == right.lval ? 1 : 0);
        case TOK_OP_NEQ:
            return mk_long(left.lval != right.lval ? 1 : 0);
        case TOK_OP_AND:
            return mk_long((left.lval && right.lval) ? 1 : 0);
        case TOK_OP_OR:
            return mk_long((left.lval || right.lval) ? 1 : 0);
        case TOK_OP_BAND:
            return mk_long(left.lval & right.lval);
        case TOK_OP_BOR:
            return mk_long(left.lval | right.lval);
        case TOK_OP_BXOR:
            return mk_long(left.lval ^ right.lval);
        case TOK_OP_LSHIFT:
            if (right.lval < 0 || right.lval >= 64)
            {
                return mk_long(0);
            }
            return mk_long(left.lval << right.lval);
        case TOK_OP_RSHIFT:
            if (right.lval < 0 || right.lval >= 64)
            {
                return mk_long(0);
            }
            return mk_long(left.lval >> right.lval);
        default:
            break;
        }
    }

    /* any numeric mix -> double */
    {
        double lv = to_double(left);
        double rv = to_double(right);

        switch (op)
        {
        case TOK_OP_PLUS:
            return mk_double(lv + rv);
        case TOK_OP_MINUS:
            return mk_double(lv - rv);
        case TOK_OP_STAR:
            return mk_double(lv * rv);
        case TOK_OP_SLASH:
            if (rv == 0.0)
            {
                parse_errmsg("Division by zero");
                return mk_double(0);
            }
            return mk_double(lv / rv);
        case TOK_OP_MOD:
            if (rv == 0.0)
            {
                parse_errmsg("Modulo by zero");
                return mk_double(0);
            }
            return mk_double(fmod(lv, rv));
        case TOK_OP_CARET:
            return mk_double(pow(lv, rv));
        case TOK_OP_LT:
            return mk_long(lv < rv ? 1 : 0);
        case TOK_OP_LE:
            return mk_long(lv <= rv ? 1 : 0);
        case TOK_OP_GT:
            return mk_long(lv > rv ? 1 : 0);
        case TOK_OP_GE:
            return mk_long(lv >= rv ? 1 : 0);
        case TOK_OP_EQ:
            return mk_long(lv == rv ? 1 : 0);
        case TOK_OP_NEQ:
            return mk_long(lv != rv ? 1 : 0);
        case TOK_OP_AND:
            return mk_long((lv != 0 && rv != 0) ? 1 : 0);
        case TOK_OP_OR:
            return mk_long((lv != 0 || rv != 0) ? 1 : 0);
        case TOK_OP_BAND:
            return mk_long((long) lv & (long) rv);
        case TOK_OP_BOR:
            return mk_long((long) lv | (long) rv);
        case TOK_OP_BXOR:
            return mk_long((long) lv ^ (long) rv);
        case TOK_OP_LSHIFT:
        {
            long s = (long) rv;
            if (s < 0 || s >= 64)
            {
                return mk_long(0);
            }
            return mk_long((long) lv << s);
        }
        case TOK_OP_RSHIFT:
        {
            long s = (long) rv;
            if (s < 0 || s >= 64)
            {
                return mk_long(0);
            }
            return mk_long((long) lv >> s);
        }
        default:
            break;
        }
    }

    parse_errmsg("Unknown operator");
    return mk_long(0);
}

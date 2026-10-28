// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_binops.c
 * @brief Binary operations for the CLI calculator.
 *
 * Role & Goal:
 *   Dispatches binary operators between scalars, strings, complex images, and real images.
 *   Same-shape real image operations and image-scalar operations use the SIMD block evaluator
 *   (`cli_calc_eval_binop_im()`); multi-axis broadcasting between differently shaped images
 *   delegates to `COREMOD_arith`.
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"

/**
 * binop_img_img - Dispatch a broadcasting binary operation between two real images
 * @op:    Operator token
 * @lname: Left image name
 * @rname: Right image name
 * @tmpn:  Output temporary image name
 *
 * Return: 1 if handled, 0 if @op is unsupported for broadcasting
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
 * images_broadcastable - Return 1 if images @ia and @ib can be combined pixel by pixel
 * @ia: First image
 * @ib: Second image
 *
 * Return: 1 if every axis is equal or 1, 0 otherwise
 */
static int images_broadcastable(const IMAGE *ia, const IMAGE *ib)
{
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

/**
 * is_text - Return 1 if @v is a string that names no existing image
 * @v: Value to inspect
 *
 * Return: 1 if @v is a pure string, 0 otherwise
 */
static int is_text(const val_t *v)
{
    return v->type == VAL_STRING &&
           image_ID(v->sval, data.core.image, data.core.NB_MAX_IMAGE) == -1;
}

/**
 * eval_concat - Concatenate @left and @right as one string
 * @left:  Left operand
 * @right: Right operand
 *
 * Return: Concatenated `VAL_STRING` value
 */
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

/**
 * eval_binop - Apply binary operator @op to scalars, images, or strings
 * @op:    Operator token
 * @left:  Left operand
 * @right: Right operand
 *
 * Return: Evaluated `val_t` result
 */
val_t eval_binop(cli_token_type op, val_t left, val_t right)
{
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

        const IMAGE *li = NULL;
        const IMAGE *ri = NULL;
        if (left.type == VAL_STRING)
        {
            if (!check_image(left.sval))
            {
                return mk_string("");
            }
            li = &data.core.image[image_ID(left.sval, data.core.image, data.core.NB_MAX_IMAGE)];
        }
        if (right.type == VAL_STRING)
        {
            if (!check_image(right.sval))
            {
                return mk_string("");
            }
            ri = &data.core.image[image_ID(right.sval, data.core.image, data.core.NB_MAX_IMAGE)];
        }

        double lv = (li == NULL) ? to_double(left) : 0.0;
        double rv = (ri == NULL) ? to_double(right) : 0.0;
        if (ri == NULL && (op == TOK_OP_SLASH || op == TOK_OP_MOD) && rv == 0.0)
        {
            parse_errmsg((op == TOK_OP_SLASH) ? "Division by zero" : "Modulo by zero");
            return mk_string("");
        }

        const char *fused = cli_calc_eval_binop_im(op, li, lv, ri, rv);
        if (fused != NULL)
        {
            return mk_string(fused);
        }
        if (li != NULL && ri != NULL)
        {
            if (!images_broadcastable(li, ri))
            {
                return mk_string("");
            }
            const char *tmpn = alloc_tmpname();
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
        if (!parse_error)
        {
            parse_errmsg("Unsupported image op");
        }
        return mk_string("");
    }

    val_t       res;
    const char *err = cli_calc_scalar_binop(op, &left, &right, &res);
    if (err != NULL)
    {
        parse_errmsg(err);
    }
    return res;
}

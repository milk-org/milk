/**
 * @file cli_calc_binops.c
 *
 * @brief Binary operations for CLI calculator
 *
 * Implements binary arithmetic dispatching between images, floats, and ints.
 * Extracted from cli_calc_functions.c to reduce file size.
 */

#include <ctype.h>
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
static int binop_img_img(
    cli_token_type op,
    const char     *lname,
    const char     *rname,
    const char     *tmpn)
{
    switch(op)
    {
    case TOK_OP_PLUS:
        arith_image_add(
            lname, rname, tmpn);
        break;
    case TOK_OP_MINUS:
        arith_image_sub(
            lname, rname, tmpn);
        break;
    case TOK_OP_STAR:
        arith_image_mult(
            lname, rname, tmpn);
        break;
    case TOK_OP_SLASH:
        arith_image_div(
            lname, rname, tmpn);
        break;
    case TOK_OP_MOD:
        arith_image_fmod(
            lname, rname, tmpn);
        break;
    case TOK_OP_CARET:
        arith_image_pow(
            lname, rname, tmpn);
        break;
    case TOK_OP_LT:
        arith_image_testlt(
            lname, rname, tmpn);
        break;
    case TOK_OP_LE:
        arith_image_testle(
            lname, rname, tmpn);
        break;
    case TOK_OP_GT:
        arith_image_testmt(
            lname, rname, tmpn);
        break;
    case TOK_OP_GE:
        arith_image_testge(
            lname, rname, tmpn);
        break;
    case TOK_OP_EQ:
        arith_image_teste(
            lname, rname, tmpn);
        break;
    case TOK_OP_NEQ:
        arith_image_testne(
            lname, rname, tmpn);
        break;
    case TOK_OP_AND:
        arith_image_and(
            lname, rname, tmpn);
        break;
    case TOK_OP_OR:
        arith_image_or(
            lname, rname, tmpn);
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
static int binop_img_scalar(
    cli_token_type op,
    const char     *iname,
    double         rv,
    const char     *tmpn)
{
    switch(op)
    {
    case TOK_OP_PLUS:
        arith_image_cstadd(
            iname, rv, tmpn);
        break;
    case TOK_OP_MINUS:
        arith_image_cstadd(
            iname, -rv, tmpn);
        break;
    case TOK_OP_STAR:
        arith_image_cstmult(
            iname, rv, tmpn);
        break;
    case TOK_OP_SLASH:
        arith_image_cstdiv(
            iname, rv, tmpn);
        break;
    case TOK_OP_MOD:
        arith_image_cstfmod(
            iname, rv, tmpn);
        break;
    case TOK_OP_CARET:
        arith_image_cstpow(
            iname, rv, tmpn);
        break;
    case TOK_OP_LT:
        arith_image_csttestlt(
            iname, rv, tmpn);
        break;
    case TOK_OP_LE:
        arith_image_csttestle(
            iname, rv, tmpn);
        break;
    case TOK_OP_GT:
        arith_image_csttestmt(
            iname, rv, tmpn);
        break;
    case TOK_OP_GE:
        arith_image_csttestge(
            iname, rv, tmpn);
        break;
    case TOK_OP_EQ:
        arith_image_cstteste(
            iname, rv, tmpn);
        break;
    case TOK_OP_NEQ:
        arith_image_csttestne(
            iname, rv, tmpn);
        break;
    case TOK_OP_AND:
        arith_image_cstand(
            iname, rv, tmpn);
        break;
    case TOK_OP_OR:
        arith_image_cstor(
            iname, rv, tmpn);
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
static int binop_scalar_img(
    cli_token_type op,
    double         lv,
    const char     *iname,
    const char     *tmpn)
{
    switch(op)
    {
    case TOK_OP_PLUS:
        arith_image_cstadd(
            iname, lv, tmpn);
        break;
    case TOK_OP_MINUS:
        arith_image_cstsubm(
            iname, lv, tmpn);
        break;
    case TOK_OP_STAR:
        arith_image_cstmult(
            iname, lv, tmpn);
        break;
    case TOK_OP_SLASH:
        arith_image_cstdiv1(
            iname, lv, tmpn);
        break;
    case TOK_OP_LT:
        arith_image_csttestmt(
            iname, lv, tmpn);
        break;
    case TOK_OP_LE:
        arith_image_csttestge(
            iname, lv, tmpn);
        break;
    case TOK_OP_GT:
        arith_image_csttestlt(
            iname, lv, tmpn);
        break;
    case TOK_OP_GE:
        arith_image_csttestle(
            iname, lv, tmpn);
        break;
    case TOK_OP_EQ:
        arith_image_cstteste(
            iname, lv, tmpn);
        break;
    case TOK_OP_NEQ:
        arith_image_csttestne(
            iname, lv, tmpn);
        break;
    case TOK_OP_AND:
        arith_image_cstand(
            iname, lv, tmpn);
        break;
    case TOK_OP_OR:
        arith_image_cstor(
            iname, lv, tmpn);
        break;
    default:
        return 0;
    }
    return 1;
}


/**
 * @brief Dispatch binary operations during expression parsing
 *
 * Evaluates binary operators (+, -, *, /, ^, %, <, <=, ==, !=, &&, ||).
 * Dispatches to specialized arithmetic routines based on operand types:
 * scalar-scalar (int/float), image-scalar (modifying a stream by a constant),
 * scalar-image, or image-image (element-wise operations).
 *
 * @param op    The token representing the binary operator
 * @param left  Left operand value (can be long, double, or string/image-name)
 * @param right Right operand value (can be long, double, or string/image-name)
 * @return val_t Result of the binary operation
 */
val_t eval_binop(
    cli_token_type op,
    val_t          left,
    val_t          right)
{
    /* string (image) operands */
    if(left.type == VAL_STRING
            || right.type == VAL_STRING)
    {
        const char *tmpn = alloc_tmpname();

        /* image OP image */
        if(left.type == VAL_STRING
                && right.type == VAL_STRING)
        {
            if(!check_image(left.sval)
                    || !check_image(right.sval))
            {
                return mk_string("");
            }
            if(!binop_img_img(
                        op, left.sval,
                        right.sval, tmpn))
            {
                parse_errmsg(
                    "Unsupported image op"
                );
                return mk_string("");
            }
            return mk_string(tmpn);
        }

        /* image OP scalar */
        if(left.type == VAL_STRING)
        {
            if(!check_image(left.sval))
            {
                return mk_string("");
            }
            double rv = to_double(right);
            if(!binop_img_scalar(
                        op, left.sval,
                        rv, tmpn))
            {
                parse_errmsg(
                    "Unsupported image op"
                );
                return mk_string("");
            }
            return mk_string(tmpn);
        }

        /* scalar OP image */
        if(right.type == VAL_STRING)
        {
            if(!check_image(right.sval))
            {
                return mk_string("");
            }
            double lv = to_double(left);
            if(!binop_scalar_img(
                        op,         lv,
                        right.sval, tmpn))
            {
                parse_errmsg(
                    "Unsupported image op"
                );
                return mk_string("");
            }
            return mk_string(tmpn);
        }
    }

    /* Both numeric (no string) */

    /* long OP long -> long (except / -> double) */
    if(left.type == VAL_LONG
            && right.type == VAL_LONG
            && op != TOK_OP_SLASH)
    {
        switch(op)
        {
        case TOK_OP_PLUS:
            return mk_long(
                       left.lval + right.lval
                   );
        case TOK_OP_MINUS:
            return mk_long(
                       left.lval - right.lval
                   );
        case TOK_OP_STAR:
            return mk_long(
                       left.lval * right.lval
                   );
        case TOK_OP_MOD:
            if(right.lval == 0)
            {
                parse_errmsg("Modulo by zero");
                return mk_long(0);
            }
            return mk_long(
                       left.lval % right.lval
                   );
        case TOK_OP_CARET:
            return mk_long(
                       (long) pow(
                           left.lval, right.lval)
                   );
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
            return mk_long(
                       left.lval & right.lval);
        case TOK_OP_BOR:
            return mk_long(
                       left.lval | right.lval);
        case TOK_OP_BXOR:
            return mk_long(
                       left.lval ^ right.lval);
        case TOK_OP_LSHIFT:
            return mk_long(
                       left.lval << right.lval);
        case TOK_OP_RSHIFT:
            return mk_long(
                       left.lval >> right.lval);
        default:
            break;
        }
    }

    /* any numeric mix -> double */
    {
        double lv = to_double(left);
        double rv = to_double(right);

        switch(op)
        {
        case TOK_OP_PLUS:
            return mk_double(lv + rv);
        case TOK_OP_MINUS:
            return mk_double(lv - rv);
        case TOK_OP_STAR:
            return mk_double(lv * rv);
        case TOK_OP_SLASH:
            if(rv == 0.0)
            {
                parse_errmsg("Division by zero");
                return mk_double(0);
            }
            return mk_double(lv / rv);
        case TOK_OP_MOD:
            if(rv == 0.0)
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
            return mk_long(
                       (long) lv & (long) rv);
        case TOK_OP_BOR:
            return mk_long(
                       (long) lv | (long) rv);
        case TOK_OP_BXOR:
            return mk_long(
                       (long) lv ^ (long) rv);
        case TOK_OP_LSHIFT:
            return mk_long(
                       (long) lv << (long) rv);
        case TOK_OP_RSHIFT:
            return mk_long(
                       (long) lv >> (long) rv);
        default:
            break;
        }
    }

    parse_errmsg("Unknown operator");
    return mk_long(0);
}

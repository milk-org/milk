#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/stream_slice.h"

#include "CLIcore_script.h"
#include "cli_calc_internal.h"

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
    const char *lname,
    const char *rname,
    const char *tmpn
)
{
    switch (op)
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
    const char *iname,
    double rv,
    const char *tmpn
)
{
    switch (op)
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
    double lv,
    const char *iname,
    const char *tmpn
)
{
    switch (op)
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


val_t eval_binop(
    cli_token_type op,
    val_t left,
    val_t right
)
{
    /* string (image) operands */
    if (left.type == VAL_STRING
        || right.type == VAL_STRING)
    {
        const char *tmpn = alloc_tmpname();

        /* image OP image */
        if (left.type == VAL_STRING
            && right.type == VAL_STRING)
        {
            if (!check_image(left.sval)
                || !check_image(right.sval))
            {
                return mk_string("");
            }
            if (!binop_img_img(
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
        if (left.type == VAL_STRING)
        {
            if (!check_image(left.sval))
            {
                return mk_string("");
            }
            double rv = to_double(right);
            if (!binop_img_scalar(
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
        if (right.type == VAL_STRING)
        {
            if (!check_image(right.sval))
            {
                return mk_string("");
            }
            double lv = to_double(left);
            if (!binop_scalar_img(
                    op, lv,
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
    if (left.type == VAL_LONG
        && right.type == VAL_LONG
        && op != TOK_OP_SLASH)
    {
        switch (op)
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
                if (right.lval == 0) {
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

        switch (op)
        {
            case TOK_OP_PLUS:
                return mk_double(lv + rv);
            case TOK_OP_MINUS:
                return mk_double(lv - rv);
            case TOK_OP_STAR:
                return mk_double(lv * rv);
            case TOK_OP_SLASH:
                if (rv == 0.0) {
                    parse_errmsg("Division by zero");
                    return mk_double(0);
                }
                return mk_double(lv / rv);
            case TOK_OP_MOD:
                if (rv == 0.0) {
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

/* --------------------------------------------------------
 * Function call parsing
 *
 * Called when a TOK_FUNC_* token has been consumed.
 * Parses the argument list (arguments already
 * follow the opening '(' which was consumed as part
 * of the function token).
 * -------------------------------------------------------- */

/**
 * @brief Parse a built-in function call
 *
 * The function name token (including opening paren)
 * has already been consumed.  We parse the argument
 * expressions separated by commas and expect a
 * closing ')'.
 */
val_t parse_funccall(cli_token *ftok)
{
    cli_token *(*cur_func)(void);
    cli_token *(*advance_func)(void);

    if (parse_error || eval_error) { // Check both error flags
        return mk_double(0);
    }

    // Determine which token stream to use based on which error flag is active
    // This is a bit of a hack, ideally the parser state would be passed explicitly
    if (parse_mode == 0) { // cli_parse is active
        cur_func = cur_parse;
        advance_func = advance_parse;
    } else { // cli_calc_eval_line is active
        cur_func = cur_eval;
        advance_func = advance_eval;
    }


    if (ftok->type == TOK_FUNC_D_D)
    {
        /*
         * Function: double -> double
         * Also handles image -> image
         */
        val_t arg = parse_expr(0);
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

        if (arg.type == VAL_STRING)
        {
            /* image -> image via function */
            if (!check_image(arg.sval))
            {
                return mk_string("");
            }
            const char *tmpn = alloc_tmpname();
            arith_image_function_im_im__d_d(
                arg.sval, tmpn, ftok->fnctptr
            );
            if (data.core.Debug > 0)
            {
                printf("double_func(double)\n");
            }
            return mk_string(tmpn);
        }

        double darg = to_double(arg);
        double res  = ((double (*)(double)) ftok->fnctptr)(darg);
        if (data.core.Debug > 0)
        {
            printf("double=func(double)\n");
        }
        return mk_double(res);
    }

    if (ftok->type == TOK_FUNC_DD_D)
    {
        /*
         * Function: (double, double) -> double
         * Also handles (image, double) -> image
         * Also handles (image, image) -> image for specific functions
         */
        val_t arg1 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }

        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance_func();

        val_t arg2 = parse_expr(0);
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

        if (arg1.type == VAL_STRING && arg2.type == VAL_STRING)
        {
            if (!check_image(arg1.sval) || !check_image(arg2.sval))
            {
                return mk_string("");
            }
            const char *tmpn = alloc_tmpname();
            if (ftok->fnctptr == fmin)
            {
                arith_image_minv(arg1.sval, arg2.sval, tmpn);
            }
            else if (ftok->fnctptr == fmax)
            {
                arith_image_maxv(arg1.sval, arg2.sval, tmpn);
            }
            else if (ftok->fnctptr == fmod)
            {
                arith_image_fmod(arg1.sval, arg2.sval, tmpn);
            }
            else if (ftok->fnctptr == pow)
            {
                arith_image_pow(arg1.sval, arg2.sval, tmpn);
            }
            else
            {
                parse_errmsg("Unsupported (image, image) function");
                return mk_string("");
            }
            return mk_string(tmpn);
        }

        if (arg1.type == VAL_STRING)
        {
            if (!check_image(arg1.sval))
            {
                return mk_string("");
            }
            const char *tmpn = alloc_tmpname();
            arith_image_function_imd_im__dd_d(
                arg1.sval,
                to_double(arg2),
                tmpn,
                ftok->fnctptr
            );
            return mk_string(tmpn);
        }

        double d1 = to_double(arg1);
        double d2 = to_double(arg2);
        return mk_double(((double (*)(double, double)) ftok->fnctptr)(d1, d2));
    }

    if (ftok->type == TOK_FUNC_DDD_D)
    {
        /*
         * Function: (d, d, d) -> double
         * Also handles (image, d, d) -> image
         */
        val_t arg1 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }

        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance_func();

        val_t arg2 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }

        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance_func();

        val_t arg3 = parse_expr(0);
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

        if (arg1.type == VAL_STRING)
        {
            if (!check_image(arg1.sval))
            {
                return mk_string("");
            }
            const char *tmpn = alloc_tmpname();
            arith_image_function_imdd_im__ddd_d(
                arg1.sval,
                to_double(arg2),
                to_double(arg3),
                tmpn,
                ftok->fnctptr
            );
            return mk_string(tmpn);
        }

        double d1 = to_double(arg1);
        double d2 = to_double(arg2);
        double d3 = to_double(arg3);
        return mk_double(
            ((double (*)(double, double, double)) ftok->fnctptr)(d1, d2, d3)
        );
    }

    if (ftok->type == TOK_FUNC_WHERE)
    {
        val_t cond = parse_expr(0);
        if (parse_error || eval_error)
            return mk_double(0);

        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance_func();

        val_t argT = parse_expr(0);
        if (parse_error || eval_error)
            return mk_double(0);

        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance_func();

        val_t argF = parse_expr(0);
        if (parse_error || eval_error)
            return mk_double(0);

        if (cur_func()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_double(0);
        }
        advance_func();

        /* If cond is scalar */
        if (cond.type != VAL_STRING)
        {
            double c = to_double(cond);
            if (c != 0) return argT;
            else return argF;
        }

        /* cond is an image. Handle T/F types */
        char tmp_mask[200], tmp_imask[200], tmp_tpart[200], tmp_fpart[200], tmpn[200];
        snprintf(tmp_mask, 200, "%s", alloc_tmpname());
        snprintf(tmp_imask, 200, "%s", alloc_tmpname());
        snprintf(tmp_tpart, 200, "%s", alloc_tmpname());
        snprintf(tmp_fpart, 200, "%s", alloc_tmpname());
        snprintf(tmpn, 200, "%s", alloc_tmpname());

        if (!check_image(cond.sval))
            return mk_string("");

        /* mask = (cond != 0) */
        arith_image_csttestne(cond.sval, 0.0, tmp_mask);
        /* imask = (cond == 0) */
        arith_image_cstteste(cond.sval, 0.0, tmp_imask);

        if (argT.type == VAL_STRING)
        {
            if (!check_image(argT.sval)) return mk_string("");
            arith_image_mult(argT.sval, tmp_mask, tmp_tpart);
        }
        else
        {
            arith_image_cstmult(tmp_mask, to_double(argT), tmp_tpart);
        }

        if (argF.type == VAL_STRING)
        {
            if (!check_image(argF.sval)) return mk_string("");
            arith_image_mult(argF.sval, tmp_imask, tmp_fpart);
        }
        else
        {
            arith_image_cstmult(tmp_imask, to_double(argF), tmp_fpart);
        }

        arith_image_add(tmp_tpart, tmp_fpart, tmpn);
        return mk_string(tmpn);
    }

    if (ftok->type == TOK_FUNC_IMIM_D)
    {
        val_t arg1 = parse_expr(0);
        if (parse_error || eval_error) return mk_double(0);
        if (cur_func()->type != TOK_COMMA) { parse_errmsg("Expected ','"); return mk_double(0); }
        advance_func();
        val_t arg2 = parse_expr(0);
        if (parse_error || eval_error) return mk_double(0);
        if (cur_func()->type != TOK_RPAREN) { parse_errmsg("Expected ')'"); return mk_double(0); }
        advance_func();

        if (arg1.type != VAL_STRING || arg2.type != VAL_STRING) {
            parse_errmsg("Expected two image arguments");
            return mk_double(0);
        }
        if (!check_image(arg1.sval) || !check_image(arg2.sval)) return mk_double(0);

        // Call ftok->fnctptr for vector operation (e.g. dot())
        double res = ((double (*)(const char *, const char *)) ftok->fnctptr)(arg1.sval, arg2.sval);
        return mk_double(res);
    }

    if (ftok->type == TOK_FUNC_IM_D)
    {
        /* Function: image -> double */
        val_t arg = parse_expr(0);
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
            parse_errmsg(
                "Expected image argument"
            );
            return mk_double(0);
        }
        if (!check_image(arg.sval))
        {
            return mk_double(0);
        }

        double res = ((double (*)(const char *)) ftok->fnctptr)(arg.sval);
        if (data.core.Debug > 0)
        {
            printf("double=func(image)\n");
        }
        return mk_double(res);
    }

    if (ftok->type == TOK_FUNC_IMD_D)
    {
        /* Function: (image, double) -> double */
        val_t arg1 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }

        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance_func();

        val_t arg2 = parse_expr(0);
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

        if (arg1.type != VAL_STRING)
        {
            parse_errmsg(
                "Expected image argument"
            );
            return mk_double(0);
        }
        if (!check_image(arg1.sval))
        {
            return mk_double(0);
        }

        double res = ((double (*)(const char *, double)) ftok->fnctptr)(
            arg1.sval, to_double(arg2)
        );
        return mk_double(res);
    }

    /* strlen(string) -> long */
    if (ftok->type == TOK_FUNC_S_D)
    {
        val_t arg = parse_expr(0);
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

        /* Argument is a variable name or
         * image name stored in sval */
        if (arg.type == VAL_STRING)
        {
            /* Try CLI var first */
            const char *cv =
                cli_var_get(arg.sval);
            if (cv != NULL)
            {
                return mk_long(
                    (long) strlen(cv)
                );
            }
            /* Fallback: length of name */
            return mk_long(
                (long) strlen(arg.sval)
            );
        }
        /* If numeric, convert to string */
        char numstr[64];
        if (arg.type == VAL_LONG)
        {
            snprintf(numstr, 64,
                     "%ld", arg.lval);
        }
        else
        {
            snprintf(numstr, 64,
                     "%g", arg.dval);
        }
        return mk_long(
            (long) strlen(numstr)
        );
    }

    /* toupper(s), tolower(s) -> string */
    if (ftok->type == TOK_FUNC_S_S)
    {
        val_t arg = parse_expr(0);
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

        /* Resolve string value */
        const char *sv = NULL;
        char numbuf[64];
        if (arg.type == VAL_STRING)
        {
            const char *cv =
                cli_var_get(arg.sval);
            sv = cv ? cv : arg.sval;
        }
        else if (arg.type == VAL_LONG)
        {
            snprintf(numbuf, 64,
                     "%ld", arg.lval);
            sv = numbuf;
        }
        else
        {
            snprintf(numbuf, 64,
                     "%g", arg.dval);
            sv = numbuf;
        }

        char result[CLI_CALC_TOKEN_MAXLEN];
        strncpy(result, sv,
                sizeof(result) - 1);
        result[sizeof(result) - 1] = '\0';

        const char *fn = ftok->sval;
        if (strncmp(fn, "toupper(", 8) == 0)
        {
            for (int i = 0;
                 result[i]; i++)
            {
                result[i] = (char) toupper(
                    (unsigned char) result[i]);
            }
        }
        else /* tolower */
        {
            for (int i = 0;
                 result[i]; i++)
            {
                result[i] = (char) tolower(
                    (unsigned char) result[i]);
            }
        }

        /* Store as temp CLI var */
        const char *tmpn = alloc_tmpname();
        cli_var_set(tmpn, result);
        return mk_string(tmpn);
    }

    /* substr(s, off, len) -> string */
    if (ftok->type == TOK_FUNC_SDD_S)
    {
        val_t arg1 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }
        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("substr: need 3 args");
            return mk_double(0);
        }
        advance_func();
        val_t arg2 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }
        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg("substr: need 3 args");
            return mk_double(0);
        }
        advance_func();
        val_t arg3 = parse_expr(0);
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

        const char *sv = NULL;
        if (arg1.type == VAL_STRING)
        {
            const char *cv =
                cli_var_get(arg1.sval);
            sv = cv ? cv : arg1.sval;
        }
        else
        {
            parse_errmsg(
                "substr: first arg "
                "must be string");
            return mk_double(0);
        }

        int slen = (int) strlen(sv);
        int off = (int) to_double(arg2);
        int len = (int) to_double(arg3);
        if (off < 0) { off = 0; }
        if (off > slen) { off = slen; }
        if (len < 0) { len = 0; }
        if (off + len > slen)
        {
            len = slen - off;
        }

        char result[CLI_CALC_TOKEN_MAXLEN];
        memcpy(result, sv + off,
               (size_t) len);
        result[len] = '\0';

        const char *tmpn = alloc_tmpname();
        cli_var_set(tmpn, result);
        return mk_string(tmpn);
    }

    /* replace(s, old, new) -> string */
    if (ftok->type == TOK_FUNC_SSS_S)
    {
        val_t arg1 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }
        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg(
                "replace: need 3 args");
            return mk_double(0);
        }
        advance_func();
        val_t arg2 = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }
        if (cur_func()->type != TOK_COMMA)
        {
            parse_errmsg(
                "replace: need 3 args");
            return mk_double(0);
        }
        advance_func();
        val_t arg3 = parse_expr(0);
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

        const char *s1 = NULL;
        const char *s2 = NULL;
        const char *s3 = NULL;
        if (arg1.type == VAL_STRING)
        {
            const char *cv =
                cli_var_get(arg1.sval);
            s1 = cv ? cv : arg1.sval;
        }
        if (arg2.type == VAL_STRING)
        {
            const char *cv =
                cli_var_get(arg2.sval);
            s2 = cv ? cv : arg2.sval;
        }
        if (arg3.type == VAL_STRING)
        {
            const char *cv =
                cli_var_get(arg3.sval);
            s3 = cv ? cv : arg3.sval;
        }
        if (!s1 || !s2 || !s3)
        {
            parse_errmsg(
                "replace: all args "
                "must be strings");
            return mk_double(0);
        }

        char result[CLI_CALC_TOKEN_MAXLEN];
        result[0] = '\0';
        int s2len = (int) strlen(s2);
        if (s2len == 0)
        {
            strncpy(result, s1,
                    sizeof(result) - 1);
            result[sizeof(result) - 1]
                = '\0';
        }
        else
        {
            const char *p = s1;
            char *w = result;
            char *end = result
                + sizeof(result) - 1;
            while (*p && w < end)
            {
                if (strncmp(p, s2,
                    (size_t) s2len) == 0)
                {
                    int rlen =
                        (int) strlen(s3);
                    if (w + rlen < end)
                    {
                        memcpy(w, s3,
                            (size_t) rlen);
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

        const char *tmpn = alloc_tmpname();
        cli_var_set(tmpn, result);
        return mk_string(tmpn);
    }

    /* hex(n), oct(n), bin(n) -> string */
    if (ftok->type == TOK_FUNC_D_S)
    {
        val_t arg = parse_expr(0);
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

        long iv = (arg.type == VAL_LONG)
            ? arg.lval
            : (long) to_double(arg);

        char result[CLI_CALC_TOKEN_MAXLEN];
        const char *fn = ftok->sval;
        if (strncmp(fn, "hex(", 4) == 0)
        {
            snprintf(result,
                     sizeof(result),
                     "0x%lx", iv);
        }
        else if (strncmp(fn, "oct(", 4)
                 == 0)
        {
            snprintf(result,
                     sizeof(result),
                     "0o%lo", iv);
        }
        else /* bin */
        {
            char *w = result;
            *w++ = '0';
            *w++ = 'b';
            if (iv == 0)
            {
                *w++ = '0';
            }
            else
            {
                long v = iv;
                char bits[65];
                int bi = 0;
                while (v > 0 && bi < 64)
                {
                    bits[bi++] =
                        (v & 1) ? '1' : '0';
                    v >>= 1;
                }
                for (int i = bi - 1;
                     i >= 0; i--)
                {
                    *w++ = bits[i];
                }
            }
            *w = '\0';
        }

        const char *tmpn = alloc_tmpname();
        cli_var_set(tmpn, result);
        return mk_string(tmpn);
    }

    parse_errmsg("Unknown function type");
    return mk_double(0);
}

/* --------------------------------------------------------
 * Primary expression parser
 *
 * Handles atoms: numbers, parenthesized expressions,
 * unary minus, function calls, and identifiers
 * (with optional assignment).
 * -------------------------------------------------------- */

/**
 * @brief Parse a primary (non-operator) expression
 */

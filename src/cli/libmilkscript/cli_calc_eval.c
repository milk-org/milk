#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "CLIcore_script.h"
#include "COREMOD_memory/stream_slice.h"
#include "cli_calc_internal.h"

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
    switch(t)
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
    return (t == TOK_OP_CARET)
           || (t == TOK_EQUAL)
           || (t == TOK_OP_PLUS_EQ)
           || (t == TOK_OP_MINUS_EQ)
           || (t == TOK_OP_STAR_EQ)
           || (t == TOK_OP_SLASH_EQ)
           || (t == TOK_OP_QUESTION);
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

    if(parse_error || eval_error)    // Check both error flags
    {
        return mk_long(0);
    }

    // Determine which token stream to use based on which error flag is active
    if(parse_mode == 0)    // cli_parse is active
    {
        cur_func = cur_parse;
        advance_func = advance_parse;
    }
    else     // cli_calc_eval_line is active
    {
        cur_func = cur_eval;
        advance_func = advance_eval;
    }

    t = cur_func();

    /* number literals */
    if(t->type == TOK_LONG)
    {
        advance_func();
        if(data.core.Debug > 0)
        {
            printf("this is a long\n");
        }
        return mk_long(t->val_l);
    }

    if(t->type == TOK_DOUBLE)
    {
        advance_func();
        if(data.core.Debug > 0)
        {
            printf("this is a double\n");
        }
        return mk_double(t->val_d);
    }

    /* unary minus */
    if(t->type == TOK_OP_MINUS)
    {
        advance_func();
        val_t v = parse_primary();
        if(parse_error || eval_error)
        {
            return mk_long(0);
        }
        if(v.type == VAL_LONG)
        {
            if(data.core.Debug > 0)
            {
                printf("-long\n");
            }
            return mk_long(-v.lval);
        }
        if(v.type == VAL_DOUBLE)
        {
            if(data.core.Debug > 0)
            {
                printf("-double\n");
            }
            return mk_double(-v.dval);
        }
        if(v.type == VAL_STRING)
        {
            if(data.core.Debug > 0)
            {
                printf("-image\n");
            }
            if(!check_image(v.sval))
            {
                return mk_double(0);
            }
            const char *tmpn = alloc_tmpname();
            arith_image_cstsubm(v.sval, 0.0, tmpn);
            return mk_string(tmpn);
        }
        parse_errmsg(
            "Cannot negate expression"
        );
        return mk_double(0);
    }

    /* unary bitwise NOT */
    if(t->type == TOK_OP_BNOT)
    {
        advance_func();
        val_t v = parse_primary();
        if(parse_error || eval_error)
        {
            return mk_long(0);
        }
        long iv = (v.type == VAL_LONG)
                  ? v.lval : (long) v.dval;
        return mk_long(~iv);
    }

    /* parenthesized expression */
    if(t->type == TOK_LPAREN)
    {
        advance_func();
        val_t v = parse_expr(0);
        if(parse_error || eval_error)
        {
            return v;
        }
        if(cur_func()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_long(0);
        }
        advance_func();
        return v;
    }

    /* function calls */
    if(t->type == TOK_FUNC_D_D
            || t->type == TOK_FUNC_DD_D
            || t->type == TOK_FUNC_DDD_D
            || t->type == TOK_FUNC_IM_D
            || t->type == TOK_FUNC_IMD_D
            || t->type == TOK_FUNC_WHERE
            || t->type == TOK_FUNC_IMIM_D
            || t->type == TOK_FUNC_S_D
            || t->type == TOK_FUNC_S_S
            || t->type == TOK_FUNC_SDD_S
            || t->type == TOK_FUNC_SSS_S
            || t->type == TOK_FUNC_D_S)
    {
        advance_func();
        return parse_funccall(t);
    }

    /* existing variable: might be assignment */
    if(t->type == TOK_VAR)
    {
        advance_func();
        cli_token_type aop = cur_func()->type;
        if(aop == TOK_EQUAL
                || aop == TOK_OP_PLUS_EQ
                || aop == TOK_OP_MINUS_EQ
                || aop == TOK_OP_STAR_EQ
                || aop == TOK_OP_SLASH_EQ)
        {
            advance_func();
            val_t v = parse_expr(0);
            if(parse_error || eval_error)
            {
                return v;
            }

            /* compound: fetch old, apply op */
            if(aop != TOK_EQUAL)
            {
                long vid = variable_ID(t->sval);
                if(vid == -1)
                {
                    parse_errmsg(
                        "Variable not found"
                        " for compound assign"
                    );
                    return mk_double(0);
                }
                double old;
                if(data.core.variable[vid]
                        .type == 1)
                {
                    old = (double)
                          data.core.variable[vid]
                          .value.l;
                }
                else
                {
                    old = data.core.variable[vid]
                          .value.f;
                }
                double nv = to_double(v);
                switch(aop)
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
                    if(nv == 0.0)
                    {
                        parse_errmsg(
                            "Division by zero"
                        );
                        return mk_double(0);
                    }
                    nv = old / nv;
                    break;
                default:
                    break;
                }
                /* check if result is integer */
                if(v.type == VAL_LONG
                        && data.core.variable[vid]
                        .type == 1
                        && aop != TOK_OP_SLASH_EQ)
                {
                    v = mk_long((long) nv);
                }
                else
                {
                    v = mk_double(nv);
                }
            }

            if(v.type == VAL_STRING)
            {
                /* var = image -> rename */
                chname_image_ID(
                    v.sval, t->sval
                );
                if(data.core.Debug > 0)
                {
                    printf("changing name\n");
                }
                return mk_string(t->sval);
            }
            if(v.type == VAL_LONG)
            {
                create_variable_long_ID(
                    t->sval, v.lval
                );
                return mk_long(v.lval);
            }
            create_variable_ID(
                t->sval, to_double(v)
            );
            return mk_double(to_double(v));
        }
        /* just a variable reference */
        long vID = variable_ID(t->sval);
        if(vID == -1)
        {
            char msg[2048];
            snprintf(msg, sizeof(msg),
                     "Variable '%s' not found",
                     t->sval);
            parse_errmsg(msg);
            return mk_double(0);
        }
        if(data.core.variable[vID].type == 1)
        {
            return mk_long(
                       data.core.variable[vID].value.l
                   );
        }
        return mk_double(
                   data.core.variable[vID].value.f
               );
    }

    /* new variable: must be assignment, or it's
     * a new string/image name */
    if(t->type == TOK_NVAR)
    {
        advance_func();
        if(cur_func()->type == TOK_EQUAL)
        {
            advance_func();
            val_t v = parse_expr(0);
            if(parse_error || eval_error)
            {
                return v;
            }
            if(v.type == VAL_STRING)
            {
                if(image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) == -1)
                {
                    parse_errmsg("Source image does not exist");
                    return mk_double(0);
                }
                chname_image_ID(
                    v.sval, t->sval
                );
                if(data.core.Debug > 0)
                {
                    printf("changing name\n");
                }
                return mk_string(t->sval);
            }
            if(v.type == VAL_LONG)
            {
                if(data.core.Debug > 0)
                {
                    printf("creating long\n");
                }
            }
            else
            {
                if(data.core.Debug > 0)
                {
                    printf("creating double\n");
                }
            }
            if(v.type == VAL_LONG)
            {
                create_variable_long_ID(
                    t->sval, v.lval
                );
                char numv[64];
                snprintf(
                    numv, 64, "%ld", v.lval
                );
                if(parse_mode == 1)
                {
                    cli_var_set(t->sval, numv);
                }
                return mk_long(v.lval);
            }
            create_variable_ID(
                t->sval, to_double(v)
            );
            {
                char numv[64];
                snprintf(
                    numv, 64, "%.*g",
                    cli_float_digits,
                    to_double(v)
                );
                if(parse_mode == 1)
                {
                    cli_var_set(t->sval, numv);
                }
            }
            return mk_double(to_double(v));
        }
        /* standalone new variable/image name */
        // This path is only for cli_parse, not cli_calc_eval_line
        if(parse_mode == 0)    // If not in eval_line context
        {
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_STRING;
            if(data.core.Debug > 0)
            {
                printf(
                    "this is a string "
                    "(new variable/image)\n"
                );
            }
        }
        return mk_string(t->sval);
    }

    /* existing image: might be assignment */
    if(t->type == TOK_IMAGE)
    {
        advance_func();
        if(cur_func()->type == TOK_EQUAL)
        {
            advance_func();
            val_t v = parse_expr(0);
            if(parse_error || eval_error)
            {
                return v;
            }
            if(v.type == VAL_STRING)
            {
                if(image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) == -1)
                {
                    parse_errmsg("Source image does not exist");
                    return mk_double(0);
                }
                delete_image_ID(
                    t->sval,
                    1
                );
                chname_image_ID(
                    v.sval, t->sval
                );
                if(data.core.Debug > 0)
                {
                    printf("changing name\n");
                }
                return mk_string(t->sval);
            }
            parse_errmsg(
                "Cannot assign scalar to image"
            );
            return mk_double(0);
        }
        // This path is only for cli_parse, not cli_calc_eval_line
        if(!eval_error)    // If not in eval_line context
        {
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_EXISTINGIMAGE;
            if(data.core.Debug > 0)
            {
                printf(
                    "this is a string "
                    "(existing image)\n"
                );
            }
        }
        /* Check for slice bracket syntax.
         * Materialize the sliced region into
         * a temporary image and return that
         * name instead. */
        if(strchr(t->sval, '[') != NULL)
        {
            char bare[200];
            char btext[200];
            int has_brk =
                imgid_slice_split_name(
                    t->sval,
                    bare, (int) sizeof(bare),
                    btext,
                    (int) sizeof(btext));
            if(has_brk)
            {
                imageID srcid = image_ID(
                                    bare,
                                    data.core.image,
                                    data.core.NB_MAX_IMAGE);
                if(srcid == -1)
                {
                    parse_errmsg(
                        "Source image "
                        "not found");
                    return mk_double(0);
                }
                IMAGE *srcim =
                    &data.core.image[srcid];
                IMGID_SLICE slc =
                    imgid_slice_parse(btext);
                if(slc.error)
                {
                    parse_errmsg(
                        slc.errmsg);
                    return mk_double(0);
                }
                uint32_t outsz[3] = {0};
                int snax =
                    srcim->md[0].naxis;
                uint32_t ssz[3];
                for(int a = 0;
                        a < snax && a < 3;
                        a++)
                {
                    ssz[a] =
                        srcim->md[0].size[a];
                }
                if(imgid_slice_output_size(
                            &slc, snax,
                            ssz, outsz) != 0)
                {
                    parse_errmsg(
                        "Bad slice dims");
                    return mk_double(0);
                }
                /* Count output axes */
                int onax = 0;
                for(int a = 0; a < 3; a++)
                {
                    if(outsz[a] > 0)
                    {
                        onax = a + 1;
                    }
                }
                if(onax == 0)
                {
                    onax = 1;
                }
                const char *tmpn =
                    alloc_tmpname();
                imageID tid =
                    create_image_ID(
                        tmpn,
                        onax,
                        outsz,
                        srcim->md[0].datatype,
                        0, 10, 0, NULL);
                if(tid != -1)
                {
                    IMGID simg;
                    memset(&simg, 0,
                           sizeof(simg));
                    simg.ID = srcid;
                    simg.im = srcim;
                    simg.md = srcim->md;
                    simg.slice = slc;
                    simg.slice_im =
                        &data.core.image[tid];
                    imgid_slice_materialize(
                        &simg);
                }
                return mk_string(tmpn);
            }
        }
        return mk_string(t->sval);
    }

    /* command */
    if(t->type == TOK_COMMAND)
    {
        advance_func();
        // This path is only for cli_parse, not cli_calc_eval_line
        if(parse_mode == 0)    // If not in eval_line context
        {
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_COMMAND;
            if(data.core.Debug > 0)
            {
                printf(
                    "this is a string (command)\n"
                );
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

    if(parse_error || eval_error)    // Check both error flags
    {
        return mk_long(0);
    }

    // Determine which token stream to use based on which error flag is active
    if(parse_mode == 0)    // cli_parse is active
    {
        cur_func = cur_parse;
        advance_func = advance_parse;
    }
    else     // cli_calc_eval_line is active
    {
        cur_func = cur_eval;
        advance_func = advance_eval;
    }


    val_t left = parse_primary();
    if(parse_error || eval_error)
    {
        return left;
    }

    for(;;)
    {
        cli_token_type op = cur_func()->type;
        int prec = get_prec(op);

        if(prec < min_prec)
        {
            break;
        }

        advance_func();

        /* Ternary ? : operator */
        if(op == TOK_OP_QUESTION)
        {
            val_t true_val = parse_expr(-1);
            if(parse_error || eval_error)
            {
                return left;
            }
            if(cur_func()->type
                    != TOK_OP_COLON)
            {
                parse_errmsg(
                    "Expected ':' in ternary"
                );
                return mk_double(0);
            }
            advance_func();
            val_t false_val = parse_expr(-1);
            if(parse_error || eval_error)
            {
                return left;
            }
            /* Scalar ternary */
            if(left.type != VAL_STRING)
            {
                double c = to_double(left);
                left = (c != 0.0)
                       ? true_val : false_val;
            }
            else
            {
                /* Image ternary -> where() */
                if(!check_image(left.sval))
                {
                    return mk_string("");
                }
                const char *tmask =
                    alloc_tmpname();
                const char *timask =
                    alloc_tmpname();
                const char *tpart =
                    alloc_tmpname();
                const char *fpart =
                    alloc_tmpname();
                const char *tmpn =
                    alloc_tmpname();
                arith_image_csttestne(
                    left.sval, 0.0,
                    tmask
                );
                arith_image_cstteste(
                    left.sval, 0.0,
                    timask
                );
                if(true_val.type
                        == VAL_STRING)
                {
                    if(!check_image(
                                true_val.sval))
                    {
                        return mk_string("");
                    }
                    arith_image_mult(
                        true_val.sval,
                        tmask, tpart
                    );
                }
                else
                {
                    arith_image_cstmult(
                        tmask,
                        to_double(true_val),
                        tpart
                    );
                }
                if(false_val.type
                        == VAL_STRING)
                {
                    if(!check_image(
                                false_val.sval))
                    {
                        return mk_string("");
                    }
                    arith_image_mult(
                        false_val.sval,
                        timask, fpart
                    );
                }
                else
                {
                    arith_image_cstmult(
                        timask,
                        to_double(false_val),
                        fpart
                    );
                }
                arith_image_add(
                    tpart, fpart, tmpn
                );
                left = mk_string(tmpn);
            }
            continue;
        }

        int next_min = is_right_assoc(op)
                       ? prec
                       : prec + 1;
        val_t right = parse_expr(next_min);
        if(parse_error || eval_error)
        {
            return left;
        }

        left = eval_binop(op, left, right);
        if(parse_error || eval_error)
        {
            return left;
        }
    }

    return left;
}

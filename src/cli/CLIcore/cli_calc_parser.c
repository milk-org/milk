/**
 * @file cli_calc_parser.c
 * @brief Hand-written Pratt parser for CLI expressions
 *
 * Replaces the bison-generated parser (calc_bison.y).
 *
 * Approach:
 * Uses precedence-climbing (Pratt parsing) to handle
 * operator precedence for +, -, *, /, ^ with three
 * value types: long, double, and string (image name).
 * The parser tokenizes the input, then recursively
 * evaluates expressions, populating
 * data.cmdargtoken[data.cmdNBarg] with the result.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"

#include "cli_calc_tokenizer.h"
#include "cli_calc_parser.h"

/* --------------------------------------------------------
 * Parser state
 * -------------------------------------------------------- */

/** token stream and current index */
static cli_token  parse_tokens[CLI_CALC_MAX_TOKENS];
static int        parse_pos;
static int        parse_ntok;
static int        parse_error;

static char calctmpimname[200];

/* --------------------------------------------------------
 * Value types for the expression evaluator
 * -------------------------------------------------------- */

/** Expression result type tag */
typedef enum
{
    VAL_LONG,
    VAL_DOUBLE,
    VAL_STRING
} val_type;

/** Expression result value */
typedef struct
{
    val_type type;
    long     lval;
    double   dval;
    char     sval[CLI_CALC_TOKEN_MAXLEN];
} val_t;

/* --------------------------------------------------------
 * Forward declarations
 * -------------------------------------------------------- */

static val_t parse_expr(int min_prec);
static val_t parse_primary(void);

/* --------------------------------------------------------
 * Token stream helpers
 * -------------------------------------------------------- */

static cli_token *cur(void)
{
    return &parse_tokens[parse_pos];
}

static cli_token *advance(void)
{
    cli_token *t = &parse_tokens[parse_pos];
    if (parse_pos < parse_ntok)
    {
        parse_pos++;
    }
    return t;
}

/**
 * @brief Report a parse error and set error flag
 */
static void parse_errmsg(const char *msg)
{
    printf(
        "\033[31mPARSING ERROR ON COMMAND LINE "
        "ARG %ld: %s\033[0m\n",
        data.cmdNBarg,
        msg
    );
    data.parseerror = 1;
    parse_error = 1;
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

static int get_prec(cli_token_type t)
{
    switch (t)
    {
        case TOK_OP_PLUS:
        case TOK_OP_MINUS:
            return 1;
        case TOK_OP_STAR:
        case TOK_OP_SLASH:
            return 2;
        case TOK_OP_CARET:
            return 3;
        default:
            return -1;
    }
}

static int is_right_assoc(cli_token_type t)
{
    return t == TOK_OP_CARET;
}

/* --------------------------------------------------------
 * Type promotion helpers
 *
 * When mixing long and double in an operation, promote
 * the long to double.  When an image (string) is
 * involved, dispatch to the COREMOD_arith functions.
 * -------------------------------------------------------- */

/**
 * @brief Promote a val_t to double if it is long
 */
static double to_double(val_t v)
{
    if (v.type == VAL_LONG)
    {
        return (double) v.lval;
    }
    return v.dval;
}

/**
 * @brief Make a long result
 */
static val_t mk_long(long v)
{
    val_t r;
    r.type = VAL_LONG;
    r.lval = v;
    r.dval = 0.0;
    r.sval[0] = '\0';
    return r;
}

/**
 * @brief Make a double result
 */
static val_t mk_double(double v)
{
    val_t r;
    r.type = VAL_DOUBLE;
    r.lval = 0;
    r.dval = v;
    r.sval[0] = '\0';
    return r;
}

/**
 * @brief Make a string/image result
 */
static val_t mk_string(const char *s)
{
    val_t r;
    r.type = VAL_STRING;
    r.lval = 0;
    r.dval = 0.0;
    snprintf(r.sval, CLI_CALC_TOKEN_MAXLEN,
             "%s", s);
    return r;
}

/**
 * @brief Check if an image name is valid
 *
 * If the image is not found, reports a parse error
 * and returns 0.
 */
static int check_image(const char *name)
{
    if (image_ID(
            name,
            data.core.image,
            data.core.NB_MAX_IMAGE) == -1)
    {
        char msg[200];
        snprintf(msg, 200,
                 "Image '%s' not found", name);
        parse_errmsg(msg);
        return 0;
    }
    return 1;
}

/**
 * @brief Allocate a temporary image name
 */
static const char *alloc_tmpname(void)
{
    snprintf(calctmpimname, 200,
             "_tmpcalc%ld",
             data.calctmp_imindex);
    data.calctmp_imindex++;
    return calctmpimname;
}

/* --------------------------------------------------------
 * Binary operator dispatch
 *
 * Handles all combinations of (long, double, string)
 * for +, -, *, /, ^ to match the bison grammar exactly.
 * -------------------------------------------------------- */

/**
 * @brief Evaluate a binary operation
 *
 * Implements type-aware arithmetic matching the
 * original bison grammar's precedence and type rules.
 */
static val_t eval_binop(
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
            switch (op)
            {
                case TOK_OP_PLUS:
                    arith_image_add(
                        left.sval,
                        right.sval,
                        tmpn
                    );
                    break;
                case TOK_OP_MINUS:
                    arith_image_sub(
                        left.sval,
                        right.sval,
                        tmpn
                    );
                    break;
                case TOK_OP_STAR:
                    arith_image_mult(
                        left.sval,
                        right.sval,
                        tmpn
                    );
                    break;
                case TOK_OP_SLASH:
                    arith_image_div(
                        left.sval,
                        right.sval,
                        tmpn
                    );
                    break;
                default:
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

            switch (op)
            {
                case TOK_OP_PLUS:
                    arith_image_cstadd(
                        left.sval, rv, tmpn);
                    break;
                case TOK_OP_MINUS:
                    arith_image_cstadd(
                        left.sval, -rv, tmpn);
                    break;
                case TOK_OP_STAR:
                    arith_image_cstmult(
                        left.sval, rv, tmpn);
                    break;
                case TOK_OP_SLASH:
                    arith_image_cstdiv(
                        left.sval, rv, tmpn);
                    break;
                case TOK_OP_CARET:
                    arith_image_cstpow(
                        left.sval, rv, tmpn);
                    break;
                default:
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

            switch (op)
            {
                case TOK_OP_PLUS:
                    arith_image_cstadd(
                        right.sval, lv, tmpn);
                    break;
                case TOK_OP_MINUS:
                    arith_image_cstsubm(
                        right.sval, lv, tmpn);
                    break;
                case TOK_OP_STAR:
                    arith_image_cstmult(
                        right.sval, lv, tmpn);
                    break;
                case TOK_OP_SLASH:
                    arith_image_cstdiv1(
                        right.sval, lv, tmpn);
                    break;
                default:
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
            case TOK_OP_CARET:
                return mk_long(
                    (long) pow(
                        left.lval, right.lval)
                );
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
                return mk_double(lv / rv);
            case TOK_OP_CARET:
                return mk_double(pow(lv, rv));
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
static val_t parse_funccall(cli_token *ftok)
{
    if (ftok->type == TOK_FUNC_D_D)
    {
        /*
         * Function: double -> double
         * Also handles image -> image
         */
        val_t arg = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_double(0);
        }
        advance();

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
        double res  = ftok->fnctptr(darg);
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
         */
        val_t arg1 = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance();

        val_t arg2 = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_double(0);
        }
        advance();

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
        return mk_double(ftok->fnctptr(d1, d2));
    }

    if (ftok->type == TOK_FUNC_DDD_D)
    {
        /*
         * Function: (d, d, d) -> double
         * Also handles (image, d, d) -> image
         */
        val_t arg1 = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance();

        val_t arg2 = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance();

        val_t arg3 = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_double(0);
        }
        advance();

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
            ftok->fnctptr(d1, d2, d3)
        );
    }

    if (ftok->type == TOK_FUNC_IM_D)
    {
        /* Function: image -> double */
        val_t arg = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_double(0);
        }
        advance();

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

        double res = ftok->fnctptr(arg.sval);
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
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_COMMA)
        {
            parse_errmsg("Expected ','");
            return mk_double(0);
        }
        advance();

        val_t arg2 = parse_expr(0);
        if (parse_error)
        {
            return mk_double(0);
        }

        if (cur()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_double(0);
        }
        advance();

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

        double res = ftok->fnctptr(
            arg1.sval, to_double(arg2)
        );
        return mk_double(res);
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
static val_t parse_primary(void)
{
    cli_token *t = cur();

    /* number literals */
    if (t->type == TOK_LONG)
    {
        advance();
        if (data.core.Debug > 0)
        {
            printf("this is a long\n");
        }
        return mk_long(t->val_l);
    }

    if (t->type == TOK_DOUBLE)
    {
        advance();
        if (data.core.Debug > 0)
        {
            printf("this is a double\n");
        }
        return mk_double(t->val_d);
    }

    /* unary minus */
    if (t->type == TOK_OP_MINUS)
    {
        advance();
        val_t v = parse_primary();
        if (parse_error)
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
        parse_errmsg(
            "Cannot negate image expression"
        );
        return mk_double(0);
    }

    /* parenthesized expression */
    if (t->type == TOK_LPAREN)
    {
        advance();
        val_t v = parse_expr(0);
        if (parse_error)
        {
            return v;
        }
        if (cur()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_long(0);
        }
        advance();
        return v;
    }

    /* function calls */
    if (t->type == TOK_FUNC_D_D
        || t->type == TOK_FUNC_DD_D
        || t->type == TOK_FUNC_DDD_D
        || t->type == TOK_FUNC_IM_D
        || t->type == TOK_FUNC_IMD_D)
    {
        advance();
        return parse_funccall(t);
    }

    /* existing variable: might be assignment */
    if (t->type == TOK_VAR)
    {
        advance();
        if (cur()->type == TOK_EQUAL)
        {
            advance();
            val_t v = parse_expr(0);
            if (parse_error)
            {
                return v;
            }
            if (v.type == VAL_STRING)
            {
                /* var = image -> rename */
                chname_image_ID(
                    v.sval, t->sval
                );
                if (data.core.Debug > 0)
                {
                    printf("changing name\n");
                }
                return mk_string(t->sval);
            }
            create_variable_ID(
                t->sval, to_double(v)
            );
            return mk_double(to_double(v));
        }
        /* just a variable reference */
        long vID = variable_ID(t->sval);
        if (vID == -1)
        {
            char msg[200];
            snprintf(msg, 200,
                     "Variable '%s' not found",
                     t->sval);
            parse_errmsg(msg);
            return mk_double(0);
        }
        return mk_double(
            data.core.variable[vID].value.f
        );
    }

    /* new variable: must be assignment, or it's
     * a new string/image name */
    if (t->type == TOK_NVAR)
    {
        advance();
        if (cur()->type == TOK_EQUAL)
        {
            advance();
            val_t v = parse_expr(0);
            if (parse_error)
            {
                return v;
            }
            if (v.type == VAL_STRING)
            {
                chname_image_ID(
                    v.sval, t->sval
                );
                if (data.core.Debug > 0)
                {
                    printf("changing name\n");
                }
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
            create_variable_ID(
                t->sval, to_double(v)
            );
            return mk_double(to_double(v));
        }
        /* standalone new variable/image name */
        data.cmdargtoken[data.cmdNBarg].type =
            CMDARGTOKEN_TYPE_STRING;
        if (data.core.Debug > 0)
        {
            printf(
                "this is a string "
                "(new variable/image)\n"
            );
        }
        return mk_string(t->sval);
    }

    /* existing image: might be assignment */
    if (t->type == TOK_IMAGE)
    {
        advance();
        if (cur()->type == TOK_EQUAL)
        {
            advance();
            val_t v = parse_expr(0);
            if (parse_error)
            {
                return v;
            }
            if (v.type == VAL_STRING)
            {
                delete_image_ID(
                    t->sval,
                    1
                );
                chname_image_ID(
                    v.sval, t->sval
                );
                if (data.core.Debug > 0)
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
        data.cmdargtoken[data.cmdNBarg].type =
            CMDARGTOKEN_TYPE_EXISTINGIMAGE;
        if (data.core.Debug > 0)
        {
            printf(
                "this is a string "
                "(existing image)\n"
            );
        }
        return mk_string(t->sval);
    }

    /* command */
    if (t->type == TOK_COMMAND)
    {
        advance();
        data.cmdargtoken[data.cmdNBarg].type =
            CMDARGTOKEN_TYPE_COMMAND;
        if (data.core.Debug > 0)
        {
            printf(
                "this is a string (command)\n"
            );
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
static val_t parse_expr(int min_prec)
{
    val_t left = parse_primary();
    if (parse_error)
    {
        return left;
    }

    for (;;)
    {
        cli_token_type op = cur()->type;
        int prec = get_prec(op);

        if (prec < min_prec)
        {
            break;
        }

        advance();

        int next_min = is_right_assoc(op)
                       ? prec
                       : prec + 1;
        val_t right = parse_expr(next_min);
        if (parse_error)
        {
            return left;
        }

        left = eval_binop(op, left, right);
        if (parse_error)
        {
            return left;
        }
    }

    return left;
}

/* --------------------------------------------------------
 * Public entry point
 * -------------------------------------------------------- */

/**
 * @brief Parse a CLI input token and populate
 *        data.cmdargtoken[data.cmdNBarg]
 *
 * This replaces the old yy_scan_string + yyparse +
 * yylex_destroy sequence.  The input should be a
 * single token/expression terminated by '\\n'.
 *
 * After parsing:
 *  - long result -> type = CMDARGTOKEN_TYPE_LONG
 *  - double result -> type = CMDARGTOKEN_TYPE_FLOAT
 *  - string result -> type set during parse
 *    (COMMAND, EXISTINGIMAGE, STRING, etc.)
 *    and val.string is filled
 */
void cli_parse(const char *input)
{
    parse_error = 0;

    parse_ntok = cli_tokenize(
        input,
        parse_tokens,
        CLI_CALC_MAX_TOKENS
    );

    if (parse_ntok <= 0)
    {
        return;
    }

    parse_pos = 0;

    /* skip if only a newline */
    if (parse_tokens[0].type == TOK_NEWLINE
        || parse_tokens[0].type == TOK_EOF)
    {
        return;
    }

    val_t result = parse_expr(0);
    if (parse_error)
    {
        return;
    }

    /* consume the trailing newline if present */
    if (cur()->type == TOK_NEWLINE)
    {
        advance();
    }

    /* store result in data.cmdargtoken */
    switch (result.type)
    {
        case VAL_DOUBLE:
            printf("\t double: %.10g\n",
                   result.dval);
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_FLOAT;
            data.cmdargtoken[data.cmdNBarg].val
                .numf = result.dval;
            break;

        case VAL_LONG:
            printf("\t long:   %ld\n",
                   result.lval);
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_LONG;
            data.cmdargtoken[data.cmdNBarg].val
                .numl = result.lval;
            break;

        case VAL_STRING:
            if (data.core.Debug > 0)
            {
                printf("\t string: %s\n",
                       result.sval);
            }
            snprintf(
                data.cmdargtoken[data.cmdNBarg]
                    .val.string,
                STRINGMAXLEN_CMDARGTOKEN_VAL,
                "%s",
                result.sval
            );
            break;
    }
}

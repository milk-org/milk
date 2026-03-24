/**
 * @file cli_calc_parser.c
 * @brief Hand-written Pratt parser for CLI expressions
 *
 * Replaces the bison-generated parser (calc_bison.y).
 *
 * Approach:
 * Uses precedence-climbing (Pratt parsing) to evaluate
 * expressions. It supports +, -, *, /, ^, % arithmetic, 
 * logical/relational operators (<, <=, >, >=, ==, !=, &&, ||, !),
 * and dynamic variable assignments (=). Supports three value
 * types: long, double, and string (image name). Includes functions
 * like round, min, max, abs, dot, norm, and ternary conditionals (where).
 * Evaluates expressions immediately, supporting math on images as well
 * as per-pixel masking. Populates data.cmdargtoken[data.cmdNBarg].
 */

#include <math.h>
#include <stdio.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/stream_slice.h"

#include "cli_calc_tokenizer.h"
#include "cli_calc_parser.h"
#include "CLIcore_script.h"

/* --------------------------------------------------------
 * Parser state
 * -------------------------------------------------------- */

/** token stream and current index for cli_parse */
static cli_token  parse_tokens[CLI_CALC_MAX_TOKENS];
static int        parse_pos;
static int        parse_ntok;
static int        parse_error;

/** token stream and current index for cli_calc_eval_line */
static cli_token  eval_tokens[CLI_CALC_MAX_TOKENS];
static int        eval_pos;
static int        eval_ntok;
static int        eval_error;


static char calctmpimname[200];

/* --------------------------------------------------------
 * Value types for the expression evaluator
 * -------------------------------------------------------- */

/** Expression result type tag */
typedef enum
{
    VAL_LONG,
    VAL_DOUBLE,
    VAL_STRING,
    VAL_GENERIC // For functions that don't return a specific value type
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

static int parse_mode = 0; // 0 for cli_parse, 1 for cli_calc_eval_line

// Helper functions for cli_parse
static inline cli_token *cur_parse(void)
{
    return &parse_tokens[parse_pos];
}

static inline cli_token *advance_parse(void)
{
    cli_token *t = &parse_tokens[parse_pos];
    if (parse_pos < parse_ntok)
    {
        parse_pos++;
    }
    return t;
}

// Helper functions for cli_calc_eval_line
static inline cli_token *cur_eval(void)
{
    return &eval_tokens[eval_pos];
}

static inline cli_token *advance_eval(void)
{
    cli_token *t = &eval_tokens[eval_pos];
    if (eval_pos < eval_ntok)
    {
        eval_pos++;
    }
    return t;
}


/**
 * @brief Report a parse error and set error flag
 */
static void parse_errmsg(const char *msg)
{
    if (parse_mode == 1 || data.core.Debug > 0)
    {
        fprintf(stderr, "   [CALC_PARSER_ERROR] %s\n", msg);
    }
    data.parseerror = 1;
    parse_error = 1;
    if (parse_mode == 1) {
        eval_error = 1;
    }
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
        case TOK_OP_OR:
            return 1;
        case TOK_OP_AND:
            return 2;
        case TOK_OP_EQ:
        case TOK_OP_NEQ:
            return 3;
        case TOK_OP_LT:
        case TOK_OP_LE:
        case TOK_OP_GT:
        case TOK_OP_GE:
            return 4;
        case TOK_OP_PLUS:
        case TOK_OP_MINUS:
            return 5;
        case TOK_OP_STAR:
        case TOK_OP_SLASH:
        case TOK_OP_MOD:
            return 6;
        case TOK_OP_CARET:
            return 7;
        case TOK_EQUAL:
            return 0;
        default:
            return -1;
    }
}

static inline int is_right_assoc(cli_token_type t)
{
    return (t == TOK_OP_CARET) || (t == TOK_EQUAL);
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
static inline double to_double(val_t v)
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
static inline val_t mk_long(long v)
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
static inline val_t mk_double(double v)
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
static inline val_t mk_string(const char *s)
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
                case TOK_OP_MOD:
                    arith_image_fmod(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_CARET:
                    arith_image_pow(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_LT:
                    arith_image_testlt(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_LE:
                    arith_image_testle(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_GT:
                    arith_image_testmt(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_GE:
                    arith_image_testge(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_EQ:
                    arith_image_teste(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_NEQ:
                    arith_image_testne(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_AND:
                    arith_image_and(left.sval, right.sval, tmpn);
                    break;
                case TOK_OP_OR:
                    arith_image_or(left.sval, right.sval, tmpn);
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
                case TOK_OP_MOD:
                    arith_image_cstfmod(
                        left.sval, rv, tmpn);
                    break;
                case TOK_OP_CARET:
                    arith_image_cstpow(
                        left.sval, rv, tmpn);
                    break;
                case TOK_OP_LT:
                    arith_image_csttestlt(left.sval, rv, tmpn);
                    break;
                case TOK_OP_LE:
                    arith_image_csttestle(left.sval, rv, tmpn);
                    break;
                case TOK_OP_GT:
                    arith_image_csttestmt(left.sval, rv, tmpn);
                    break;
                case TOK_OP_GE:
                    arith_image_csttestge(left.sval, rv, tmpn);
                    break;
                case TOK_OP_EQ:
                    arith_image_cstteste(left.sval, rv, tmpn);
                    break;
                case TOK_OP_NEQ:
                    arith_image_csttestne(left.sval, rv, tmpn);
                    break;
                case TOK_OP_AND:
                    arith_image_cstand(left.sval, rv, tmpn);
                    break;
                case TOK_OP_OR:
                    arith_image_cstor(left.sval, rv, tmpn);
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
                case TOK_OP_LT:
                    arith_image_csttestmt(right.sval, lv, tmpn);
                    break;
                case TOK_OP_LE:
                    arith_image_csttestge(right.sval, lv, tmpn);
                    break;
                case TOK_OP_GT:
                    arith_image_csttestlt(right.sval, lv, tmpn);
                    break;
                case TOK_OP_GE:
                    arith_image_csttestle(right.sval, lv, tmpn);
                    break;
                case TOK_OP_EQ:
                    arith_image_cstteste(right.sval, lv, tmpn);
                    break;
                case TOK_OP_NEQ:
                    arith_image_csttestne(right.sval, lv, tmpn);
                    break;
                case TOK_OP_AND:
                    arith_image_cstand(right.sval, lv, tmpn);
                    break;
                case TOK_OP_OR:
                    arith_image_cstor(right.sval, lv, tmpn);
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
    cli_token *t;
    cli_token *(*cur_func)(void);
    cli_token *(*advance_func)(void);

    if (parse_error || eval_error) { // Check both error flags
        return mk_long(0);
    }

    // Determine which token stream to use based on which error flag is active
    if (parse_mode == 0) { // cli_parse is active
        cur_func = cur_parse;
        advance_func = advance_parse;
    } else { // cli_calc_eval_line is active
        cur_func = cur_eval;
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
        parse_errmsg(
            "Cannot negate expression"
        );
        return mk_double(0);
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
    if (t->type == TOK_FUNC_D_D
        || t->type == TOK_FUNC_DD_D
        || t->type == TOK_FUNC_DDD_D
        || t->type == TOK_FUNC_IM_D
        || t->type == TOK_FUNC_IMD_D
        || t->type == TOK_FUNC_WHERE
        || t->type == TOK_FUNC_IMIM_D) // Added TOK_FUNC_IMIM_D and TOK_FUNC_WHERE
    {
        advance_func();
        return parse_funccall(t);
    }

    /* existing variable: might be assignment */
    if (t->type == TOK_VAR)
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
            char msg[2048];
            snprintf(msg, sizeof(msg),
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
                if (image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) == -1) {
                    parse_errmsg("Source image does not exist");
                    return mk_double(0);
                }
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
            
            char numv[64];
            snprintf(numv, 64, "%g", to_double(v));
            if (parse_mode == 1) {
                cli_var_set(t->sval, numv);
            }
            
            return mk_double(to_double(v));
        }
        /* standalone new variable/image name */
        // This path is only for cli_parse, not cli_calc_eval_line
        if (parse_mode == 0) { // If not in eval_line context
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_STRING;
            if (data.core.Debug > 0)
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
    if (t->type == TOK_IMAGE)
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
                if (image_ID(v.sval, data.core.image, data.core.NB_MAX_IMAGE) == -1) {
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
        // This path is only for cli_parse, not cli_calc_eval_line
        if (!eval_error) { // If not in eval_line context
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_EXISTINGIMAGE;
            if (data.core.Debug > 0)
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
        if (strchr(t->sval, '[') != NULL)
        {
            char bare[200];
            char btext[200];
            int has_brk =
                imgid_slice_split_name(
                    t->sval,
                    bare, (int) sizeof(bare),
                    btext,
                    (int) sizeof(btext));
            if (has_brk)
            {
                imageID srcid = image_ID(
                    bare,
                    data.core.image,
                    data.core.NB_MAX_IMAGE);
                if (srcid == -1)
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
                if (slc.error)
                {
                    parse_errmsg(
                        slc.errmsg);
                    return mk_double(0);
                }
                uint32_t outsz[3] = {0};
                int snax =
                    srcim->md[0].naxis;
                uint32_t ssz[3];
                for (int a = 0;
                     a < snax && a < 3;
                     a++)
                {
                    ssz[a] =
                        srcim->md[0].size[a];
                }
                if (imgid_slice_output_size(
                        &slc, snax,
                        ssz, outsz) != 0)
                {
                    parse_errmsg(
                        "Bad slice dims");
                    return mk_double(0);
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
                const char *tmpn =
                    alloc_tmpname();
                imageID tid =
                    create_image_ID(
                        tmpn,
                        onax,
                        outsz,
                        srcim->md[0].datatype,
                        0, 10, 0, NULL);
                if (tid != -1)
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
    if (t->type == TOK_COMMAND)
    {
        advance_func();
        // This path is only for cli_parse, not cli_calc_eval_line
        if (parse_mode == 0) { // If not in eval_line context
            data.cmdargtoken[data.cmdNBarg].type =
                CMDARGTOKEN_TYPE_COMMAND;
            if (data.core.Debug > 0)
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
static val_t parse_expr(int min_prec)
{
    cli_token *(*cur_func)(void);
    cli_token *(*advance_func)(void);

    if (parse_error || eval_error) { // Check both error flags
        return mk_long(0);
    }

    // Determine which token stream to use based on which error flag is active
    if (parse_mode == 0) { // cli_parse is active
        cur_func = cur_parse;
        advance_func = advance_parse;
    } else { // cli_calc_eval_line is active
        cur_func = cur_eval;
        advance_func = advance_eval;
    }


    val_t left = parse_primary();
    if (parse_error || eval_error)
    {
        return left;
    }

    for (;;)
    {
        cli_token_type op = cur_func()->type;
        int prec = get_prec(op);

        if (prec < min_prec)
        {
            break;
        }

        advance_func();

        int next_min = is_right_assoc(op)
                       ? prec
                       : prec + 1;
        val_t right = parse_expr(next_min);
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
    parse_mode = 0;
    parse_error = 0; // Reset parse_error for cli_parse
    eval_error = 0; // Do not trip error abortion logic


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
        /* Fallback: if it's not a valid expression, treat it as a raw string */
        data.parseerror = 0;
        parse_error = 0;
        
        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_RAWSTRING;
        snprintf(data.cmdargtoken[data.cmdNBarg].val.string,
                 STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", input);
        
        /* Remove trailing newline if `input` had one (which it does via snprintf earlier) */
        size_t len = strlen(data.cmdargtoken[data.cmdNBarg].val.string);
        if (len > 0 && data.cmdargtoken[data.cmdNBarg].val.string[len - 1] == '\n')
        {
            data.cmdargtoken[data.cmdNBarg].val.string[len - 1] = '\0';
        }
        return;
    }

    /* consume the trailing newline if present */
    if (cur_parse()->type == TOK_NEWLINE)
    {
        advance_parse();
    }

    /* store result in data.cmdargtoken */
    if (result.type == VAL_DOUBLE)
    {
        if (data.core.Debug > 0)
        {
            printf("\t double: %.10g\n",
                   result.dval);
        }
        data.cmdargtoken[data.cmdNBarg].type =
            CMDARGTOKEN_TYPE_FLOAT;
        data.cmdargtoken[data.cmdNBarg]
            .val.numf = result.dval;
    }
    else if (result.type == VAL_LONG)
    {
        if (data.core.Debug > 0)
        {
            printf("\t long:   %ld\n",
                   result.lval);
        }
        data.cmdargtoken[data.cmdNBarg].type =
            CMDARGTOKEN_TYPE_LONG;
        data.cmdargtoken[data.cmdNBarg].val
            .numl = result.lval;
    }
    else if (result.type == VAL_STRING)
    {
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
        // The type for STRING was already set by AST logic
    }
}

/**
 * @brief Evaluate an entire line as a math expression.
 *
 * If the line successfully evaluates as a generic math
 * expression, print the result and return 1.
 * If it contains syntax errors, return 0.
 */
int cli_calc_eval_line(const char *input)
{
    parse_mode = 1;
    /* Use tokenizer but ensure it handles everything */
    char tbuf[8192];
    snprintf(tbuf, 8192, "%s\n", input);

    eval_ntok = cli_tokenize(tbuf, eval_tokens, CLI_CALC_MAX_TOKENS);

    parse_error = 0;
    eval_error  = 0;
    eval_pos    = 0;

    if (eval_ntok <= 0 || cur_eval()->type == TOK_NEWLINE || cur_eval()->type == TOK_EOF)
    {
        return 0; // empty expression
    }

    val_t result = parse_expr(0);

    /* if there is any parse error or trailing garbage */
    if (parse_error || eval_error || (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE))
    {
        return 0; /* not a pure math expression */
    }

    /* Success! Print output and return 1 */
    if (result.type == VAL_LONG)
    {
        printf("    long: %ld\n", result.lval);
    }
    else if (result.type == VAL_DOUBLE)
    {
        printf("    double: %g\n", result.dval);
    }
    else if (result.type == VAL_STRING)
    {
        /* Just string returned, maybe "ls" etc */
        /* To prevent capturing generic shell commands that happen to be single string tokens */
        if (eval_ntok > 2)
        {   /* it took operators to combine them into string? Rare... */
            printf("    string: %s\n", result.sval);
        }
        else
        {
            return 0; /* It was probably a generic 1-word shell command like `ls` */
        }
    }
    else if (result.type == VAL_GENERIC)
    {
         /* generic usually means function evaluated but no specific printable value returned */
         //printf("    generic\n");
    }
    
    return 1;
}

/**
 * @brief Evaluate a string as a pure math expression, returning the result value silently.
 * 
 * @param input     Expression string
 * @param out_type  Pointer to receive the parsed type (1=long, 2=double)
 * @param out_lval  Pointer to receive long value
 * @param out_dval  Pointer to receive double value
 * @return 1 on success (pure math), 0 on failure/string
 */
int cli_calc_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval)
{
    parse_mode = 1;
    char tbuf[8192];
    snprintf(tbuf, 8192, "%s\n", input);

    eval_ntok = cli_tokenize(tbuf, eval_tokens, CLI_CALC_MAX_TOKENS);
    parse_error = 0;
    eval_error  = 0;
    eval_pos    = 0;

    if (eval_ntok <= 0 || cur_eval()->type == TOK_NEWLINE || cur_eval()->type == TOK_EOF)
    {
        return 0; // empty expression
    }

    val_t result = parse_expr(0);

    /* if there is any parse error or trailing garbage */
    if (parse_error || eval_error || (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE))
    {
        return 0; /* not a pure math expression */
    }

    /* Success! If it's a string, it's not pure math unless it was evaluated from an operator */
    if (result.type == VAL_STRING && eval_ntok <= 2)
    {
        return 0;
    }

    /* Output values */
    if (result.type == VAL_LONG) {
        if (out_type) *out_type = 1;
        if (out_lval) *out_lval = result.lval;
    } else if (result.type == VAL_DOUBLE) {
        if (out_type) *out_type = 2;
        if (out_dval) *out_dval = result.dval;
    } else {
        return 0; // Not a numeric result
    }

    return 1;
}

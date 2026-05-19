/**
 * @file cli_calc_parser.c
 * @brief Hand-written Pratt parser for CLI expressions
 *
 * Architecture Overview:
 * The Pratt (precedence-climbing) parser replaces legacy bison-generated parsers
 * for expression evaluation. It parses operator precedence natively supporting 
 * +, -, *, /, ^, % arithmetic, logical/relational operators, and dynamic variable
 * assignments. Supports `long`, `double`, and `string` (image name) evaluation.
 * Evaluates expressions immediately, supporting math on images as well
 * as per-pixel masking. Populates the global execution context `data.cmdargtoken`.
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/stream_slice.h"

#include "cli_calc_tokenizer.h"
#include "cli_calc_parser.h"
#include "CLIcore_script.h"
#include "cli_calc_internal.h"
/* --------------------------------------------------------
 * Parser state
 * -------------------------------------------------------- */

/** token stream and current index for cli_parse */
static cli_token  parse_tokens[CLI_CALC_MAX_TOKENS];
static int        parse_pos;
static int        parse_ntok;
int               parse_error;

/** token stream and current index for cli_calc_eval_line */
static cli_token  eval_tokens[CLI_CALC_MAX_TOKENS];
static int        eval_pos;
static int        eval_ntok;
int               eval_error;


static char calctmpimname[200];

/* --------------------------------------------------------
 * Token stream helpers
 * -------------------------------------------------------- */

int parse_mode = 0; // 0 for cli_parse, 1 for cli_calc_eval_line

// Helper functions for cli_parse
cli_token *cur_parse(void)
{
    return &parse_tokens[parse_pos];
}

cli_token *advance_parse(void)
{
    cli_token *t = &parse_tokens[parse_pos];
    if (parse_pos < parse_ntok)
    {
        parse_pos++;
    }
    return t;
}

// Helper functions for cli_calc_eval_line
cli_token *cur_eval(void)
{
    return &eval_tokens[eval_pos];
}

cli_token *advance_eval(void)
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
void parse_errmsg(const char *msg)
{
    if ((parse_mode == 1 || data.core.Debug > 0) && dcquiet == 0)
    {
        PRINT_ERROR("   [CALC_PARSER_ERROR] %s", msg);
    }
    data.parseerror = 1;
    parse_error = 1;
    if (parse_mode == 1) {
        eval_error = 1;
    }
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
double to_double(val_t v)
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
val_t mk_long(long v)
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
val_t mk_double(double v)
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
val_t mk_string(const char *s)
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
int check_image(const char *name)
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
const char *alloc_tmpname(void)
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

    /* Check if this was a top-level assignment for output formatting */
    int is_assignment = 0;
    const char *assign_var_name = NULL;
    if (eval_ntok >= 3)
    {
        if ((eval_tokens[0].type == TOK_VAR || eval_tokens[0].type == TOK_NVAR || eval_tokens[0].type == TOK_IMAGE) &&
            (eval_tokens[1].type == TOK_EQUAL || eval_tokens[1].type == TOK_OP_PLUS_EQ || eval_tokens[1].type == TOK_OP_MINUS_EQ || eval_tokens[1].type == TOK_OP_STAR_EQ || eval_tokens[1].type == TOK_OP_SLASH_EQ))
        {
            is_assignment = 1;
            assign_var_name = eval_tokens[0].sval;
        }
    }

    /* Success! Print output and return 1 */
    if (result.type == VAL_LONG)
    {
        if (is_assignment)
        {
            printf("    %s long: %ld\n", assign_var_name, result.lval);
        }
        else
        {
            printf("    long: %ld\n", result.lval);
        }
    }
    else if (result.type == VAL_DOUBLE)
    {
        if (is_assignment)
        {
            printf("    %s double: %.*g\n",
                   assign_var_name,
                   cli_float_digits,
                   result.dval);
        }
        else
        {
            printf("    double: %.*g\n",
                   cli_float_digits,
                   result.dval);
        }
    }
    else if (result.type == VAL_STRING)
    {
        /* Just string returned, maybe "ls" etc */
        /* To prevent capturing generic shell commands that happen to be single string tokens */
        if (eval_ntok > 2)
        {   /* it took operators to combine them into string? Rare... */
            if (is_assignment)
            {
                printf("    %s string: %s\n", assign_var_name, result.sval);
            }
            else
            {
                printf("    string: %s\n", result.sval);
            }
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

    /* Clean up temporary images created
     * during expression evaluation (e.g.
     * intermediate arithmetic results and
     * slice materialization buffers). */
    {
        char tmpn[200];
        for (long i = 0;
             i < data.calctmp_imindex;
             i++)
        {
            snprintf(tmpn, sizeof(tmpn),
                     "_tmpcalc%ld", i);
            imageID tid = image_ID(
                tmpn,
                data.core.image,
                data.core.NB_MAX_IMAGE);
            if (tid != -1)
            {
                delete_image_ID(
                    tmpn,
                    DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        data.calctmp_imindex = 0;
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
int cli_calc_eval_math_to_val(
    const char *input,
    int *out_type,
    long *out_lval,
    double *out_dval)
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

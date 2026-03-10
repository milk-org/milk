/**
 * @file cli_calc_tokenizer.c
 * @brief Hand-written lexer for CLI expression parsing
 *
 * Replaces the flex-generated lexer (calc_flex.l).
 * Scans input text and produces an array of tokens:
 * integers, floats, operators, function names, and
 * identifiers classified as variable/image/command.
 */

#include <ctype.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"

#include "cli_calc_tokenizer.h"

/**
 * @brief Built-in function table entry
 *
 * Maps a function name prefix (including '(') to
 * its C function pointer and token type.
 */
typedef struct
{
    const char     *name;
    double        (*fptr)();
    cli_token_type  ttype;
} builtin_func;

/**
 * @brief Table of built-in functions recognized by
 *        the expression parser
 *
 * Each entry maps a string like "sin(" to the
 * corresponding C math function and a token type
 * indicating the function signature.
 */
static const builtin_func builtins[] =
{
    /* double -> double */
    {"sin(",   (double(*)()) sin,   TOK_FUNC_D_D},
    {"cos(",   (double(*)()) cos,   TOK_FUNC_D_D},
    {"exp(",   (double(*)()) exp,   TOK_FUNC_D_D},
    {"ln(",    (double(*)()) log,   TOK_FUNC_D_D},
    {"log(",   (double(*)()) log10, TOK_FUNC_D_D},
    {"tan(",   (double(*)()) tan,   TOK_FUNC_D_D},
    {"atan(",  (double(*)()) atan,  TOK_FUNC_D_D},
    {"sqrt(",  (double(*)()) sqrt,  TOK_FUNC_D_D},
    {"cbrt(",  (double(*)()) cbrt,  TOK_FUNC_D_D},
    {"ceil(",  (double(*)()) ceil,  TOK_FUNC_D_D},
    {"floor(", (double(*)()) floor, TOK_FUNC_D_D},
    {"asin(",  (double(*)()) asin,  TOK_FUNC_D_D},
    {"acos(",  (double(*)()) acos,  TOK_FUNC_D_D},
    {"posi(",  (double(*)()) Ppositive, TOK_FUNC_D_D},

    /* double, double -> double */
    {"atan2(", (double(*)()) atan2, TOK_FUNC_DD_D},

    /* double, double, double -> double */
    {"trunc(", (double(*)()) Ptrunc, TOK_FUNC_DDD_D},

    /* image -> double */
    {"itot(",  (double(*)()) arith_image_total,
        TOK_FUNC_IM_D},
    {"imin(",  (double(*)()) arith_image_min,
        TOK_FUNC_IM_D},
    {"imax(",  (double(*)()) arith_image_max,
        TOK_FUNC_IM_D},
    {"imean(", (double(*)()) arith_image_mean,
        TOK_FUNC_IM_D},

    /* image, double -> double */
    {"perc(",  (double(*)()) arith_image_percentile,
        TOK_FUNC_IMD_D},

    {NULL, NULL, TOK_EOF}
};

/**
 * @brief Check if character is valid in an identifier
 *
 * Identifiers may contain alphanumerics and the
 * characters _ . $ : ?
 */
static int is_ident_char(int c)
{
    return isalnum(c)
        || c == '_'
        || c == '.'
        || c == '$'
        || c == ':'
        || c == '?';
}

/**
 * @brief Check if character can start an identifier
 */
static int is_ident_start(int c)
{
    return isalpha(c) || c == '?';
}

/**
 * @brief Tokenize a CLI input string
 *
 * Scans the null-terminated input and fills the
 * tokens array.  Returns the number of tokens, or
 * -1 on error.
 *
 * Classification of string identifiers:
 *  - If it matches a known variable -> TOK_VAR
 *  - If it matches a known image   -> TOK_IMAGE
 *  - If cmdNBarg==0 and it matches a
 *    registered command             -> TOK_COMMAND
 *  - Otherwise                      -> TOK_NVAR
 */
int cli_tokenize(
    const char *input,
    cli_token  *tokens,
    int         max_tok
)
{
    const char *p  = input;
    int         nt = 0;

    while (*p != '\0' && nt < max_tok - 1)
    {
        /* skip whitespace (not newline) */
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }

        if (*p == '\0')
        {
            break;
        }

        /* newline */
        if (*p == '\n')
        {
            tokens[nt].type = TOK_NEWLINE;
            nt++;
            p++;
            continue;
        }

        /* operators and punctuation */
        {
            cli_token_type optype = TOK_EOF;

            switch (*p)
            {
                case '+':
                    optype = TOK_OP_PLUS;
                    break;
                case '-':
                    optype = TOK_OP_MINUS;
                    break;
                case '*':
                    optype = TOK_OP_STAR;
                    break;
                case '/':
                    optype = TOK_OP_SLASH;
                    break;
                case '^':
                    optype = TOK_OP_CARET;
                    break;
                case '(':
                    optype = TOK_LPAREN;
                    break;
                case ')':
                    optype = TOK_RPAREN;
                    break;
                case ',':
                    optype = TOK_COMMA;
                    break;
                case '=':
                    optype = TOK_EQUAL;
                    break;
                default:
                    break;
            }

            if (optype != TOK_EOF)
            {
                if (data.core.Debug > 0)
                {
                    printf(
                        "DEBUG: TOKENIZER: \"%c\" "
                        "is an operator/punct\n",
                        *p
                    );
                }
                tokens[nt].type = optype;
                nt++;
                p++;
                continue;
            }
        }

        /*
         * Number: integer or floating-point
         *
         * We try to parse a double first to detect
         * decimal points and exponents.  If none are
         * found, treat as long.
         */
        if (isdigit((unsigned char) *p)
            || (*p == '.' && isdigit(
                (unsigned char) *(p + 1))))
        {
            const char *start = p;
            int         is_float = 0;

            /* integer part */
            while (isdigit((unsigned char) *p))
            {
                p++;
            }

            /* fractional part */
            if (*p == '.')
            {
                is_float = 1;
                p++;
                while (isdigit((unsigned char) *p))
                {
                    p++;
                }
            }

            /* exponent */
            if (*p == 'e' || *p == 'E')
            {
                is_float = 1;
                p++;
                if (*p == '+' || *p == '-')
                {
                    p++;
                }
                while (isdigit((unsigned char) *p))
                {
                    p++;
                }
            }

            if (is_float)
            {
                tokens[nt].type  = TOK_DOUBLE;
                tokens[nt].val_d = strtod(start, NULL);
                if (data.core.Debug > 0)
                {
                    printf(
                        "DEBUG: TOKENIZER: \"%.*s\""
                        " is a float -> %f\n",
                        (int)(p - start),
                        start,
                        tokens[nt].val_d
                    );
                }
            }
            else
            {
                tokens[nt].type  = TOK_LONG;
                tokens[nt].val_l = strtol(start,
                                          NULL, 10);
                if (data.core.Debug > 0)
                {
                    printf(
                        "DEBUG: TOKENIZER: \"%.*s\""
                        " is a long -> %ld\n",
                        (int)(p - start),
                        start,
                        tokens[nt].val_l
                    );
                }
            }
            nt++;
            continue;
        }

        /*
         * Identifier or built-in function
         *
         * First check if the text matches a built-in
         * function name (e.g. "sin(").  If not, read
         * the whole identifier and classify it.
         */
        if (is_ident_start((unsigned char) *p))
        {
            /* try built-in function match */
            {
                int matched = 0;
                for (int i = 0;
                     builtins[i].name != NULL;
                     i++)
                {
                    size_t blen =
                        strlen(builtins[i].name);

                    if (strncmp(p,
                                builtins[i].name,
                                blen) == 0)
                    {
                        tokens[nt].type =
                            builtins[i].ttype;
                        tokens[nt].fnctptr =
                            builtins[i].fptr;
                        if (data.core.Debug > 0)
                        {
                            printf(
                                "DEBUG: TOKENIZER:"
                                " \"%.*s\" is a "
                                "function\n",
                                (int) blen, p
                            );
                        }
                        nt++;
                        p += blen;
                        matched = 1;
                        break;
                    }
                }
                if (matched)
                {
                    continue;
                }
            }

            /* read identifier string */
            const char *start = p;
            while (is_ident_char(
                (unsigned char) *p))
            {
                p++;
            }

            size_t slen = (size_t)(p - start);
            if (slen >= CLI_CALC_TOKEN_MAXLEN)
            {
                slen = CLI_CALC_TOKEN_MAXLEN - 1;
            }
            memcpy(tokens[nt].sval, start, slen);
            tokens[nt].sval[slen] = '\0';

            if (data.core.Debug > 0)
            {
                printf(
                    "Found string %s\n",
                    tokens[nt].sval
                );
            }

            /* classify the identifier */
            const char *s = tokens[nt].sval;

            if (variable_ID(s) != -1)
            {
                tokens[nt].type = TOK_VAR;
                if (data.core.Debug > 0)
                {
                    printf(
                        "DEBUG: TOKENIZER: \"%s\""
                        " IS A VARIABLE\n", s
                    );
                }
            }
            else if (image_ID(
                s,
                data.core.image,
                data.core.NB_MAX_IMAGE) != -1)
            {
                tokens[nt].type = TOK_IMAGE;
                if (data.core.Debug > 0)
                {
                    printf(
                        "DEBUG: TOKENIZER: \"%s\""
                        " IS AN IMAGE\n", s
                    );
                }
            }
            else if (data.cmdNBarg == 0)
            {
                /* first argument: check commands */
                int found_cmd = 0;
                data.cmdindex = -1;

                for (long i = 0;
                     i < (long) data.NBcmd;
                     i++)
                {
                    size_t cmdlen =
                        strlen(data.cmd[i].key);

                    if (strncmp(
                            s,
                            data.cmd[i].key,
                            cmdlen) == 0
                        && (s[cmdlen] == '\0'
                            || s[cmdlen] == ':'
                            || s[cmdlen] == ' '))
                    {
                        data.cmdindex = i;
                        tokens[nt].type =
                            TOK_COMMAND;
                        found_cmd = 1;
                        if (data.core.Debug > 0)
                        {
                            printf(
                                "DEBUG: TOKENIZER:"
                                " \"%s\" IS A "
                                "COMMAND (cmd"
                                " %ld)\n",
                                s, i
                            );
                        }
                        break;
                    }
                }

                if (!found_cmd)
                {
                    tokens[nt].type = TOK_NVAR;
                    if (data.core.Debug > 0)
                    {
                        printf(
                            "DEBUG: TOKENIZER:"
                            " \"%s\" IS A NEW"
                            " VARIABLE\n", s
                        );
                    }
                }
            }
            else
            {
                tokens[nt].type = TOK_NVAR;
                if (data.core.Debug > 0)
                {
                    printf(
                        "DEBUG: TOKENIZER: \"%s\""
                        " IS A NEW VARIABLE\n", s
                    );
                }
            }

            nt++;
            continue;
        }

        /* unrecognized character */
        printf(
            "DEBUG: TOKENIZER: unrecognised char"
            " [hex %02X] length 1\n",
            (unsigned char) *p
        );
        p++;
    }

    /* sentinel */
    tokens[nt].type = TOK_EOF;
    return nt;
}

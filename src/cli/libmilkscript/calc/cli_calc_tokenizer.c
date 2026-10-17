// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#include <sys/stat.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include "cli_calc_internal.h"
#include "cli_calc_scalar.h"
#include "cli_calc_tokenizer.h"

/**
 * @brief Built-in function table entry
 *
 * Maps a function name prefix (including '(') to
 * its C function pointer and token type.
 */
typedef struct
{
    const char *name;
    double (*fptr)();
    cli_token_type ttype;
} builtin_func;

/**
 * @brief Table of built-in functions recognized by
 *        the expression parser
 *
 * Each entry maps a string like "sin(" to the
 * corresponding C math function and a token type
 * indicating the function signature.
 */
static const builtin_func builtins[] = {
    /* the scalar math functions (sin, min, trunc, ...) are cli_calc_scalar_fn()'s */

    /* image -> double */
    { "itot(", (double (*)()) arith_image_total, TOK_FUNC_IM_D },
    { "imin(", (double (*)()) arith_image_min, TOK_FUNC_IM_D },
    { "imax(", (double (*)()) arith_image_max, TOK_FUNC_IM_D },
    { "imean(", (double (*)()) arith_image_mean, TOK_FUNC_IM_D },
    { "istd(", (double (*)()) arith_image_rms, TOK_FUNC_IM_D },
    { "irms(", (double (*)()) arith_image_rms, TOK_FUNC_IM_D }, /* old name of istd */
    { "ivar(", (double (*)()) arith_image_var, TOK_FUNC_IM_D },
    { "imedian(", (double (*)()) arith_image_median, TOK_FUNC_IM_D },

    /* image -> long */
    { "icount(", (double (*)()) arith_image_count, TOK_FUNC_IM_L },
    { "iargmax(", (double (*)()) arith_image_argmax, TOK_FUNC_IM_L },
    { "iargmin(", (double (*)()) arith_image_argmin, TOK_FUNC_IM_L },

    /* image, double -> double */
    { "perc(", (double (*)()) arith_image_percentile, TOK_FUNC_IMD_D },

    /* special -> parser handles manually */
    { "where(", NULL, TOK_FUNC_WHERE },
    { "dot(", (double (*)()) arith_image_dot, TOK_FUNC_IMIM_D },
    { "norm(", (double (*)()) arith_image_norm, TOK_FUNC_IM_D },
    { "matmul(", NULL, TOK_FUNC_IMIM_IM },

    /* complex images */
    { "real(", NULL, TOK_FUNC_CPLX },
    { "imag(", NULL, TOK_FUNC_CPLX },
    { "arg(", NULL, TOK_FUNC_CPLX },
    { "conj(", NULL, TOK_FUNC_CPLX },
    { "cplx(", NULL, TOK_FUNC_CPLX },

    /* string -> double */
    { "strlen(", NULL, TOK_FUNC_S_D },

    /* string -> string */
    { "toupper(", NULL, TOK_FUNC_S_S },
    { "tolower(", NULL, TOK_FUNC_S_S },

    /* string,int,int -> string */
    { "substr(", NULL, TOK_FUNC_SDD_S },

    /* string,string,string -> string */
    { "replace(", NULL, TOK_FUNC_SSS_S },

    /* double -> string (format) */
    { "hex(", NULL, TOK_FUNC_D_S },
    { "oct(", NULL, TOK_FUNC_D_S },
    { "bin(", NULL, TOK_FUNC_D_S },

    { NULL, NULL, TOK_EOF }
};

/**
 * @brief Check if character is valid in an identifier
 *
 * Identifiers may contain alphanumerics and _ . $ ; command-argument parsing (parse_mode 0)
 * also keeps ':' and '?' inside words. In expressions they are the ternary operator, except
 * for the ':' of an S:/L: location prefix (see ident_len()).
 */
static inline int is_ident_char(int c)
{
    return isalnum(c) || c == '_' || c == '.' || c == '$' ||
           (parse_mode == 0 && (c == ':' || c == '?'));
}

/** ident_len() - length of the identifier starting at @start. */
static size_t ident_len(const char *start)
{
    const char *p = start;
    while (is_ident_char((unsigned char) *p) ||
           (*p == ':' && p - start == 1 && (start[0] == 'S' || start[0] == 'L')))
    {
        p++;
    }
    return (size_t) (p - start);
}

/**
 * @brief Check if character can start an identifier
 */
static inline int is_ident_start(int c)
{
    return isalpha(c) || c == '_' || c == '?' || c == '.';
}

/**
 * @brief Check if an environment variable exists and holds a pure numeric value.
 */
static int is_numeric_env_var(const char *s)
{
    const char *ev = getenv(s);
    if (ev == NULL || ev[0] == '\0')
    {
        return 0;
    }
    char *endp = NULL;
    (void) strtod(ev, &endp);
    return (endp != ev && *endp == '\0');
}

/**
 * @brief Resolve an image by name, auto-connecting from /dev/shm/<name>.im.shm if allowed.
 */
static imageID resolve_or_autoconnect_image(const char *name, int allow_autoconnect)
{
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    if (sp.error)
    {
        return -1;
    }
    const char *bare = (sp.name != NULL && sp.name[0] != '\0') ? sp.name : name;

    imageID id = image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE);
    if (id != -1 && data.core.image[id].md != NULL && data.core.image[id].md->shared == 1)
    {
        /* a stream re-created by another process has a new file: map that one, not the old one */
        char        shmpath[512];
        struct stat st;
        if (ImageStreamIO_filename(shmpath, sizeof(shmpath), bare) == IMAGESTREAMIO_SUCCESS &&
            stat(shmpath, &st) == 0 && st.st_ino != data.core.image[id].md->inode)
        {
            IMAGE fresh;
            memset(&fresh, 0, sizeof(fresh));
            if (ImageStreamIO_read_sharedmem_image_toIMAGE(bare, &fresh) == IMAGESTREAMIO_SUCCESS)
            {
                ImageStreamIO_closeIm(&data.core.image[id]);
                memcpy(&data.core.image[id], &fresh, sizeof(IMAGE));
                data.core.image[id].used = 1;
            }
        }
    }
    if (id != -1 || !allow_autoconnect || sp.loc == 'L')
    {
        return id;
    }
    char shmpath[512];
    if (ImageStreamIO_filename(shmpath, sizeof(shmpath), bare) == IMAGESTREAMIO_SUCCESS &&
        access(shmpath, R_OK) == 0)
    {
        IMAGE tmp_im;
        memset(&tmp_im, 0, sizeof(tmp_im));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(bare, &tmp_im) == IMAGESTREAMIO_SUCCESS)
        {
            id = next_avail_image_ID(-1);
            if (id != -1)
            {
                memcpy(&data.core.image[id], &tmp_im, sizeof(IMAGE));
                data.core.image[id].used = 1;
            }
            else
            {
                ImageStreamIO_closeIm(&tmp_im);
            }
        }
    }
    return id;
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
int cli_tokenize(const char *input, cli_token *tokens, int max_tok)
{
    const char *p  = input;
    int         nt = 0;

    while (*p != '\0' && nt < max_tok - 1)
    {
        tokens[nt].literal = 0;
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

        /*
         * Check for registered CLI command when parsing the first argument
         * of a command line (data.cmdNBarg == 0 && parse_mode == 0 && nt == 0).
         * This allows commands containing hyphens (e.g. list-streams, list-fps)
         * to be recognized as a single command token rather than being split
         * by the arithmetic minus operator.
         */
        if (data.cmdNBarg == 0 && parse_mode == 0 && nt == 0)
        {
            long   best_cmd    = -1;
            size_t max_cmd_len = 0;

            for (long i = 0; i < (long) data.NBcmd; i++)
            {
                size_t cmdlen = strlen(data.cmd[i].key);
                if (strncmp(p, data.cmd[i].key, cmdlen) == 0 &&
                    (p[cmdlen] == '\0' || p[cmdlen] == ' ' || p[cmdlen] == '\t' ||
                     p[cmdlen] == '\n' || p[cmdlen] == ';'))
                {
                    /* Ensure not immediately followed by assignment operator '=' */
                    const char *after = p + cmdlen;
                    while (*after == ' ' || *after == '\t')
                    {
                        after++;
                    }
                    if (*after == '=' || (*after == '+' && *(after + 1) == '=') ||
                        (*after == '-' && *(after + 1) == '=') ||
                        (*after == '*' && *(after + 1) == '=') ||
                        (*after == '/' && *(after + 1) == '='))
                    {
                        continue;
                    }

                    if (cmdlen > max_cmd_len)
                    {
                        max_cmd_len = cmdlen;
                        best_cmd    = i;
                    }
                }
            }

            if (best_cmd >= 0)
            {
                data.cmdindex   = best_cmd;
                tokens[nt].type = TOK_COMMAND;
                strncpy(tokens[nt].sval, data.cmd[best_cmd].key, sizeof(tokens[nt].sval) - 1);
                tokens[nt].sval[sizeof(tokens[nt].sval) - 1] = '\0';
                if (data.core.Debug > 0)
                {
                    printf("DEBUG: TOKENIZER: \"%s\" IS A COMMAND (cmd %ld)\n", tokens[nt].sval,
                           best_cmd);
                }
                nt++;
                p += max_cmd_len;
                continue;
            }
            else
            {
                data.cmdindex = -1;
            }
        }


        /* operators and punctuation */
        {
            int            oplen  = 1;
            cli_token_type optype = cli_calc_scan_op(p, &oplen);
            if (optype != TOK_EOF)
            {
                if (data.core.Debug > 0)
                {
                    printf("DEBUG: TOKENIZER: \"%.*s\" "
                           "is an operator/punct\n",
                           oplen, p);
                }
                tokens[nt].type = optype;
                nt++;
                p += oplen;
                continue;
            }
        }

        /* Number: decimal (a '.' or an exponent makes it a double), 0x, 0o or 0b */
        {
            size_t nlen =
                cli_calc_scan_number(p, &tokens[nt].type, &tokens[nt].val_l, &tokens[nt].val_d);
            if (nlen > 0)
            {
                if (data.core.Debug > 0)
                {
                    printf("DEBUG: TOKENIZER: \"%.*s\" is a number\n", (int) nlen, p);
                }
                nt++;
                p += nlen;
                continue;
            }
        }

        /* Quoted string literal: "hello" */
        if (*p == '"')
        {
            p++;
            const char *start = p;
            while (*p != '"' && *p != '\0' && *p != '\n')
            {
                p++;
            }
            size_t slen = (size_t) (p - start);
            if (slen >= CLI_CALC_TOKEN_MAXLEN)
            {
                slen = CLI_CALC_TOKEN_MAXLEN - 1;
            }
            memcpy(tokens[nt].sval, start, slen);
            tokens[nt].sval[slen] = '\0';
            tokens[nt].type       = TOK_NVAR;
            tokens[nt].literal    = 1;
            if (*p == '"')
            {
                p++;
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
                int                         matched = 0;
                const cli_calc_scalar_fn_t *sf      = cli_calc_scalar_fn(p);
                if (sf != NULL)
                {
                    tokens[nt].type    = (sf->nargs == 1)   ? TOK_FUNC_D_D
                                         : (sf->nargs == 2) ? TOK_FUNC_DD_D
                                                            : TOK_FUNC_DDD_D;
                    tokens[nt].fnctptr = sf->fn;
                    snprintf(tokens[nt].sval, sizeof(tokens[nt].sval), "%s", sf->name);
                    nt++;
                    p += strlen(sf->name);
                    continue;
                }
                for (int i = 0; builtins[i].name != NULL; i++)
                {
                    size_t blen = strlen(builtins[i].name);

                    if (strncmp(p, builtins[i].name, blen) == 0)
                    {
                        tokens[nt].type    = builtins[i].ttype;
                        tokens[nt].fnctptr = builtins[i].fptr;
                        strncpy(tokens[nt].sval, builtins[i].name, sizeof(tokens[nt].sval) - 1);
                        tokens[nt].sval[sizeof(tokens[nt].sval) - 1] = '\0';
                        if (data.core.Debug > 0)
                        {
                            printf("DEBUG: TOKENIZER:"
                                   " \"%.*s\" is a "
                                   "function\n",
                                   (int) blen, p);
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
            p += ident_len(p);

            /* Consume trailing bracket block
             * for stream slicing syntax
             * e.g. wfs[0:63,10:73] */
            if (*p == '[')
            {
                int bdepth = 0;
                do
                {
                    if (*p == '[')
                    {
                        bdepth++;
                    }
                    else if (*p == ']')
                    {
                        bdepth--;
                    }
                    p++;
                } while (bdepth > 0 && *p != '\0');
            }

            /* Detect unknown function call:
             * identifier immediately followed
             * by '(' means the user tried to
             * call an unrecognized function. */
            if (*p == '(')
            {
                size_t nlen = (size_t) (p - start);
                if (nlen >= CLI_CALC_TOKEN_MAXLEN)
                {
                    nlen = CLI_CALC_TOKEN_MAXLEN - 1;
                }
                char fname[CLI_CALC_TOKEN_MAXLEN];
                memcpy(fname, start, nlen);
                fname[nlen] = '\0';
                fprintf(stderr,
                        "ERROR: unknown "
                        "function "
                        "'%s'\n",
                        fname);
                return -1;
            }

            size_t slen = (size_t) (p - start);
            if (slen >= CLI_CALC_TOKEN_MAXLEN)
            {
                slen = CLI_CALC_TOKEN_MAXLEN - 1;
            }
            memcpy(tokens[nt].sval, start, slen);
            tokens[nt].sval[slen] = '\0';

            if (data.core.Debug > 0)
            {
                printf("Found string %s\n", tokens[nt].sval);
            }

            /* classify the identifier */
            const char *s = tokens[nt].sval;

            /* Determine whether auto-connecting /dev/shm/<name>.im.shm is safe here */
            int allow_autoconnect = 0;
            if (strchr(s, '[') != NULL || nt > 0 || parse_mode == 2)
            {
                allow_autoconnect = 1;
            }
            else if (parse_mode != 0 && nt == 0)
            {
                const char *after = p;
                while (*after == ' ' || *after == '\t')
                {
                    after++;
                }
                /* Allow auto-connect at nt==0 if followed by an operator (not a CLI flag like -la) */
                if (*after == '=' || *after == '+' || *after == '*' || *after == '/' ||
                    *after == '^' || *after == '%' || *after == '<' || *after == '>' ||
                    *after == '!' || *after == '&' || *after == '|' || *after == '?' ||
                    (*after == '-' && !isalpha((unsigned char) *(after + 1))))
                {
                    allow_autoconnect = 1;
                }
            }

            if (variable_ID(s) != -1)
            {
                tokens[nt].type = TOK_VAR;
                if (data.core.Debug > 0)
                {
                    printf("DEBUG: TOKENIZER: \"%s\""
                           " IS A VARIABLE\n",
                           s);
                }
            }
            else if (strchr(s, '[') == NULL &&
                     resolve_or_autoconnect_image(s, allow_autoconnect) != -1)
            {
                tokens[nt].type          = TOK_IMAGE;
                FPS_STREAMNAME_PARSED sp = fps_streamname_parse(s);
                if (!sp.error && sp.name != NULL && sp.name[0] != '\0' && strcmp(sp.name, s) != 0)
                {
                    snprintf(tokens[nt].sval, CLI_CALC_TOKEN_MAXLEN, "%s", sp.name);
                }
                if (data.core.Debug > 0)
                {
                    printf("DEBUG: TOKENIZER: \"%s\""
                           " IS AN IMAGE\n",
                           tokens[nt].sval);
                }
            }
            /* Bracket token: try bare name
             * for sliced stream references */
            else if (strchr(s, '[') != NULL)
            {
                char        bare[CLI_CALC_TOKEN_MAXLEN];
                const char *bk = strchr(s, '[');
                size_t      bn = (size_t) (bk - s);
                if (bn > 0 && bn < CLI_CALC_TOKEN_MAXLEN)
                {
                    memcpy(bare, s, bn);
                    bare[bn] = '\0';
                    if (resolve_or_autoconnect_image(bare, 1) != -1)
                    {
                        tokens[nt].type          = TOK_IMAGE;
                        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(bare);
                        if (!sp.error && sp.name != NULL && sp.name[0] != '\0' &&
                            strcmp(sp.name, bare) != 0)
                        {
                            char slice_tail[CLI_CALC_TOKEN_MAXLEN];
                            snprintf(slice_tail, sizeof(slice_tail), "%s", bk);
                            snprintf(tokens[nt].sval, CLI_CALC_TOKEN_MAXLEN, "%s%s", sp.name,
                                     slice_tail);
                        }
                    }
                    else
                    {
                        tokens[nt].type = TOK_NVAR;
                    }
                }
                else
                {
                    tokens[nt].type = TOK_NVAR;
                }
            }
            else if (parse_mode != 0 &&
                     (cli_var_get(s) != NULL || (allow_autoconnect && is_numeric_env_var(s))))
            {
                tokens[nt].type = TOK_VAR;
            }
            else if (data.cmdNBarg == 0)
            {
                /* first argument: check commands */
                int found_cmd = 0;
                data.cmdindex = -1;

                for (long i = 0; i < (long) data.NBcmd; i++)
                {
                    size_t cmdlen = strlen(data.cmd[i].key);

                    if (strncmp(s, data.cmd[i].key, cmdlen) == 0 &&
                        (s[cmdlen] == '\0' || s[cmdlen] == ' '))
                    {
                        data.cmdindex   = i;
                        tokens[nt].type = TOK_COMMAND;
                        found_cmd       = 1;
                        if (data.core.Debug > 0)
                        {
                            printf("DEBUG: TOKENIZER:"
                                   " \"%s\" IS A "
                                   "COMMAND (cmd"
                                   " %ld)\n",
                                   s, i);
                        }
                        break;
                    }
                }

                if (!found_cmd)
                {
                    tokens[nt].type = TOK_NVAR;
                    if (data.core.Debug > 0)
                    {
                        printf("DEBUG: TOKENIZER:"
                               " \"%s\" IS A NEW"
                               " VARIABLE\n",
                               s);
                    }
                }
            }
            else
            {
                tokens[nt].type = TOK_NVAR;
                if (data.core.Debug > 0)
                {
                    printf("DEBUG: TOKENIZER: \"%s\""
                           " IS A NEW VARIABLE\n",
                           s);
                }
            }

            nt++;
            continue;
        }

        /* unrecognized character */
        if (data.core.Debug > 0)
        {
            printf("DEBUG: TOKENIZER: unrecognised char"
                   " [hex %02X] length 1\n",
                   (unsigned char) *p);
        }
        return -1;
    }

    /* token array full: expressions must not be silently truncated */
    while (*p == ' ' || *p == '\t' || *p == '\n')
    {
        p++;
    }
    if (*p != '\0' && parse_mode != 0)
    {
        fprintf(stderr, "ERROR: expression too long (max %d tokens)\n", max_tok - 1);
        return -1;
    }

    /* sentinel */
    tokens[nt].type = TOK_EOF;
    return nt;
}

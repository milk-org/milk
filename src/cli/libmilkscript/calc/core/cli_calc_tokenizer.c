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
#include "libmilkdata/milkdata.h"
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
    { "iamax(", (double (*)()) arith_image_iamax, TOK_FUNC_IM_L },

    /* image, double -> double */
    { "perc(", (double (*)()) arith_image_percentile, TOK_FUNC_IMD_D },

    /* image[, cx, cy] -> the pixel coordinates, centred on (xsize/2, ysize/2), or SIMD PRNG */
    { "ix(", NULL, TOK_FUNC_COORD },
    { "iy(", NULL, TOK_FUNC_COORD },
    { "ir(", NULL, TOK_FUNC_COORD },
    { "itheta(", NULL, TOK_FUNC_COORD },
    { "rand(", NULL, TOK_FUNC_COORD },
    { "randn(", NULL, TOK_FUNC_COORD },
    { "urand(", NULL, TOK_FUNC_COORD },
    { "grand(", NULL, TOK_FUNC_COORD },

    /* special -> parser handles manually */
    { "where(", NULL, TOK_FUNC_WHERE },
    { "dot(", (double (*)()) arith_image_dot, TOK_FUNC_IMIM_D },
    { "norm(", (double (*)()) arith_image_norm, TOK_FUNC_IM_D },
    { "nrm2(", (double (*)()) arith_image_norm, TOK_FUNC_IM_D },
    { "asum(", (double (*)()) arith_image_asum, TOK_FUNC_IM_D },
    { "trace(", (double (*)()) arith_image_trace, TOK_FUNC_IM_D },
    { "matmul(", (double (*)()) cli_calc_func_matmul, TOK_FUNC_IMIM_IM },
    { "gemm(", (double (*)()) cli_calc_func_gemm_op, TOK_FUNC_IMIM_IM },
    { "gemv(", (double (*)()) cli_calc_func_gemv_op, TOK_FUNC_IMIM_IM },
    { "ger(", (double (*)()) cli_calc_func_ger, TOK_FUNC_IMIM_IM },
    { "outer(", (double (*)()) cli_calc_func_outer, TOK_FUNC_IMIM_IM },
    { "syrk(", (double (*)()) cli_calc_func_syrk, TOK_FUNC_IMIM_IM },
    { "symv(", (double (*)()) cli_calc_func_symv, TOK_FUNC_IMIM_IM },
    { "axpy(", (double (*)()) cli_calc_func_axpy, TOK_FUNC_IMIM_IM },
    { "scal(", (double (*)()) cli_calc_func_scal, TOK_FUNC_IMIM_IM },
    { "trsv(", (double (*)()) cli_calc_func_trsv_op, TOK_FUNC_IMIM_IM },
    { "trsm(", (double (*)()) cli_calc_func_trsm_op, TOK_FUNC_IMIM_IM },
    { "gesv(", (double (*)()) cli_calc_func_gesv, TOK_FUNC_IMIM_IM },
    { "posv(", (double (*)()) cli_calc_func_posv, TOK_FUNC_IMIM_IM },
    { "gels(", (double (*)()) cli_calc_func_gels, TOK_FUNC_IMIM_IM },
    { "solve(", (double (*)()) cli_calc_func_solve, TOK_FUNC_IMIM_IM },
    { "inv(", (double (*)()) cli_calc_func_inv, TOK_FUNC_IMIM_IM },
    { "pinv(", (double (*)()) cli_calc_func_pinv, TOK_FUNC_IMIM_IM },
    { "eye(", (double (*)()) cli_calc_func_eye, TOK_FUNC_IMIM_IM },
    { "diag(", (double (*)()) cli_calc_func_diag, TOK_FUNC_IMIM_IM },
    { "transpose(", (double (*)()) cli_calc_func_transpose, TOK_FUNC_IMIM_IM },
    { "reshape(", (double (*)()) cli_calc_func_reshape, TOK_FUNC_IMIM_IM },
    { "vec(", (double (*)()) cli_calc_func_vec, TOK_FUNC_IMIM_IM },
    { "zeros(", (double (*)()) cli_calc_func_zeros, TOK_FUNC_IMIM_IM },

    /* complex images */
    { "real(", NULL, TOK_FUNC_CPLX },
    { "imag(", NULL, TOK_FUNC_CPLX },
    { "arg(", NULL, TOK_FUNC_CPLX },
    { "conj(", NULL, TOK_FUNC_CPLX },
    { "cplx(", NULL, TOK_FUNC_CPLX },
    { "expi(", NULL, TOK_FUNC_CPLX },

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

    /* stencil, lookup-table, spatial kernel, and multi-stat unpacking functions */
    { "shift(", NULL, TOK_FUNC_SHIFT },
    { "lut(", NULL, TOK_FUNC_LUT },
    { "medfilt3(", NULL, TOK_FUNC_MEDFILT },
    { "medfilt5(", NULL, TOK_FUNC_MEDFILT },
    { "medfilt7(", NULL, TOK_FUNC_MEDFILT },
    { "pixremap(", NULL, TOK_FUNC_PIXREMAP },
    { "pixunmap(", NULL, TOK_FUNC_PIXUNMAP },
    { "rprof(", NULL, TOK_FUNC_PIXUNMAP },
    { "gauss(", NULL, TOK_FUNC_GAUSS },
    { "cumsum(", NULL, TOK_FUNC_CUMSUM },
    { "interp2(", NULL, TOK_FUNC_INTERP2 },
    { "istats(", NULL, TOK_FUNC_IM_D },
    { "icentroid(", NULL, TOK_FUNC_IM_D },

    { NULL, NULL, TOK_EOF }
};

/** is_ident_char() - 1 if @c is valid inside an identifier. */
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

/** is_ident_start() - 1 if @c can start an identifier. */
static inline int is_ident_start(int c)
{
    return isalpha(c) || c == '_' || c == '?' || c == '.';
}

/** is_numeric_env_var() - 1 if environment variable @s exists and holds a pure numeric value. */
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
 * cli_calc_refresh_shared_image - Remap a shared image if its `/dev/shm` file has a new inode
 * @im: Image table slot to inspect and refresh
 */
void cli_calc_refresh_shared_image(IMAGE *im)
{
    if (im == NULL || !im->used || im->md == NULL || im->md[0].shared != 1)
    {
        return;
    }
    char        shmpath[512];
    struct stat st;
    if (ImageStreamIO_filename(shmpath, sizeof(shmpath), im->name) == IMAGESTREAMIO_SUCCESS &&
        stat(shmpath, &st) == 0 && st.st_ino != im->md[0].inode)
    {
        IMAGE fresh;
        memset(&fresh, 0, sizeof(fresh));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(im->name, &fresh) == IMAGESTREAMIO_SUCCESS)
        {
            ImageStreamIO_closeIm(im);
            memcpy(im, &fresh, sizeof(IMAGE));
            im->used = 1;
            milk_table_bump();
        }
    }
}

/**
 * resolve_or_autoconnect_image - Resolve an image by name, auto-connecting from `/dev/shm` if allowed
 * @name:              Image or stream name (with optional `@S:` / `@L:` prefix)
 * @allow_autoconnect: 1 to auto-connect from `/dev/shm/<name>.im.shm` when absent from table
 *
 * Return: Image ID in `data.core.image[]`, or -1 if not found
 */
static imageID resolve_or_autoconnect_image(const char *name, int allow_autoconnect)
{
    if (data.core.image == NULL || data.core.NB_MAX_IMAGE <= 0)
    {
        return -1;
    }
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    if (sp.error)
    {
        return -1;
    }
    const char *bare = (sp.name != NULL && sp.name[0] != '\0') ? sp.name : name;

    imageID id = image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE);
    if (id != -1)
    {
        cli_calc_refresh_shared_image(&data.core.image[id]);
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
 * match_first_arg_command - Match a registered CLI command at token 0 in command mode
 * @p:   Input pointer at start of line
 * @tok: Output token populated with `TOK_COMMAND` on match
 *
 * Return: Matched command length in bytes, or 0 if no command matched
 */
static size_t match_first_arg_command(const char *p, cli_token *tok)
{
    long   best_cmd    = -1;
    size_t max_cmd_len = 0;

    for (long i = 0; i < (long) data.NBcmd; i++)
    {
        size_t cmdlen = strlen(data.cmd[i].key);
        if (strncmp(p, data.cmd[i].key, cmdlen) == 0 &&
            (p[cmdlen] == '\0' || p[cmdlen] == ' ' || p[cmdlen] == '\t' || p[cmdlen] == '\n' ||
             p[cmdlen] == ';'))
        {
            const char *after = p + cmdlen;
            while (*after == ' ' || *after == '\t')
            {
                after++;
            }
            if (*after == '=' || (*after == '+' && *(after + 1) == '=') ||
                (*after == '-' && *(after + 1) == '=') || (*after == '*' && *(after + 1) == '=') ||
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
    if (best_cmd < 0)
    {
        data.cmdindex = -1;
        return 0;
    }
    data.cmdindex = best_cmd;
    tok->type     = TOK_COMMAND;
    snprintf(tok->sval, sizeof(tok->sval), "%s", data.cmd[best_cmd].key);
    if (data.core.Debug > 0)
    {
        printf("DEBUG: TOKENIZER: \"%s\" IS A COMMAND (cmd %ld)\n", tok->sval, best_cmd);
    }
    return max_cmd_len;
}

/**
 * match_builtin_func - Match a `<name>(` function call prefix of length @flen
 * @p:    Pointer to function name start
 * @flen: Length of `<name>(` including the opening `'('`
 * @tok:  Output token populated on match
 *
 * Return: 1 if matched, 0 if unrecognized
 */
static int match_builtin_func(const char *p, size_t flen, cli_token *tok)
{
    const cli_calc_scalar_fn_t *sf = cli_calc_scalar_fn(p);
    if (sf != NULL && strlen(sf->name) == flen)
    {
        tok->type    = (sf->nargs == 1)   ? TOK_FUNC_D_D
                       : (sf->nargs == 2) ? TOK_FUNC_DD_D
                                          : TOK_FUNC_DDD_D;
        tok->fnctptr = sf->fn;
        snprintf(tok->sval, sizeof(tok->sval), "%s", sf->name);
        return 1;
    }
    for (int i = 0; builtins[i].name != NULL; i++)
    {
        if (strncmp(p, builtins[i].name, flen) == 0 && builtins[i].name[flen] == '\0')
        {
            tok->type    = builtins[i].ttype;
            tok->fnctptr = builtins[i].fptr;
            snprintf(tok->sval, sizeof(tok->sval), "%s", builtins[i].name);
            if (data.core.Debug > 0)
            {
                printf("DEBUG: TOKENIZER: \"%.*s\" is a function\n", (int) flen, p);
            }
            return 1;
        }
    }
    int uidx = cli_calc_udef_find(p, flen - 1);
    if (uidx >= 0)
    {
        tok->type    = TOK_FUNC_UDEF;
        tok->val_l   = uidx;
        tok->fnctptr = NULL;
        snprintf(tok->sval, sizeof(tok->sval), "%.*s", (int) flen, p);
        return 1;
    }
    size_t fn_len = 0;
    while (fn_len + 1 < flen && p[fn_len] != ':')
    {
        fn_len++;
    }
    if (fn_len > 0 && fn_len < 64)
    {
        char fn_buf[64];
        memcpy(fn_buf, p, fn_len);
        fn_buf[fn_len] = '\0';
        int ext_idx    = cli_calc_ext_find(fn_buf);
        if (ext_idx >= 0)
        {
            tok->type    = TOK_FUNC_USER;
            tok->val_l   = ext_idx;
            tok->fnctptr = NULL;
            snprintf(tok->sval, sizeof(tok->sval), "%.*s", (int) flen, p);
            return 1;
        }
    }
    return 0;
}

/**
 * expand_string_var_alias - Expand string variable (`type == 3`) aliases in `tok->sval`
 * @tok: Token whose `sval` is inspected and rewritten in place if aliased
 *
 * Return: Resolved variable ID when `tok->sval` is not a string alias, or -2 if rewritten/sliced
 */
static long expand_string_var_alias(cli_token *tok)
{
    const char *s  = tok->sval;
    const char *bk = strchr(s, '[');

    if (bk != NULL)
    {
        char   bare[CLI_CALC_TOKEN_MAXLEN];
        size_t bn = (size_t) (bk - s);
        if (bn > 0 && bn < CLI_CALC_TOKEN_MAXLEN)
        {
            memcpy(bare, s, bn);
            bare[bn]  = '\0';
            long bvid = variable_ID(bare);
            if (bvid != -1 && data.core.variable[bvid].type == 3 &&
                data.core.variable[bvid].value.s[0] != '\0')
            {
                char tail[CLI_CALC_TOKEN_MAXLEN];
                snprintf(tail, sizeof(tail), "%s", bk);
                snprintf(tok->sval, CLI_CALC_TOKEN_MAXLEN, "%s%s", data.core.variable[bvid].value.s,
                         tail);
            }
        }
        return -2;
    }

    long svid = variable_ID(s);
    if (svid != -1 && data.core.variable[svid].type == 3 &&
        data.core.variable[svid].value.s[0] != '\0')
    {
        snprintf(tok->sval, CLI_CALC_TOKEN_MAXLEN, "%s", data.core.variable[svid].value.s);
        return -2;
    }
    return svid;
}

/**
 * classify_identifier - Classify a scanned identifier token as variable, image, command, or new var
 * @tok:     Token whose `sval` holds the raw identifier (with optional `[...]` slice suffix)
 * @after_p: Pointer in input string immediately following the identifier
 * @nt:      Index of @tok in the token stream
 */
static void classify_identifier(cli_token *tok, const char *after_p, int nt)
{
    long        tok_vid = expand_string_var_alias(tok);
    const char *s       = tok->sval;

    /* Determine whether auto-connecting /dev/shm/<name>.im.shm is safe here */
    int allow_autoconnect = (strchr(s, '[') != NULL || nt > 0 || parse_mode == 2);
    if (!allow_autoconnect && parse_mode != 0 && nt == 0)
    {
        const char *after = after_p;
        while (*after == ' ' || *after == '\t')
        {
            after++;
        }
        if (*after == '=' || *after == '+' || *after == '*' || *after == '/' || *after == '^' ||
            *after == '%' || *after == '<' || *after == '>' || *after == '!' || *after == '&' ||
            *after == '|' || *after == '?' ||
            (*after == '-' && !isalpha((unsigned char) *(after + 1))))
        {
            allow_autoconnect = 1;
        }
    }

    if (tok_vid == -2)
    {
        tok_vid = variable_ID(s);
    }
    if (tok_vid != -1 && data.core.variable[tok_vid].type != 3)
    {
        tok->type = TOK_VAR;
        return;
    }

    const char *bk = strchr(s, '[');
    if (bk == NULL && resolve_or_autoconnect_image(s, allow_autoconnect) != -1)
    {
        tok->type                = TOK_IMAGE;
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(s);
        if (!sp.error && sp.name != NULL && sp.name[0] != '\0' && strcmp(sp.name, s) != 0)
        {
            snprintf(tok->sval, CLI_CALC_TOKEN_MAXLEN, "%s", sp.name);
        }
        return;
    }
    if (bk != NULL)
    {
        char   bare[CLI_CALC_TOKEN_MAXLEN];
        size_t bn = (size_t) (bk - s);
        if (bn > 0 && bn < CLI_CALC_TOKEN_MAXLEN)
        {
            memcpy(bare, s, bn);
            bare[bn] = '\0';
            if (resolve_or_autoconnect_image(bare, 1) != -1)
            {
                tok->type                = TOK_IMAGE;
                FPS_STREAMNAME_PARSED sp = fps_streamname_parse(bare);
                if (!sp.error && sp.name != NULL && sp.name[0] != '\0' &&
                    strcmp(sp.name, bare) != 0)
                {
                    char slice_tail[CLI_CALC_TOKEN_MAXLEN];
                    snprintf(slice_tail, sizeof(slice_tail), "%s", bk);
                    snprintf(tok->sval, CLI_CALC_TOKEN_MAXLEN, "%s%s", sp.name, slice_tail);
                }
                return;
            }
        }
        tok->type = TOK_NVAR;
        return;
    }
    if (parse_mode != 0 && (cli_var_get(s) != NULL || (allow_autoconnect && is_numeric_env_var(s))))
    {
        tok->type = TOK_VAR;
        return;
    }
    if (data.cmdNBarg == 0)
    {
        data.cmdindex = -1;
        for (long i = 0; i < (long) data.NBcmd; i++)
        {
            size_t cmdlen = strlen(data.cmd[i].key);
            if (strncmp(s, data.cmd[i].key, cmdlen) == 0 && (s[cmdlen] == '\0' || s[cmdlen] == ' '))
            {
                data.cmdindex = i;
                tok->type     = TOK_COMMAND;
                return;
            }
        }
    }
    tok->type = TOK_NVAR;
}

/**
 * lex_identifier_or_func - Lex an identifier, sliced stream, or function call token
 * @p_io: In/out scan pointer
 * @tok:  Output token slot
 * @nt:   Current token index
 *
 * Return: 0 on success, -1 on unknown function error
 */
static int lex_identifier_or_func(const char **p_io, cli_token *tok, int nt)
{
    const char *p     = *p_io;
    const char *start = p;
    size_t      idlen = ident_len(p);

    if (p[idlen] == '(' && match_builtin_func(p, idlen + 1, tok))
    {
        *p_io = p + idlen + 1;
        return 0;
    }
    if (p[idlen] == ':' && is_ident_start((unsigned char) p[idlen + 1]))
    {
        size_t lbl_len = ident_len(p + idlen + 1);
        size_t tot_len = idlen + 1 + lbl_len + 1;
        if (p[idlen + 1 + lbl_len] == '(' && match_builtin_func(p, tot_len, tok))
        {
            *p_io = p + tot_len;
            return 0;
        }
    }
    p += idlen;
    if (*p == '[')
    {
        int bdepth = 0;
        do
        {
            bdepth += (*p == '[') - (*p == ']');
            p++;
        } while (bdepth > 0 && *p != '\0');
    }
    if (*p == '(')
    {
        size_t nlen = (size_t) (p - start);
        if (nlen >= CLI_CALC_TOKEN_MAXLEN)
        {
            nlen = CLI_CALC_TOKEN_MAXLEN - 1;
        }
        fprintf(stderr, "milk: calc error: unknown function '%.*s'\n", (int) nlen, start);
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "unknown function '%.*s'", (int) nlen,
                 start);
        return -1;
    }

    size_t slen = (size_t) (p - start);
    if (slen >= CLI_CALC_TOKEN_MAXLEN)
    {
        slen = CLI_CALC_TOKEN_MAXLEN - 1;
    }
    if ((start[0] == 'L' || start[0] == 'S') && start[1] == ':' && slen + 1 < CLI_CALC_TOKEN_MAXLEN)
    {
        tok->sval[0] = '@';
        memcpy(tok->sval + 1, start, slen);
        tok->sval[slen + 1] = '\0';
    }
    else
    {
        memcpy(tok->sval, start, slen);
        tok->sval[slen] = '\0';
    }
    classify_identifier(tok, p, nt);
    *p_io = p;
    return 0;
}

/** lex_quoted_string - Scan a double-quoted string literal into @tok. */
static void lex_quoted_string(const char **p_io, cli_token *tok)
{
    const char *p     = *p_io + 1;
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
    memcpy(tok->sval, start, slen);
    tok->sval[slen] = '\0';
    tok->type       = TOK_NVAR;
    tok->literal    = 1;
    if (*p == '"')
    {
        p++;
    }
    *p_io = p;
}

/**
 * cli_tokenize - Tokenize a CLI input string into `tokens`
 * @input:   Null-terminated input string
 * @tokens:  Output token array of capacity @max_tok
 * @max_tok: Maximum capacity of @tokens
 *
 * Return: Number of tokens emitted (excluding sentinel `TOK_EOF`), or -1 on error
 */
int cli_tokenize(const char *input, cli_token *tokens, int max_tok)
{
    const char *p  = input;
    int         nt = 0;

    while (*p != '\0' && nt < max_tok - 1)
    {
        tokens[nt].literal = 0;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }
        if (*p == '\n')
        {
            tokens[nt++].type = TOK_NEWLINE;
            p++;
            continue;
        }
        if (data.cmdNBarg == 0 && parse_mode == 0 && nt == 0)
        {
            size_t cmd_len = match_first_arg_command(p, &tokens[nt]);
            if (cmd_len > 0)
            {
                nt++;
                p += cmd_len;
                continue;
            }
        }
        int            oplen  = 1;
        cli_token_type optype = cli_calc_scan_op(p, &oplen);
        if (optype != TOK_EOF)
        {
            tokens[nt++].type = optype;
            p += oplen;
            continue;
        }
        size_t nlen =
            cli_calc_scan_number(p, &tokens[nt].type, &tokens[nt].val_l, &tokens[nt].val_d);
        if (nlen > 0)
        {
            nt++;
            p += nlen;
            continue;
        }
        if (*p == '"')
        {
            lex_quoted_string(&p, &tokens[nt++]);
            continue;
        }
        if (*p == '@' && is_ident_start((unsigned char) p[1]))
        {
            p++;
        }
        if (is_ident_start((unsigned char) *p))
        {
            if (lex_identifier_or_func(&p, &tokens[nt], nt) != 0)
            {
                return -1;
            }
            nt++;
            continue;
        }
        return -1;
    }

    while (*p == ' ' || *p == '\t' || *p == '\n')
    {
        p++;
    }
    if (*p != '\0' && parse_mode != 0)
    {
        fprintf(stderr, "ERROR: expression too long (max %d tokens)\n", max_tok - 1);
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "expression too long (max %d tokens)",
                 max_tok - 1);
        return -1;
    }

    tokens[nt].type = TOK_EOF;
    return nt;
}

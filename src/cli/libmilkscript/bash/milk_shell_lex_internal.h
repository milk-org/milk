// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shell_lex_internal.h
 * @brief Internal frame stack and scanner state shared between `milk_shell_lex.c` and
 *        `milk_shell_lex_code.c`.
 */

#ifndef MILK_SHELL_LEX_INTERNAL_H
#define MILK_SHELL_LEX_INTERNAL_H

#include <stddef.h>
#include <string.h>

#include "milk_shell_lex.h"

#define MSL_MAX_FRAMES 64
#define MSL_MAX_KW 64
#define MSL_MAX_HEREDOC 16

enum frame_type
{
    F_CODE,
    F_SQ,
    F_DQ,
    F_ANSI,
    F_PARAM,
    F_ARITH
};

enum frame_closer
{
    C_TOP,      /* top level: never closed */
    C_PAREN,    /* $( <( >(  -> ) */
    C_BACKTICK, /* `           -> ` */
    C_BRACE,    /* ${| ${<sp>  -> } */
    A_DPAREN,   /* $(( ((      -> )) */
    A_BRACKET   /* $[          -> ] */
};

typedef struct
{
    int    type;
    int    closer;
    int    nest;         /* CODE: open ( subshells; PARAM: { depth; ARITH: ( or [ depth */
    int    inline_brace; /* CODE: open { depth inside @{...} */
    int    cmdpos;       /* CODE: next word is in command position */
    int    fn_name;      /* CODE: after `function`, the name word is not a command */
    int    in_dq;        /* PARAM: inside double quotes (single quotes are literal) */
    int    depth;        /* command-substitution depth of this frame's bytes */
    size_t start;
    char   kw[MSL_MAX_KW];
    int    nkw;
} frame_t;

typedef struct
{
    char delim[128];
    int  strip_tabs;
} heredoc_t;

typedef struct
{
    const char *s;
    size_t      len;
    msl_map_t  *m;
    frame_t     f[MSL_MAX_FRAMES];
    frame_t     ovf_frame;
    int         nf;
    int         overflow;
    heredoc_t   hd[MSL_MAX_HEREDOC];
    int         nhd;
    int         trailing_op; /* top level ends with | && || */
} lex_t;

/** Record lexical context @ctx and substitution @depth for source byte @i. */
static inline void msl_mark(lex_t *L, size_t i, int ctx, int depth)
{
    if (i < L->len)
    {
        L->m->ctx[i]   = (unsigned char) ctx;
        L->m->depth[i] = (unsigned char) (depth > 255 ? 255 : depth);
    }
}

/** Push a new lexical frame onto the scanner stack (tracking overflow safely). */
static inline frame_t *msl_push(lex_t *L, int type, int closer, int depth, size_t start)
{
    frame_t *f = (L->nf >= MSL_MAX_FRAMES) ? (L->overflow++, &L->ovf_frame) : &L->f[L->nf++];
    memset(f, 0, sizeof(*f));
    f->type   = type;
    f->closer = closer;
    f->depth  = depth;
    f->start  = start;
    f->cmdpos = 1;
    return f;
}

/** Pop the innermost lexical frame from the scanner stack. */
static inline void msl_pop(lex_t *L)
{
    if (L->overflow > 0)
    {
        L->overflow--;
    }
    else if (L->nf > 1)
    {
        L->nf--;
    }
}

size_t msl_open_expansion(lex_t *L, size_t i, const frame_t *f);

size_t msl_scan_frame_code(lex_t *L, frame_t *f, size_t i);

#endif /* MILK_SHELL_LEX_INTERNAL_H */

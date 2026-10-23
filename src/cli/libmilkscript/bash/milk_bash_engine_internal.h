// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_engine_internal.h
 * @brief Internal declarations shared across `bash/` translator, modernizer, and co-process files.
 *
 * Role & Goal:
 *   Shares dynamic string-buffer helpers (`sbuf_t`), lexical context state
 *   (`g_mbt_*`), and recursive statement-walker entry points between
 *   `milk_bash_translate.c`, `milk_bash_modernize.c`, and `milk_bash_coproc.c`.
 */

#ifndef MILK_BASH_ENGINE_INTERNAL_H
#define MILK_BASH_ENGINE_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_bash_engine.h"
#include "milk_shell_lex.h"

/* ============================================================
 *  Dynamic String Buffer Helper
 * ============================================================ */

typedef struct
{
    char  *data;
    size_t len;
    size_t cap;
} sbuf_t;

/**
 * sbuf_init - Initialize a growable NUL-terminated string buffer
 * @b: Buffer to initialize
 *
 * Pre-allocates a 512-byte buffer to avoid repeated reallocations on short lines.
 */
static inline void sbuf_init(sbuf_t *b)
{
    b->cap  = 512;
    b->len  = 0;
    b->data = (char *) malloc(b->cap);
    if (b->data != NULL)
    {
        b->data[0] = '\0';
    }
}

/**
 * sbuf_reserve - Ensure @b has capacity for at least @extra additional bytes plus NUL
 * @b:     Target string buffer
 * @extra: Additional byte count required
 *
 * Doubles buffer capacity geometrically to maintain amortized O(1) appends.
 */
static inline void sbuf_reserve(sbuf_t *b, size_t extra)
{
    if (b->data == NULL)
    {
        return;
    }
    if (b->len + extra + 1 > b->cap)
    {
        size_t ncap = b->cap * 2;
        while (ncap < b->len + extra + 1)
        {
            ncap *= 2;
        }
        char *nd = (char *) realloc(b->data, ncap);
        if (nd == NULL)
        {
            return;
        }
        b->data = nd;
        b->cap  = ncap;
    }
}

/**
 * sbuf_append_mem - Append @n bytes from @s to string buffer @b
 * @b: Target string buffer
 * @s: Source byte slice
 * @n: Number of bytes to append
 */
static inline void sbuf_append_mem(sbuf_t *b, const char *s, size_t n)
{
    if (b->data == NULL || s == NULL || n == 0)
    {
        return;
    }
    sbuf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

/**
 * sbuf_append_str - Append NUL-terminated string @s to string buffer @b
 * @b: Target string buffer
 * @s: NUL-terminated string to append
 */
static inline void sbuf_append_str(sbuf_t *b, const char *s)
{
    if (s != NULL)
    {
        sbuf_append_mem(b, s, strlen(s));
    }
}

/**
 * sbuf_append_ch - Append single character @c to string buffer @b
 * @b: Target string buffer
 * @c: Character to append
 */
static inline void sbuf_append_ch(sbuf_t *b, char c)
{
    sbuf_reserve(b, 1);
    if (b->data != NULL)
    {
        b->data[b->len++] = c;
        b->data[b->len]   = '\0';
    }
}

/**
 * sbuf_append_dq_escaped - Append @s[0..@n) escaped for inclusion inside double quotes
 * @b: Target string buffer
 * @s: Source byte slice
 * @n: Number of bytes in @s
 *
 * Escapes `"`, `\`, and `` ` `` with a backslash while leaving `$` intact so bash
 * variable expansion still occurs inside `milkcalc "..."` arguments.
 */
static inline void sbuf_append_dq_escaped(sbuf_t *b, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (s[i] == '"' || s[i] == '\\' || s[i] == '`')
        {
            sbuf_append_ch(b, '\\');
        }
        sbuf_append_ch(b, s[i]);
    }
}

/* ============================================================
 *  Shared Translation / Modernization State
 * ============================================================ */

extern const char *g_mbt_src;
extern msl_map_t   g_mbt_map;
extern int         g_mbt_allow_legacy;
extern int         g_mbt_skip_dq_ent;
extern int         g_mbt_modernize;
extern FILE       *g_mbt_report;
extern const char *g_mbt_fname;
extern int         g_mbt_nchanges;

/**
 * mbt_cx - Return the `msl_ctx_t` lexical context of byte @p within `g_mbt_src`
 * @p: Pointer into `g_mbt_src`
 *
 * Return: `MSL_*` context constant for @p.
 */
static inline int mbt_cx(const char *p)
{
    return g_mbt_map.ctx[p - g_mbt_src];
}

/**
 * mbt_dp - Return the command-substitution nesting depth of byte @p within `g_mbt_src`
 * @p: Pointer into `g_mbt_src`
 *
 * Return: Nesting depth (`0` at top level).
 */
static inline int mbt_dp(const char *p)
{
    return g_mbt_map.depth[p - g_mbt_src];
}

/**
 * mbt_code_at - Check whether @p is unquoted shell code at substitution depth @d
 * @p: Pointer into `g_mbt_src`
 * @d: Expected command-substitution depth
 *
 * Return: 1 if @p is `MSL_CODE` at depth @d, 0 otherwise.
 */
static inline int mbt_code_at(const char *p, int d)
{
    return mbt_cx(p) == MSL_CODE && mbt_dp(p) == d;
}

/**
 * mbt_sig_len - Return length of the `@` sigil if @p[0..@n) starts with `@`, else 0
 * @p: Input slice
 * @n: Available length in bytes
 *
 * Return: 1 if @n >= 1 and `p[0] == '@'`, 0 otherwise.
 */
static inline size_t mbt_sig_len(const char *p, size_t n)
{
    return (n >= 1 && p[0] == '@') ? 1 : 0;
}

int milk_bash_funsub_disabled(void);

size_t mbt_scan_entity_path(const char *s, int *has_eq);

size_t mbt_entity_ref_len(const char *p);

size_t mbt_inline_span(const char *p, size_t n, size_t i, int d);

size_t mbt_word_len(const char *p, size_t n, int d);

size_t mbt_skip_flow_prefix(const char *s, size_t n);

int mbt_is_calc_shorthand(const char *s, size_t n);

int mbt_is_qualified_milk_command(const char *w);

void mbt_emit_src(const char *p, size_t n, int d, int ent, sbuf_t *out);

void mbt_translate_code(const char *p, size_t n, int d, sbuf_t *out);

void mbt_modernize_stmt_dispatch(const char *stmt, size_t len, int d, sbuf_t *out);

void mbt_report_note(const char *at, const char *word, const char *why);

#endif /* MILK_BASH_ENGINE_INTERNAL_H */

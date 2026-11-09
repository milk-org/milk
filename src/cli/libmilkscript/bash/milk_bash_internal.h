// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_internal.h
 * @brief Internal declarations shared by the translator and the co-process files.
 *
 * Purpose:     what the translator and co-process files share, kept out of the
 *              public milk_bash.h.
 * Does:        sbuf_t, a growing string buffer (inline helpers; after a failed
 *              allocation its data is NULL and appends do nothing);
 *              milk_bash_funsub_disabled() (bash 5.3 `${| ...; }`); the co-process
 *              calls of milk_bash_exec.c (milk_bash_coproc_ensure(),
 *              milk_bash_coproc_run(), milk_bash_set_label()); milk_bash_exec_raw(),
 *              which the co-process start also uses.
 * Used by:     milk_bash_translate.c, milk_bash_coproc.c, milk_bash_exec.c;
 *              milk_unit_script.c, for the translator.
 * Uses:        milk_bash.h, milk_shell_lex.h.
 * See also:    milk_bash.h.
 */

#ifndef MILK_BASH_INTERNAL_H
#define MILK_BASH_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_bash.h"
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
            free(b->data);
            b->data = NULL;
            b->len  = 0;
            b->cap  = 0;
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
    if (b->data == NULL)
    {
        return;
    }
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

/** Check whether bash 5.3+ `${| ...; }` function substitution is unavailable. */
int milk_bash_funsub_disabled(void);

/** Set the command-file label used by `milk_bash_exec_raw()` for diagnostics. */
void milk_bash_set_label(const char *label);

/** Ensure the bash co-process is alive, restarting it if needed. */
int milk_bash_coproc_ensure(void);

/** Write @translated to the command file, run it in the co-process, and await status. */
int milk_bash_coproc_run(char *translated, int *out_exit_req, int *out_lost);

/** Translate and execute a raw bash command block in the supervised bash co-process. */
int milk_bash_exec_raw(const char *cmdline, int *out_exit_req);

extern int g_milk_bash_exec_depth;

#endif /* MILK_BASH_INTERNAL_H */

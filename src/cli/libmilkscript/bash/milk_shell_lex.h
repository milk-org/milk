// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shell_lex.h
 * @brief Bash-aware context scanner: tells, for every byte of a script, which syntactic
 *        context it is in, and whether the text is an incomplete construct.
 *
 * Purpose:     milk reads bash text without running bash (to rewrite `@{...}`, to see where a
 *              command ends, to split `@pipe` stages, to highlight); one scanner decides where
 *              quotes, substitutions, comments and here-documents start and end, so they agree.
 * Does:        msl_scan() gives each byte its context (enum msl_ctx) and its
 *              command-substitution depth, and tells whether the text needs more input
 *              (incomplete) or leaves a quote or substitution open (unterminated, and where);
 *              milk_shell_incomplete() is its yes-or-no form.
 * Used by:     the translator (milk_bash_translate.c) and milk_bash_exec.c; milk-cli's
 *              continuation lines (CLIcore_UI_completion.c), piped and FIFO input (CLIcore.c),
 *              highlighting and diagnostics (CLIcore/treesitter: cli_treesitter_lexical.c,
 *              cli_treesitter_diag.c); the server's `@pipe` and `@assert` parsing
 *              (milk_server_eval.c); the milktranslate builtin (milk_bash_hooks.c).
 * Uses:        libc only.
 * See also:    milk_shell_lex.c, milk_shell_lex_code.c, milk_shell_lex_internal.h.
 * Constraints: also compiled into libmilkbash.so, the thin client every shell loads.
 */

#ifndef MILK_SHELL_LEX_H
#define MILK_SHELL_LEX_H

#include <stddef.h>

/** Per-byte context. Delimiters (`$(`, `)`, backticks, `${|`, `}`) are MSL_OPEN / MSL_CLOSE. */
enum msl_ctx
{
    MSL_CODE = 0, /**< shell code (statements, words, operators) */
    MSL_SQ,       /**< '...' (quote characters included) */
    MSL_DQ,       /**< "..." (quote characters included) */
    MSL_ANSI,     /**< $'...' */
    MSL_PARAM,    /**< ${...} parameter expansion */
    MSL_ARITH,    /**< $((...)), ((...)), $[...] */
    MSL_COMMENT,  /**< # to end of line */
    MSL_HEREDOC,  /**< here-document body and terminator line */
    MSL_OPEN,     /**< opening delimiter of a nested command (depth of the inner code) */
    MSL_CLOSE     /**< closing delimiter of a nested command (depth of the inner code) */
};

typedef struct
{
    unsigned char *ctx;          /**< enum msl_ctx per byte */
    unsigned char *depth;        /**< command-substitution nesting per byte (0 = top level) */
    size_t         len;          /**< number of bytes scanned */
    int            incomplete;   /**< 1 if more input is needed to complete the text */
    int            unterminated; /**< 1 if a quote or substitution is still open at the end */
    int            open_ctx;     /**< then the innermost open one: MSL_SQ, MSL_DQ, MSL_ANSI,
                                      MSL_PARAM, MSL_ARITH, or MSL_OPEN (a substitution) */
    size_t         open_at;      /**< and its first byte inside (the delimiter is just before) */
} msl_map_t;

int msl_scan(const char *src, size_t len, msl_map_t *map);

void msl_free(msl_map_t *map);

int milk_shell_incomplete(const char *src);

/** milk_calc_incomplete() - 1 if calculator text @s has an open quote, brace, paren or bracket. */
int milk_calc_incomplete(const char *s);

#endif /* MILK_SHELL_LEX_H */

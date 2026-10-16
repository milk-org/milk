// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shell_lex.h
 * @brief Bash-aware context scanner: tells, for every byte of a script, which syntactic
 *        context it is in, and whether the text is an incomplete construct.
 *
 * Single source of truth for quoting/nesting used by the `@` translator, the REPL
 * continuation prompt and FIFO line buffering.
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
} msl_map_t;

int msl_scan(const char *src, size_t len, msl_map_t *map);

void msl_free(msl_map_t *map);

int milk_shell_incomplete(const char *src);

#endif /* MILK_SHELL_LEX_H */

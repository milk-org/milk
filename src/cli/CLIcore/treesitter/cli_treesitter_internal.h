// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_treesitter_internal.h
 * @brief Internal types and cross-file prototypes for the `treesitter/` subsystem.
 *
 * Role & Goal:
 *   Shares highlight span structures, parser state accessors, normalized AST
 *   parsing (`cli_ts_parse_normalized_tree()`), and lexical fallback analyzers
 *   across `cli_treesitter.c`, `cli_treesitter_lexical.c`,
 *   `cli_treesitter_diag.c`, and `cli_treesitter_format.c`.
 */

#ifndef CLI_TREESITTER_INTERNAL_H
#define CLI_TREESITTER_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cli_treesitter.h"

#ifdef USE_TREESITTER
#    include <tree_sitter/api.h>
#endif

/** Syntax / diagnostic / match-pair highlight span */
typedef struct
{
    uint32_t    start_byte;
    uint32_t    end_byte;
    const char *color;
    int         priority; /**< 0 = normal syntax, 1 = error/diag, 2 = match pair */
} HighlightSpan;

/* ---- Lexical helpers (cli_treesitter_lexical.c) ---- */

int cli_ts_check_prefix_or_boundary_mode(const char *line, int start, const char *text);

int cli_determine_mode_lexical(const char *line,
                               int         start,
                               const char *text,
                               char       *out_cmdname,
                               size_t      cmdname_size,
                               int        *out_argidx);

bool find_match_pair_lexical(const char *line, int len, int cursor_pos, CLI_MATCH_PAIR *pair);

int cli_ts_has_unclosed_quotes(const char *s);

bool cli_ts_is_statement_boundary_before(const char *s, size_t i);

int cli_ts_count_shell_keyword(const char *s, const char *kw);

void cli_ts_count_block_braces(const char *s, int *out_open, int *out_close);

bool cli_ts_is_closer_token_line(const char *trimmed);

bool cli_ts_is_transition_token_line(const char *trimmed);

#ifdef USE_TREESITTER

/* ---- Core parser accessors (cli_treesitter.c) ---- */

TSParser *cli_ts_get_parser(void);

TSTree *cli_ts_parse_normalized_tree(const char *line, size_t len);

/* ---- Diagnostic span collector (cli_treesitter_diag.c) ---- */

void cli_ts_collect_error_spans(TSNode         node,
                                const char    *source,
                                size_t         linelen,
                                HighlightSpan *spans,
                                int           *num_spans,
                                int            max_spans,
                                int            color_lvl);

#endif /* USE_TREESITTER */

#endif /* CLI_TREESITTER_INTERNAL_H */

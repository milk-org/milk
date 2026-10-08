// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_treesitter.h
 *
 * @brief Tree-sitter syntax highlighting for the
 *        milk-cli interactive readline loop
 *
 * Provides full-line syntax coloring by parsing the
 * input with the milkcli tree-sitter grammar and
 * mapping capture groups to ANSI terminal colors.
 */

#ifndef CLI_TREESITTER_H
#define CLI_TREESITTER_H

#include <stdio.h>
#include <stdint.h>

#define CLI_DIAG_SEVERITY_NONE  0
#define CLI_DIAG_SEVERITY_INFO  1
#define CLI_DIAG_SEVERITY_ERROR 2

typedef struct
{
    int      severity;
    uint32_t start_byte;
    uint32_t end_byte;
    char     message[128];
    char     token[64];
} CLI_SYNTAX_DIAG;

/**
 * @brief Initialize tree-sitter parser and query
 *
 * Creates the TSParser, sets the milkcli language,
 * and compiles the highlight query from the embedded
 * .scm string. Call once at startup.
 *
 * @return 0 on success, -1 on failure
 */
int cli_ts_init(void);

/**
 * @brief Highlight a line and write colored output
 *
 * Parses the line with tree-sitter, walks highlight
 * captures, and writes ANSI-colored text to @p out.
 * The output includes a leading cursor-save and
 * trailing cursor-restore so readline state is
 * preserved.
 *
 * @param line  Null-terminated input line
 * @param len   Length of the line in bytes
 * @param out   Output stream (typically rl_outstream)
 */
void cli_ts_highlight_line(const char *line, int len, FILE *out);

/**
 * @brief Detect if terminal supports 256 colors
 *
 * Checks the TERM and COLORTERM environment variables
 * to determine color capability.
 *
 * @return 2 if 256-color capable, 1 otherwise
 */
int cli_ts_detect_color_level(void);

/**
 * @brief Free tree-sitter resources
 *
 * Deletes the parser, query, and cursor. Call at
 * shutdown.
 */
void cli_ts_cleanup(void);

/**
 * @brief Check if an input line/buffer is syntactically incomplete
 *
 * Uses tree-sitter AST to inspect if the buffer has missing closing
 * tokens (such as done, fi, }, unclosed quotes or brackets) or
 * ends with continuation operators (|, &&, ||, \\).
 *
 * @param buffer  Input string to test
 * @return 1 if input is incomplete and needs continuation lines,
 *         0 if complete or empty
 */
int cli_ts_is_incomplete(const char *buffer);

/**
 * @brief Determine completion mode and command context using tree-sitter AST
 *
 * Inspects the input buffer up to cursor position @p start to determine
 * whether the token at @p start is a command, file, image stream, FPS parameter,
 * or variable.
 *
 * @param line         Full command line buffer
 * @param start        Byte offset where the token to complete starts
 * @param text         Token string to complete
 * @param out_cmdname  Output buffer for extracted command name (can be NULL)
 * @param cmdname_size Size of out_cmdname buffer
 * @param out_argidx   Output pointer for 0-indexed argument position (can be NULL)
 * @return Completion mode (CLICOMPLETIONMODE_*), or -1 if a command was identified
 *         and the caller should check command argument types.
 */
int cli_ts_determine_completion_mode(
    const char *line,
    int         start,
    const char *text,
    char       *out_cmdname,
    size_t      cmdname_size,
    int        *out_argidx);

/**
 * @brief Get real-time syntax diagnostic for current input buffer
 *
 * Inspects the input line for incomplete constructs (unclosed quotes, open
 * control blocks, dangling pipes/operators) and syntax errors (unexpected
 * tokens, malformed statements) using Tree-sitter AST or lexical analysis.
 *
 * @param line Input line buffer
 * @param diag Output structure populated with severity, span, and message
 * @return 1 if a diagnostic was detected, 0 if clean/valid
 */
int cli_ts_get_diagnostic(
    const char      *line,
    CLI_SYNTAX_DIAG *diag);

/**
 * @brief Compute block nesting depth for auto-indentation
 *
 * Calculates the current block nesting depth (loops, conditionals, functions)
 * of the buffer to determine how many indentation levels should be applied.
 *
 * @param buffer Input code buffer
 * @return Nesting depth (>= 0)
 */
int cli_ts_compute_indent_depth(const char *buffer);

/**
 * @brief Format script code with semantic AST indentation
 *
 * Re-indents multi-line milk script code using Tree-sitter block scopes.
 *
 * @param code         Input script string
 * @param indent_width Number of spaces per indentation level (typically 2 or 4)
 * @return Dynamically allocated formatted string (caller must free), or NULL on error
 */
char *cli_ts_format_code(
    const char *code,
    int         indent_width);

/**
 * @brief Print structural outline of AST blocks in code
 *
 * Traverses compound statement blocks (for, while, if, case, functions)
 * and prints line ranges, line counts, and header summaries to @p out.
 *
 * @param code  Input code buffer
 * @param label Descriptive label or filename for header
 * @param out   Output stream (typically stdout)
 * @return Number of blocks found
 */
int cli_ts_print_block_folds(
    const char *code,
    const char *label,
    FILE       *out);

/**
 * @brief Check if CLI is currently prompting for a multi-line continuation line
 *
 * @return 1 if inside continuation prompt (PS2), 0 otherwise
 */
int cli_is_continuation_prompt(void);

#endif /* CLI_TREESITTER_H */

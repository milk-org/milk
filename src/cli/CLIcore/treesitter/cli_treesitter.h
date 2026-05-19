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
void cli_ts_highlight_line(
    const char *line,
    int        len,
    FILE       *out);

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

#endif /* CLI_TREESITTER_H */

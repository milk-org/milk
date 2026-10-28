// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_treesitter_format.c
 * @brief Auto-indentation depth calculation, AST script formatting, and block fold outlines.
 *
 * Role & Goal:
 *   Computes block nesting depth for interactive PS2 continuation prompts
 *   (`cli_ts_compute_indent_depth()`), formats multi-line milk scripts with
 *   AST-scoped indentation (`cli_ts_format_code()`), and prints structural
 *   block outlines (`cli_ts_print_block_folds()`).
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli_treesitter.h"
#include "cli_treesitter_internal.h"

/**
 * @brief Compute block nesting depth for auto-indentation.
 */
int cli_ts_compute_indent_depth(const char *buffer)
{
    if (!buffer || buffer[0] == '\0')
    {
        return 0;
    }

    int loops  = cli_ts_count_shell_keyword(buffer, "for") +
                 cli_ts_count_shell_keyword(buffer, "while") +
                 cli_ts_count_shell_keyword(buffer, "until");
    int dones  = cli_ts_count_shell_keyword(buffer, "done");
    int loop_d = (loops > dones) ? (loops - dones) : 0;

    int ifs  = cli_ts_count_shell_keyword(buffer, "if");
    int fis  = cli_ts_count_shell_keyword(buffer, "fi");
    int if_d = (ifs > fis) ? (ifs - fis) : 0;

    int cases  = cli_ts_count_shell_keyword(buffer, "case");
    int esacs  = cli_ts_count_shell_keyword(buffer, "esac");
    int case_d = (cases > esacs) ? (cases - esacs) : 0;

    int obrace = 0;
    int cbrace = 0;
    cli_ts_count_block_braces(buffer, &obrace, &cbrace);
    int brc_d = (obrace > cbrace) ? (obrace - cbrace) : 0;

    return loop_d + if_d + case_d + brc_d;
}

#ifdef USE_TREESITTER

/**
 * is_ast_compound_block - Check if a Tree-sitter node type represents a compound shell block.
 * @type: Null-terminated AST node type string from `ts_node_type()`.
 *
 * Identifies multi-line compound statements (`for_statement`, `while_statement`,
 * `until_statement`, `if_statement`, `case_statement`, `function_definition`,
 * and `subshell`) that introduce an indentation level and a foldable block.
 *
 * Return: true if @type is a compound block node, or false otherwise.
 */
static bool is_ast_compound_block(const char *type)
{
    return (strcmp(type, "for_statement") == 0 || strcmp(type, "while_statement") == 0 ||
            strcmp(type, "until_statement") == 0 || strcmp(type, "if_statement") == 0 ||
            strcmp(type, "case_statement") == 0 || strcmp(type, "function_definition") == 0 ||
            strcmp(type, "subshell") == 0);
}

/**
 * compute_ast_line_depths - Recursively accumulate per-line indentation depths from an AST.
 * @node:       Current Tree-sitter AST node being visited.
 * @line_depth: Array of length @num_lines receiving the nesting depth for each 0-based line.
 * @num_lines:  Total number of lines in the source script.
 * @lines:      Array of trimmed line strings used to exclude closing-keyword lines.
 *
 * Increments @line_depth for every line spanned by a multi-line compound block
 * node (excluding the final line if it starts with a closing keyword such as
 * `done`, `fi`, `esac`, `}`, or `)`), then recurses over all child nodes.
 *
 * Return: Nothing.
 */
static void compute_ast_line_depths(TSNode             node,
                                    int               *line_depth,
                                    int                num_lines,
                                    const char *const *lines)
{
    if (is_ast_compound_block(ts_node_type(node)))
    {
        TSPoint sp = ts_node_start_point(node);
        TSPoint ep = ts_node_end_point(node);
        if (ep.row > sp.row)
        {
            uint32_t end_row = ep.row;
            if (end_row < (uint32_t) num_lines && lines &&
                cli_ts_is_closer_token_line(lines[end_row]))
            {
                end_row = ep.row - 1;
            }
            for (uint32_t r = sp.row + 1; r <= end_row && r < (uint32_t) num_lines; r++)
            {
                line_depth[r]++;
            }
        }
    }

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        compute_ast_line_depths(ts_node_child(node, i), line_depth, num_lines, lines);
    }
}

/**
 * print_folds_ast_walk - Recursively walk the AST and print multi-line compound block summaries.
 * @node:        Current Tree-sitter AST node being visited.
 * @source:      Full source code string of the script.
 * @depth:       Current compound block nesting depth for indentation.
 * @block_count: In/out counter incremented for each multi-line block printed.
 * @out:         Output stream receiving the formatted outline lines.
 *
 * Prints the 1-based start/end line numbers, line span, and a 60-character
 * preview of the opening line for every multi-line compound block in @node,
 * recursing through child nodes with increased @depth.
 *
 * Return: Nothing.
 */
static void print_folds_ast_walk(TSNode      node,
                                 const char *source,
                                 int         depth,
                                 int        *block_count,
                                 FILE       *out)
{
    bool is_block = is_ast_compound_block(ts_node_type(node));
    if (is_block)
    {
        TSPoint sp = ts_node_start_point(node);
        TSPoint ep = ts_node_end_point(node);
        if (ep.row > sp.row)
        {
            uint32_t    sb   = ts_node_start_byte(node);
            const char *p    = source + sb;
            const char *eol  = strchr(p, '\n');
            size_t      flen = eol ? (size_t) (eol - p) : strlen(p);
            if (flen > 60)
            {
                flen = 60;
            }
            char first_line[64];
            memcpy(first_line, p, flen);
            first_line[flen] = '\0';

            fprintf(out, "  %*sLines %3u-%-3u (%2u lines): %s ...\n", depth * 2, "", sp.row + 1,
                    ep.row + 1, ep.row - sp.row + 1, first_line);
            (*block_count)++;
        }
    }

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        print_folds_ast_walk(ts_node_child(node, i), source, depth + (is_block ? 1 : 0),
                             block_count, out);
    }
}

#endif /* USE_TREESITTER */

/**
 * @brief Lexical fallback to populate per-line indentation depths when AST is unavailable.
 */
static void compute_lexical_line_depths(char **trimmed_lines, int *line_depth, int num_lines)
{
    int depth = 0;
    for (int i = 0; i < num_lines; i++)
    {
        const char *start = trimmed_lines[i] ? trimmed_lines[i] : "";
        if (*start == '\0')
        {
            line_depth[i] = depth;
            continue;
        }
        if (cli_ts_is_closer_token_line(start) || cli_ts_is_transition_token_line(start))
        {
            line_depth[i] = (depth > 0) ? (depth - 1) : 0;
        }
        else
        {
            line_depth[i] = depth;
        }

        int loops = cli_ts_count_shell_keyword(start, "for") +
                    cli_ts_count_shell_keyword(start, "while") +
                    cli_ts_count_shell_keyword(start, "until");
        int dones = cli_ts_count_shell_keyword(start, "done");
        depth += (loops - dones);

        int ifs = cli_ts_count_shell_keyword(start, "if");
        int fis = cli_ts_count_shell_keyword(start, "fi");
        depth += (ifs - fis);

        int cases = cli_ts_count_shell_keyword(start, "case");
        int esacs = cli_ts_count_shell_keyword(start, "esac");
        depth += (cases - esacs);

        int obrace = 0;
        int cbrace = 0;
        cli_ts_count_block_braces(start, &obrace, &cbrace);
        depth += (obrace - cbrace);

        if (depth < 0)
        {
            depth = 0;
        }
    }
}

/**
 * cli_ts_format_code - Format a multi-line milk script with consistent block indentation.
 * @code:         Null-terminated source code string to format.
 * @indent_width: Number of spaces per indentation level (clamped to 1..16, default 4).
 *
 * Splits @code into trimmed lines, computes per-line block nesting depths via
 * the normalized Tree-sitter AST (falling back to lexical keyword counting if
 * the parser is unavailable), and reconstructs the formatted script with
 * @indent_width spaces per nesting level.
 *
 * Return: Newly heap-allocated formatted string (caller must `free()`), or NULL on error.
 */
char *cli_ts_format_code(const char *code, int indent_width)
{
    if (!code)
    {
        return NULL;
    }

#ifndef USE_TREESITTER
    (void) indent_width;
    return strdup(code);
#else
    if (indent_width <= 0)
    {
        indent_width = 4;
    }
    if (indent_width > 16)
    {
        indent_width = 16;
    }

    int num_lines = 1;
    for (const char *p = code; *p != '\0'; p++)
    {
        if (*p == '\n')
        {
            num_lines++;
        }
    }

    char **raw_lines     = (char **) calloc((size_t) num_lines, sizeof(char *));
    char **trimmed_lines = (char **) calloc((size_t) num_lines, sizeof(char *));
    int   *line_depth    = (int *) calloc((size_t) num_lines, sizeof(int));
    if (!raw_lines || !trimmed_lines || !line_depth)
    {
        free(raw_lines);
        free(trimmed_lines);
        free(line_depth);
        return NULL;
    }

    const char *p        = code;
    int         line_idx = 0;
    while (*p != '\0' && line_idx < num_lines)
    {
        const char *nl      = strchr(p, '\n');
        size_t      linelen = nl ? (size_t) (nl - p) : strlen(p);
        char       *line    = (char *) malloc(linelen + 1);
        if (line)
        {
            memcpy(line, p, linelen);
            line[linelen] = '\0';
        }
        raw_lines[line_idx] = line;

        const char *start = line ? line : "";
        while (*start == ' ' || *start == '\t' || *start == '\r')
        {
            start++;
        }
        size_t slen = strlen(start);
        while (slen > 0 &&
               (start[slen - 1] == ' ' || start[slen - 1] == '\t' || start[slen - 1] == '\r'))
        {
            slen--;
        }
        char *tline = (char *) malloc(slen + 1);
        if (tline)
        {
            memcpy(tline, start, slen);
            tline[slen] = '\0';
        }
        trimmed_lines[line_idx] = tline;
        line_idx++;
        if (!nl)
        {
            break;
        }
        p = nl + 1;
    }
    num_lines = line_idx;

    bool used_ast = false;
    if (cli_ts_get_parser() != NULL || cli_ts_init() == 0)
    {
        TSTree *tree = cli_ts_parse_normalized_tree(code, strlen(code));
        if (tree)
        {
            compute_ast_line_depths(ts_tree_root_node(tree), line_depth, num_lines,
                                    (const char *const *) trimmed_lines);
            ts_tree_delete(tree);
            used_ast = true;
        }
    }
    if (!used_ast)
    {
        compute_lexical_line_depths(trimmed_lines, line_depth, num_lines);
    }

    size_t out_cap = strlen(code) * 2 + 4096;
    char  *out     = (char *) malloc(out_cap);
    if (out)
    {
        out[0]         = '\0';
        size_t out_len = 0;
        for (int i = 0; i < num_lines; i++)
        {
            const char *tline = trimmed_lines[i] ? trimmed_lines[i] : "";
            if (*tline == '\0')
            {
                if (out_len + 2 < out_cap)
                {
                    out[out_len++] = '\n';
                    out[out_len]   = '\0';
                }
                continue;
            }
            int d = line_depth[i];
            if (cli_ts_is_transition_token_line(tline) && d > 0)
            {
                d--;
            }
            int    nspaces = d * indent_width;
            size_t needed  = out_len + (size_t) nspaces + strlen(tline) + 2;
            if (needed >= out_cap)
            {
                out_cap       = needed * 2 + 4096;
                char *new_out = (char *) realloc(out, out_cap);
                if (!new_out)
                {
                    free(out);
                    out = NULL;
                    break;
                }
                out = new_out;
            }
            for (int s = 0; s < nspaces; s++)
            {
                out[out_len++] = ' ';
            }
            size_t slen = strlen(tline);
            memcpy(out + out_len, tline, slen);
            out_len += slen;
            out[out_len++] = '\n';
            out[out_len]   = '\0';
        }
    }

    for (int i = 0; i < num_lines; i++)
    {
        free(raw_lines[i]);
        free(trimmed_lines[i]);
    }
    free(raw_lines);
    free(trimmed_lines);
    free(line_depth);
    return out;
#endif
}

/**
 * cli_ts_print_block_folds - Print a hierarchical outline of multi-line blocks in a script.
 * @code:  Null-terminated script source code string.
 * @label: Optional script or file label for the outline header (may be NULL).
 * @out:   Output stream to write the outline to (defaults to `stdout` if NULL).
 *
 * Parses @code into a normalized Tree-sitter AST and walks its compound block
 * nodes via print_folds_ast_walk() to display line ranges and opening-line
 * previews.
 *
 * Return: Number of multi-line compound blocks printed.
 */
int cli_ts_print_block_folds(const char *code, const char *label, FILE *out)
{
#ifndef USE_TREESITTER
    (void) code;
    (void) label;
    (void) out;
    return 0;
#else
    if (!code || code[0] == '\0')
    {
        return 0;
    }
    if (!out)
    {
        out = stdout;
    }
    if (!cli_ts_get_parser() && cli_ts_init() != 0)
    {
        return 0;
    }

    TSTree *tree = cli_ts_parse_normalized_tree(code, strlen(code));
    if (!tree)
    {
        return 0;
    }

    if (label && label[0] != '\0')
    {
        fprintf(out, "Script Block Outline for '%s':\n", label);
    }
    else
    {
        fprintf(out, "Script Block Outline:\n");
    }

    int block_count = 0;
    print_folds_ast_walk(ts_tree_root_node(tree), code, 0, &block_count, out);
    ts_tree_delete(tree);
    return block_count;
#endif
}

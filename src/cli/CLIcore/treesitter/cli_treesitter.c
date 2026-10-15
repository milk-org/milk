// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_treesitter.c
 * @brief Tree-sitter parser lifecycle, normalized AST parsing, and line syntax highlighting.
 *
 * Role & Goal:
 *   Manages the Tree-sitter parser and highlight query (`cli_ts_init()`,
 *   `cli_ts_cleanup()`), normalizes `milk.bash` / `milk-cli` `@` and slice constructs
 *   into length-preserving shell tokens (`cli_ts_parse_normalized_tree()`),
 *   and renders ANSI-colored command lines (`cli_ts_highlight_line()`) with
 *   structural delimiter/keyword pair matching (`cli_ts_find_match_pair()`).
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_treesitter.h"
#include "cli_treesitter_internal.h"
#include "milkcli_highlights.h"

#ifdef USE_TREESITTER

TSLanguage *tree_sitter_milkcli(void);

static TSParser *ts_parser   = NULL;
static TSQuery  *ts_query    = NULL;
static int       color_level = 1; /* 1 = 16-color, 2 = 256-color */

TSParser *cli_ts_get_parser(void)
{
    return ts_parser;
}

int cli_ts_get_color_level(void)
{
    return color_level;
}

struct color_mapping
{
    const char *capture;
    const char *ansi256;
    const char *ansi16;
};

static const struct color_mapping colormap[] = {
    { "comment", "\033[38;5;60m", "\033[2;32m" },
    { "string", "\033[38;5;150m", "\033[32m" },
    { "number", "\033[38;5;209m", "\033[33m" },
    { "keyword", "\033[38;5;176m", "\033[35m" },
    { "keyword.return", "\033[38;5;176m", "\033[35m" },
    { "function.builtin", "\033[38;5;117m", "\033[36m" },
    { "function.macro", "\033[38;5;114m", "\033[1;32m" },
    { "function.call", "\033[38;5;111m", "\033[36m" },
    { "property", "\033[38;5;114m", "\033[1;32m" },
    { "type", "\033[38;5;86m", "\033[1;36m" },
    { "variable.builtin", "\033[38;5;178m", "\033[33m" },
    { "variable.parameter", "\033[38;5;255m", "\033[37m" },
    { "operator", "\033[38;5;117m", "\033[36m" },
    { "punctuation.bracket", "\033[38;5;117m", "\033[36m" },
    { "boolean", "\033[38;5;209m", "\033[33m" },
    { "string.special.path", "\033[38;5;150m", "\033[32m" },
    { "embedded", "\033[38;5;255m", "\033[37m" },
    { "keyword.operator", "\033[38;5;117m", "\033[36m" },
    { "function", "\033[38;5;111m", "\033[36m" },
    { NULL, NULL, NULL }
};

static const char *get_color_for_capture(const char *capture_name)
{
    for (int i = 0; colormap[i].capture != NULL; i++)
    {
        if (strcmp(colormap[i].capture, capture_name) == 0)
        {
            return (color_level >= 2) ? colormap[i].ansi256 : colormap[i].ansi16;
        }
    }
    return NULL;
}

int cli_ts_detect_color_level(void)
{
    const char *term      = getenv("TERM");
    const char *colorterm = getenv("COLORTERM");
    if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
    {
        return 2;
    }
    if (term && strstr(term, "256color"))
    {
        return 2;
    }
    return 1;
}

int cli_ts_init(void)
{
    if (ts_parser != NULL)
    {
        return 0;
    }

    color_level = cli_ts_detect_color_level();
    ts_parser   = ts_parser_new();
    if (!ts_parser)
    {
        return -1;
    }
    ts_parser_set_language(ts_parser, tree_sitter_milkcli());

    uint32_t     error_offset;
    TSQueryError error_type;
    ts_query = ts_query_new(tree_sitter_milkcli(), milkcli_highlights_scm,
                            (uint32_t) strlen(milkcli_highlights_scm), &error_offset, &error_type);
    if (!ts_query)
    {
        ts_parser_delete(ts_parser);
        ts_parser = NULL;
        return -1;
    }
    return 0;
}

void cli_ts_cleanup(void)
{
    if (ts_query)
    {
        ts_query_delete(ts_query);
        ts_query = NULL;
    }
    if (ts_parser)
    {
        ts_parser_delete(ts_parser);
        ts_parser = NULL;
    }
}

static int compare_spans(const void *a, const void *b)
{
    const HighlightSpan *sa = (const HighlightSpan *) a;
    const HighlightSpan *sb = (const HighlightSpan *) b;
    if (sa->start_byte != sb->start_byte)
    {
        return (int) (sa->start_byte - sb->start_byte);
    }
    if (sa->end_byte != sb->end_byte)
    {
        return (int) (sb->end_byte - sa->end_byte);
    }
    return sa->priority - sb->priority;
}

/**
 * @brief Find the first child of @parent whose `ts_node_type()` matches any entry in @types.
 */
static bool find_child_matching_types(TSNode parent, const char *const *types, TSNode *out_node)
{
    uint32_t ccount = ts_node_child_count(parent);
    for (uint32_t i = 0; i < ccount; i++)
    {
        TSNode      ch = ts_node_child(parent, i);
        const char *ct = ts_node_type(ch);
        for (int k = 0; types[k] != NULL; k++)
        {
            if (strcmp(ct, types[k]) == 0)
            {
                *out_node = ch;
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Search AST for a matching delimiter or block keyword pair at @cursor_pos.
 */
static bool find_match_pair_ast(TSNode          root,
                                const char     *line,
                                int             len,
                                int             cursor_pos,
                                CLI_MATCH_PAIR *pair)
{
    memset(pair, 0, sizeof(*pair));
    if (cursor_pos < 0 || len <= 0 || line == NULL)
    {
        return false;
    }

    int test_positions[2];
    int npos = 0;
    if (cursor_pos < len)
    {
        test_positions[npos++] = cursor_pos;
    }
    if (cursor_pos > 0)
    {
        test_positions[npos++] = cursor_pos - 1;
    }

    static const char *const open_parens[]  = { "(", "$(", "$(( ", "$((", NULL };
    static const char *const close_parens[] = { ")", "))", NULL };
    static const char *const open_brks[]    = { "[", "[[", NULL };
    static const char *const close_brks[]   = { "]", "]]", NULL };
    static const char *const open_brcs[]    = { "{", "${", NULL };
    static const char *const close_brcs[]   = { "}", NULL };
    static const char *const kw_if[]        = { "if", NULL };
    static const char *const kw_fi[]        = { "fi", NULL };
    static const char *const kw_do[]        = { "do", NULL };
    static const char *const kw_done[]      = { "done", NULL };
    static const char *const kw_loops[]     = { "for", "while", "until", NULL };
    static const char *const kw_case[]      = { "case", NULL };
    static const char *const kw_esac[]      = { "esac", NULL };
    static const char *const kw_dsemi[]     = { ";;", NULL };
    static const char *const kw_rparen[]    = { ")", NULL };

    for (int p = 0; p < npos; p++)
    {
        uint32_t pos  = (uint32_t) test_positions[p];
        TSNode   node = ts_node_descendant_for_byte_range(root, pos, pos + 1);
        if (ts_node_is_null(node))
        {
            continue;
        }
        const char *type   = ts_node_type(node);
        TSNode      parent = ts_node_parent(node);
        if (ts_node_is_null(parent))
        {
            continue;
        }

        TSNode target = { 0 };
        bool   found  = false;

        if (strcmp(type, "(") == 0 || strcmp(type, "$(") == 0 || strcmp(type, "$(( ") == 0 ||
            strcmp(type, "$((") == 0)
        {
            found = find_child_matching_types(parent, close_parens, &target);
        }
        else if (strcmp(type, ")") == 0 || strcmp(type, "))") == 0)
        {
            if (strcmp(ts_node_type(parent), "case_item") == 0)
            {
                found = find_child_matching_types(parent, kw_dsemi, &target);
            }
            else
            {
                found = find_child_matching_types(parent, open_parens, &target);
            }
        }
        else if (strcmp(type, ";;") == 0)
        {
            found = find_child_matching_types(parent, kw_rparen, &target);
        }
        else if (strcmp(type, "[") == 0 || strcmp(type, "[[") == 0)
        {
            found = find_child_matching_types(parent, close_brks, &target);
        }
        else if (strcmp(type, "]") == 0 || strcmp(type, "]]") == 0)
        {
            found = find_child_matching_types(parent, open_brks, &target);
        }
        else if (strcmp(type, "{") == 0 || strcmp(type, "${") == 0)
        {
            found = find_child_matching_types(parent, close_brcs, &target);
        }
        else if (strcmp(type, "}") == 0)
        {
            found = find_child_matching_types(parent, open_brcs, &target);
        }
        else if (strcmp(type, "if") == 0)
        {
            found = find_child_matching_types(parent, kw_fi, &target);
        }
        else if (strcmp(type, "fi") == 0)
        {
            found = find_child_matching_types(parent, kw_if, &target);
        }
        else if (strcmp(type, "do") == 0 || strcmp(type, "for") == 0 ||
                 strcmp(type, "while") == 0 || strcmp(type, "until") == 0)
        {
            found = find_child_matching_types(parent, kw_done, &target);
        }
        else if (strcmp(type, "done") == 0)
        {
            found = find_child_matching_types(parent, kw_do, &target);
            if (!found)
            {
                found = find_child_matching_types(parent, kw_loops, &target);
            }
        }
        else if (strcmp(type, "case") == 0)
        {
            found = find_child_matching_types(parent, kw_esac, &target);
        }
        else if (strcmp(type, "esac") == 0)
        {
            found = find_child_matching_types(parent, kw_case, &target);
        }

        if (found && !ts_node_is_null(target))
        {
            pair->token_start = ts_node_start_byte(node);
            pair->token_end   = ts_node_end_byte(node);
            pair->match_start = ts_node_start_byte(target);
            pair->match_end   = ts_node_end_byte(target);
            pair->has_match   = true;
            return true;
        }
    }
    return false;
}

/**
 * @brief Neutralize non-alphanumeric characters of an RHS expression up to `;`, `\n`, or `#`.
 */
static size_t mask_rhs_statement(const char *line, char *norm, size_t len, size_t start_r)
{
    size_t r  = start_r;
    int    sq = 0;
    int    dq = 0;
    while (r < len)
    {
        char rc = line[r];
        if (rc == '\\' && !sq && r + 1 < len)
        {
            r += 2;
            continue;
        }
        if (rc == '\'' && !dq)
        {
            sq = !sq;
            r++;
            continue;
        }
        if (rc == '"' && !sq)
        {
            dq = !dq;
            r++;
            continue;
        }
        if (!sq && !dq)
        {
            if (rc == '\n' || rc == ';' || (rc == '#' && isspace((unsigned char) line[r - 1])))
            {
                break;
            }
            if (!isspace((unsigned char) rc) && !isalnum((unsigned char) rc))
            {
                norm[r] = '_';
            }
        }
        r++;
    }
    return r;
}

/**
 * @brief Normalize milk.bash / milk-cli shorthand constructs into length-preserving
 *        shell-compatible characters so the Tree-sitter parser AST has no false
 *        ERROR nodes and all byte offsets remain 1-to-1 with @line.
 */
TSTree *cli_ts_parse_normalized_tree(const char *line, size_t len)
{
    if (!ts_parser || !line)
    {
        return NULL;
    }

    char *norm = (char *) malloc(len + 1);
    if (!norm)
    {
        return ts_parser_parse_string(ts_parser, NULL, line, (uint32_t) len);
    }
    memcpy(norm, line, len);
    norm[len] = '\0';

    int in_sq         = 0;
    int in_dq         = 0;
    int in_comment    = 0;
    int at_stmt_start = 1;

    for (size_t i = 0; i < len; i++)
    {
        char c = line[i];
        if (in_comment)
        {
            if (c == '\n')
            {
                in_comment    = 0;
                at_stmt_start = 1;
            }
            continue;
        }
        if (c == '\\' && !in_sq && i + 1 < len)
        {
            i++;
            at_stmt_start = 0;
            continue;
        }
        if (c == '\'' && !in_dq)
        {
            in_sq         = !in_sq;
            at_stmt_start = 0;
            continue;
        }
        if (c == '"' && !in_sq)
        {
            in_dq         = !in_dq;
            at_stmt_start = 0;
            continue;
        }
        if (in_sq)
        {
            continue;
        }
        if (in_dq)
        {
            if (c == '@' && i + 1 < len && line[i + 1] == '{')
            {
                size_t k     = i + 2;
                int    depth = 1;
                while (k < len && line[k] != '"' && depth > 0)
                {
                    if (line[k] == '{')
                    {
                        depth++;
                    }
                    else if (line[k] == '}' && --depth == 0)
                    {
                        break;
                    }
                    k++;
                }
                if (k < len && depth == 0)
                {
                    memset(norm + i, '_', k - i + 1);
                    i = k;
                }
            }
            continue;
        }

        if (c == ' ' || c == '\t' || c == '\r')
        {
            continue;
        }
        if (c == '|' && i + 1 < len && line[i + 1] == '>')
        {
            norm[i + 1] = ' ';
            i += 1;
            at_stmt_start = 1;
            continue;
        }
        if (c == ';' || c == '\n' || c == '|' || c == '&' || c == '(' || c == '{')
        {
            at_stmt_start = 1;
            continue;
        }
        if (c == '#' && (i == 0 || isspace((unsigned char) line[i - 1]) ||
                         strchr(";|(&{", line[i - 1]) != NULL))
        {
            in_comment = 1;
            continue;
        }

        /* 1. Statement-leading calculator expression: '@ <expr>' */
        if (at_stmt_start && c == '@' && (i + 1 == len || isspace((unsigned char) line[i + 1])))
        {
            norm[i]   = ':';
            size_t k  = i + 1;
            int    sq = 0;
            int    dq = 0;
            int    pd = 0;
            int    bd = 0;
            while (k < len)
            {
                char kc = line[k];
                if (kc == '\\' && !sq && k + 1 < len)
                {
                    k += 2;
                    continue;
                }
                if (kc == '\'' && !dq)
                {
                    sq = !sq;
                    k++;
                    continue;
                }
                if (kc == '"' && !sq)
                {
                    dq = !dq;
                    k++;
                    continue;
                }
                if (!sq && !dq)
                {
                    if (kc == '(')
                    {
                        pd++;
                    }
                    else if (kc == ')' && pd > 0)
                    {
                        pd--;
                    }
                    else if (kc == '[')
                    {
                        bd++;
                    }
                    else if (kc == ']' && bd > 0)
                    {
                        bd--;
                    }
                    else if (pd == 0 && bd == 0 &&
                             (kc == '\n' || kc == ';' ||
                              (kc == '#' && isspace((unsigned char) line[k - 1]))))
                    {
                        break;
                    }
                    if (!isspace((unsigned char) kc))
                    {
                        norm[k] = '_';
                    }
                }
                k++;
            }
            i             = k - 1;
            at_stmt_start = 0;
            continue;
        }

        /* 2. Inline calculator substitution: '@{<expr>}' */
        if (c == '@' && i + 1 < len && line[i + 1] == '{')
        {
            size_t k     = i + 2;
            int    depth = 1;
            while (k < len && depth > 0)
            {
                if (line[k] == '{')
                {
                    depth++;
                }
                else if (line[k] == '}' && --depth == 0)
                {
                    break;
                }
                k++;
            }
            if (k < len && depth == 0)
            {
                memset(norm + i, '_', k - i + 1);
                i             = k;
                at_stmt_start = 0;
                continue;
            }
        }

        /* 3. '@'-prefixed token: @S:im1, @STF64:im, @s.im.kw.K, @mem.mk2Dim, etc. */
        if (c == '@' && (i == 0 || isspace((unsigned char) line[i - 1]) ||
                         strchr(";|(&=", line[i - 1]) != NULL))
        {
            size_t k  = i;
            int    bd = 0;
            while (k < len)
            {
                char kc = line[k];
                if (kc == '[')
                {
                    bd++;
                }
                else if (kc == ']' && bd > 0)
                {
                    bd--;
                }
                else if (bd == 0 && (isspace((unsigned char) kc) || strchr(";|&)\"'=", kc) != NULL))
                {
                    break;
                }
                k++;
            }
            for (size_t m = i; m < k; m++)
            {
                if (!isalnum((unsigned char) norm[m]))
                {
                    norm[m] = '_';
                }
            }
            if (at_stmt_start)
            {
                size_t sp = k;
                while (sp < len && (line[sp] == ' ' || line[sp] == '\t'))
                {
                    sp++;
                }
                if (sp < len && line[sp] == '=')
                {
                    norm[sp] = '_';
                    k        = mask_rhs_statement(line, norm, len, sp + 1);
                }
            }
            i             = (k > i) ? (k - 1) : i;
            at_stmt_start = 0;
            continue;
        }

        /* 4. Identifier with optional slice suffix im[...] or shorthand assignment */
        if (isalnum((unsigned char) c) || c == '_')
        {
            size_t k = i;
            while (k < len &&
                   (isalnum((unsigned char) line[k]) || line[k] == '_' || line[k] == '.'))
            {
                k++;
            }
            size_t word_len = k - i;
            if (at_stmt_start && ((word_len == 2 && (strncmp(line + i, "do", 2) == 0 ||
                                                     strncmp(line + i, "if", 2) == 0)) ||
                                  (word_len == 4 && (strncmp(line + i, "then", 4) == 0 ||
                                                     strncmp(line + i, "else", 4) == 0 ||
                                                     strncmp(line + i, "elif", 4) == 0))))
            {
                i = k - 1;
                continue;
            }
            int has_lhs_slice = 0;
            if (k < len && line[k] == '[')
            {
                size_t b  = k + 1;
                int    bd = 1;
                while (b < len && bd > 0)
                {
                    if (line[b] == '[')
                    {
                        bd++;
                    }
                    else if (line[b] == ']' && --bd == 0)
                    {
                        break;
                    }
                    b++;
                }
                if (b < len && bd == 0)
                {
                    memset(norm + k, '_', b - k + 1);
                    k             = b + 1;
                    has_lhs_slice = 1;
                }
            }
            if (at_stmt_start)
            {
                size_t sp = k;
                while (sp < len && (line[sp] == ' ' || line[sp] == '\t'))
                {
                    sp++;
                }
                int is_compound_op =
                    (sp + 1 < len && line[sp + 1] == '=' && strchr("+-*/", line[sp]) != NULL);
                int is_eq_op =
                    (sp < len && line[sp] == '=' && (sp + 1 >= len || line[sp + 1] != '='));
                int is_shorthand_assign =
                    is_compound_op || (is_eq_op && (has_lhs_slice || sp > k ||
                                                    (sp + 1 < len && line[sp + 1] == '(')));
                if (is_shorthand_assign)
                {
                    k = mask_rhs_statement(line, norm, len, sp);
                }
                else if (is_eq_op && sp == k)
                {
                    i             = k;
                    at_stmt_start = 0;
                    continue;
                }
            }
            i             = (k > i) ? (k - 1) : i;
            at_stmt_start = 0;
            continue;
        }

        at_stmt_start = 0;
    } // for (size_t i = 0; i < len; i++)

    TSTree *tree = ts_parser_parse_string(ts_parser, NULL, norm, (uint32_t) len);
    free(norm);
    return tree;
}

/**
 * @brief Append lexical highlight spans for `@<cmd>`, `@<mod>:`, `@{...}`, `@ <expr>`, and `@s.`.
 */
static void collect_at_token_spans(const char    *line,
                                   int            len,
                                   HighlightSpan *spans,
                                   int           *num_spans,
                                   int            max_spans)
{
    int         in_sq     = 0;
    int         in_dq     = 0;
    const char *mod_color = (color_level >= 2) ? "\033[38;5;214m" : "\033[33m";
    const char *cmd_color = (color_level >= 2) ? "\033[1;38;5;39m" : "\033[1;36m";

    for (int i = 0; i < len && *num_spans < max_spans; i++)
    {
        if (line[i] == '\\' && !in_sq && i + 1 < len)
        {
            i++;
            continue;
        }
        if (line[i] == '\'' && !in_dq)
        {
            in_sq = !in_sq;
            continue;
        }
        if (line[i] == '"' && !in_sq)
        {
            in_dq = !in_dq;
            continue;
        }
        if (in_sq || line[i] != '@' ||
            !(i == 0 || isspace((unsigned char) line[i - 1]) ||
              strchr(";|(&=", line[i - 1]) != NULL))
        {
            continue;
        }

        if (!in_dq && (i + 1 == len || isspace((unsigned char) line[i + 1])))
        {
            spans[*num_spans].start_byte = (uint32_t) i;
            spans[*num_spans].end_byte   = (uint32_t) (i + 1);
            spans[*num_spans].color      = cmd_color;
            spans[*num_spans].priority   = 1;
            (*num_spans)++;
            continue;
        }
        if (i + 1 < len && line[i + 1] == '{')
        {
            spans[*num_spans].start_byte = (uint32_t) i;
            spans[*num_spans].end_byte   = (uint32_t) (i + 2);
            spans[*num_spans].color      = cmd_color;
            spans[*num_spans].priority   = 1;
            (*num_spans)++;
            int k     = i + 2;
            int depth = 1;
            while (k < len && depth > 0)
            {
                if (line[k] == '{')
                {
                    depth++;
                }
                else if (line[k] == '}' && --depth == 0)
                {
                    break;
                }
                k++;
            }
            if (k < len && depth == 0 && *num_spans < max_spans)
            {
                spans[*num_spans].start_byte = (uint32_t) k;
                spans[*num_spans].end_byte   = (uint32_t) (k + 1);
                spans[*num_spans].color      = cmd_color;
                spans[*num_spans].priority   = 1;
                (*num_spans)++;
            }
            continue;
        }

        int j         = i + 1;
        int colon_end = -1;
        while (j < len && !isspace((unsigned char) line[j]) && strchr(";|&)\"'=[", line[j]) == NULL)
        {
            if (line[j] == ':')
            {
                colon_end = j + 1;
            }
            j++;
        }
        if (colon_end > i + 1)
        {
            spans[*num_spans].start_byte = (uint32_t) i;
            spans[*num_spans].end_byte   = (uint32_t) colon_end;
            spans[*num_spans].color      = mod_color;
            spans[*num_spans].priority   = 1;
            (*num_spans)++;
            i = colon_end - 1;
        }
        else if (j > i + 1)
        {
            int is_ent =
                (j > i + 2 &&
                 (strncmp(line + i + 1, "s.", 2) == 0 || strncmp(line + i + 1, "f.", 2) == 0 ||
                  strncmp(line + i + 1, "p.", 2) == 0 || strncmp(line + i + 1, "q.", 2) == 0 ||
                  strncmp(line + i + 1, "stream.", 7) == 0 ||
                  strncmp(line + i + 1, "fps.", 4) == 0 || strncmp(line + i + 1, "proc.", 5) == 0 ||
                  strncmp(line + i + 1, "procinfo.", 9) == 0 ||
                  strncmp(line + i + 1, "seq.", 4) == 0));
            spans[*num_spans].start_byte = (uint32_t) i;
            spans[*num_spans].end_byte   = (uint32_t) j;
            spans[*num_spans].color      = is_ent ? mod_color : cmd_color;
            spans[*num_spans].priority   = 1;
            (*num_spans)++;
            i = j - 1;
        }
    }
}

void cli_ts_highlight_line(const char *line, int len, int cursor_pos, FILE *out)
{
    if (!ts_parser || !ts_query || !line || len == 0)
    {
        fprintf(out, "%s", line);
        return;
    }

    TSTree *tree = cli_ts_parse_normalized_tree(line, (size_t) len);
    if (!tree)
    {
        fprintf(out, "%s", line);
        return;
    }

    TSNode         root_node = ts_tree_root_node(tree);
    TSQueryCursor *cursor    = ts_query_cursor_new();
    ts_query_cursor_exec(cursor, ts_query, root_node);

    TSQueryMatch  match;
    uint32_t      capture_index;
    HighlightSpan spans[1024];
    int           num_spans = 0;

    while (ts_query_cursor_next_capture(cursor, &match, &capture_index) && num_spans < 1024)
    {
        TSNode      node          = match.captures[capture_index].node;
        uint32_t    id            = match.captures[capture_index].index;
        uint32_t    name_len      = 0;
        const char *capture_name  = ts_query_capture_name_for_id(ts_query, id, &name_len);
        char        name_buf[128] = { 0 };
        if (name_len < sizeof(name_buf))
        {
            memcpy(name_buf, capture_name, name_len);
            const char *color = get_color_for_capture(name_buf);
            if (color)
            {
                spans[num_spans].start_byte = ts_node_start_byte(node);
                spans[num_spans].end_byte   = ts_node_end_byte(node);
                spans[num_spans].color      = color;
                spans[num_spans].priority   = 0;
                num_spans++;
            }
        }
    }
    ts_query_cursor_delete(cursor);

    collect_at_token_spans(line, len, spans, &num_spans, 1024);

    if (data.syntax_diagnostics && ts_node_has_error(root_node))
    {
        cli_ts_collect_error_spans(root_node, line, (size_t) len, spans, &num_spans, 1024,
                                   color_level);
    }

    if (data.show_match && cursor_pos >= 0)
    {
        CLI_MATCH_PAIR mp;
        bool           ok = find_match_pair_ast(root_node, line, len, cursor_pos, &mp);
        if (!ok || !mp.has_match)
        {
            ok = find_match_pair_lexical(line, len, cursor_pos, &mp);
        }
        if (ok && mp.has_match)
        {
            const char *match_style = "\033[7m";
            if (num_spans < 1024)
            {
                spans[num_spans].start_byte = mp.token_start;
                spans[num_spans].end_byte   = mp.token_end;
                spans[num_spans].color      = match_style;
                spans[num_spans].priority   = 2;
                num_spans++;
            }
            if (num_spans < 1024)
            {
                spans[num_spans].start_byte = mp.match_start;
                spans[num_spans].end_byte   = mp.match_end;
                spans[num_spans].color      = match_style;
                spans[num_spans].priority   = 2;
                num_spans++;
            }
        }
    }

    ts_tree_delete(tree);
    qsort(spans, (size_t) num_spans, sizeof(HighlightSpan), compare_spans);

    const char *RESET           = "\033[0m";
    const char *color_stack[64] = { 0 };
    int         stack_depth     = 0;

    for (int i = 0; i < len; i++)
    {
        int old_depth = stack_depth;
        for (int k = 0; k < num_spans; k++)
        {
            if (spans[k].end_byte == (uint32_t) i)
            {
                for (int j = 0; j < stack_depth; j++)
                {
                    if (color_stack[j] == spans[k].color)
                    {
                        for (int m = j; m < stack_depth - 1; m++)
                        {
                            color_stack[m] = color_stack[m + 1];
                        }
                        stack_depth--;
                        break;
                    }
                }
            }
        }

        int pushed = 0;
        for (int k = 0; k < num_spans; k++)
        {
            if (spans[k].start_byte == (uint32_t) i && stack_depth < 64)
            {
                color_stack[stack_depth++] = spans[k].color;
                pushed                     = 1;
            }
        }

        if (pushed || stack_depth < old_depth)
        {
            fprintf(out, "%s", RESET);
            if (stack_depth > 0)
            {
                fprintf(out, "%s", color_stack[stack_depth - 1]);
            }
        }
        fputc(line[i], out);
    }

    fprintf(out, "%s", RESET);
    fflush(out);
}

bool cli_ts_find_match_pair(const char *line, int cursor_pos, CLI_MATCH_PAIR *pair)
{
    memset(pair, 0, sizeof(*pair));
    if (line == NULL || cursor_pos < 0)
    {
        return false;
    }
    int len = (int) strlen(line);
    if (len <= 0)
    {
        return false;
    }

    if (ts_parser != NULL)
    {
        TSTree *tree = cli_ts_parse_normalized_tree(line, (size_t) len);
        if (tree != NULL)
        {
            TSNode root = ts_tree_root_node(tree);
            bool   ok   = find_match_pair_ast(root, line, len, cursor_pos, pair);
            ts_tree_delete(tree);
            if (ok && pair->has_match)
            {
                return true;
            }
        }
    }
    return find_match_pair_lexical(line, len, cursor_pos, pair);
}

#else /* !USE_TREESITTER */

int cli_ts_init(void)
{
    return 0;
}

int cli_ts_detect_color_level(void)
{
    return 1;
}

void cli_ts_cleanup(void)
{
}

void cli_ts_highlight_line(const char *line, int len, int cursor_pos, FILE *out)
{
    if (line == NULL || len == 0)
    {
        return;
    }
    if (data.show_match && cursor_pos >= 0)
    {
        CLI_MATCH_PAIR mp;
        if (find_match_pair_lexical(line, len, cursor_pos, &mp) && mp.has_match)
        {
            for (int i = 0; i < len; i++)
            {
                if (i == (int) mp.token_start || i == (int) mp.match_start)
                {
                    fprintf(out, "\033[7m");
                }
                fputc(line[i], out);
                if (i + 1 == (int) mp.token_end || i + 1 == (int) mp.match_end)
                {
                    fprintf(out, "\033[0m");
                }
            }
            fflush(out);
            return;
        }
    }
    fprintf(out, "%s", line);
    fflush(out);
}

bool cli_ts_find_match_pair(const char *line, int cursor_pos, CLI_MATCH_PAIR *pair)
{
    if (line == NULL || cursor_pos < 0)
    {
        if (pair != NULL)
        {
            memset(pair, 0, sizeof(*pair));
        }
        return false;
    }
    return find_match_pair_lexical(line, (int) strlen(line), cursor_pos, pair);
}

#endif /* USE_TREESITTER */

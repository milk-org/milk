// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "cli_treesitter.h"
#include "milkcli_highlights.h"

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#include "CLIcore.h"

/**
 * @brief Lexical fallback for determining completion mode and active command
 *
 * Scans backward from @p start to identify statement boundaries, command
 * positions, and command arguments.
 *
 * @param line         Full command line buffer
 * @param start        Byte offset where the token to complete starts
 * @param text         Token string to complete
 * @param out_cmdname  Output buffer for extracted command name (can be NULL)
 * @param cmdname_size Size of out_cmdname buffer
 * @param out_argidx   Output pointer for 0-indexed argument position (can be NULL)
 * @return Completion mode (CLICOMPLETIONMODE_*), or -1 if a command was identified
 */
static int cli_determine_mode_lexical(
    const char *line,
    int         start,
    const char *text,
    char       *out_cmdname,
    size_t      cmdname_size,
    int        *out_argidx)
{
    if (out_cmdname && cmdname_size > 0)
    {
        out_cmdname[0] = '\0';
    }
    if (out_argidx)
    {
        *out_argidx = 0;
    }

    if (!line)
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    /* 1. Direct token prefix overrides */
    if (text)
    {
        if (strncmp(text, "${s.", 4) == 0 || strncmp(text, "@s.", 3) == 0)
        {
            return CLICOMPLETIONMODE_VARS_STREAM;
        }
        if (strncmp(text, "@fps.", 5) == 0)
        {
            return CLICOMPLETIONMODE_VARS_FPS;
        }
        if (strncmp(text, "@seq.", 5) == 0)
        {
            return CLICOMPLETIONMODE_VARS_SEQ;
        }
        if (text[0] == '$')
        {
            return CLICOMPLETIONMODE_VARS_ENV;
        }
        if (strncmp(text, "./", 2) == 0 || strncmp(text, "../", 3) == 0 ||
            text[0] == '/' || text[0] == '~')
        {
            return CLICOMPLETIONMODE_FILES;
        }
    }

    /* 2. Backward check from start to find statement boundary or command position */
    int prev_idx = start - 1;
    while (prev_idx >= 0 && (line[prev_idx] == ' ' || line[prev_idx] == '\t'))
    {
        prev_idx--;
    }
    if (prev_idx < 0)
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    char prev_c = line[prev_idx];
    if (prev_c == ';' || prev_c == '|' || prev_c == '&' ||
        prev_c == '(' || prev_c == '{' || prev_c == '\n' || prev_c == '`')
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_c == '>' || prev_c == '<')
    {
        return CLICOMPLETIONMODE_FILES;
    }

    /* Check keyword separators */
    if (prev_idx >= 1 && strncmp(&line[prev_idx - 1], "do", 2) == 0 &&
        (prev_idx - 1 == 0 || isspace((unsigned char) line[prev_idx - 2]) ||
         line[prev_idx - 2] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "then", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4]) ||
         line[prev_idx - 4] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "else", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4]) ||
         line[prev_idx - 4] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "elif", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4]) ||
         line[prev_idx - 4] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 1 && strncmp(&line[prev_idx - 1], "if", 2) == 0 &&
        (prev_idx - 1 == 0 || isspace((unsigned char) line[prev_idx - 2]) ||
         line[prev_idx - 2] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 4 && strncmp(&line[prev_idx - 4], "while", 5) == 0 &&
        (prev_idx - 4 == 0 || isspace((unsigned char) line[prev_idx - 5]) ||
         line[prev_idx - 5] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 4 && strncmp(&line[prev_idx - 4], "until", 5) == 0 &&
        (prev_idx - 4 == 0 || isspace((unsigned char) line[prev_idx - 5]) ||
         line[prev_idx - 5] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 0 && line[prev_idx] == '!')
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "time", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4])))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 4 && strncmp(&line[prev_idx - 4], "watch", 5) == 0 &&
        (prev_idx - 4 == 0 || isspace((unsigned char) line[prev_idx - 5])))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    /* 3. Find start of current statement by scanning backwards */
    int in_sq      = 0;
    int in_dq      = 0;
    int stmt_start = 0;

    for (int i = 0; i < start; i++)
    {
        char c = line[i];
        if (c == '\\' && line[i + 1] != '\0' && !in_sq)
        {
            i++;
            continue;
        }
        if (c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
        }
        else if (c == '"' && !in_sq)
        {
            in_dq = !in_dq;
        }
        else if (!in_sq && !in_dq)
        {
            if (c == ';' || c == '|' || c == '&' || c == '(' || c == '{' ||
                c == '\n' || c == '`')
            {
                stmt_start = i + 1;
            }
        }
    }

    const char *p = line + stmt_start;
    while (*p && (p - line) < start && isspace((unsigned char) *p))
    {
        p++;
    }

    const char *cmd_start = p;
    while (*p && (p - line) < start && !isspace((unsigned char) *p) &&
           *p != ';' && *p != '|' && *p != '&' && *p != '(' && *p != ')')
    {
        p++;
    }
    const char *cmd_end = p;

    if (cmd_start >= line + start || (line + start) <= cmd_end)
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    size_t clen = (size_t) (cmd_end - cmd_start);
    if (out_cmdname && cmdname_size > 0)
    {
        if (clen >= cmdname_size)
        {
            clen = cmdname_size - 1;
        }
        strncpy(out_cmdname, cmd_start, clen);
        out_cmdname[clen] = '\0';
    }

    int arg_count = 0;
    int in_word   = 0;
    in_sq         = 0;
    in_dq         = 0;
    for (const char *ap = cmd_end; ap < line + start; ap++)
    {
        char ac = *ap;
        if (ac == '\\' && *(ap + 1) != '\0' && !in_sq)
        {
            ap++;
            in_word = 1;
            continue;
        }
        if (ac == '\'' && !in_dq)
        {
            in_sq   = !in_sq;
            in_word = 1;
        }
        else if (ac == '"' && !in_sq)
        {
            in_dq   = !in_dq;
            in_word = 1;
        }
        else if (!in_sq && !in_dq)
        {
            if (isspace((unsigned char) ac))
            {
                in_word = 0;
            }
            else
            {
                if (!in_word)
                {
                    arg_count++;
                    in_word = 1;
                }
            }
        }
    }

    if (out_argidx)
    {
        *out_argidx = (arg_count > 0) ? (arg_count - 1) : 0;
    }

    return -1;
}

#ifdef USE_TREESITTER

#    include <tree_sitter/api.h>

/* Declare the language function generated by tree-sitter */
TSLanguage *tree_sitter_milkcli(void);

static TSParser *ts_parser   = NULL;
static TSQuery  *ts_query    = NULL;
static int       color_level = 1; // 1 = 16-color, 2 = 256-color

struct color_mapping
{
    const char *capture;
    const char *ansi256;
    const char *ansi16;
};

// Neovim Material Palenight approximation
static const struct color_mapping colormap[] = {
    { "comment", "\033[38;5;60m", "\033[2;32m" }, // Dark green/grey
    { "string", "\033[38;5;150m", "\033[32m" },   // Green
    { "number", "\033[38;5;209m", "\033[33m" },   // Orange/Yellow
    { "keyword", "\033[38;5;176m", "\033[35m" },  // Purple
    { "keyword.return", "\033[38;5;176m", "\033[35m" },
    { "function.builtin", "\033[38;5;117m", "\033[36m" },   // Cyan
    { "function.macro", "\033[38;5;114m", "\033[1;32m" },   // Bright Green
    { "function.call", "\033[38;5;111m", "\033[36m" },      // Blue/Cyan
    { "property", "\033[38;5;114m", "\033[1;32m" },         // Bright Green
    { "type", "\033[38;5;86m", "\033[1;36m" },              // Bright Cyan
    { "variable.builtin", "\033[38;5;178m", "\033[33m" },   // Amber/Yellow
    { "variable.parameter", "\033[38;5;255m", "\033[37m" }, // White
    { "operator", "\033[38;5;117m", "\033[36m" },           // Cyan
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

/**
 * @brief Detect terminal color capability.
 *
 * Checks COLORTERM and TERM environment variables
 * to determine 256-color or truecolor support.
 */
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

/**
 * @brief Initialize treesitter syntax highlighting.
 *
 * Loads the milk grammar and sets up the
 * parser instance.
 */
int cli_ts_init(void)
{
    if (ts_parser != NULL)
    {
        return 0; // Already initialized
    }

    color_level = cli_ts_detect_color_level();

    ts_parser = ts_parser_new();
    if (!ts_parser)
    {
        return -1;
    }

    ts_parser_set_language(ts_parser, tree_sitter_milkcli());

    uint32_t     error_offset;
    TSQueryError error_type;
    ts_query = ts_query_new(tree_sitter_milkcli(), milkcli_highlights_scm,
                            strlen(milkcli_highlights_scm), &error_offset, &error_type);

    if (!ts_query)
    {
        // Query compilation failed
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

// Spans for highlighting
typedef struct
{
    uint32_t    start_byte;
    uint32_t    end_byte;
    const char *color;
} HighlightSpan;

static int compare_spans(const void *a, const void *b)
{
    const HighlightSpan *sa = (const HighlightSpan *) a;
    const HighlightSpan *sb = (const HighlightSpan *) b;
    if (sa->start_byte != sb->start_byte)
    {
        return sa->start_byte - sb->start_byte;
    }
    // If they start at the same place, earlier end_byte goes first so outer spans
    // enclose inner spans
    return sb->end_byte - sa->end_byte;
}

void cli_ts_highlight_line(const char *line, int len, FILE *out)
{
    if (!ts_parser || !ts_query || !line || len == 0)
    {
        fprintf(out, "%s", line);
        return;
    }

    TSTree *tree = ts_parser_parse_string(ts_parser, NULL, line, len);
    if (!tree)
    {
        fprintf(out, "%s", line);
        return;
    }

    TSNode         root_node = ts_tree_root_node(tree);
    TSQueryCursor *cursor    = ts_query_cursor_new();

    ts_query_cursor_exec(cursor, ts_query, root_node);

    TSQueryMatch match;
    uint32_t     capture_index;

    // Max 1024 spans per line should be plenty
    HighlightSpan spans[1024];
    int           num_spans = 0;

    // Collect all captures
    while (ts_query_cursor_next_capture(cursor, &match, &capture_index) && num_spans < 1024)
    {
        TSNode   node = match.captures[capture_index].node;
        uint32_t id   = match.captures[capture_index].index;

        uint32_t    name_len     = 0;
        const char *capture_name = ts_query_capture_name_for_id(ts_query, id, &name_len);

        // Ensure null-termination or safe comparison
        char name_buf[128] = { 0 };
        if (name_len < sizeof(name_buf))
        {
            memcpy(name_buf, capture_name, name_len);
            const char *color = get_color_for_capture(name_buf);
            if (color)
            {
                spans[num_spans].start_byte = ts_node_start_byte(node);
                spans[num_spans].end_byte   = ts_node_end_byte(node);
                spans[num_spans].color      = color;
                num_spans++;
            }
        }
    }

    ts_query_cursor_delete(cursor);
    ts_tree_delete(tree);

    // Sort spans by start_byte, then by size (largest first)
    qsort(spans, num_spans, sizeof(HighlightSpan), compare_spans);

    // Render: caller positions cursor at input start

    const char *RESET        = "\033[0m";
    int         current_byte = 0;

    // Keep track of active style stack to return to outer colors
    const char *color_stack[64] = { 0 };
    int         stack_depth     = 0;

    for (int i = 0; i < len; i++)
    {
        // Pop expired spans
        int old_depth = stack_depth;
        for (int k = 0; k < num_spans; k++)
        {
            if (spans[k].end_byte == i)
            {
                // Find and remove this span from stack
                for (int j = 0; j < stack_depth; j++)
                {
                    if (color_stack[j] == spans[k].color)
                    {
                        // Shift remaining down
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

        // Push new spans starting here
        int pushed = 0;
        for (int k = 0; k < num_spans; k++)
        {
            if (spans[k].start_byte == i)
            {
                if (stack_depth < 64)
                {
                    color_stack[stack_depth++] = spans[k].color;
                    pushed                     = 1;
                }
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

static int has_unclosed_quotes(const char *s)
{
    int in_dquote = 0;
    int in_squote = 0;

    for (size_t i = 0; s[i] != '\0'; i++)
    {
        if (s[i] == '\\' && s[i + 1] != '\0' && !in_squote)
        {
            i++;
            continue;
        }
        if (s[i] == '"' && !in_squote)
        {
            in_dquote = !in_dquote;
        }
        else if (s[i] == '\'' && !in_dquote)
        {
            in_squote = !in_squote;
        }
    }

    return in_dquote || in_squote;
}

/**
 * @brief Checks if a statement boundary precedes the token at index i.
 *
 * A shell reserved word (such as done, fi, esac) can only appear at a
 * statement boundary: start of line/buffer, newline, semicolon, pipe,
 * ampersand, or block keywords (do, then, else, elif).
 *
 * @param s Input string
 * @param i Index of token start
 * @return true if preceded by statement boundary, false otherwise
 */
static bool is_statement_boundary_before(
    const char *s,
    size_t      i)
{
    while (i > 0 && (s[i - 1] == ' ' || s[i - 1] == '\t' || s[i - 1] == '\r'))
    {
        i--;
    }
    if (i == 0)
    {
        return true;
    }
    char prev = s[i - 1];
    if (prev == '\n' || prev == ';' || prev == '|' || prev == '&' ||
        prev == '(' || prev == '{')
    {
        return true;
    }

    if (i >= 2 && strncmp(&s[i - 2], "do", 2) == 0 &&
        (i == 2 || s[i - 3] == ' ' || s[i - 3] == '\t' ||
         s[i - 3] == '\n' || s[i - 3] == ';'))
    {
        return true;
    }
    if (i >= 4 && strncmp(&s[i - 4], "then", 4) == 0 &&
        (i == 4 || s[i - 5] == ' ' || s[i - 5] == '\t' ||
         s[i - 5] == '\n' || s[i - 5] == ';'))
    {
        return true;
    }
    if (i >= 4 && strncmp(&s[i - 4], "else", 4) == 0 &&
        (i == 4 || s[i - 5] == ' ' || s[i - 5] == '\t' ||
         s[i - 5] == '\n' || s[i - 5] == ';'))
    {
        return true;
    }
    if (i >= 4 && strncmp(&s[i - 4], "elif", 4) == 0 &&
        (i == 4 || s[i - 5] == ' ' || s[i - 5] == '\t' ||
         s[i - 5] == '\n' || s[i - 5] == ';'))
    {
        return true;
    }

    return false;
}

/**
 * @brief Count occurrences of a shell reserved word in buffer.
 *
 * Ignores occurrences inside single/double quotes and comments,
 * and requires the keyword to appear at a statement boundary.
 *
 * @param s  Input buffer
 * @param kw Reserved word to count
 * @return Number of valid keyword occurrences
 */
static int count_shell_keyword(
    const char *s,
    const char *kw)
{
    size_t kwlen = strlen(kw);
    int count = 0;
    int in_dquote = 0;
    int in_squote = 0;
    int in_comment = 0;

    for (size_t i = 0; s[i] != '\0'; i++)
    {
        if (s[i] == '\n')
        {
            in_comment = 0;
        }
        if (in_comment)
        {
            continue;
        }
        if (s[i] == '\\' && s[i + 1] != '\0' && !in_squote)
        {
            i++;
            continue;
        }
        if (s[i] == '"' && !in_squote)
        {
            in_dquote = !in_dquote;
            continue;
        }
        if (s[i] == '\'' && !in_dquote)
        {
            in_squote = !in_squote;
            continue;
        }
        if (in_dquote || in_squote)
        {
            continue;
        }
        if (s[i] == '#')
        {
            if (is_statement_boundary_before(s, i))
            {
                in_comment = 1;
                continue;
            }
        }

        if (strncmp(&s[i], kw, kwlen) == 0)
        {
            char rc = s[i + kwlen];
            bool right_ok = (rc == '\0' || rc == ' ' || rc == '\t' ||
                             rc == '\n' || rc == '\r' || rc == ';' ||
                             rc == ')' || rc == '}' || rc == '|' || rc == '&');
            if (right_ok && is_statement_boundary_before(s, i))
            {
                count++;
                i += kwlen - 1;
            }
        }
    } // for (size_t i = 0; s[i] != '\0'; i++)

    return count;
}

static bool node_tree_has_missing(
    TSNode      node,
    const char *buffer)
{
    if (ts_node_is_missing(node))
    {
        const char *mtype = ts_node_type(node);
        if (strcmp(mtype, "done") == 0)
        {
            int starters = count_shell_keyword(buffer, "for") +
                           count_shell_keyword(buffer, "while") +
                           count_shell_keyword(buffer, "until");
            int closers  = count_shell_keyword(buffer, "done");
            if (starters <= closers)
            {
                return false;
            }
            return true;
        }
        if (strcmp(mtype, "fi") == 0)
        {
            int starters = count_shell_keyword(buffer, "if");
            int closers  = count_shell_keyword(buffer, "fi");
            if (starters <= closers)
            {
                return false;
            }
            return true;
        }
        if (strcmp(mtype, "esac") == 0)
        {
            int starters = count_shell_keyword(buffer, "case");
            int closers  = count_shell_keyword(buffer, "esac");
            if (starters <= closers)
            {
                return false;
            }
            return true;
        }
        if (strcmp(mtype, "}") == 0)
        {
            int o = count_shell_keyword(buffer, "{");
            int c = count_shell_keyword(buffer, "}");
            if (o <= c)
            {
                return false;
            }
            return true;
        }
        return true;
    }

    const char *type = ts_node_type(node);
    if (strcmp(type, "ERROR") == 0)
    {
        uint32_t count = ts_node_child_count(node);
        for (uint32_t i = 0; i < count; i++)
        {
            TSNode child = ts_node_child(node, i);
            const char *ctype = ts_node_type(child);
            if (strcmp(ctype, "for") == 0 ||
                strcmp(ctype, "while") == 0 ||
                strcmp(ctype, "until") == 0 ||
                strcmp(ctype, "do") == 0)
            {
                int starters = count_shell_keyword(buffer, "for") +
                               count_shell_keyword(buffer, "while") +
                               count_shell_keyword(buffer, "until");
                int closers  = count_shell_keyword(buffer, "done");
                if (starters > closers)
                {
                    return true;
                }
            }
            else if (strcmp(ctype, "if") == 0 ||
                     strcmp(ctype, "then") == 0 ||
                     strcmp(ctype, "elif") == 0 ||
                     strcmp(ctype, "else") == 0)
            {
                int starters = count_shell_keyword(buffer, "if");
                int closers  = count_shell_keyword(buffer, "fi");
                if (starters > closers)
                {
                    return true;
                }
            }
            else if (strcmp(ctype, "case") == 0)
            {
                int starters = count_shell_keyword(buffer, "case");
                int closers  = count_shell_keyword(buffer, "esac");
                if (starters > closers)
                {
                    return true;
                }
            }
        }
    }

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        if (node_tree_has_missing(ts_node_child(node, i), buffer))
        {
            return true;
        }
    }

    return false;
}

int cli_ts_is_incomplete(const char *buffer)
{
    if (buffer == NULL || buffer[0] == '\0')
    {
        return 0;
    }

    if (has_unclosed_quotes(buffer))
    {
        return 1;
    }

    /* Check trailing continuation operators */
    size_t len = strlen(buffer);
    while (len > 0 && (buffer[len - 1] == ' ' || buffer[len - 1] == '\t' ||
                       buffer[len - 1] == '\n' || buffer[len - 1] == '\r'))
    {
        len--;
    }
    if (len > 0)
    {
        if (buffer[len - 1] == '\\' && (len == 1 || buffer[len - 2] != '\\'))
        {
            return 1;
        }
        if (buffer[len - 1] == '|' && (len == 1 || buffer[len - 2] != '|'))
        {
            return 1;
        }
        if (len >= 2 && buffer[len - 1] == '&' && buffer[len - 2] == '&')
        {
            return 1;
        }
        if (len >= 2 && buffer[len - 1] == '|' && buffer[len - 2] == '|')
        {
            return 1;
        }
    }

    if (ts_parser == NULL)
    {
        if (cli_ts_init() != 0)
        {
            return 0;
        }
    }

    TSTree *tree = ts_parser_parse_string(ts_parser, NULL, buffer, (uint32_t) strlen(buffer));
    if (!tree)
    {
        return 0;
    }

    TSNode root = ts_tree_root_node(tree);
    bool incomplete = false;
    if (ts_node_has_error(root))
    {
        incomplete = node_tree_has_missing(root, buffer);
    }
    ts_tree_delete(tree);

    return incomplete ? 1 : 0;
}

/**
 * @brief Determine completion mode and command context using tree-sitter AST
 */
int cli_ts_determine_completion_mode(
    const char *line,
    int         start,
    const char *text,
    char       *out_cmdname,
    size_t      cmdname_size,
    int        *out_argidx)
{
    if (out_cmdname && cmdname_size > 0)
    {
        out_cmdname[0] = '\0';
    }
    if (out_argidx)
    {
        *out_argidx = 0;
    }

    if (!line)
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    /* 1. Direct token prefix overrides */
    if (text)
    {
        if (strncmp(text, "${s.", 4) == 0 || strncmp(text, "@s.", 3) == 0)
        {
            return CLICOMPLETIONMODE_VARS_STREAM;
        }
        if (strncmp(text, "@fps.", 5) == 0)
        {
            return CLICOMPLETIONMODE_VARS_FPS;
        }
        if (strncmp(text, "@seq.", 5) == 0)
        {
            return CLICOMPLETIONMODE_VARS_SEQ;
        }
        if (text[0] == '$')
        {
            return CLICOMPLETIONMODE_VARS_ENV;
        }
        if (strncmp(text, "./", 2) == 0 || strncmp(text, "../", 3) == 0 ||
            text[0] == '/' || text[0] == '~')
        {
            return CLICOMPLETIONMODE_FILES;
        }
    }

    /* 2. Boundary / whitespace check */
    int prev_idx = start - 1;
    while (prev_idx >= 0 && (line[prev_idx] == ' ' || line[prev_idx] == '\t'))
    {
        prev_idx--;
    }
    if (prev_idx < 0)
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    char prev_c = line[prev_idx];
    if (prev_c == ';' || prev_c == '|' || prev_c == '&' ||
        prev_c == '(' || prev_c == '{' || prev_c == '\n' || prev_c == '`')
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_c == '>' || prev_c == '<')
    {
        return CLICOMPLETIONMODE_FILES;
    }

    /* Check keyword separators */
    if (prev_idx >= 1 && strncmp(&line[prev_idx - 1], "do", 2) == 0 &&
        (prev_idx - 1 == 0 || isspace((unsigned char) line[prev_idx - 2]) ||
         line[prev_idx - 2] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "then", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4]) ||
         line[prev_idx - 4] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "else", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4]) ||
         line[prev_idx - 4] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "elif", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4]) ||
         line[prev_idx - 4] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 1 && strncmp(&line[prev_idx - 1], "if", 2) == 0 &&
        (prev_idx - 1 == 0 || isspace((unsigned char) line[prev_idx - 2]) ||
         line[prev_idx - 2] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 4 && strncmp(&line[prev_idx - 4], "while", 5) == 0 &&
        (prev_idx - 4 == 0 || isspace((unsigned char) line[prev_idx - 5]) ||
         line[prev_idx - 5] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 4 && strncmp(&line[prev_idx - 4], "until", 5) == 0 &&
        (prev_idx - 4 == 0 || isspace((unsigned char) line[prev_idx - 5]) ||
         line[prev_idx - 5] == ';'))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 0 && line[prev_idx] == '!')
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 3 && strncmp(&line[prev_idx - 3], "time", 4) == 0 &&
        (prev_idx - 3 == 0 || isspace((unsigned char) line[prev_idx - 4])))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_idx >= 4 && strncmp(&line[prev_idx - 4], "watch", 5) == 0 &&
        (prev_idx - 4 == 0 || isspace((unsigned char) line[prev_idx - 5])))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    /* 3. Tree-sitter AST analysis */
    if (ts_parser != NULL)
    {
        TSTree *tree = ts_parser_parse_string(ts_parser, NULL, line, (uint32_t) strlen(line));
        if (tree != NULL)
        {
            TSNode root = ts_tree_root_node(tree);
            uint32_t cpos = start > 0 ? (uint32_t) start : 0;
            TSNode node = ts_node_descendant_for_byte_range(
                root, cpos > 0 ? cpos - 1 : 0, cpos);

            TSNode curr = node;
            bool in_expansion = false;
            bool in_redirect  = false;
            TSNode cmd_node   = { 0 };
            bool found_cmd    = false;

            while (!ts_node_is_null(curr))
            {
                const char *ntype = ts_node_type(curr);
                if (strcmp(ntype, "simple_expansion") == 0 ||
                    strcmp(ntype, "expansion") == 0)
                {
                    in_expansion = true;
                    break;
                }
                if (strcmp(ntype, "fps_variable") == 0)
                {
                    ts_tree_delete(tree);
                    return CLICOMPLETIONMODE_VARS_FPS;
                }
                if (strcmp(ntype, "seq_variable") == 0)
                {
                    ts_tree_delete(tree);
                    return CLICOMPLETIONMODE_VARS_SEQ;
                }
                if (strcmp(ntype, "stream_metadata") == 0)
                {
                    ts_tree_delete(tree);
                    return CLICOMPLETIONMODE_VARS_STREAM;
                }
                if (strcmp(ntype, "file_path") == 0 ||
                    strcmp(ntype, "io_redirect") == 0)
                {
                    in_redirect = true;
                    break;
                }
                if (!found_cmd && strcmp(ntype, "command") == 0)
                {
                    cmd_node  = curr;
                    found_cmd = true;
                }
                curr = ts_node_parent(curr);
            }

            if (in_expansion)
            {
                ts_tree_delete(tree);
                return CLICOMPLETIONMODE_VARS_ENV;
            }
            if (in_redirect)
            {
                ts_tree_delete(tree);
                return CLICOMPLETIONMODE_FILES;
            }

            if (found_cmd)
            {
                uint32_t ccount = ts_node_child_count(cmd_node);
                if (ccount > 0)
                {
                    TSNode first = ts_node_child(cmd_node, 0);
                    uint32_t fstart = ts_node_start_byte(first);
                    uint32_t fend   = ts_node_end_byte(first);
                    if (cpos <= fend)
                    {
                        ts_tree_delete(tree);
                        return CLICOMPLETIONMODE_COMMANDS;
                    }

                    if (out_cmdname && cmdname_size > 0)
                    {
                        uint32_t len = fend - fstart;
                        if (len >= cmdname_size)
                        {
                            len = (uint32_t) cmdname_size - 1;
                        }
                        strncpy(out_cmdname, line + fstart, len);
                        out_cmdname[len] = '\0';
                    }

                    int argi = 0;
                    for (uint32_t ci = 1; ci < ccount; ci++)
                    {
                        TSNode child = ts_node_child(cmd_node, ci);
                        uint32_t ch_start = ts_node_start_byte(child);
                        if (ch_start >= cpos)
                        {
                            break;
                        }
                        argi++;
                    }
                    if (out_argidx)
                    {
                        *out_argidx = (argi > 0) ? (argi - 1) : 0;
                    }

                    ts_tree_delete(tree);
                    return -1;
                }
            }

            ts_tree_delete(tree);
        }
    }

    return cli_determine_mode_lexical(
        line, start, text, out_cmdname, cmdname_size, out_argidx);
}

#else

// Stubs when USE_TREESITTER is not defined
/**
 * @brief Initialize treesitter syntax highlighting.
 *
 * Loads the milk grammar and sets up the
 * parser instance.
 */
int cli_ts_init(void)
{
    return 0;
}
/**
 * @brief Detect terminal color capability.
 *
 * Checks COLORTERM and TERM environment variables
 * to determine 256-color or truecolor support.
 */
int cli_ts_detect_color_level(void)
{
    return 1;
}
void cli_ts_cleanup(void)
{
}
void cli_ts_highlight_line(const char *line, int len, FILE *out)
{
    (void) len;
    fprintf(out, "%s", line);
    fflush(out);
}

int cli_ts_is_incomplete(const char *buffer)
{
    if (buffer == NULL || buffer[0] == '\0')
    {
        return 0;
    }

    int in_dquote = 0;
    int in_squote = 0;
    for (size_t i = 0; buffer[i] != '\0'; i++)
    {
        if (buffer[i] == '\\' && buffer[i + 1] != '\0' && !in_squote)
        {
            i++;
            continue;
        }
        if (buffer[i] == '"' && !in_squote)
        {
            in_dquote = !in_dquote;
        }
        else if (buffer[i] == '\'' && !in_dquote)
        {
            in_squote = !in_squote;
        }
    }
    if (in_dquote || in_squote)
    {
        return 1;
    }

    size_t len = strlen(buffer);
    while (len > 0 && (buffer[len - 1] == ' ' || buffer[len - 1] == '\t' ||
                       buffer[len - 1] == '\n' || buffer[len - 1] == '\r'))
    {
        len--;
    }
    if (len > 0)
    {
        if (buffer[len - 1] == '\\' && (len == 1 || buffer[len - 2] != '\\'))
        {
            return 1;
        }
        if (buffer[len - 1] == '|' && (len == 1 || buffer[len - 2] != '|'))
        {
            return 1;
        }
        if (len >= 2 && buffer[len - 1] == '&' && buffer[len - 2] == '&')
        {
            return 1;
        }
        if (len >= 2 && buffer[len - 1] == '|' && buffer[len - 2] == '|')
        {
            return 1;
        }
    }

    return 0;
}

int cli_ts_determine_completion_mode(
    const char *line,
    int         start,
    const char *text,
    char       *out_cmdname,
    size_t      cmdname_size,
    int        *out_argidx)
{
    return cli_determine_mode_lexical(
        line, start, text, out_cmdname, cmdname_size, out_argidx);
}

#endif

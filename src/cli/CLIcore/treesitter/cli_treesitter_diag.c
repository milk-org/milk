// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_treesitter_diag.c
 * @brief Real-time syntax diagnostics, incomplete statement checks, and AST completion context.
 *
 * Role & Goal:
 *   Provides AST + lexical syntax error and incomplete-construct diagnostics
 *   (`cli_ts_get_diagnostic()`, `cli_ts_collect_error_spans()`), multi-line
 *   continuation detection (`cli_ts_is_incomplete()`), and AST-assisted
 *   tab-completion mode determination (`cli_ts_determine_completion_mode()`).
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_entity_ns.h"
#include "CLIcore.h"
#include "cli_treesitter.h"
#include "cli_treesitter_internal.h"

#ifdef USE_TREESITTER

/**
 * @brief Check if an AST keyword type (`for`, `if`, `case`, `{`) belongs to an unclosed block in @source.
 */
static bool is_unclosed_block_keyword(const char *ctype,
                                      const char *source,
                                      char       *out_msg,
                                      size_t      msg_sz)
{
    if (strcmp(ctype, "for") == 0 || strcmp(ctype, "while") == 0 || strcmp(ctype, "until") == 0 ||
        strcmp(ctype, "do") == 0)
    {
        int starters = cli_ts_count_shell_keyword(source, "for") +
                       cli_ts_count_shell_keyword(source, "while") +
                       cli_ts_count_shell_keyword(source, "until");
        int closers  = cli_ts_count_shell_keyword(source, "done");
        if (starters > closers)
        {
            if (out_msg)
            {
                snprintf(out_msg, msg_sz, "loop open: missing 'done'");
            }
            return true;
        }
    }
    else if (strcmp(ctype, "if") == 0 || strcmp(ctype, "then") == 0 || strcmp(ctype, "elif") == 0 ||
             strcmp(ctype, "else") == 0)
    {
        int starters = cli_ts_count_shell_keyword(source, "if");
        int closers  = cli_ts_count_shell_keyword(source, "fi");
        if (starters > closers)
        {
            if (out_msg)
            {
                snprintf(out_msg, msg_sz, "if block open: missing 'fi'");
            }
            return true;
        }
    }
    else if (strcmp(ctype, "case") == 0)
    {
        int starters = cli_ts_count_shell_keyword(source, "case");
        int closers  = cli_ts_count_shell_keyword(source, "esac");
        if (starters > closers)
        {
            if (out_msg)
            {
                snprintf(out_msg, msg_sz, "case block open: missing 'esac'");
            }
            return true;
        }
    }
    else if (strcmp(ctype, "function") == 0 || strcmp(ctype, "{") == 0)
    {
        int o = 0;
        int c = 0;
        cli_ts_count_block_braces(source, &o, &c);
        if (o > c)
        {
            if (out_msg)
            {
                snprintf(out_msg, msg_sz, "unclosed '{': missing '}'");
            }
            return true;
        }
    }
    return false;
}

/**
 * @brief True if @mtype is a statement terminator; recovery inserts one before the missing closer of
 *        `for i in 1; do cmd` (no `;` after the last command), so it stands for that closer.
 */
static bool is_terminator_type(const char *mtype)
{
    return strcmp(mtype, ";") == 0 || strcmp(mtype, "\n") == 0 || strcmp(mtype, "&") == 0;
}

/**
 * @brief Check whether any loop, if, case or `{` block of @source is unclosed (message as for one block).
 */
static bool has_unclosed_block(const char *source, char *out_msg, size_t msg_sz)
{
    return is_unclosed_block_keyword("for", source, out_msg, msg_sz) ||
           is_unclosed_block_keyword("if", source, out_msg, msg_sz) ||
           is_unclosed_block_keyword("case", source, out_msg, msg_sz) ||
           is_unclosed_block_keyword("{", source, out_msg, msg_sz);
}

/**
 * is_open_block_error - Check if an AST ERROR node stems from an unclosed compound block.
 * @node:   Tree-sitter ERROR node to inspect.
 * @source: Full source code buffer being parsed.
 *
 * Iterates over the direct children of @node and checks whether any child is
 * an unclosed block keyword (`for`, `while`, `until`, `do`, `if`, `then`,
 * `elif`, `else`, `case`, `function`, or `{`) in @source.
 *
 * Return: true if @node represents an unclosed block rather than a syntax error, or false otherwise.
 */
static bool is_open_block_error(TSNode node, const char *source)
{
    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        TSNode child = ts_node_child(node, i);
        if (is_unclosed_block_keyword(ts_node_type(child), source, NULL, 0))
        {
            return true;
        }
    }
    return false;
}

/**
 * cli_ts_collect_error_spans - Append underline highlight spans for AST syntax error nodes.
 * @node:      Current Tree-sitter AST node being traversed.
 * @source:    Original command line string.
 * @linelen:   Length of @source in bytes.
 * @spans:     Destination array of `HighlightSpan` entries.
 * @num_spans: In/out count of spans populated in @spans.
 * @max_spans: Maximum capacity of @spans.
 * @color_lvl: Terminal color capability level (1 for 16-color, 2 for 256-color).
 *
 * Recursively traverses @node and records red-underlined error spans for
 * `ERROR` nodes while suppressing false positives on unclosed multi-line
 * blocks, continuation closer/transition lines, `@` tokens, and trailing `$`
 * characters being typed.
 *
 * Return: Nothing.
 */
void cli_ts_collect_error_spans(TSNode         node,
                                const char    *source,
                                size_t         linelen,
                                HighlightSpan *spans,
                                int           *num_spans,
                                int            max_spans,
                                int            color_lvl)
{
    const char *type = ts_node_type(node);
    if (strcmp(type, "ERROR") == 0)
    {
        if (is_open_block_error(node, source))
        {
            return;
        }
        if (cli_is_continuation_prompt())
        {
            const char *start = source;
            while (*start == ' ' || *start == '\t')
            {
                start++;
            }
            if (cli_ts_is_closer_token_line(start) || cli_ts_is_transition_token_line(start))
            {
                return;
            }
        }

        if (*num_spans < max_spans)
        {
            uint32_t sb = ts_node_start_byte(node);
            uint32_t eb = ts_node_end_byte(node);
            if (eb > sb)
            {
                if (source[sb] == '@')
                {
                    return;
                }
                spans[*num_spans].start_byte = sb;
                spans[*num_spans].end_byte   = eb;
                if (eb >= linelen && source[sb] == '$' && eb - sb == 1)
                {
                    spans[*num_spans].color = (color_lvl >= 2) ? "\033[38;5;178m" : "\033[33m";
                }
                else
                {
                    spans[*num_spans].color = (color_lvl >= 2) ? "\033[4;38;5;203m" : "\033[4;31m";
                }
                spans[*num_spans].priority = 1;
                (*num_spans)++;
            }
        }
        return;
    }

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        cli_ts_collect_error_spans(ts_node_child(node, i), source, linelen, spans, num_spans,
                                   max_spans, color_lvl);
    }
}

/**
 * handle_missing_node_diag - Populate a syntax diagnostic from a missing Tree-sitter AST node.
 * @node:   Missing Tree-sitter AST node (`ts_node_is_missing(node) == true`).
 * @source: Full source buffer being diagnosed.
 * @diag:   Output diagnostic structure to populate.
 *
 * Classifies missing closers (`done`, `fi`, `esac`, `then`, `do`, `}`, `)`,
 * `]`, `]]`) or missing statement terminators inside unclosed blocks as
 * `CLI_DIAG_SEVERITY_INFO` hints, and other missing tokens as
 * `CLI_DIAG_SEVERITY_ERROR`.
 *
 * Return: true if @diag was populated with a valid diagnostic, or false if suppressed.
 */
static bool handle_missing_node_diag(TSNode node, const char *source, CLI_SYNTAX_DIAG *diag)
{
    const char *mtype = ts_node_type(node);
    if (strcmp(mtype, "done") == 0)
    {
        if (!is_unclosed_block_keyword("for", source, diag->message, sizeof(diag->message)))
        {
            return false;
        }
        diag->severity = CLI_DIAG_SEVERITY_INFO;
    }
    else if (strcmp(mtype, "fi") == 0)
    {
        if (!is_unclosed_block_keyword("if", source, diag->message, sizeof(diag->message)))
        {
            return false;
        }
        diag->severity = CLI_DIAG_SEVERITY_INFO;
    }
    else if (strcmp(mtype, "esac") == 0)
    {
        if (!is_unclosed_block_keyword("case", source, diag->message, sizeof(diag->message)))
        {
            return false;
        }
        diag->severity = CLI_DIAG_SEVERITY_INFO;
    }
    else if (is_terminator_type(mtype))
    {
        if (!has_unclosed_block(source, diag->message, sizeof(diag->message)))
        {
            return false;
        }
        diag->severity = CLI_DIAG_SEVERITY_INFO;
    }
    else if (strcmp(mtype, "then") == 0)
    {
        diag->severity = CLI_DIAG_SEVERITY_INFO;
        snprintf(diag->message, sizeof(diag->message), "condition open: missing 'then'");
    }
    else if (strcmp(mtype, "do") == 0)
    {
        diag->severity = CLI_DIAG_SEVERITY_INFO;
        snprintf(diag->message, sizeof(diag->message), "loop open: missing 'do'");
    }
    else if (strcmp(mtype, "}") == 0)
    {
        int o = cli_ts_count_shell_keyword(source, "{");
        int c = cli_ts_count_shell_keyword(source, "}");
        if (o <= c)
        {
            return false;
        }
        diag->severity = CLI_DIAG_SEVERITY_INFO;
        snprintf(diag->message, sizeof(diag->message), "unclosed '{': missing '}'");
    }
    else if (strcmp(mtype, ")") == 0)
    {
        int o = cli_ts_count_shell_keyword(source, "(");
        int c = cli_ts_count_shell_keyword(source, ")");
        if (o <= c)
        {
            return false;
        }
        diag->severity = CLI_DIAG_SEVERITY_INFO;
        snprintf(diag->message, sizeof(diag->message), "unclosed '(': missing ')'");
    }
    else if (strcmp(mtype, "]") == 0 || strcmp(mtype, "]]") == 0)
    {
        diag->severity = CLI_DIAG_SEVERITY_INFO;
        snprintf(diag->message, sizeof(diag->message), "unclosed test bracket: missing '%s'",
                 mtype);
    }
    else
    {
        diag->severity = CLI_DIAG_SEVERITY_ERROR;
        snprintf(diag->message, sizeof(diag->message), "unexpected token");
    }

    diag->start_byte = ts_node_start_byte(node);
    diag->end_byte   = ts_node_end_byte(node);
    snprintf(diag->token, sizeof(diag->token), "%s", mtype);
    return true;
}

/**
 * find_first_error - Locate the first actionable syntax error or incomplete construct in an AST.
 * @node:   Current Tree-sitter AST node being inspected.
 * @source: Original command line string.
 * @diag:   Output diagnostic structure populated when an issue is found.
 *
 * Checks @node for missing tokens via handle_missing_node_diag() or `ERROR`
 * nodes (distinguishing unclosed blocks, unclosed `${`/`$(`/`$((` expansions,
 * missing redirection operands, and unexpected tokens), then recurses into
 * child nodes until the first diagnostic is produced.
 *
 * Return: true if a diagnostic was recorded in @diag, or false otherwise.
 */
static bool find_first_error(TSNode node, const char *source, CLI_SYNTAX_DIAG *diag)
{
    if (ts_node_is_missing(node))
    {
        return handle_missing_node_diag(node, source, diag);
    }

    const char *type = ts_node_type(node);
    if (strcmp(type, "ERROR") == 0)
    {
        uint32_t count = ts_node_child_count(node);
        for (uint32_t i = 0; i < count; i++)
        {
            TSNode      child = ts_node_child(node, i);
            const char *ctype = ts_node_type(child);
            if ((strcmp(ctype, "for") == 0 || strcmp(ctype, "while") == 0 ||
                 strcmp(ctype, "until") == 0 || strcmp(ctype, "do") == 0 ||
                 strcmp(ctype, "if") == 0 || strcmp(ctype, "then") == 0 ||
                 strcmp(ctype, "elif") == 0 || strcmp(ctype, "else") == 0 ||
                 strcmp(ctype, "case") == 0) &&
                is_unclosed_block_keyword(ctype, source, diag->message, sizeof(diag->message)))
            {
                diag->severity   = CLI_DIAG_SEVERITY_INFO;
                diag->start_byte = ts_node_start_byte(child);
                diag->end_byte   = ts_node_end_byte(child);
                snprintf(diag->token, sizeof(diag->token), "%s", ctype);
                return true;
            }
        }

        uint32_t sb      = ts_node_start_byte(node);
        uint32_t eb      = ts_node_end_byte(node);
        size_t   linelen = strlen(source);

        diag->start_byte = sb;
        diag->end_byte   = eb;
        int toklen       = (int) (eb - sb);
        if (toklen > (int) sizeof(diag->token) - 1)
        {
            toklen = (int) sizeof(diag->token) - 1;
        }
        if (toklen > 0)
        {
            memcpy(diag->token, source + sb, (size_t) toklen);
            diag->token[toklen] = '\0';
        }
        else
        {
            diag->token[0] = '\0';
        }

        if (diag->token[0] == '@')
        {
            return false;
        }

        if (eb >= linelen)
        {
            if (strstr(diag->token, "${") != NULL)
            {
                diag->severity = CLI_DIAG_SEVERITY_INFO;
                snprintf(diag->message, sizeof(diag->message), "unclosed '${': missing '}'");
                return true;
            }
            if (strstr(diag->token, "$(") != NULL)
            {
                diag->severity = CLI_DIAG_SEVERITY_INFO;
                snprintf(diag->message, sizeof(diag->message), "unclosed '$(': missing ')'");
                return true;
            }
            if (strstr(diag->token, "((") != NULL)
            {
                diag->severity = CLI_DIAG_SEVERITY_INFO;
                snprintf(diag->message, sizeof(diag->message),
                         "unclosed arithmetic '$((': missing '))'");
                return true;
            }
            if (diag->token[0] == '@' || strncmp(diag->token, "${s.", 4) == 0 ||
                strncmp(diag->token, "${stream.", 9) == 0 ||
                strncmp(diag->token, "${fps.", 6) == 0 || strncmp(diag->token, "${f.", 4) == 0 ||
                strncmp(diag->token, "${proc.", 7) == 0 ||
                strncmp(diag->token, "${procinfo.", 11) == 0 ||
                strncmp(diag->token, "${p.", 4) == 0)
            {
                return false;
            }
        }

        diag->severity = CLI_DIAG_SEVERITY_ERROR;
        if (strcmp(diag->token, "<") == 0 || strcmp(diag->token, ">") == 0 ||
            strcmp(diag->token, ">>") == 0)
        {
            snprintf(diag->message, sizeof(diag->message), "missing redirection file operand");
        }
        else if (diag->token[0] != '\0')
        {
            snprintf(diag->message, sizeof(diag->message), "unexpected token '%s'", diag->token);
        }
        else
        {
            snprintf(diag->message, sizeof(diag->message), "syntax error");
        }
        return true;
    }

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        if (find_first_error(ts_node_child(node, i), source, diag))
        {
            return true;
        }
    }
    return false;
}

#endif /* USE_TREESITTER */

/**
 * @brief Check if @buffer ends with a line-continuation operator (`\`, `|`, `&&`, `||`).
 */
static int has_trailing_continuation_op(const char *line, CLI_SYNTAX_DIAG *diag)
{
    size_t len = strlen(line);
    while (len > 0 && isspace((unsigned char) line[len - 1]))
    {
        len--;
    }
    if (len == 0)
    {
        return 0;
    }

    if (line[len - 1] == '\\' && (len == 1 || line[len - 2] != '\\'))
    {
        if (diag)
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 1);
            diag->end_byte   = (uint32_t) len;
            snprintf(diag->message, sizeof(diag->message), "trailing '\\' (line continuation)");
            snprintf(diag->token, sizeof(diag->token), "\\");
        }
        return 1;
    }
    if (line[len - 1] == '|' && (len == 1 || line[len - 2] != '|'))
    {
        if (diag)
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 1);
            diag->end_byte   = (uint32_t) len;
            snprintf(diag->message, sizeof(diag->message),
                     "trailing pipe '|' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "|");
        }
        return 1;
    }
    if (len >= 2 && line[len - 1] == '&' && line[len - 2] == '&')
    {
        if (diag)
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 2);
            diag->end_byte   = (uint32_t) len;
            snprintf(diag->message, sizeof(diag->message), "trailing '&&' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "&&");
        }
        return 1;
    }
    if (len >= 2 && line[len - 1] == '|' && line[len - 2] == '|')
    {
        if (diag)
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 2);
            diag->end_byte   = (uint32_t) len;
            snprintf(diag->message, sizeof(diag->message), "trailing '||' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "||");
        }
        return 1;
    }
    return 0;
}

/**
 * cli_ts_get_diagnostic - Inspect a command line for syntax errors or incomplete constructs.
 * @line: Null-terminated command line string to analyze.
 * @diag: Output structure populated with diagnostic severity, span, and message.
 *
 * Checks for unclosed quotes, unclosed `@{` calculator expressions, trailing
 * continuation operators (`\`, `|`, `&&`, `||`), and Tree-sitter AST errors or
 * missing block closers.
 *
 * Return: 1 if a diagnostic was recorded in @diag, or 0 if no issue was found.
 */
int cli_ts_get_diagnostic(const char *line, CLI_SYNTAX_DIAG *diag)
{
    if (diag)
    {
        memset(diag, 0, sizeof(*diag));
    }
    if (!line || line[0] == '\0' || !diag)
    {
        return 0;
    }

    int in_dquote = 0;
    int in_squote = 0;
    int q_start   = -1;
    for (int i = 0; line[i] != '\0'; i++)
    {
        if (line[i] == '\\' && line[i + 1] != '\0' && !in_squote)
        {
            i++;
            continue;
        }
        if (line[i] == '"' && !in_squote)
        {
            if (!in_dquote)
            {
                q_start = i;
            }
            in_dquote = !in_dquote;
        }
        else if (line[i] == '\'' && !in_dquote)
        {
            if (!in_squote)
            {
                q_start = i;
            }
            in_squote = !in_squote;
        }
    }
    if (in_dquote || in_squote)
    {
        diag->severity   = CLI_DIAG_SEVERITY_INFO;
        diag->start_byte = (uint32_t) q_start;
        diag->end_byte   = (uint32_t) strlen(line);
        snprintf(diag->message, sizeof(diag->message),
                 in_dquote ? "unclosed double quote \"" : "unclosed single quote '");
        snprintf(diag->token, sizeof(diag->token), in_dquote ? "\"" : "'");
        return 1;
    }

    /* Unclosed '@{' inline calculator expression outside single quotes */
    {
        int sq       = 0;
        int at_start = -1;
        int at_depth = 0;
        for (int i = 0; line[i] != '\0'; i++)
        {
            if (line[i] == '\\' && line[i + 1] != '\0' && !sq)
            {
                i++;
                continue;
            }
            if (line[i] == '\'')
            {
                sq = !sq;
                continue;
            }
            if (!sq)
            {
                if (at_depth == 0 && line[i] == '@' && line[i + 1] == '{')
                {
                    at_start = i;
                    at_depth = 1;
                    i++;
                }
                else if (at_depth > 0)
                {
                    if (line[i] == '{')
                    {
                        at_depth++;
                    }
                    else if (line[i] == '}')
                    {
                        at_depth--;
                    }
                }
            }
        }
        if (at_depth > 0 && at_start >= 0)
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) at_start;
            diag->end_byte   = (uint32_t) strlen(line);
            snprintf(diag->message, sizeof(diag->message), "unclosed '@{': missing '}'");
            snprintf(diag->token, sizeof(diag->token), "@{");
            return 1;
        }
    }

    if (has_trailing_continuation_op(line, diag))
    {
        return 1;
    }

#ifdef USE_TREESITTER
    if (!cli_ts_get_parser() && cli_ts_init() != 0)
    {
        return 0;
    }
    if (cli_is_continuation_prompt())
    {
        const char *start = line;
        while (*start == ' ' || *start == '\t')
        {
            start++;
        }
        if (cli_ts_is_closer_token_line(start) || cli_ts_is_transition_token_line(start))
        {
            return 0;
        }
    }

    TSTree *tree = cli_ts_parse_normalized_tree(line, strlen(line));
    if (!tree)
    {
        return 0;
    }
    TSNode root = ts_tree_root_node(tree);
    if (ts_node_has_error(root))
    {
        find_first_error(root, line, diag);
    }
    ts_tree_delete(tree);
    return (diag->severity != CLI_DIAG_SEVERITY_NONE);
#else
    size_t len = strlen(line);
    while (len > 0 && isspace((unsigned char) line[len - 1]))
    {
        len--;
    }
    if (len > 0 && (line[len - 1] == '<' || line[len - 1] == '>'))
    {
        diag->severity   = CLI_DIAG_SEVERITY_ERROR;
        diag->start_byte = (uint32_t) (len - 1);
        diag->end_byte   = (uint32_t) len;
        snprintf(diag->message, sizeof(diag->message), "missing redirection file operand");
        snprintf(diag->token, sizeof(diag->token), "%c", line[len - 1]);
        return 1;
    }
    int obrace = 0;
    int cbrace = 0;
    cli_ts_count_block_braces(line, &obrace, &cbrace);
    if (obrace > cbrace)
    {
        diag->severity = CLI_DIAG_SEVERITY_INFO;
        snprintf(diag->message, sizeof(diag->message), "unclosed '{': missing '}'");
        snprintf(diag->token, sizeof(diag->token), "}");
        return 1;
    }
    return 0;
#endif
}

#ifdef USE_TREESITTER
/** entity_completion_mode() - completion mode for the entity path at @at ("s.cam.x", "f.loop.gain"). */
static int entity_completion_mode(const char *at)
{
    switch (milk_entity_ns_kind(at, NULL))
    {
    case MILK_NS_FPS:
        return CLICOMPLETIONMODE_VARS_FPS;
    case MILK_NS_PROC:
        return CLICOMPLETIONMODE_VARS_PROC;
    default:
        return CLICOMPLETIONMODE_VARS_STREAM;
    }
}
#endif

/**
 * cli_ts_determine_completion_mode - Determine the Readline tab-completion context at @start.
 * @line:         Full input command line buffer (`rl_line_buffer`).
 * @start:        Byte offset in @line where the token being completed begins.
 * @text:         Prefix string of the token being completed.
 * @out_cmdname:  Optional output buffer receiving the active command name when completing args.
 * @cmdname_size: Capacity of @out_cmdname in bytes.
 * @out_argidx:   Optional output pointer receiving the 0-based argument index.
 *
 * Checks fast lexical prefix/boundary rules first, then walks the normalized
 * Tree-sitter AST around @start to identify variable expansions, entity paths,
 * I/O redirections, or command/argument positions, falling back to
 * cli_determine_mode_lexical().
 *
 * Return: A `CLICOMPLETIONMODE_*` constant (>= 0), or -1 when completing arguments
 *         for @out_cmdname at @out_argidx.
 */
int cli_ts_determine_completion_mode(const char *line,
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

    int early = cli_ts_check_prefix_or_boundary_mode(line, start, text);
    if (early >= 0)
    {
        return early;
    }

#ifdef USE_TREESITTER
    if (cli_ts_get_parser() != NULL)
    {
        TSTree *tree = cli_ts_parse_normalized_tree(line, strlen(line));
        if (tree != NULL)
        {
            TSNode   root = ts_tree_root_node(tree);
            uint32_t cpos = start > 0 ? (uint32_t) start : 0;
            TSNode   node = ts_node_descendant_for_byte_range(root, cpos > 0 ? cpos - 1 : 0, cpos);

            TSNode curr         = node;
            bool   in_expansion = false;
            bool   in_redirect  = false;
            TSNode cmd_node     = { 0 };
            bool   found_cmd    = false;

            while (!ts_node_is_null(curr))
            {
                const char *ntype = ts_node_type(curr);
                if (strcmp(ntype, "simple_expansion") == 0 || strcmp(ntype, "expansion") == 0)
                {
                    in_expansion = true;
                    break;
                }
                if (strcmp(ntype, "entity_path") == 0)
                {
                    int mode = entity_completion_mode(line + ts_node_start_byte(curr));
                    ts_tree_delete(tree);
                    return mode;
                }
                if (strcmp(ntype, "file_path") == 0 || strcmp(ntype, "io_redirect") == 0)
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
                    TSNode   first  = ts_node_child(cmd_node, 0);
                    uint32_t fstart = ts_node_start_byte(first);
                    uint32_t fend   = ts_node_end_byte(first);
                    if (cpos <= fend)
                    {
                        ts_tree_delete(tree);
                        return CLICOMPLETIONMODE_COMMANDS;
                    }
                    if (out_cmdname && cmdname_size > 0)
                    {
                        uint32_t clen = fend - fstart;
                        if (clen >= cmdname_size)
                        {
                            clen = (uint32_t) cmdname_size - 1;
                        }
                        strncpy(out_cmdname, line + fstart, clen);
                        out_cmdname[clen] = '\0';
                    }
                    int argi = 0;
                    for (uint32_t ci = 1; ci < ccount; ci++)
                    {
                        TSNode child = ts_node_child(cmd_node, ci);
                        if (ts_node_start_byte(child) >= cpos)
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
#endif

    return cli_determine_mode_lexical(line, start, text, out_cmdname, cmdname_size, out_argidx);
}

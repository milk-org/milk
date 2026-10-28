// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_treesitter_lexical.c
 * @brief Lexical analyzers for completion context, delimiter/keyword matching, and block counting.
 *
 * Role & Goal:
 *   Provides fast, allocation-light lexical scanners used both as primary
 *   pre-checks and as fallbacks when Tree-sitter is unavailable or encountering
 *   in-progress edits:
 *   - Token prefix & statement boundary classification (`cli_ts_check_prefix_or_boundary_mode()`)
 *   - Lexical command/argument completion context (`cli_determine_mode_lexical()`)
 *   - Quote/comment-masked delimiter & keyword pair matching (`find_match_pair_lexical()`)
 *   - Quote, shell keyword, and block brace counters for diagnostics and auto-indent.
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_treesitter_internal.h"
#include "milk_entity_ns.h"

/**
 * is_word_char - Check if a character is an ASCII alphanumeric or underscore identifier byte.
 * @c: Character byte to test.
 *
 * Used by lexical keyword scanners to verify word boundaries around reserved
 * shell keywords.
 *
 * Return: true if @c is alphanumeric or `'_'`, or false otherwise.
 */
static bool is_word_char(char c)
{
    return isalnum((unsigned char) c) || c == '_';
}

/**
 * is_word_at - Check if a standalone identifier word matches at a given byte offset.
 * @line: Input command line string.
 * @len:  Length of @line in bytes.
 * @pos:  Zero-based byte offset in @line to test.
 * @word: Null-terminated keyword string to match.
 * @wlen: Optional output pointer receiving `strlen(word)`.
 *
 * Verifies that @word occurs at `line[pos]` and is neither preceded nor
 * followed by a word character (`is_word_char()`).
 *
 * Return: true if @word matches as a standalone word at @pos, or false otherwise.
 */
static bool is_word_at(const char *line, int len, int pos, const char *word, int *wlen)
{
    int wl = (int) strlen(word);
    if (wlen != NULL)
    {
        *wlen = wl;
    }
    if (pos < 0 || pos + wl > len)
    {
        return false;
    }
    if (strncmp(&line[pos], word, (size_t) wl) != 0)
    {
        return false;
    }
    if (pos > 0 && is_word_char(line[pos - 1]))
    {
        return false;
    }
    if (pos + wl < len && is_word_char(line[pos + wl]))
    {
        return false;
    }
    return true;
}

/**
 * ends_with_sep_keyword - Check if the token ending at @prev_idx is a command-introducing keyword.
 * @line:       Input command line string.
 * @prev_idx:   Index of the last non-whitespace byte preceding the current token.
 * @kw:         Null-terminated keyword string to test (e.g., `"do"`, `"then"`, `"time"`).
 * @allow_semi: Non-zero if `;` immediately before @kw is also accepted as a boundary.
 *
 * Tests whether `line[0..prev_idx]` ends with @kw preceded by the start of
 * the line, whitespace, or (when @allow_semi is non-zero) a semicolon.
 *
 * Return: true if @kw ends at @prev_idx on a valid boundary, or false otherwise.
 */
static bool ends_with_sep_keyword(const char *line, int prev_idx, const char *kw, int allow_semi)
{
    int kwlen = (int) strlen(kw);
    int start = prev_idx - (kwlen - 1);
    if (start < 0 || strncmp(&line[start], kw, (size_t) kwlen) != 0)
    {
        return false;
    }
    if (start == 0 || isspace((unsigned char) line[start - 1]))
    {
        return true;
    }
    return (allow_semi && line[start - 1] == ';');
}

/**
 * @brief Check direct token prefixes (entity paths, `@S:`, `$`, paths) and statement boundaries before @start.
 *
 * @return A `CLICOMPLETIONMODE_*` constant (>= 0) if resolved immediately, or -1 to continue.
 */
int cli_ts_check_prefix_or_boundary_mode(const char *line, int start, const char *text)
{
    if (!line)
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    if (text)
    {
        const char *ent = text + ((strncmp(text, "@{", 2) == 0) ? 2 : 0); /* s.cam.x, @{s.cam.x */
        switch (milk_entity_ns_kind(ent, NULL))
        {
        case MILK_NS_STREAM:
        case MILK_NS_LOCAL:
            return CLICOMPLETIONMODE_VARS_STREAM;
        case MILK_NS_FPS:
            return CLICOMPLETIONMODE_VARS_FPS;
        case MILK_NS_PROC:
            return CLICOMPLETIONMODE_VARS_PROC;
        default:
            break;
        }
        if (start > 0 && text[0] == '@' && strchr("SLFENT", text[1]) != NULL)
        {
            return CLICOMPLETIONMODE_IMAGES;
        }
        if (text[0] == '$')
        {
            return CLICOMPLETIONMODE_VARS_ENV;
        }
        if (strncmp(text, "./", 2) == 0 || strncmp(text, "../", 3) == 0 || text[0] == '/' ||
            text[0] == '~')
        {
            return CLICOMPLETIONMODE_FILES;
        }
    }

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
    if (prev_c == ';' || prev_c == '|' || prev_c == '&' || prev_c == '(' || prev_c == '{' ||
        prev_c == '\n' || prev_c == '`' || prev_c == '!')
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }
    if (prev_c == '>' || prev_c == '<')
    {
        return CLICOMPLETIONMODE_FILES;
    }

    static const char *const semi_kws[] = { "do", "then",  "else",  "elif",
                                            "if", "while", "until", NULL };
    for (int i = 0; semi_kws[i] != NULL; i++)
    {
        if (ends_with_sep_keyword(line, prev_idx, semi_kws[i], 1))
        {
            return CLICOMPLETIONMODE_COMMANDS;
        }
    }
    if (ends_with_sep_keyword(line, prev_idx, "time", 0) ||
        ends_with_sep_keyword(line, prev_idx, "watch", 0))
    {
        return CLICOMPLETIONMODE_COMMANDS;
    }

    return -1;
}

/**
 * @brief Lexical fallback for determining completion mode and active command.
 */
int cli_determine_mode_lexical(const char *line,
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
            if (c == ';' || c == '|' || c == '&' || c == '(' || c == '{' || c == '\n' || c == '`')
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
    while (*p && (p - line) < start && !isspace((unsigned char) *p) && *p != ';' && *p != '|' &&
           *p != '&' && *p != '(' && *p != ')')
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
            else if (!in_word)
            {
                arg_count++;
                in_word = 1;
            }
        }
    }

    if (out_argidx)
    {
        *out_argidx = (arg_count > 0) ? (arg_count - 1) : 0;
    }
    return -1;
}

/**
 * build_quote_comment_mask - Classify each byte of @line by quote or comment state.
 * @line: Input command line string.
 * @len:  Length of @line in bytes.
 * @mask: Output array of @len bytes receiving the lexical state per byte.
 *
 * Populates `mask[i]` with 0 for unquoted code, 1 for single-quoted text,
 * 2 for double-quoted text, and 3 for `#` comments so delimiter and keyword
 * matchers can skip non-code regions in O(1) time per byte.
 *
 * Return: Nothing.
 */
static void build_quote_comment_mask(const char *line, int len, uint8_t *mask)
{
    int in_sq  = 0;
    int in_dq  = 0;
    int in_cmt = 0;
    for (int i = 0; i < len; i++)
    {
        char c = line[i];
        if (in_cmt)
        {
            mask[i] = 3;
            if (c == '\n')
            {
                in_cmt = 0;
            }
            continue;
        }
        if (c == '\\' && i + 1 < len && !in_sq)
        {
            mask[i]     = in_dq ? 2 : 0;
            mask[i + 1] = in_dq ? 2 : 0;
            i++;
            continue;
        }
        if (c == '\'' && !in_dq)
        {
            in_sq   = !in_sq;
            mask[i] = 1;
            continue;
        }
        if (c == '"' && !in_sq)
        {
            in_dq   = !in_dq;
            mask[i] = 2;
            continue;
        }
        if (c == '#' && !in_sq && !in_dq)
        {
            in_cmt  = 1;
            mask[i] = 3;
            continue;
        }
        mask[i] = in_sq ? 1 : (in_dq ? 2 : 0);
    }
}

/**
 * set_match_pair - Populate a `CLI_MATCH_PAIR` structure with matched token spans.
 * @pair:   Output match pair structure.
 * @tstart: Start byte offset of the token under the cursor.
 * @tend:   End byte offset of the token under the cursor.
 * @mstart: Start byte offset of the matching partner token.
 * @mend:   End byte offset of the matching partner token.
 *
 * Stores the byte ranges and sets `pair->has_match = true`.
 *
 * Return: Nothing.
 */
static void set_match_pair(CLI_MATCH_PAIR *pair, int tstart, int tend, int mstart, int mend)
{
    pair->token_start = (uint32_t) tstart;
    pair->token_end   = (uint32_t) tend;
    pair->match_start = (uint32_t) mstart;
    pair->match_end   = (uint32_t) mend;
    pair->has_match   = true;
}

/**
 * match_single_char_forward - Scan forward for the matching closing delimiter of @open_c.
 * @line:    Input command line string.
 * @len:     Length of @line in bytes.
 * @mask:    Per-byte quote/comment mask from build_quote_comment_mask().
 * @pos:     Byte offset of the opening delimiter @open_c.
 * @open_c:  Opening delimiter character (`'('`, `'['`, or `'{'`).
 * @close_c: Closing delimiter character (`')'`, `']'`, or `'}'`).
 * @pair:    Output match pair structure populated when a match is found.
 *
 * Tracks nesting depth across unquoted bytes (`mask[i] == 0`) after @pos and
 * records the matching closing delimiter when depth reaches zero.
 *
 * Return: true if a matching closing delimiter was found, or false otherwise.
 */
static bool match_single_char_forward(const char     *line,
                                      int             len,
                                      const uint8_t  *mask,
                                      int             pos,
                                      char            open_c,
                                      char            close_c,
                                      CLI_MATCH_PAIR *pair)
{
    int depth = 1;
    for (int i = pos + 1; i < len; i++)
    {
        if (mask[i] != 0)
        {
            continue;
        }
        if (line[i] == open_c)
        {
            depth++;
        }
        else if (line[i] == close_c)
        {
            depth--;
            if (depth == 0)
            {
                set_match_pair(pair, pos, pos + 1, i, i + 1);
                return true;
            }
        }
    }
    return false;
}

/**
 * match_single_char_backward - Scan backward for the matching opening delimiter of @close_c.
 * @line:                  Input command line string.
 * @mask:                  Per-byte quote/comment mask from build_quote_comment_mask().
 * @pos:                   Byte offset of the closing delimiter @close_c.
 * @open_c:                Opening delimiter character (`'('`, `'['`, or `'{'`).
 * @close_c:               Closing delimiter character (`')'`, `']'`, or `'}'`).
 * @allow_dq_mask:         Non-zero to allow matching inside double-quoted expansions (`${...}`).
 * @include_dollar_prefix: Non-zero to include a preceding `'$'` in the matched opening span.
 * @pair:                  Output match pair structure populated when a match is found.
 *
 * Tracks nesting depth backward from `@pos - 1` and records the matching
 * opening delimiter (plus any leading `'$'` when @include_dollar_prefix is set)
 * when depth reaches zero.
 *
 * Return: true if a matching opening delimiter was found, or false otherwise.
 */
static bool match_single_char_backward(const char     *line,
                                       const uint8_t  *mask,
                                       int             pos,
                                       char            open_c,
                                       char            close_c,
                                       int             allow_dq_mask,
                                       int             include_dollar_prefix,
                                       CLI_MATCH_PAIR *pair)
{
    int depth = 1;
    for (int i = pos - 1; i >= 0; i--)
    {
        if (allow_dq_mask ? (mask[i] == 1 || mask[i] == 3) : (mask[i] != 0))
        {
            continue;
        }
        if (line[i] == close_c)
        {
            depth++;
        }
        else if (line[i] == open_c)
        {
            depth--;
            if (depth == 0)
            {
                int start_idx =
                    (include_dollar_prefix && i > 0 && line[i - 1] == '$') ? (i - 1) : i;
                set_match_pair(pair, pos, pos + 1, start_idx, i + 1);
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Try matching multi-character or single-character delimiters at @pos.
 */
static bool try_match_delimiter_at(const char     *line,
                                   int             len,
                                   const uint8_t  *mask,
                                   int             pos,
                                   CLI_MATCH_PAIR *pair)
{
    if (pos + 2 < len && strncmp(&line[pos], "$((", 3) == 0)
    {
        int depth = 1;
        for (int i = pos + 3; i + 1 < len; i++)
        {
            if (mask[i] == 1 || mask[i] == 3)
            {
                continue;
            }
            if (strncmp(&line[i], "$((", 3) == 0)
            {
                depth++;
                i += 2;
            }
            else if (strncmp(&line[i], "))", 2) == 0)
            {
                if (--depth == 0)
                {
                    set_match_pair(pair, pos, pos + 3, i, i + 2);
                    return true;
                }
                i++;
            }
        }
    }
    if (pos + 1 < len && strncmp(&line[pos], "))", 2) == 0)
    {
        int depth = 1;
        for (int i = pos - 1; i >= 0; i--)
        {
            if (mask[i] == 1 || mask[i] == 3)
            {
                continue;
            }
            if (i >= 2 && strncmp(&line[i - 2], "$((", 3) == 0)
            {
                if (--depth == 0)
                {
                    set_match_pair(pair, pos, pos + 2, i - 2, i + 1);
                    return true;
                }
                i -= 2;
            }
            else if (i >= 1 && strncmp(&line[i - 1], "))", 2) == 0)
            {
                depth++;
                i--;
            }
        }
    }
    if (pos + 1 < len && strncmp(&line[pos], "$(", 2) == 0)
    {
        int depth = 1;
        for (int i = pos + 2; i < len; i++)
        {
            if (mask[i] == 1 || mask[i] == 3)
            {
                continue;
            }
            if (strncmp(&line[i], "$(", 2) == 0)
            {
                depth++;
                i++;
            }
            else if (line[i] == '(')
            {
                depth++;
            }
            else if (line[i] == ')' && --depth == 0)
            {
                set_match_pair(pair, pos, pos + 2, i, i + 1);
                return true;
            }
        }
    }
    if ((pos + 1 < len && strncmp(&line[pos], "${", 2) == 0) ||
        (pos > 0 && line[pos - 1] == '$' && line[pos] == '{'))
    {
        int tstart = (line[pos] == '$') ? pos : (pos - 1);
        int depth  = 1;
        for (int i = tstart + 2; i < len; i++)
        {
            if (mask[i] == 1 || mask[i] == 3)
            {
                continue;
            }
            if (strncmp(&line[i], "${", 2) == 0)
            {
                depth++;
                i++;
            }
            else if (line[i] == '{')
            {
                depth++;
            }
            else if (line[i] == '}' && --depth == 0)
            {
                set_match_pair(pair, tstart, tstart + 2, i, i + 1);
                return true;
            }
        }
    }
    if (pos + 1 < len && strncmp(&line[pos], "[[", 2) == 0 && mask[pos] == 0)
    {
        for (int i = pos + 2; i + 1 < len; i++)
        {
            if (mask[i] == 0 && strncmp(&line[i], "]]", 2) == 0)
            {
                set_match_pair(pair, pos, pos + 2, i, i + 2);
                return true;
            }
        }
    }
    if (pos + 1 < len && strncmp(&line[pos], "]]", 2) == 0 && mask[pos] == 0)
    {
        for (int i = pos - 2; i >= 0; i--)
        {
            if (mask[i] == 0 && strncmp(&line[i], "[[", 2) == 0)
            {
                set_match_pair(pair, pos, pos + 2, i, i + 2);
                return true;
            }
        }
    }

    char c = line[pos];
    if (c == '(' && (pos == 0 || line[pos - 1] != '$') && mask[pos] == 0)
    {
        return match_single_char_forward(line, len, mask, pos, '(', ')', pair);
    }
    if (c == ')' && mask[pos] == 0)
    {
        return match_single_char_backward(line, mask, pos, '(', ')', 0, 1, pair);
    }
    if (c == '[' && (pos + 1 >= len || line[pos + 1] != '[') &&
        (pos == 0 || line[pos - 1] != '[') && mask[pos] == 0)
    {
        return match_single_char_forward(line, len, mask, pos, '[', ']', pair);
    }
    if (c == ']' && (pos + 1 >= len || line[pos + 1] != ']') &&
        (pos == 0 || line[pos - 1] != ']') && mask[pos] == 0)
    {
        return match_single_char_backward(line, mask, pos, '[', ']', 0, 0, pair);
    }
    if (c == '{' && (pos == 0 || line[pos - 1] != '$') && mask[pos] == 0)
    {
        return match_single_char_forward(line, len, mask, pos, '{', '}', pair);
    }
    if (c == '}')
    {
        return match_single_char_backward(line, mask, pos, '{', '}', 1, 1, pair);
    }
    return false;
}

/**
 * match_kw_forward - Scan forward for the matching closing keyword of a compound block.
 * @line:            Input command line string.
 * @len:             Length of @line in bytes.
 * @mask:            Per-byte quote/comment mask from build_quote_comment_mask().
 * @wstart:          Start byte offset of the opening keyword under the cursor.
 * @wend:            End byte offset of the opening keyword under the cursor.
 * @open_kw:         Opening keyword string (e.g., `"if"`, `"do"`, `"case"`).
 * @close_kw:        Closing keyword string (e.g., `"fi"`, `"done"`, `"esac"`).
 * @match_all_loops: Non-zero if `"for"`, `"while"`, and `"until"` all increment loop depth.
 * @pair:            Output match pair structure populated when a match is found.
 *
 * Scans unquoted words after @wend, incrementing depth on nested opening
 * keywords and decrementing on @close_kw until depth reaches zero.
 *
 * Return: true if the matching closing keyword was found, or false otherwise.
 */
static bool match_kw_forward(const char     *line,
                             int             len,
                             const uint8_t  *mask,
                             int             wstart,
                             int             wend,
                             const char     *open_kw,
                             const char     *close_kw,
                             int             match_all_loops,
                             CLI_MATCH_PAIR *pair)
{
    int depth    = 1;
    int close_wl = (int) strlen(close_kw);
    int dummy;
    for (int i = wend; i < len; i++)
    {
        if (mask[i] != 0)
        {
            continue;
        }
        bool is_open = match_all_loops ? (is_word_at(line, len, i, "for", &dummy) ||
                                          is_word_at(line, len, i, "while", &dummy) ||
                                          is_word_at(line, len, i, "until", &dummy))
                                       : is_word_at(line, len, i, open_kw, &dummy);
        if (is_open)
        {
            depth++;
        }
        else if (is_word_at(line, len, i, close_kw, &dummy))
        {
            if (--depth == 0)
            {
                set_match_pair(pair, wstart, wend, i, i + close_wl);
                return true;
            }
        }
    }
    return false;
}

/**
 * match_kw_backward - Scan backward for the matching opening keyword of a compound block.
 * @line:            Input command line string.
 * @len:             Length of @line in bytes.
 * @mask:            Per-byte quote/comment mask from build_quote_comment_mask().
 * @wstart:          Start byte offset of the closing keyword under the cursor.
 * @wend:            End byte offset of the closing keyword under the cursor.
 * @open_kw:         Opening keyword string (e.g., `"if"`, `"do"`, `"case"`).
 * @close_kw:        Closing keyword string (e.g., `"fi"`, `"done"`, `"esac"`).
 * @match_all_loops: Non-zero to match `"do"`, `"for"`, `"while"`, or `"until"` when scanning back.
 * @pair:            Output match pair structure populated when a match is found.
 *
 * Scans unquoted words backward from `@wstart - 1`, incrementing depth on
 * nested @close_kw tokens and decrementing on matching opening keywords until
 * depth reaches zero.
 *
 * Return: true if the matching opening keyword was found, or false otherwise.
 */
static bool match_kw_backward(const char     *line,
                              int             len,
                              const uint8_t  *mask,
                              int             wstart,
                              int             wend,
                              const char     *open_kw,
                              const char     *close_kw,
                              int             match_all_loops,
                              CLI_MATCH_PAIR *pair)
{
    int depth = 1;
    int dummy;
    for (int i = wstart - 1; i >= 0; i--)
    {
        if (mask[i] != 0)
        {
            continue;
        }
        if (is_word_at(line, len, i, close_kw, &dummy))
        {
            depth++;
        }
        else
        {
            int open_wl = 0;
            if (match_all_loops)
            {
                if (is_word_at(line, len, i, "do", &dummy))
                {
                    open_wl = 2;
                }
                else if (is_word_at(line, len, i, "for", &dummy))
                {
                    open_wl = 3;
                }
                else if (is_word_at(line, len, i, "while", &dummy) ||
                         is_word_at(line, len, i, "until", &dummy))
                {
                    open_wl = 5;
                }
            }
            else if (is_word_at(line, len, i, open_kw, &open_wl))
            {
                /* open_wl populated */
            }
            if (open_wl > 0 && --depth == 0)
            {
                set_match_pair(pair, wstart, wend, i, i + open_wl);
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Lexical delimiter and block keyword match scanner.
 */
bool find_match_pair_lexical(const char *line, int len, int cursor_pos, CLI_MATCH_PAIR *pair)
{
    memset(pair, 0, sizeof(*pair));
    if (line == NULL || len <= 0 || cursor_pos < 0)
    {
        return false;
    }

    uint8_t  mask_buf[1024];
    uint8_t *mask = (len < 1024) ? mask_buf : (uint8_t *) malloc((size_t) (len + 1));
    if (mask == NULL)
    {
        return false;
    }
    build_quote_comment_mask(line, len, mask);

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

    bool matched = false;
    for (int p = 0; p < npos && !matched; p++)
    {
        int pos = test_positions[p];
        if (mask[pos] == 1 || mask[pos] == 3)
        {
            continue;
        }
        if (try_match_delimiter_at(line, len, mask, pos, pair))
        {
            matched = true;
            break;
        }
        if (mask[pos] != 0)
        {
            continue;
        }

        int wstart = pos;
        while (wstart > 0 && is_word_char(line[wstart - 1]))
        {
            wstart--;
        }
        int wend = pos;
        while (wend < len && is_word_char(line[wend]))
        {
            wend++;
        }
        int wlen = wend - wstart;
        if (wlen <= 0 || wlen >= 31)
        {
            continue;
        }

        char word[32];
        strncpy(word, &line[wstart], (size_t) wlen);
        word[wlen] = '\0';

        if (strcmp(word, "if") == 0)
        {
            matched = match_kw_forward(line, len, mask, wstart, wend, "if", "fi", 0, pair);
        }
        else if (strcmp(word, "fi") == 0)
        {
            matched = match_kw_backward(line, len, mask, wstart, wend, "if", "fi", 0, pair);
        }
        else if (strcmp(word, "for") == 0 || strcmp(word, "while") == 0 ||
                 strcmp(word, "until") == 0)
        {
            matched = match_kw_forward(line, len, mask, wstart, wend, word, "done", 1, pair);
        }
        else if (strcmp(word, "do") == 0)
        {
            matched = match_kw_forward(line, len, mask, wstart, wend, "do", "done", 0, pair);
        }
        else if (strcmp(word, "done") == 0)
        {
            matched = match_kw_backward(line, len, mask, wstart, wend, "do", "done", 1, pair);
        }
        else if (strcmp(word, "case") == 0)
        {
            matched = match_kw_forward(line, len, mask, wstart, wend, "case", "esac", 0, pair);
        }
        else if (strcmp(word, "esac") == 0)
        {
            matched = match_kw_backward(line, len, mask, wstart, wend, "case", "esac", 0, pair);
        }
    }

    if (mask != mask_buf)
    {
        free(mask);
    }
    return matched;
}

/**
 * @brief Check if a statement boundary precedes the token at index @i.
 */
bool cli_ts_is_statement_boundary_before(const char *s, size_t i)
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
    if (prev == '\n' || prev == ';' || prev == '|' || prev == '&' || prev == '(' || prev == '{')
    {
        return true;
    }

    static const char *const bnd_kws[] = { "do", "then", "else", "elif", NULL };
    for (int k = 0; bnd_kws[k] != NULL; k++)
    {
        size_t kl = strlen(bnd_kws[k]);
        if (i >= kl && strncmp(&s[i - kl], bnd_kws[k], kl) == 0 &&
            (i == kl || s[i - kl - 1] == ' ' || s[i - kl - 1] == '\t' || s[i - kl - 1] == '\n' ||
             s[i - kl - 1] == ';'))
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Count occurrences of a shell reserved word @kw in buffer @s.
 */
int cli_ts_count_shell_keyword(const char *s, const char *kw)
{
    size_t kwlen      = strlen(kw);
    int    count      = 0;
    int    in_dquote  = 0;
    int    in_squote  = 0;
    int    in_comment = 0;

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
        if (s[i] == '#' && cli_ts_is_statement_boundary_before(s, i))
        {
            in_comment = 1;
            continue;
        }

        if (strncmp(&s[i], kw, kwlen) == 0)
        {
            char rc       = s[i + kwlen];
            bool right_ok = (rc == '\0' || rc == ' ' || rc == '\t' || rc == '\n' || rc == '\r' ||
                             rc == ';' || rc == ')' || rc == '}' || rc == '|' || rc == '&');
            if (right_ok && cli_ts_is_statement_boundary_before(s, i))
            {
                count++;
                i += kwlen - 1;
            }
        }
    }
    return count;
}

/**
 * @brief Count unquoted block opening and closing braces (`{` and `}` excluding `${...}`).
 */
void cli_ts_count_block_braces(const char *s, int *out_open, int *out_close)
{
    int open_cnt        = 0;
    int close_cnt       = 0;
    int in_dquote       = 0;
    int in_squote       = 0;
    int in_comment      = 0;
    int var_brace_depth = 0;

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
        if (s[i] == '#' && cli_ts_is_statement_boundary_before(s, i))
        {
            in_comment = 1;
            continue;
        }

        if (s[i] == '$' && s[i + 1] == '{')
        {
            var_brace_depth++;
            i++;
            continue;
        }
        if (s[i] == '{')
        {
            open_cnt++;
        }
        else if (s[i] == '}')
        {
            if (var_brace_depth > 0)
            {
                var_brace_depth--;
            }
            else
            {
                close_cnt++;
            }
        }
    }

    if (out_open)
    {
        *out_open = open_cnt;
    }
    if (out_close)
    {
        *out_close = close_cnt;
    }
}

/**
 * @brief Check if a trimmed line starts with a block-closing keyword (`done`, `fi`, `esac`, `}`).
 */
bool cli_ts_is_closer_token_line(const char *trimmed)
{
    if (strncmp(trimmed, "done", 4) == 0 &&
        (trimmed[4] == '\0' || isspace((unsigned char) trimmed[4]) || trimmed[4] == ';'))
    {
        return true;
    }
    if (strncmp(trimmed, "fi", 2) == 0 &&
        (trimmed[2] == '\0' || isspace((unsigned char) trimmed[2]) || trimmed[2] == ';'))
    {
        return true;
    }
    if (strncmp(trimmed, "esac", 4) == 0 &&
        (trimmed[4] == '\0' || isspace((unsigned char) trimmed[4]) || trimmed[4] == ';'))
    {
        return true;
    }
    if (trimmed[0] == '}' &&
        (trimmed[1] == '\0' || isspace((unsigned char) trimmed[1]) || trimmed[1] == ';'))
    {
        return true;
    }
    return false;
}

/**
 * @brief Check if a trimmed line starts with an intermediate transition keyword (`else`, `elif`, `;;`).
 */
bool cli_ts_is_transition_token_line(const char *trimmed)
{
    if (strncmp(trimmed, "else", 4) == 0 &&
        (trimmed[4] == '\0' || isspace((unsigned char) trimmed[4]) || trimmed[4] == ';'))
    {
        return true;
    }
    if (strncmp(trimmed, "elif", 4) == 0 &&
        (trimmed[4] == '\0' || isspace((unsigned char) trimmed[4]) || trimmed[4] == ';'))
    {
        return true;
    }
    if (strncmp(trimmed, ";;", 2) == 0 &&
        (trimmed[2] == '\0' || isspace((unsigned char) trimmed[2])))
    {
        return true;
    }
    return false;
}

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

static bool is_word_char(char c)
{
    return isalnum((unsigned char) c) || c == '_';
}

static bool is_word_at(
    const char *line,
    int         len,
    int         pos,
    const char *word,
    int        *wlen)
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
 * @brief Lexical delimiter and block keyword match scanner
 *
 * Scans @p line for matching pairs of delimiters ((), [], {}, ${}, $(( )))
 * and block keywords (if/fi, do/done, for/while/until/done, case/esac)
 * ignoring matches inside string literals and comments.
 *
 * @param line       Input command line string
 * @param len        Length of input line
 * @param cursor_pos Current cursor position
 * @param pair       Output structure with match byte offsets
 * @return true if a match was found, false otherwise
 */
static bool find_match_pair_lexical(
    const char     *line,
    int             len,
    int             cursor_pos,
    CLI_MATCH_PAIR *pair)
{
    memset(pair, 0, sizeof(*pair));
    if (line == NULL || len <= 0 || cursor_pos < 0)
    {
        return false;
    }

    uint8_t  mask_buf[1024];
    uint8_t *mask =
        (len < 1024) ? mask_buf : (uint8_t *) malloc((size_t) (len + 1));
    if (mask == NULL)
    {
        return false;
    }

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

    for (int p = 0; p < npos; p++)
    {
        int pos = test_positions[p];
        if (mask[pos] == 1 || mask[pos] == 3)
        {
            continue;
        }

        // Multi-char delimiters: $(( and ))
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
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 3;
                        pair->match_start = i;
                        pair->match_end   = i + 2;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                    i++;
                }
            }
            if (matched)
            {
                break;
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
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 2;
                        pair->match_start = i - 2;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                    i -= 2;
                }
                else if (i >= 1 && strncmp(&line[i - 1], "))", 2) == 0)
                {
                    depth++;
                    i--;
                }
            }
            if (matched)
            {
                break;
            }
        }

        // $( and )
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
                else if (line[i] == ')')
                {
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 2;
                        pair->match_start = i;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }

        // ${ and }
        if (pos + 1 < len && strncmp(&line[pos], "${", 2) == 0)
        {
            int depth = 1;
            for (int i = pos + 2; i < len; i++)
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
                else if (line[i] == '}')
                {
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 2;
                        pair->match_start = i;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }
        if (pos > 0 && line[pos - 1] == '$' && line[pos] == '{')
        {
            int depth = 1;
            for (int i = pos + 1; i < len; i++)
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
                else if (line[i] == '}')
                {
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos - 1;
                        pair->token_end   = pos + 1;
                        pair->match_start = i;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }

        // [[ and ]]
        if (pos + 1 < len && strncmp(&line[pos], "[[", 2) == 0 && mask[pos] == 0)
        {
            for (int i = pos + 2; i + 1 < len; i++)
            {
                if (mask[i] != 0)
                {
                    continue;
                }
                if (strncmp(&line[i], "]]", 2) == 0)
                {
                    pair->token_start = pos;
                    pair->token_end   = pos + 2;
                    pair->match_start = i;
                    pair->match_end   = i + 2;
                    pair->has_match   = true;
                    matched           = true;
                    break;
                }
            }
            if (matched)
            {
                break;
            }
        }
        if (pos + 1 < len && strncmp(&line[pos], "]]", 2) == 0 && mask[pos] == 0)
        {
            for (int i = pos - 2; i >= 0; i--)
            {
                if (mask[i] != 0)
                {
                    continue;
                }
                if (strncmp(&line[i], "[[", 2) == 0)
                {
                    pair->token_start = pos;
                    pair->token_end   = pos + 2;
                    pair->match_start = i;
                    pair->match_end   = i + 2;
                    pair->has_match   = true;
                    matched           = true;
                    break;
                }
            }
            if (matched)
            {
                break;
            }
        }

        // Single delimiters: (, ), [, ], {, }
        char c = line[pos];
        if (c == '(' && (pos == 0 || line[pos - 1] != '$') && mask[pos] == 0)
        {
            int depth = 1;
            for (int i = pos + 1; i < len; i++)
            {
                if (mask[i] != 0)
                {
                    continue;
                }
                if (line[i] == '(')
                {
                    depth++;
                }
                else if (line[i] == ')')
                {
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 1;
                        pair->match_start = i;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }
        else if (c == ')' && mask[pos] == 0)
        {
            int depth = 1;
            for (int i = pos - 1; i >= 0; i--)
            {
                if (mask[i] != 0)
                {
                    continue;
                }
                if (line[i] == ')')
                {
                    depth++;
                }
                else if (line[i] == '(')
                {
                    depth--;
                    if (depth == 0)
                    {
                        int start_idx = (i > 0 && line[i - 1] == '$') ? i - 1 : i;
                        pair->token_start = pos;
                        pair->token_end   = pos + 1;
                        pair->match_start = start_idx;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }
        else if (c == '[' && (pos + 1 >= len || line[pos + 1] != '[') &&
                 (pos == 0 || line[pos - 1] != '[') && mask[pos] == 0)
        {
            int depth = 1;
            for (int i = pos + 1; i < len; i++)
            {
                if (mask[i] != 0)
                {
                    continue;
                }
                if (line[i] == '[')
                {
                    depth++;
                }
                else if (line[i] == ']')
                {
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 1;
                        pair->match_start = i;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }
        else if (c == ']' && (pos + 1 >= len || line[pos + 1] != ']') &&
                 (pos == 0 || line[pos - 1] != ']') && mask[pos] == 0)
        {
            int depth = 1;
            for (int i = pos - 1; i >= 0; i--)
            {
                if (mask[i] != 0)
                {
                    continue;
                }
                if (line[i] == ']')
                {
                    depth++;
                }
                else if (line[i] == '[')
                {
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 1;
                        pair->match_start = i;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }
        else if (c == '{' && (pos == 0 || line[pos - 1] != '$') && mask[pos] == 0)
        {
            int depth = 1;
            for (int i = pos + 1; i < len; i++)
            {
                if (mask[i] != 0)
                {
                    continue;
                }
                if (line[i] == '{')
                {
                    depth++;
                }
                else if (line[i] == '}')
                {
                    depth--;
                    if (depth == 0)
                    {
                        pair->token_start = pos;
                        pair->token_end   = pos + 1;
                        pair->match_start = i;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }
        else if (c == '}')
        {
            int depth = 1;
            for (int i = pos - 1; i >= 0; i--)
            {
                if (mask[i] == 1 || mask[i] == 3)
                {
                    continue;
                }
                if (line[i] == '}')
                {
                    depth++;
                }
                else if (line[i] == '{')
                {
                    depth--;
                    if (depth == 0)
                    {
                        int start_idx = (i > 0 && line[i - 1] == '$') ? i - 1 : i;
                        pair->token_start = pos;
                        pair->token_end   = pos + 1;
                        pair->match_start = start_idx;
                        pair->match_end   = i + 1;
                        pair->has_match   = true;
                        matched           = true;
                        break;
                    }
                }
            }
            if (matched)
            {
                break;
            }
        }

        // Keywords (outside quotes & comments)
        if (mask[pos] == 0)
        {
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

            if (wend > wstart)
            {
                int  wlen = wend - wstart;
                char word[32];
                if (wlen < 31)
                {
                    strncpy(word, &line[wstart], (size_t) wlen);
                    word[wlen] = '\0';

                    if (strcmp(word, "if") == 0)
                    {
                        int depth = 1;
                        for (int i = wend; i < len; i++)
                        {
                            if (mask[i] != 0)
                            {
                                continue;
                            }
                            int dummy;
                            if (is_word_at(line, len, i, "if", &dummy))
                            {
                                depth++;
                            }
                            else if (is_word_at(line, len, i, "fi", &dummy))
                            {
                                depth--;
                                if (depth == 0)
                                {
                                    pair->token_start = wstart;
                                    pair->token_end   = wend;
                                    pair->match_start = i;
                                    pair->match_end   = i + 2;
                                    pair->has_match   = true;
                                    matched           = true;
                                    break;
                                }
                            }
                        }
                        if (matched)
                        {
                            break;
                        }
                    }
                    else if (strcmp(word, "fi") == 0)
                    {
                        int depth = 1;
                        for (int i = wstart - 1; i >= 0; i--)
                        {
                            if (mask[i] != 0)
                            {
                                continue;
                            }
                            int dummy;
                            if (is_word_at(line, len, i, "fi", &dummy))
                            {
                                depth++;
                            }
                            else if (is_word_at(line, len, i, "if", &dummy))
                            {
                                depth--;
                                if (depth == 0)
                                {
                                    pair->token_start = wstart;
                                    pair->token_end   = wend;
                                    pair->match_start = i;
                                    pair->match_end   = i + 2;
                                    pair->has_match   = true;
                                    matched           = true;
                                    break;
                                }
                            }
                        }
                        if (matched)
                        {
                            break;
                        }
                    }
                    else if (strcmp(word, "for") == 0 || strcmp(word, "while") == 0 ||
                             strcmp(word, "until") == 0)
                    {
                        int depth = 1;
                        for (int i = wend; i < len; i++)
                        {
                            if (mask[i] != 0)
                            {
                                continue;
                            }
                            int dummy;
                            if (is_word_at(line, len, i, "for", &dummy) ||
                                is_word_at(line, len, i, "while", &dummy) ||
                                is_word_at(line, len, i, "until", &dummy))
                            {
                                depth++;
                            }
                            else if (is_word_at(line, len, i, "done", &dummy))
                            {
                                depth--;
                                if (depth == 0)
                                {
                                    pair->token_start = wstart;
                                    pair->token_end   = wend;
                                    pair->match_start = i;
                                    pair->match_end   = i + 4;
                                    pair->has_match   = true;
                                    matched           = true;
                                    break;
                                }
                            }
                        }
                        if (matched)
                        {
                            break;
                        }
                    }
                    else if (strcmp(word, "do") == 0)
                    {
                        int depth = 1;
                        for (int i = wend; i < len; i++)
                        {
                            if (mask[i] != 0)
                            {
                                continue;
                            }
                            int dummy;
                            if (is_word_at(line, len, i, "do", &dummy))
                            {
                                depth++;
                            }
                            else if (is_word_at(line, len, i, "done", &dummy))
                            {
                                depth--;
                                if (depth == 0)
                                {
                                    pair->token_start = wstart;
                                    pair->token_end   = wend;
                                    pair->match_start = i;
                                    pair->match_end   = i + 4;
                                    pair->has_match   = true;
                                    matched           = true;
                                    break;
                                }
                            }
                        }
                        if (matched)
                        {
                            break;
                        }
                    }
                    else if (strcmp(word, "done") == 0)
                    {
                        int depth = 1;
                        for (int i = wstart - 1; i >= 0; i--)
                        {
                            if (mask[i] != 0)
                            {
                                continue;
                            }
                            int dummy;
                            if (is_word_at(line, len, i, "done", &dummy))
                            {
                                depth++;
                            }
                            else if (is_word_at(line, len, i, "do", &dummy) ||
                                     is_word_at(line, len, i, "for", &dummy) ||
                                     is_word_at(line, len, i, "while", &dummy) ||
                                     is_word_at(line, len, i, "until", &dummy))
                            {
                                depth--;
                                if (depth == 0)
                                {
                                    int match_len =
                                        is_word_at(line, len, i, "do", &dummy) ? 2 :
                                        is_word_at(line, len, i, "for", &dummy) ? 3 :
                                        is_word_at(line, len, i, "while", &dummy) ? 5 : 5;
                                    pair->token_start = wstart;
                                    pair->token_end   = wend;
                                    pair->match_start = i;
                                    pair->match_end   = i + match_len;
                                    pair->has_match   = true;
                                    matched           = true;
                                    break;
                                }
                            }
                        }
                        if (matched)
                        {
                            break;
                        }
                    }
                    else if (strcmp(word, "case") == 0)
                    {
                        int depth = 1;
                        for (int i = wend; i < len; i++)
                        {
                            if (mask[i] != 0)
                            {
                                continue;
                            }
                            int dummy;
                            if (is_word_at(line, len, i, "case", &dummy))
                            {
                                depth++;
                            }
                            else if (is_word_at(line, len, i, "esac", &dummy))
                            {
                                depth--;
                                if (depth == 0)
                                {
                                    pair->token_start = wstart;
                                    pair->token_end   = wend;
                                    pair->match_start = i;
                                    pair->match_end   = i + 4;
                                    pair->has_match   = true;
                                    matched           = true;
                                    break;
                                }
                            }
                        }
                        if (matched)
                        {
                            break;
                        }
                    }
                    else if (strcmp(word, "esac") == 0)
                    {
                        int depth = 1;
                        for (int i = wstart - 1; i >= 0; i--)
                        {
                            if (mask[i] != 0)
                            {
                                continue;
                            }
                            int dummy;
                            if (is_word_at(line, len, i, "esac", &dummy))
                            {
                                depth++;
                            }
                            else if (is_word_at(line, len, i, "case", &dummy))
                            {
                                depth--;
                                if (depth == 0)
                                {
                                    pair->token_start = wstart;
                                    pair->token_end   = wend;
                                    pair->match_start = i;
                                    pair->match_end   = i + 4;
                                    pair->has_match   = true;
                                    matched           = true;
                                    break;
                                }
                            }
                        }
                        if (matched)
                        {
                            break;
                        }
                    }
                }
            }
        }
    }

    if (mask != mask_buf)
    {
        free(mask);
    }
    return matched;
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
    int         priority; // 0 = normal syntax, 1 = error/diag, 2 = match pair
} HighlightSpan;

static void collect_error_spans(
    TSNode         node,
    const char    *source,
    size_t         linelen,
    HighlightSpan *spans,
    int           *num_spans,
    int            max_spans,
    int            color_lvl);

static int compare_spans(const void *a, const void *b)
{
    const HighlightSpan *sa = (const HighlightSpan *) a;
    const HighlightSpan *sb = (const HighlightSpan *) b;
    if (sa->start_byte != sb->start_byte)
    {
        return (int) (sa->start_byte - sb->start_byte);
    }
    // If they start at the same place, earlier end_byte goes first so outer spans
    // enclose inner spans
    if (sa->end_byte != sb->end_byte)
    {
        return (int) (sb->end_byte - sa->end_byte);
    }
    // Lower priority first so higher priority is pushed later and takes precedence
    return sa->priority - sb->priority;
}

/**
 * @brief Search AST for matching delimiter or block keyword pair
 *
 * Traverses parent/sibling nodes of the AST to match:
 *  - Parentheses: ( and )
 *  - Subshells and command substitutions: $( and )
 *  - Arithmetic expansions: $(( and ))
 *  - Test brackets: [ and ], [[ and ]]
 *  - Braces: { and }, ${ and }
 *  - Conditionals: if and fi
 *  - Loops: do/for/while/until and done
 *  - Case blocks: case and esac, ) and ;;
 *
 * @param root       Tree-sitter root node
 * @param line       Command line string
 * @param len        Length of line
 * @param cursor_pos Current cursor position
 * @param pair       Output structure with matched byte ranges
 * @return true if match found, false otherwise
 */
static bool find_match_pair_ast(
    TSNode          root,
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

        uint32_t ccount = ts_node_child_count(parent);
        TSNode   target = { 0 };
        bool     found  = false;

        if (strcmp(type, "(") == 0 || strcmp(type, "$(") == 0 ||
            strcmp(type, "$(( ") == 0 || strcmp(type, "$((") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode      ch = ts_node_child(parent, i);
                const char *ct = ts_node_type(ch);
                if (strcmp(ct, ")") == 0 || strcmp(ct, "))") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, ")") == 0 || strcmp(type, "))") == 0)
        {
            /* If inside case_item: ')' matches ';;' */
            if (strcmp(ts_node_type(parent), "case_item") == 0)
            {
                for (uint32_t i = 0; i < ccount; i++)
                {
                    TSNode ch = ts_node_child(parent, i);
                    if (strcmp(ts_node_type(ch), ";;") == 0)
                    {
                        target = ch;
                        found  = true;
                        break;
                    }
                }
            }
            else
            {
                for (uint32_t i = 0; i < ccount; i++)
                {
                    TSNode      ch = ts_node_child(parent, i);
                    const char *ct = ts_node_type(ch);
                    if (strcmp(ct, "(") == 0 || strcmp(ct, "$(") == 0 ||
                        strcmp(ct, "$(( ") == 0 || strcmp(ct, "$((") == 0)
                    {
                        target = ch;
                        found  = true;
                        break;
                    }
                }
            }
        }
        else if (strcmp(type, ";;") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), ")") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "[") == 0 || strcmp(type, "[[") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode      ch = ts_node_child(parent, i);
                const char *ct = ts_node_type(ch);
                if (strcmp(ct, "]") == 0 || strcmp(ct, "]]") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "]") == 0 || strcmp(type, "]]") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode      ch = ts_node_child(parent, i);
                const char *ct = ts_node_type(ch);
                if (strcmp(ct, "[") == 0 || strcmp(ct, "[[") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "{") == 0 || strcmp(type, "${") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "}") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "}") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode      ch = ts_node_child(parent, i);
                const char *ct = ts_node_type(ch);
                if (strcmp(ct, "{") == 0 || strcmp(ct, "${") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "if") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "fi") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "fi") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "if") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "do") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "done") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "for") == 0 || strcmp(type, "while") == 0 ||
                 strcmp(type, "until") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "done") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "done") == 0)
        {
            /* Match 'do' first, then loop header */
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "do") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
            if (!found)
            {
                for (uint32_t i = 0; i < ccount; i++)
                {
                    TSNode      ch = ts_node_child(parent, i);
                    const char *ct = ts_node_type(ch);
                    if (strcmp(ct, "for") == 0 || strcmp(ct, "while") == 0 ||
                        strcmp(ct, "until") == 0)
                    {
                        target = ch;
                        found  = true;
                        break;
                    }
                }
            }
        }
        else if (strcmp(type, "case") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "esac") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
        }
        else if (strcmp(type, "esac") == 0)
        {
            for (uint32_t i = 0; i < ccount; i++)
            {
                TSNode ch = ts_node_child(parent, i);
                if (strcmp(ts_node_type(ch), "case") == 0)
                {
                    target = ch;
                    found  = true;
                    break;
                }
            }
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

void cli_ts_highlight_line(
    const char *line,
    int         len,
    int         cursor_pos,
    FILE       *out)
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
                spans[num_spans].priority   = 0;
                num_spans++;
            }
        }
    }

    ts_query_cursor_delete(cursor);

    if (data.syntax_diagnostics && ts_node_has_error(root_node))
    {
        collect_error_spans(
            root_node, line, (size_t) len, spans, &num_spans, 1024, color_level);
    }

    if (data.show_match && cursor_pos >= 0)
    {
        CLI_MATCH_PAIR mp;
        bool ok = find_match_pair_ast(root_node, line, len, cursor_pos, &mp);
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

/**
 * @brief Find matching structural delimiter or block keyword pair
 *
 * Inspects the token at or adjacent to @p cursor_pos in @p line.
 * If the cursor is on or next to an opening or closing delimiter
 * ((), [], {}, ${...}, $((...))) or block keyword (if/fi, do/done,
 * for/while/until/done, case/esac), finds the corresponding matching
 * token's start and end byte offsets.
 *
 * Uses Tree-sitter AST with lexical fallback.
 *
 * @param line       Input line buffer
 * @param cursor_pos Current cursor position (0 <= cursor_pos <= strlen(line))
 * @param pair       Output structure with matched byte ranges
 * @return true if a matching pair was found, false otherwise
 */
bool cli_ts_find_match_pair(
    const char     *line,
    int             cursor_pos,
    CLI_MATCH_PAIR *pair)
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
        TSTree *tree = ts_parser_parse_string(ts_parser, NULL, line, len);
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

static bool is_closer_token_line(const char *trimmed);
static bool is_transition_token_line(const char *trimmed);

/**
 * @brief Count unquoted block opening and closing braces.
 *
 * Distinguishes command/function block braces '{' and '}' from parameter
 * expansions like '${var}'. Parameter expansions have '{' immediately preceded
 * by '$'.
 *
 * @param s         Input buffer
 * @param out_open  Receives count of opening block braces
 * @param out_close Receives count of closing block braces
 */
static void count_block_braces(
    const char *s,
    int        *out_open,
    int        *out_close)
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
        if (s[i] == '#')
        {
            if (is_statement_boundary_before(s, i))
            {
                in_comment = 1;
                continue;
            }
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
 * @brief Test if an ERROR node represents an unclosed control block.
 *
 * In shell syntax, incomplete blocks (like for without done, if without fi)
 * can cause Tree-sitter to generate an ERROR node enclosing the block.
 * We distinguish this from true syntax errors so in-progress blocks are not
 * displayed with jarring red underlines.
 *
 * @param node   Tree-sitter AST node to inspect
 * @param source Input line string
 * @return true if error node is due to an open block, false otherwise
 */
static bool is_open_block_error(
    TSNode      node,
    const char *source)
{
    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        TSNode      child = ts_node_child(node, i);
        const char *ctype = ts_node_type(child);
        if (strcmp(ctype, "for") == 0 ||
            strcmp(ctype, "while") == 0 ||
            strcmp(ctype, "until") == 0 ||
            strcmp(ctype, "do") == 0)
        {
            int starters = count_shell_keyword(source, "for") +
                           count_shell_keyword(source, "while") +
                           count_shell_keyword(source, "until");
            int closers  = count_shell_keyword(source, "done");
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
            int starters = count_shell_keyword(source, "if");
            int closers  = count_shell_keyword(source, "fi");
            if (starters > closers)
            {
                return true;
            }
        }
        else if (strcmp(ctype, "case") == 0)
        {
            int starters = count_shell_keyword(source, "case");
            int closers  = count_shell_keyword(source, "esac");
            if (starters > closers)
            {
                return true;
            }
        }
        else if (strcmp(ctype, "function") == 0 ||
                 strcmp(ctype, "{") == 0)
        {
            int o = 0;
            int c = 0;
            count_block_braces(source, &o, &c);
            if (o > c)
            {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Collect error and continuation highlight spans from Tree-sitter AST.
 *
 * Traverses the AST for ERROR nodes. Incomplete variable prefixes ($) at line end
 * are preserved in variable color, while unclosed strings, expansions, and genuine
 * syntax errors (such as unexpected tokens) are highlighted with red underlines.
 *
 * @param node      Current AST node
 * @param source    Input line string
 * @param linelen   Length of input line in bytes
 * @param spans     Output array of highlight spans
 * @param num_spans Current number of spans in array
 * @param max_spans Maximum capacity of spans array
 * @param color_lvl Active color capability level (1 = 16-color, 2 = 256-color)
 */
static void collect_error_spans(
    TSNode         node,
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
        /* If this error node is caused by an open control block, do not underline */
        if (is_open_block_error(node, source))
        {
            return;
        }

        /* In continuation prompt, closer and transition tokens are expected */
        if (cli_is_continuation_prompt())
        {
            const char *start = source;
            while (*start == ' ' || *start == '\t')
            {
                start++;
            }
            if (is_closer_token_line(start) || is_transition_token_line(start))
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
                spans[*num_spans].start_byte = sb;
                spans[*num_spans].end_byte   = eb;

                /* If incomplete variable prefix ($) at end of line, preserve variable color */
                if (eb >= linelen && source[sb] == '$' && eb - sb == 1)
                {
                    spans[*num_spans].color =
                        (color_lvl >= 2) ? "\033[38;5;178m" : "\033[33m";
                }
                else
                {
                    /* Real syntax error or unclosed string/token: underline red */
                    spans[*num_spans].color =
                        (color_lvl >= 2) ? "\033[4;38;5;203m" : "\033[4;31m";
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
        collect_error_spans(
            ts_node_child(node, i), source, linelen, spans, num_spans, max_spans, color_lvl);
    }
}

/**
 * @brief Search AST recursively for the first syntax error or missing token.
 *
 * Evaluates missing tokens (missing 'done', 'fi', 'esac', '}', etc.) and ERROR nodes,
 * extracting diagnostic token text, byte positions, and human-readable message.
 *
 * @param node   Root or current AST node
 * @param source Input line string
 * @param diag   Output diagnostic structure
 * @return true if an error or incomplete state was identified, false otherwise
 */
static bool find_first_error(
    TSNode           node,
    const char      *source,
    CLI_SYNTAX_DIAG *diag)
{
    if (ts_node_is_missing(node))
    {
        const char *mtype = ts_node_type(node);
        if (strcmp(mtype, "done") == 0)
        {
            int starters = count_shell_keyword(source, "for") +
                           count_shell_keyword(source, "while") +
                           count_shell_keyword(source, "until");
            int closers  = count_shell_keyword(source, "done");
            if (starters <= closers)
            {
                return false;
            }
            diag->severity = CLI_DIAG_SEVERITY_INFO;
            snprintf(diag->message, sizeof(diag->message), "loop open: missing 'done'");
        }
        else if (strcmp(mtype, "fi") == 0)
        {
            int starters = count_shell_keyword(source, "if");
            int closers  = count_shell_keyword(source, "fi");
            if (starters <= closers)
            {
                return false;
            }
            diag->severity = CLI_DIAG_SEVERITY_INFO;
            snprintf(diag->message, sizeof(diag->message), "if block open: missing 'fi'");
        }
        else if (strcmp(mtype, "esac") == 0)
        {
            int starters = count_shell_keyword(source, "case");
            int closers  = count_shell_keyword(source, "esac");
            if (starters <= closers)
            {
                return false;
            }
            diag->severity = CLI_DIAG_SEVERITY_INFO;
            snprintf(diag->message, sizeof(diag->message), "case block open: missing 'esac'");
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
            int o = count_shell_keyword(source, "{");
            int c = count_shell_keyword(source, "}");
            if (o <= c)
            {
                return false;
            }
            diag->severity = CLI_DIAG_SEVERITY_INFO;
            snprintf(diag->message, sizeof(diag->message), "unclosed '{': missing '}'");
        }
        else if (strcmp(mtype, ")") == 0)
        {
            int o = count_shell_keyword(source, "(");
            int c = count_shell_keyword(source, ")");
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
            snprintf(
                diag->message, sizeof(diag->message),
                "unclosed test bracket: missing '%s'", mtype);
        }
        else
        {
            diag->severity = CLI_DIAG_SEVERITY_INFO;
            snprintf(
                diag->message, sizeof(diag->message),
                "syntax incomplete: expected '%s'", mtype);
        }

        diag->start_byte = ts_node_start_byte(node);
        diag->end_byte   = ts_node_end_byte(node);
        snprintf(diag->token, sizeof(diag->token), "%s", mtype);
        return true;
    }

    const char *type = ts_node_type(node);
    if (strcmp(type, "ERROR") == 0)
    {
        uint32_t count = ts_node_child_count(node);
        for (uint32_t i = 0; i < count; i++)
        {
            TSNode      child = ts_node_child(node, i);
            const char *ctype = ts_node_type(child);
            if (strcmp(ctype, "for") == 0 ||
                strcmp(ctype, "while") == 0 ||
                strcmp(ctype, "until") == 0 ||
                strcmp(ctype, "do") == 0)
            {
                int starters = count_shell_keyword(source, "for") +
                               count_shell_keyword(source, "while") +
                               count_shell_keyword(source, "until");
                int closers  = count_shell_keyword(source, "done");
                if (starters > closers)
                {
                    diag->severity   = CLI_DIAG_SEVERITY_INFO;
                    diag->start_byte = ts_node_start_byte(child);
                    diag->end_byte   = ts_node_end_byte(child);
                    snprintf(diag->token, sizeof(diag->token), "%s", ctype);
                    snprintf(
                        diag->message, sizeof(diag->message), "loop open: missing 'done'");
                    return true;
                }
            }
            else if (strcmp(ctype, "if") == 0 ||
                     strcmp(ctype, "then") == 0 ||
                     strcmp(ctype, "elif") == 0 ||
                     strcmp(ctype, "else") == 0)
            {
                int starters = count_shell_keyword(source, "if");
                int closers  = count_shell_keyword(source, "fi");
                if (starters > closers)
                {
                    diag->severity   = CLI_DIAG_SEVERITY_INFO;
                    diag->start_byte = ts_node_start_byte(child);
                    diag->end_byte   = ts_node_end_byte(child);
                    snprintf(diag->token, sizeof(diag->token), "%s", ctype);
                    snprintf(
                        diag->message, sizeof(diag->message), "if block open: missing 'fi'");
                    return true;
                }
            }
            else if (strcmp(ctype, "case") == 0)
            {
                int starters = count_shell_keyword(source, "case");
                int closers  = count_shell_keyword(source, "esac");
                if (starters > closers)
                {
                    diag->severity   = CLI_DIAG_SEVERITY_INFO;
                    diag->start_byte = ts_node_start_byte(child);
                    diag->end_byte   = ts_node_end_byte(child);
                    snprintf(diag->token, sizeof(diag->token), "%s", ctype);
                    snprintf(
                        diag->message, sizeof(diag->message), "case block open: missing 'esac'");
                    return true;
                }
            }
        }

        uint32_t sb      = ts_node_start_byte(node);
        uint32_t eb      = ts_node_end_byte(node);
        size_t   linelen = strlen(source);

        diag->start_byte = sb;
        diag->end_byte   = eb;

        int toklen = (int) (eb - sb);
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

        /* Check for incomplete constructs at end of line */
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
                snprintf(
                    diag->message, sizeof(diag->message),
                    "unclosed arithmetic '$((': missing '))'");
                return true;
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

/**
 * @brief Get real-time syntax diagnostic for current input buffer.
 *
 * Inspects the input line for incomplete constructs (unclosed quotes, open
 * control blocks, dangling pipes/operators) and syntax errors (unexpected
 * tokens, malformed statements) using Tree-sitter AST.
 *
 * @param line Input line buffer
 * @param diag Output structure populated with severity, span, and message
 * @return 1 if a diagnostic was detected, 0 if clean/valid
 */
int cli_ts_get_diagnostic(
    const char      *line,
    CLI_SYNTAX_DIAG *diag)
{
    if (diag)
    {
        memset(diag, 0, sizeof(*diag));
    }
    if (!line || line[0] == '\0' || !diag)
    {
        return 0;
    }

    /* 1. Unclosed quotes */
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

    if (in_dquote)
    {
        diag->severity   = CLI_DIAG_SEVERITY_INFO;
        diag->start_byte = (uint32_t) q_start;
        diag->end_byte   = (uint32_t) strlen(line);
        snprintf(diag->message, sizeof(diag->message), "unclosed double quote \"");
        snprintf(diag->token, sizeof(diag->token), "\"");
        return 1;
    }
    if (in_squote)
    {
        diag->severity   = CLI_DIAG_SEVERITY_INFO;
        diag->start_byte = (uint32_t) q_start;
        diag->end_byte   = (uint32_t) strlen(line);
        snprintf(diag->message, sizeof(diag->message), "unclosed single quote '");
        snprintf(diag->token, sizeof(diag->token), "'");
        return 1;
    }

    /* 2. Trailing continuation operators */
    size_t len = strlen(line);
    while (len > 0 && isspace((unsigned char) line[len - 1]))
    {
        len--;
    }
    if (len > 0)
    {
        if (line[len - 1] == '\\' && (len == 1 || line[len - 2] != '\\'))
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 1);
            diag->end_byte   = (uint32_t) len;
            snprintf(diag->message, sizeof(diag->message), "trailing '\\' (line continuation)");
            snprintf(diag->token, sizeof(diag->token), "\\");
            return 1;
        }
        if (line[len - 1] == '|' && (len == 1 || line[len - 2] != '|'))
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 1);
            diag->end_byte   = (uint32_t) len;
            snprintf(
                diag->message, sizeof(diag->message),
                "trailing pipe '|' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "|");
            return 1;
        }
        if (len >= 2 && line[len - 1] == '&' && line[len - 2] == '&')
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 2);
            diag->end_byte   = (uint32_t) len;
            snprintf(
                diag->message, sizeof(diag->message),
                "trailing '&&' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "&&");
            return 1;
        }
        if (len >= 2 && line[len - 1] == '|' && line[len - 2] == '|')
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 2);
            diag->end_byte   = (uint32_t) len;
            snprintf(
                diag->message, sizeof(diag->message),
                "trailing '||' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "||");
            return 1;
        }
    }

    if (!ts_parser)
    {
        if (cli_ts_init() != 0)
        {
            return 0;
        }
    }

    /* In continuation prompt, closer and transition tokens are expected */
    if (cli_is_continuation_prompt())
    {
        const char *start = line;
        while (*start == ' ' || *start == '\t')
        {
            start++;
        }
        if (is_closer_token_line(start) || is_transition_token_line(start))
        {
            return 0;
        }
    }

    /* 3. Tree-sitter AST error detection */
    TSTree *tree = ts_parser_parse_string(ts_parser, NULL, line, (uint32_t) strlen(line));
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
}

/**
 * @brief Check if a trimmed line starts with a block-closing keyword.
 *
 * Closing keywords (done, fi, esac, }) reduce the line's visual indentation
 * so the closer aligns with its matching opening block keyword.
 *
 * @param trimmed Null-terminated string with leading whitespace stripped
 * @return true if line begins with a closing token, false otherwise
 */
static bool is_closer_token_line(const char *trimmed)
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
 * @brief Check if a trimmed line starts with an intermediate block keyword.
 *
 * Transition keywords (else, elif, ;;) align with the outer enclosing block
 * rather than the inner block statements.
 *
 * @param trimmed Null-terminated string with leading whitespace stripped
 * @return true if line begins with a transition token, false otherwise
 */
static bool is_transition_token_line(const char *trimmed)
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

/**
 * @brief Compute block nesting depth for auto-indentation.
 *
 * Evaluates the net nesting level of open loops, if statements, case statements,
 * and compound blocks.
 *
 * @param buffer Input code buffer
 * @return Nesting depth (>= 0)
 */
int cli_ts_compute_indent_depth(const char *buffer)
{
    if (!buffer || buffer[0] == '\0')
    {
        return 0;
    }

    int loops  = count_shell_keyword(buffer, "for") +
                 count_shell_keyword(buffer, "while") +
                 count_shell_keyword(buffer, "until");
    int dones  = count_shell_keyword(buffer, "done");
    int loop_d = (loops > dones) ? (loops - dones) : 0;

    int ifs    = count_shell_keyword(buffer, "if");
    int fis    = count_shell_keyword(buffer, "fi");
    int if_d   = (ifs > fis) ? (ifs - fis) : 0;

    int cases  = count_shell_keyword(buffer, "case");
    int esacs  = count_shell_keyword(buffer, "esac");
    int case_d = (cases > esacs) ? (cases - esacs) : 0;

    int obrace = 0;
    int cbrace = 0;
    count_block_braces(buffer, &obrace, &cbrace);
    int brc_d  = (obrace > cbrace) ? (obrace - cbrace) : 0;

    return loop_d + if_d + case_d + brc_d;
}

/**
 * @brief Traverse Tree-sitter AST and record block indentation depth for each line.
 *
 * Traverses compound control nodes (loops, conditionals, functions, subshells)
 * and increments depth for all lines within the block body. Closer lines starting
 * with done, fi, esac, or } are not incremented to ensure alignment with the parent.
 *
 * @param node       Root or child Tree-sitter AST node
 * @param line_depth Array mapping 0-indexed line numbers to indent depths
 * @param num_lines  Total number of lines in script
 * @param lines      Array of trimmed line strings (used to detect closer tokens)
 */
static void compute_ast_line_depths(
    TSNode              node,
    int                *line_depth,
    int                 num_lines,
    const char * const *lines)
{
    const char *type = ts_node_type(node);
    bool is_block = (strcmp(type, "for_statement") == 0 ||
                     strcmp(type, "while_statement") == 0 ||
                     strcmp(type, "until_statement") == 0 ||
                     strcmp(type, "if_statement") == 0 ||
                     strcmp(type, "case_statement") == 0 ||
                     strcmp(type, "function_definition") == 0 ||
                     strcmp(type, "subshell") == 0);

    if (is_block)
    {
        TSPoint sp = ts_node_start_point(node);
        TSPoint ep = ts_node_end_point(node);
        if (ep.row > sp.row)
        {
            uint32_t end_row = ep.row;
            if (end_row < (uint32_t) num_lines &&
                lines && is_closer_token_line(lines[end_row]))
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
 * @brief Format script code with semantic AST indentation.
 *
 * Re-indents multi-line milk script code using Tree-sitter block scopes.
 *
 * @param code         Input script string
 * @param indent_width Number of spaces per indentation level (default: 4)
 * @return Dynamically allocated formatted string (caller must free), or NULL on error
 */
char *cli_ts_format_code(
    const char *code,
    int         indent_width)
{
    if (!code)
    {
        return NULL;
    }

    if (indent_width <= 0)
    {
        indent_width = 4;
    }
    if (indent_width > 16)
    {
        indent_width = 16;
    }

    /* Count lines */
    int num_lines = 0;
    for (const char *p = code; *p != '\0'; p++)
    {
        if (*p == '\n')
        {
            num_lines++;
        }
    }
    num_lines++; /* For trailing line without newline */

    char **raw_lines     = (char **) calloc(num_lines, sizeof(char *));
    char **trimmed_lines = (char **) calloc(num_lines, sizeof(char *));
    int   *line_depth    = (int *) calloc(num_lines, sizeof(int));
    if (!raw_lines || !trimmed_lines || !line_depth)
    {
        free(raw_lines);
        free(trimmed_lines);
        free(line_depth);
        return NULL;
    }

    /* Extract lines and trimmed lines */
    const char *p        = code;
    int         line_idx = 0;
    while (*p != '\0' && line_idx < num_lines)
    {
        const char *nl      = strchr(p, '\n');
        size_t      linelen = nl ? (size_t) (nl - p) : strlen(p);

        char *line = (char *) malloc(linelen + 1);
        if (line)
        {
            memcpy(line, p, linelen);
            line[linelen] = '\0';
        }
        raw_lines[line_idx] = line;

        /* Trim */
        const char *start = line ? line : "";
        while (*start == ' ' || *start == '\t' || *start == '\r')
        {
            start++;
        }
        size_t slen = strlen(start);
        while (slen > 0 && (start[slen - 1] == ' ' || start[slen - 1] == '\t' ||
                            start[slen - 1] == '\r'))
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

    /* Compute depths using Tree-sitter AST */
    bool used_ast = false;
    if (ts_parser != NULL || cli_ts_init() == 0)
    {
        TSTree *tree = ts_parser_parse_string(ts_parser, NULL, code, (uint32_t) strlen(code));
        if (tree)
        {
            compute_ast_line_depths(
                ts_tree_root_node(tree),
                line_depth,
                num_lines,
                (const char * const *) trimmed_lines);
            ts_tree_delete(tree);
            used_ast = true;
        }
    }

    /* Fallback to lexical depth computation if Tree-sitter was not available */
    if (!used_ast)
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
            if (is_closer_token_line(start) || is_transition_token_line(start))
            {
                line_depth[i] = (depth > 0) ? (depth - 1) : 0;
            }
            else
            {
                line_depth[i] = depth;
            }

            int loops = count_shell_keyword(start, "for") +
                        count_shell_keyword(start, "while") +
                        count_shell_keyword(start, "until");
            int dones = count_shell_keyword(start, "done");
            depth += (loops - dones);

            int ifs = count_shell_keyword(start, "if");
            int fis = count_shell_keyword(start, "fi");
            depth += (ifs - fis);

            int cases = count_shell_keyword(start, "case");
            int esacs = count_shell_keyword(start, "esac");
            depth += (cases - esacs);

            int obrace = 0;
            int cbrace = 0;
            count_block_braces(start, &obrace, &cbrace);
            depth += (obrace - cbrace);

            if (depth < 0)
            {
                depth = 0;
            }
        }
    }

    /* Assemble formatted buffer */
    size_t in_len  = strlen(code);
    size_t out_cap = in_len * 2 + 4096;
    char  *out     = (char *) malloc(out_cap);
    if (!out)
    {
        for (int i = 0; i < num_lines; i++)
        {
            free(raw_lines[i]);
            free(trimmed_lines[i]);
        }
        free(raw_lines);
        free(trimmed_lines);
        free(line_depth);
        return NULL;
    }
    out[0] = '\0';
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
        if (is_transition_token_line(tline) && d > 0)
        {
            d--;
        }

        int    nspaces = d * indent_width;
        size_t needed  = out_len + nspaces + strlen(tline) + 2;
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

    /* Cleanup */
    for (int i = 0; i < num_lines; i++)
    {
        free(raw_lines[i]);
        free(trimmed_lines[i]);
    }
    free(raw_lines);
    free(trimmed_lines);
    free(line_depth);

    return out;
}

static void print_folds_ast_walk(
    TSNode      node,
    const char *source,
    int         depth,
    int        *block_count,
    FILE       *out)
{
    const char *type = ts_node_type(node);
    bool is_block = (strcmp(type, "for_statement") == 0 ||
                     strcmp(type, "while_statement") == 0 ||
                     strcmp(type, "until_statement") == 0 ||
                     strcmp(type, "if_statement") == 0 ||
                     strcmp(type, "case_statement") == 0 ||
                     strcmp(type, "function_definition") == 0 ||
                     strcmp(type, "subshell") == 0);

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

            fprintf(
                out, "  %*sLines %3u-%-3u (%2u lines): %s ...\n",
                depth * 2, "",
                sp.row + 1, ep.row + 1, ep.row - sp.row + 1,
                first_line);
            (*block_count)++;
        }
    }

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; i++)
    {
        print_folds_ast_walk(
            ts_node_child(node, i), source, depth + (is_block ? 1 : 0), block_count, out);
    }
}

/**
 * @brief Print structural outline of AST blocks in code.
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
    FILE       *out)
{
    if (!code || code[0] == '\0')
    {
        return 0;
    }
    if (!out)
    {
        out = stdout;
    }

    if (!ts_parser)
    {
        if (cli_ts_init() != 0)
        {
            return 0;
        }
    }

    TSTree *tree = ts_parser_parse_string(ts_parser, NULL, code, (uint32_t) strlen(code));
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

    TSNode root        = ts_tree_root_node(tree);
    int    block_count = 0;
    print_folds_ast_walk(root, code, 0, &block_count, out);

    ts_tree_delete(tree);
    return block_count;
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
            int o = 0;
            int c = 0;
            count_block_braces(buffer, &o, &c);
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
            else if (strcmp(ctype, "function") == 0 ||
                     strcmp(ctype, "{") == 0)
            {
                int o = 0;
                int c = 0;
                count_block_braces(buffer, &o, &c);
                if (o > c)
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
void cli_ts_highlight_line(
    const char *line,
    int         len,
    int         cursor_pos,
    FILE       *out)
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

bool cli_ts_find_match_pair(
    const char     *line,
    int             cursor_pos,
    CLI_MATCH_PAIR *pair)
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

int cli_ts_get_diagnostic(
    const char      *line,
    CLI_SYNTAX_DIAG *diag)
{
    if (diag)
    {
        memset(diag, 0, sizeof(*diag));
    }
    if (!line || line[0] == '\0' || !diag)
    {
        return 0;
    }

    /* 1. Unclosed quotes */
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

    if (in_dquote)
    {
        diag->severity   = CLI_DIAG_SEVERITY_INFO;
        diag->start_byte = (uint32_t) q_start;
        diag->end_byte   = (uint32_t) strlen(line);
        snprintf(diag->message, sizeof(diag->message), "unclosed double quote \"");
        snprintf(diag->token, sizeof(diag->token), "\"");
        return 1;
    }
    if (in_squote)
    {
        diag->severity   = CLI_DIAG_SEVERITY_INFO;
        diag->start_byte = (uint32_t) q_start;
        diag->end_byte   = (uint32_t) strlen(line);
        snprintf(diag->message, sizeof(diag->message), "unclosed single quote '");
        snprintf(diag->token, sizeof(diag->token), "'");
        return 1;
    }

    /* 2. Trailing continuation operators */
    size_t len = strlen(line);
    while (len > 0 && isspace((unsigned char) line[len - 1]))
    {
        len--;
    }
    if (len > 0)
    {
        if (line[len - 1] == '\\' && (len == 1 || line[len - 2] != '\\'))
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 1);
            diag->end_byte   = (uint32_t) len;
            snprintf(diag->message, sizeof(diag->message), "trailing '\\' (line continuation)");
            snprintf(diag->token, sizeof(diag->token), "\\");
            return 1;
        }
        if (line[len - 1] == '|' && (len == 1 || line[len - 2] != '|'))
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 1);
            diag->end_byte   = (uint32_t) len;
            snprintf(
                diag->message, sizeof(diag->message),
                "trailing pipe '|' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "|");
            return 1;
        }
        if (len >= 2 && line[len - 1] == '&' && line[len - 2] == '&')
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 2);
            diag->end_byte   = (uint32_t) len;
            snprintf(
                diag->message, sizeof(diag->message),
                "trailing '&&' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "&&");
            return 1;
        }
        if (len >= 2 && line[len - 1] == '|' && line[len - 2] == '|')
        {
            diag->severity   = CLI_DIAG_SEVERITY_INFO;
            diag->start_byte = (uint32_t) (len - 2);
            diag->end_byte   = (uint32_t) len;
            snprintf(
                diag->message, sizeof(diag->message),
                "trailing '||' (waiting for command)");
            snprintf(diag->token, sizeof(diag->token), "||");
            return 1;
        }
        if (line[len - 1] == '<' || line[len - 1] == '>')
        {
            diag->severity   = CLI_DIAG_SEVERITY_ERROR;
            diag->start_byte = (uint32_t) (len - 1);
            diag->end_byte   = (uint32_t) len;
            snprintf(diag->message, sizeof(diag->message), "missing redirection file operand");
            snprintf(diag->token, sizeof(diag->token), "%c", line[len - 1]);
            return 1;
        }
    }

    /* 3. Check for open braces and shell blocks */
    int obrace   = 0;
    int cbrace   = 0;
    int case_cnt = 0;
    int esac_cnt = 0;

    for (size_t i = 0; line[i] != '\0'; i++)
    {
        if (line[i] == '{')
        {
            obrace++;
        }
        else if (line[i] == '}')
        {
            cbrace++;
        }
        else if (line[i] == ';' && line[i + 1] == ';')
        {
            if (case_cnt <= esac_cnt)
            {
                diag->severity   = CLI_DIAG_SEVERITY_ERROR;
                diag->start_byte = (uint32_t) i;
                diag->end_byte   = (uint32_t) (i + 2);
                snprintf(diag->message, sizeof(diag->message), "unexpected token ';;'");
                snprintf(diag->token, sizeof(diag->token), ";;");
                return 1;
            }
        }
    }

    if (obrace > cbrace)
    {
        diag->severity = CLI_DIAG_SEVERITY_INFO;
        snprintf(diag->message, sizeof(diag->message), "unclosed '{': missing '}'");
        snprintf(diag->token, sizeof(diag->token), "}");
        return 1;
    }

    return (diag->severity != CLI_DIAG_SEVERITY_NONE);
}

int cli_ts_compute_indent_depth(const char *buffer)
{
    if (!buffer || buffer[0] == '\0')
    {
        return 0;
    }

    int obrace = 0;
    int cbrace = 0;
    for (size_t i = 0; buffer[i] != '\0'; i++)
    {
        if (buffer[i] == '{')
        {
            obrace++;
        }
        else if (buffer[i] == '}')
        {
            cbrace++;
        }
    }
    return (obrace > cbrace) ? (obrace - cbrace) : 0;
}

char *cli_ts_format_code(
    const char *code,
    int         indent_width)
{
    (void) indent_width;
    if (!code)
    {
        return NULL;
    }
    return strdup(code);
}

int cli_ts_print_block_folds(
    const char *code,
    const char *label,
    FILE       *out)
{
    (void) code;
    (void) label;
    (void) out;
    return 0;
}

#endif

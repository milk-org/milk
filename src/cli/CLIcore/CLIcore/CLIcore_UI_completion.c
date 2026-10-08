// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_completion.c
 *
 * @brief Readline tab-completion and prompt construction
 *
 * Provides the tab-completion engine for the milk CLI.
 * Completion matches against registered commands, shared-
 * memory image streams, FPS names, command arguments (dot-
 * prefixed FPS tags), and filesystem paths.
 *
 * Also provides the prompt builder (including PS1 support)
 * and the readline callback that hands accepted input to
 * the command execution pipeline.
 *
 * ## Key design choices
 *
 * - **Two-pass completion**: The generator first tries
 *   prefix matching, then falls back to substring (fuzzy)
 *   matching if nothing was found and fuzzy mode is on.
 *
 * - **Argument-type-aware completion**: When the cursor is
 *   on a positional argument of a known command, the
 *   completion mode switches to match the expected argument
 *   type (image stream, filename, FPS name, etc.).
 *
 * - **Levenshtein distance**: Used by the "did you mean?"
 *   suggestions when a command is not found.
 */

#include <stdio.h>
#include <stdbool.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#include "CLIcore.h"

#include "CLIcore_script.h"
#include "CLIcore_signals.h"
#include "CLIcore_UI_execute.h"
#include "treesitter/cli_treesitter.h"

#include <fnmatch.h>
#include <glob.h>
#include <sys/wait.h>


#include "timeutils.h"

#define COLORRED "\001\033[31m\002"
#define COLORHBOLDCYAN "\001\e[0;96m\002"
#define COLORDIMYELLOW "\033[2;33m"
#define COLORRST "\033[0m"
#define RL_COLORRESET "\001\033[0m\002"


/* ---- String utilities ---- */

void *xmalloc(int size)
{
    void *buf;

    buf = malloc(size);
    if (!buf)
    {
        fprintf(stderr, COLORRED "Error: Out of memory. Exiting.'n" COLORRESET);
        exit(1);
    }

    return buf;
}

/**
 * @brief Duplicate a string using xmalloc.
 *
 * Allocates memory for a copy of @s and returns
 * the copy. The caller must free() the result.
 *
 * @param s  String to duplicate
 * @return Newly allocated copy of @s
 */
char *dupstr(const char *s)
{
    char *r;

    size_t len = strlen(s) + 1;
    r          = (char *) xmalloc(len);
    memcpy(r, s, len);
    return (r);
}


/* ---- Readline callback and prompt ---- */

#ifdef USE_READLINE

/**
 * Number of ghost chars rendered on current line.
 * Set by print_ghost(), read by cli_accept_line().
 */
int ghost_chars_on_line = 0;

/**
 * @brief Custom accept-line handler for readline
 *
 * Bound to Enter key. Overwrites ghost suggestion
 * text with spaces before accepting the line, so
 * the terminal scrollback entry is clean.
 */
int cli_accept_line(int count, int key)
{
    if (ghost_chars_on_line > 0)
    {
        int n = ghost_chars_on_line;
        for (int i = 0; i < n; i++)
        {
            putchar(' ');
        }
        for (int i = 0; i < n; i++)
        {
            putchar('\b');
        }
        fflush(stdout);
        ghost_chars_on_line = 0;
    }

    return rl_newline(count, key);
}

/**
 * @brief Readline callback handler — processes a
 *        completed input line.
 *
 * Invoked by rl_callback_read_char() when the user
 * presses Enter. Copies the input into
 * data.CLIcmdline, handles backslash line
 * continuation (reading extra lines until no
 * trailing backslash), then dispatches the
 * assembled command via CLI_execute_line().
 *
 * If linein is NULL (Ctrl-D / EOF), sets
 * data.CLIloopON=0 to exit the main loop.
 *
 * @param linein  Line text from readline
 *                (caller-allocated, freed here)
 */
/**
 * @brief Execute a multi-line buffer statement by statement
 *
 * Splits the accumulated buffer by newlines occurring outside of quotes,
 * executing each line with fault isolation.
 *
 * @param buffer  Multi-line input buffer
 */
static void cli_execute_single_segment(const char *cmd)
{
    const char *p = cmd;
    while (*p == ' ' || *p == '\t' || *p == '\r')
    {
        p++;
    }
    if (*p == '\0')
    {
        return;
    }

    strncpy(data.CLIcmdline, p, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    size_t len = strlen(data.CLIcmdline);
    while (len > 0 && (data.CLIcmdline[len - 1] == ' ' || data.CLIcmdline[len - 1] == '\t' ||
                       data.CLIcmdline[len - 1] == '\r'))
    {
        data.CLIcmdline[--len] = '\0';
    }
    if (len == 0)
    {
        return;
    }

    cli_history_expand();
    cli_fault_isolation_arm();
    if (sigsetjmp(*cli_get_repl_env(), 1) == 0)
    {
        CLI_execute_line();
    }
    else
    {
        dcsigINT  = 0;
        dcsigSEGV = 0;
        dcsigBUS  = 0;
        dcsigABRT = 0;
        rl_on_new_line();
    }
    cli_fault_isolation_disarm();
}

static void cli_execute_multiline(const char *buffer)
{
    char   line[STRINGMAXLEN_CLICMDLINE];
    size_t line_len    = 0;
    int    in_dquote   = 0;
    int    in_squote   = 0;
    int    paren_depth = 0;

    for (size_t i = 0; buffer[i] != '\0'; i++)
    {
        char c = buffer[i];

        if (c == '\\' && buffer[i + 1] != '\0' && !in_squote)
        {
            if (line_len < sizeof(line) - 1)
            {
                line[line_len++] = c;
            }
            if (line_len < sizeof(line) - 1)
            {
                line[line_len++] = buffer[++i];
            }
            continue;
        }

        if (c == '"' && !in_squote)
        {
            in_dquote = !in_dquote;
        }
        else if (c == '\'' && !in_dquote)
        {
            in_squote = !in_squote;
        }
        else if (!in_dquote && !in_squote)
        {
            if (c == '(')
            {
                paren_depth++;
            }
            else if (c == ')' && paren_depth > 0)
            {
                paren_depth--;
            }
        }

        /* Check for statement separator: newline or semicolon outside quotes/parens */
        bool is_sep = false;
        if (!in_dquote && !in_squote)
        {
            if (c == '\n')
            {
                is_sep = true;
            }
            else if (c == ';' && paren_depth == 0)
            {
                /* Keep ';;' in case statements together; split on second semicolon */
                if (buffer[i + 1] != ';')
                {
                    is_sep = true;
                }
            }
        }

        if (is_sep)
        {
            line[line_len] = '\0';
            cli_execute_single_segment(line);
            line_len = 0;
            continue;
        }

        if (line_len < sizeof(line) - 1)
        {
            line[line_len++] = c;
        }
    } // for (size_t i = 0; buffer[i] != '\0'; i++)

    if (line_len > 0)
    {
        line[line_len] = '\0';
        cli_execute_single_segment(line);
    }
}

/**
 * @brief Compress a multi-line buffer into a single-line command for readline history
 *
 * Emulates bash cmdhist behavior: strips line-breaks and separates
 * statements with semicolons, avoiding embedded newlines in readline's line buffer.
 *
 * @param multiline  Input multi-line buffer
 * @param single     Output single-line buffer
 * @param maxlen     Capacity of output buffer
 */
static void cli_multiline_to_single_line(const char *multiline, char *single, size_t maxlen)
{
    single[0] = '\0';
    if (multiline == NULL || multiline[0] == '\0')
    {
        return;
    }

    size_t      out_len = 0;
    const char *p       = multiline;

    while (*p != '\0')
    {
        const char *nl      = strchr(p, '\n');
        size_t      linelen = nl ? (size_t) (nl - p) : strlen(p);

        /* Trim leading whitespace */
        const char *line = p;
        while (linelen > 0 && (*line == ' ' || *line == '\t' || *line == '\r'))
        {
            line++;
            linelen--;
        }
        /* Trim trailing whitespace */
        while (linelen > 0 &&
               (line[linelen - 1] == ' ' || line[linelen - 1] == '\t' || line[linelen - 1] == '\r'))
        {
            linelen--;
        }

        if (linelen > 0)
        {
            if (out_len > 0)
            {
                char lastc     = single[out_len - 1];
                int  need_semi = 1;

                if (lastc == ';' || lastc == '|' || lastc == '&' || lastc == '\\')
                {
                    need_semi = 0;
                }
                else
                {
                    size_t wlen = 0;
                    while (wlen < out_len && single[out_len - 1 - wlen] != ' ' &&
                           single[out_len - 1 - wlen] != '\t')
                    {
                        wlen++;
                    }
                    const char *last_word = single + out_len - wlen;
                    if (strcmp(last_word, "do") == 0 || strcmp(last_word, "then") == 0 ||
                        strcmp(last_word, "else") == 0 || strcmp(last_word, "{") == 0)
                    {
                        need_semi = 0;
                    }
                }

                if (need_semi)
                {
                    if (out_len + 2 < maxlen)
                    {
                        single[out_len++] = ';';
                        single[out_len++] = ' ';
                        single[out_len]   = '\0';
                    }
                }
                else
                {
                    if (out_len + 1 < maxlen)
                    {
                        single[out_len++] = ' ';
                        single[out_len]   = '\0';
                    }
                }
            }

            size_t copy_len = linelen;
            if (out_len + copy_len >= maxlen)
            {
                copy_len = maxlen - out_len - 1;
            }
            if (copy_len > 0)
            {
                memcpy(single + out_len, line, copy_len);
                out_len += copy_len;
                single[out_len] = '\0';
            }
        }

        if (!nl)
        {
            break;
        }
        p = nl + 1;
    }
}

static int g_auto_indent_spaces     = 0;
static int g_in_continuation_prompt = 0;

int cli_is_continuation_prompt(void)
{
    return g_in_continuation_prompt;
}

static int cli_auto_indent_startup_hook(void)
{
    if (g_auto_indent_spaces > 0)
    {
        for (int i = 0; i < g_auto_indent_spaces; i++)
        {
            rl_insert_text(" ");
        }
    }
    return 0;
}

void rl_cb_linehandler(char *linein)
{
    if (NULL == linein)
    {
        data.CLIloopON = 0;
        return;
    }

    data.CLIexecuteCMDready = 1;

    char multiline_buf[16384];
    strncpy(multiline_buf, linein, sizeof(multiline_buf) - 1);
    multiline_buf[sizeof(multiline_buf) - 1] = '\0';

    int had_continuation = 0;

    /* Handle multi-line continuation:
     * both backslash continuation and tree-sitter syntactic continuation */
    if (cli_ts_is_incomplete(multiline_buf))
    {
        had_continuation = 1;
        rl_callback_handler_remove();

        while (cli_ts_is_incomplete(multiline_buf))
        {
            size_t len       = strlen(multiline_buf);
            int    is_bslash = (len > 0 && multiline_buf[len - 1] == '\\');
            if (is_bslash)
            {
                multiline_buf[len - 1] = ' ';
            }

            const char *ps2 = cli_var_get("PS2");
            if (ps2 == NULL || ps2[0] == '\0')
            {
                ps2 = "> ";
            }

            int depth         = cli_ts_compute_indent_depth(multiline_buf);
            int indent_spaces = (data.auto_indent > 0) ? (depth * data.auto_indent) : 0;

            g_auto_indent_spaces     = indent_spaces;
            g_in_continuation_prompt = 1;
            rl_startup_hook          = cli_auto_indent_startup_hook;
            char *cont               = readline(ps2);
            rl_startup_hook          = NULL;
            g_in_continuation_prompt = 0;
            g_auto_indent_spaces     = 0;

            if (cont == NULL)
            {
                /* Interrupted or EOF (Ctrl-C / Ctrl-D) */
                multiline_buf[0] = '\0';
                break;
            }

            /* Adjust indentation for closing tokens typed by user */
            char line_to_add[4096];
            line_to_add[0] = '\0';

            const char *cstart = cont;
            while (*cstart == ' ' || *cstart == '\t')
            {
                cstart++;
            }

            if (data.auto_indent > 0 && depth > 0 &&
                (strncmp(cstart, "done", 4) == 0 || strncmp(cstart, "fi", 2) == 0 ||
                 strncmp(cstart, "esac", 4) == 0 || strncmp(cstart, "}", 1) == 0 ||
                 strncmp(cstart, "else", 4) == 0 || strncmp(cstart, "elif", 4) == 0))
            {
                int closer_depth = (depth > 0) ? (depth - 1) : 0;
                int nsp          = closer_depth * data.auto_indent;
                for (int s = 0; s < nsp && s < 64; s++)
                {
                    line_to_add[s]     = ' ';
                    line_to_add[s + 1] = '\0';
                }
                strncat(line_to_add, cstart, sizeof(line_to_add) - strlen(line_to_add) - 1);
            }
            else
            {
                strncpy(line_to_add, cont, sizeof(line_to_add) - 1);
                line_to_add[sizeof(line_to_add) - 1] = '\0';
            }

            size_t curlen  = strlen(multiline_buf);
            size_t contlen = strlen(line_to_add);
            if (curlen + 2 + contlen < sizeof(multiline_buf))
            {
                if (!is_bslash)
                {
                    /* Check if preceding non-space was pipe or logical op */
                    size_t trimmed = curlen;
                    while (trimmed > 0 && (multiline_buf[trimmed - 1] == ' ' ||
                                           multiline_buf[trimmed - 1] == '\t'))
                    {
                        trimmed--;
                    }
                    int join_space = 0;
                    if (trimmed > 0)
                    {
                        char lastc = multiline_buf[trimmed - 1];
                        if (lastc == '|' || lastc == '&')
                        {
                            join_space = 1;
                        }
                    }
                    multiline_buf[curlen++] = join_space ? ' ' : '\n';
                    multiline_buf[curlen]   = '\0';
                }
                strncat(multiline_buf, line_to_add, sizeof(multiline_buf) - curlen - 1);
            }
            free(cont);
        }
    }

    if (multiline_buf[0] == '\0')
    {
        if (had_continuation)
        {
            rl_callback_handler_install(cli_get_active_prompt(),
                                        (rl_vcpfunc_t *) &rl_cb_linehandler);
        }
        free(linein);
        return;
    }

    /* Record flattened single-line representation in history
     * to avoid embedded newlines corrupting readline cursor display */
    char history_entry[16384];
    cli_multiline_to_single_line(multiline_buf, history_entry, sizeof(history_entry));
    if (history_entry[0] != '\0')
    {
        add_history(history_entry);
        cli_history_log_prompt(history_entry);
        if (data.autocomplete_history)
        {
            append_history(1, CLI_history_file());
            history_truncate_file(CLI_history_file(), 10000);
        }
    }

    if (data.echo_input)
    {
        printf("\033[32m[echo]\033[0m \u2190 \"%s\"\n", multiline_buf);
    }

    /* Execute the accumulated multi-line block */
    cli_execute_multiline(multiline_buf);

    if (had_continuation)
    {
        rl_callback_handler_install(cli_get_active_prompt(), (rl_vcpfunc_t *) &rl_cb_linehandler);
    }

    free(linein);
}

static char cli_active_prompt[FPS_DIR_STRLENMAX] = "";

void cli_set_active_prompt(const char *prompt)
{
    if (prompt != NULL)
    {
        strncpy(cli_active_prompt, prompt, sizeof(cli_active_prompt) - 1);
        cli_active_prompt[sizeof(cli_active_prompt) - 1] = '\0';
    }
}

const char *cli_get_active_prompt(void)
{
    if (cli_active_prompt[0] == '\0')
    {
        runCLI_prompt("", cli_active_prompt);
    }
    return cli_active_prompt;
}
#endif

/**
 * @brief Build the prompt string for the CLI
 *
 * Checks for a PS1 variable in CLI vars or the
 * environment. Falls back to the default colored
 * prompt with the process name.
 */
errno_t runCLI_prompt(char *promptstring, char *prompt)
{
    /* Use PS1 only from CLI vars (set inside
     * milk-cli).  Do NOT fall back to
     * getenv("PS1") — the bash PS1 contains
     * shell-specific escapes like $(cmd) that
     * cli_expand_env cannot evaluate, which
     * would corrupt the prompt. */
    const char *ps1_val = cli_var_get("PS1");

    if (ps1_val != NULL && strlen(ps1_val) > 0)
    {
        char expanded_ps1[FPS_DIR_STRLENMAX];
        strncpy(expanded_ps1, ps1_val, FPS_DIR_STRLENMAX - 1);
        expanded_ps1[FPS_DIR_STRLENMAX - 1] = '\0';
        cli_expand_env(expanded_ps1, FPS_DIR_STRLENMAX);
        strncpy(prompt, expanded_ps1, FPS_DIR_STRLENMAX - 1);
        prompt[FPS_DIR_STRLENMAX - 1] = '\0';
        return RETURN_SUCCESS;
    }

    if (strlen(promptstring) > 0)
    {
        if (data.processnameflag == 0)
        {
            snprintf(prompt, FPS_DIR_STRLENMAX, COLORHBOLDCYAN "%s > " RL_COLORRESET, promptstring);
        }
        else
        {
            snprintf(prompt, FPS_DIR_STRLENMAX, COLORHBOLDCYAN "%s-%s > " RL_COLORRESET,
                     promptstring, data.processname);
        }
    }
    else
    {
        snprintf(prompt, FPS_DIR_STRLENMAX, COLORHBOLDCYAN "%s > " RL_COLORRESET, data.processname);
    }

    return RETURN_SUCCESS;
}


/* ---- Levenshtein distance (fuzzy matching) ---- */

#ifdef USE_READLINE

/**
 * @brief Compute Levenshtein edit distance
 *
 * Used to suggest similar commands when a typed
 * command is not found ("did you mean?").
 */
int levenshtein_distance(const char *s1, const char *s2)
{
    unsigned int  len1 = strlen(s1);
    unsigned int  len2 = strlen(s2);
    unsigned int *d    = (unsigned int *) xmalloc((len1 + 1) * (len2 + 1) * sizeof(unsigned int));

    for (unsigned int i = 0; i <= len1; i++)
    {
        d[i * (len2 + 1)] = i;
    }
    for (unsigned int j = 0; j <= len2; j++)
    {
        d[j] = j;
    }

    for (unsigned int i = 1; i <= len1; i++)
    {
        for (unsigned int j = 1; j <= len2; j++)
        {
            unsigned int cost     = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            unsigned int min1     = d[(i - 1) * (len2 + 1) + j] + 1;
            unsigned int min2     = d[i * (len2 + 1) + j - 1] + 1;
            unsigned int min3     = d[(i - 1) * (len2 + 1) + j - 1] + cost;
            unsigned int m        = (min1 < min2) ? min1 : min2;
            d[i * (len2 + 1) + j] = (m < min3) ? m : min3;
        }
    }
    int dist = d[len1 * (len2 + 1) + len2];
    free(d);
    return dist;
}

#endif

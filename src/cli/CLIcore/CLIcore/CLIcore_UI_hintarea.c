// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_hintarea.c
 *
 * @brief Inline ghost-text suggestions and hint area
 *
 * Provides the "fish-shell-like" inline suggestion
 * system for the milk CLI. When the user types, this
 * module renders ghost text (dimmed) after the cursor
 * showing the best-matching completion or history
 * entry. Pressing Right Arrow accepts the suggestion.
 *
 * ## Hint area (bottom-of-screen)
 *
 * A reserved bottom line of the terminal shows the
 * full syntax of the command being typed, with the
 * current argument position highlighted in bold.
 * This is implemented using ANSI scroll regions:
 * normal terminal output is confined to lines
 * 1..(rows-1), keeping the hint line fixed.
 *
 * ## Ghost text
 *
 * After each keystroke, `CLI_redisplay()` checks:
 * 1. History matches (prefix-based, most recent first)
 * 2. Generator matches (command/image/FPS names)
 * 3. Fuzzy generator matches (substring)
 *
 * The suggestion is rendered in dim gray after the
 * cursor. If accepted, the text is inserted into the
 * readline buffer; if rejected (any other key), it
 * disappears on next redisplay.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdint.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"


#ifdef USE_READLINE

/* ---- Suggestion state ---- */

/**
 * @brief Current inline suggestion suffix
 *
 * Set by CLI_redisplay when a suggestion is shown.
 * Consumed by accept_suggestion when Right Arrow
 * is pressed.
 */
char *pending_suggestion  = NULL;
int   pending_replace_len = 0;

/**
 * @brief Accept the inline suggestion on Right Arrow
 *
 * If cursor is at end-of-line and a pending suggestion
 * exists, insert it. Otherwise, fall through to normal
 * cursor-right movement.
 */
int accept_suggestion(int count, int key)
{
    (void) count;
    (void) key;

    if (pending_suggestion && rl_point == rl_end)
    {
        if (pending_replace_len > 0 && rl_end >= pending_replace_len)
        {
            int del_start = rl_end - pending_replace_len;
            rl_delete_text(del_start, rl_end);
            rl_point = del_start;
        }
        rl_insert_text(pending_suggestion);
        free(pending_suggestion);
        pending_suggestion  = NULL;
        pending_replace_len = 0;
        rl_redisplay();
        return 0;
    }

    /* Not at EOL or no suggestion — normal right */
    return rl_forward_char(1, key);
}

/**
 * @brief Store the suggestion suffix for Right Arrow
 */
void set_pending_suggestion(const char *text, int replace_len)
{
    free(pending_suggestion);
    pending_suggestion  = NULL;
    pending_replace_len = 0;
    if (text && strlen(text) > 0)
    {
        pending_suggestion  = dupstr((char *) text);
        pending_replace_len = replace_len;
    }
}


/* ---- Command matching helper ---- */

/**
 * @brief Find the command index matching firstword
 *
 * Searches the command table for an exact match.
 * Also sets data.cmdindex as a side effect.
 *
 * @return Command index, or -1 if not found
 */
int find_command_match(const char *firstword)
{
    for (uint32_t cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {
        if (strcmp(firstword, data.cmd[cmdi].key) == 0)
        {
            data.cmdindex = cmdi;
            return (int) cmdi;
        }
    }
    return -1;
}


/* ---- Terminal geometry helpers ---- */

/**
 * @brief Compute visible length of readline prompt
 *
 * Strips \\001..\\002 escape wrappers that readline
 * uses to mark non-printing characters.
 */
int visible_prompt_len(void)
{
    const char *p         = rl_display_prompt ? rl_display_prompt : "";
    int         len       = 0;
    int         invisible = 0;

    for (; *p; p++)
    {
        if (*p == '\001')
        {
            invisible = 1;
        }
        else if (*p == '\002')
        {
            invisible = 0;
        }
        else if (!invisible)
        {
            len++;
        }
    }
    return len;
}

/**
 * @brief Get number of ghost chars that fit on line
 *
 * Returns max chars that can be printed after the
 * cursor without wrapping to the next terminal line.
 */
int get_ghost_budget(void)
{
    struct winsize ws;
    int            cols = 80; /* fallback */

    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) >= 0 && ws.ws_col > 0)
    {
        cols = ws.ws_col;
    }

    int cursor_col = (visible_prompt_len() + rl_point) % cols;

    int budget = cols - cursor_col - 1;
    if (budget < 0)
    {
        budget = 0;
    }
    return budget;
}

/**
 * @brief Print ghost text with truncation
 *
 * Prints up to budget visible chars from text
 * in the given ANSI style, then resets style.
 *
 * @return Number of visible chars printed
 */
int print_ghost(const char *style, const char *text, int budget)
{
    int tlen = (int) strlen(text);
    int plen = tlen < budget ? tlen : budget;

    if (plen <= 0)
    {
        return 0;
    }

    printf("%s%.*s\033[0m", style, plen, text);
    ghost_chars_on_line = plen;
    return plen;
}


/* ---- Hint area (reserved bottom line) ---- */

/** @brief State for the reserved hint area */
int hint_area_active = 0;
int cached_term_rows = 0;
int cached_term_cols = 0;

/**
 * @brief Set up scroll region reserving bottom line
 *
 * Confines normal terminal output to lines
 * 1..(rows-1) so the bottom line stays fixed for
 * the command-syntax hint display.
 */
void CLI_setup_hint_area(void)
{
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) < 0 || ws.ws_row <= 3)
    {
        hint_area_active = 0;
        return;
    }

    cached_term_rows = ws.ws_row;
    cached_term_cols = ws.ws_col;

    /* Scroll up to ensure bottom line is free */
    printf("\n\033[1A");

    /* Save cursor + set scroll region + clear
     * hint line + restore cursor */
    printf("\033[s");
    printf("\033[1;%dr", cached_term_rows - 1);
    printf("\033[%d;1H\033[2K", cached_term_rows);
    printf("\033[u");
    fflush(stdout);

    hint_area_active = 1;
}

/**
 * @brief Reset scroll region to full terminal
 *
 * Call this before exiting readline mode or when
 * the CLI session ends.
 */
void CLI_cleanup_scroll_region(void)
{
    if (!hint_area_active)
    {
        return;
    }

    printf("\033[s");
    printf("\033[%d;1H\033[2K", cached_term_rows);
    printf("\033[r");
    printf("\033[u");
    fflush(stdout);

    hint_area_active = 0;
}

/**
 * @brief Update the hint area with function syntax
 *
 * Paints the reserved bottom line with the command's
 * argument syntax when a known command is being typed.
 * The current argument position is highlighted in
 * bold white; other arguments are shown dimmed.
 */
void update_hint_area(void)
{
    if (!hint_area_active || !data.autocomplete_arghint)
    {
        return;
    }

    /* Save cursor for entire operation */
    printf("\033[s");

    /* Check for terminal resize */
    {
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) >= 0 && ws.ws_row > 3 &&
            (ws.ws_row != cached_term_rows || ws.ws_col != cached_term_cols))
        {
            cached_term_rows = ws.ws_row;
            cached_term_cols = ws.ws_col;
            printf("\033[1;%dr", cached_term_rows - 1);
            printf("\033[%d;1H\033[2K", cached_term_rows);
        }
    }

    /* Move to hint line, clear it */
    printf("\033[%d;1H\033[2K", cached_term_rows);

    /* Check if first word is a known command */
    if (rl_line_buffer[0] != '\0')
    {
        char  buf[200];
        char *saveptr_hint = NULL;
        snprintf(buf, sizeof(buf), "%s", rl_line_buffer);
        char *fw = strtok_r(buf, " ", &saveptr_hint);

        if (fw != NULL)
        {
            int cmi = find_command_match(fw);
            if (cmi >= 0)
            {
                /* Count argument words after
                 * cmd to determine current
                 * argument index */
                int argidx = 0;
                {
                    const char *p = rl_line_buffer;
                    while (*p && *p != ' ')
                    {
                        p++;
                    }
                    int wcount  = 0;
                    int in_word = 0;
                    while (*p)
                    {
                        if (*p != ' ')
                        {
                            if (!in_word)
                            {
                                wcount++;
                                in_word = 1;
                            }
                        }
                        else
                        {
                            in_word = 0;
                        }
                        p++;
                    }
                    if (rl_end > 0 && rl_line_buffer[rl_end - 1] == ' ')
                    {
                        argidx = wcount;
                    }
                    else
                    {
                        argidx = wcount > 0 ? wcount - 1 : 0;
                    }
                }

                /* Print syntax with <> tokens,
                 * highlighting current arg */
                const char *syn  = data.cmd[cmi].syntax;
                int         col  = 0;
                int         tidx = 0;
                const char *p    = syn;

                while (*p && col < cached_term_cols - 2)
                {
                    if (*p == ' ')
                    {
                        printf(" ");
                        col++;
                        p++;
                        continue;
                    }

                    const char *tstart = p;
                    if (*p == '<')
                    {
                        while (*p && *p != '>')
                        {
                            p++;
                        }
                        if (*p == '>')
                        {
                            p++;
                        }
                    }
                    else
                    {
                        while (*p && *p != ' ' && *p != '<')
                        {
                            p++;
                        }
                    }
                    int tlen  = (int) (p - tstart);
                    int avail = cached_term_cols - 1 - col;
                    int plen  = tlen < avail ? tlen : avail;

                    if (*tstart == '<' && tidx == argidx)
                    {
                        printf("\033[1;97m"
                               "%.*s"
                               "\033[0m",
                               plen, tstart);
                    }
                    else
                    {
                        printf("\033[2m"
                               "%.*s"
                               "\033[0m",
                               plen, tstart);
                    }
                    col += plen;
                    if (*tstart == '<')
                    {
                        tidx++;
                    }
                }
            }
        }
    }

    /* Restore cursor */
    printf("\033[u");
    fflush(stdout);
}


/* ---- Main redisplay function ---- */

/**
 * @brief Combined redisplay: highlight + ghost + hint
 *
 * Installed as readline's redisplay function. On each
 * keystroke it:
 * 1. Calls syntax-highlighted or normal redisplay
 * 2. Checks for history-based inline suggestions
 * 3. Falls back to generator-based suggestions
 * 4. Updates the bottom hint area
 */
void CLI_redisplay(void)
{
    /* Erase stale ghost text before rl_redisplay().
     * Ghost text is written outside readline's buffer.
     * If not erased, readline's optimized redraw does
     * not know to overwrite it, and characters typed
     * into overlapping positions may not appear. */
    if (ghost_chars_on_line > 0)
    {
        printf("\033[K");
        ghost_chars_on_line = 0;
    }

#    if RL_READLINE_VERSION >= 0x0600
    /* If incremental search (Ctrl-R) is active, fall back
     * entirely to readline's built-in redisplay to avoid
     * corrupting the search prompt. */
    if (RL_ISSTATE(RL_STATE_ISEARCH) || RL_ISSTATE(RL_STATE_NSEARCH))
    {
        rl_redisplay_function = NULL;
        rl_redisplay();
        fflush(stdout);
        rl_redisplay_function = CLI_redisplay;
        return;
    }
#    endif

    /* Default or syntax-highlighted redisplay */
    rl_redisplay_function = NULL;
    if (data.syntax_highlight && rl_line_buffer[0] != '\0')
    {
        cli_highlight_redisplay();
    }
    else
    {
        rl_redisplay();
        fflush(stdout);
    }
    rl_redisplay_function = CLI_redisplay;

    /* Clear any stale suggestion */
    set_pending_suggestion(NULL, 0);

    if (data.autocomplete == 0)
    {
        return;
    }

    if (rl_line_buffer[0] == '\0')
    {
        update_hint_area();
        return;
    }

    if (rl_point != rl_end)
    {
        update_hint_area();
        return;
    }

    int budget = get_ghost_budget();
    if (budget <= 0)
    {
        update_hint_area();
        return;
    }

    int total_ghost = 0;

    /* ===== History-based suggestion ===== */
    if (data.autocomplete_history)
    {
        HIST_ENTRY **hist = history_list();
        if (hist)
        {
            int hlen = history_length;
            for (int i = hlen - 1; i >= 0; i--)
            {
                if (strncmp(hist[i]->line, rl_line_buffer, rl_end) == 0 &&
                    (int) strlen(hist[i]->line) > rl_end)
                {
                    const char *suffix = hist[i]->line + rl_end;
                    int         n      = print_ghost("\033[38;5;245m", suffix, budget);
                    if (n > 0)
                    {
                        printf("\033[K");
                        printf("\033[%dD", n);
                        fflush(stdout);
                        set_pending_suggestion(suffix, 0);
                    }
                    update_hint_area();
                    return;
                }
            }
        }
    }

    /* ===== Generator-based suggestion ===== */

    /* Find current word start */
    int start = 0;
    for (int i = rl_point - 1; i >= 0; i--)
    {
        if (rl_line_buffer[i] == ' ')
        {
            start = i + 1;
            break;
        }
    }

    char *text = rl_line_buffer + start;

    /* Determine matching mode */
    if ((start == 0) || (strncmp(rl_line_buffer, "cmd?", strlen("cmd?")) == 0))
    {
        data.CLImatchMode = 0; /* COMMANDS */
    }
    else
    {
        char  str[200];
        char *saveptr_comp = NULL;
        snprintf(str, 200, "%s", rl_line_buffer);
        char *firstword = strtok_r(str, " ", &saveptr_comp);

        int cmdimatch = -1;
        if (firstword != NULL)
        {
            cmdimatch = find_command_match(firstword);
        }

        /* If command has no <> argument tokens,
         * don't suggest arguments */
        if (cmdimatch >= 0)
        {
            const char *syn = data.cmd[cmdimatch].syntax;
            if (syn == NULL || strchr(syn, '<') == NULL)
            {
                update_hint_area();
                return;
            }
        }

        if ((cmdimatch != -1) && (text[0] == '.'))
        {
            data.CLImatchMode = 2; /* CMDARGS */
        }
        else
        {
            data.CLImatchMode = 1; /* IMAGES */
        }
    }

    /* Get best match */
    char *match = CLI_generator(text, 0);

    if (match)
    {
        if (strncmp(match, text, strlen(text)) == 0)
        {
            char *suffix = match + strlen(text);
            int   n      = print_ghost("\033[38;5;245m", suffix, budget);
            if (n > 0)
            {
                total_ghost += n;
                set_pending_suggestion(suffix, 0);
            }
        }
        else if (data.autocomplete_fuzzy)
        {
            char fzbuf[256];
            snprintf(fzbuf, sizeof(fzbuf), " [%s]", match);
            int n = print_ghost("\033[38;5;245m", fzbuf, budget);
            total_ghost += n;
            set_pending_suggestion(match, (int) strlen(text));
        }
        free(match);
    }

    /* Erase rest of line + move cursor back */
    if (total_ghost > 0)
    {
        printf("\033[K");
        printf("\033[%dD", total_ghost);
        fflush(stdout);
    }

    update_hint_area();
}


/* ---- Readline configuration ---- */

/**
 * @brief Configure readline with custom handlers
 *
 * Installs the combined redisplay function, loads
 * saved history, and binds Right Arrow to accept
 * inline suggestions.
 */
void CLI_configure_readline()
{
    rl_redisplay_function = CLI_redisplay;

    if (data.autocomplete_history)
    {
        read_history(CLI_history_file());
    }

    /* Set custom word break characters to prevent readline
     * from splitting on @ and $, ensuring full variable tokens
     * (e.g. @fps.run or ${s.sz}) reach the completion generator. */
    rl_completer_word_break_characters = " \t\n\"\\'<>=;|&(";

    /* Bind Right Arrow to accept suggestion */
    rl_bind_keyseq("\\e[C", accept_suggestion);

    /* Bind Enter to clear ghost text first */
    rl_bind_key('\r', cli_accept_line);
    rl_bind_key('\n', cli_accept_line);
}
#else
/* Stubs when readline is not available */
void CLI_configure_readline()
{
}
/**
 * @brief Set up the hint/completion area below the prompt.
 */
void CLI_setup_hint_area(void)
{
}
/**
 * @brief Restore terminal scroll region on exit.
 */
void CLI_cleanup_scroll_region(void)
{
}
#endif

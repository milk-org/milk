// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_completion_menu.c
 * @brief Interactive Zsh-style grid menu for readline tab-completion matches.
 *
 * Role & Goal:
 *   Intercepts the terminal when multiple tab-completion matches are available
 *   (`cli_menu_display_matches()`), rendering a live-highlighted multi-column
 *   grid below the prompt with arrow-key / Tab / Shift-Tab / PageUp / PageDown
 *   navigation, Enter selection, and Escape cancellation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#include "CLIcore.h"

#ifdef USE_READLINE

extern int cli_comp_start;
extern int cli_comp_end;
extern int ghost_chars_on_line;

void set_pending_suggestion(const char *text, int replace_len);

static int g_menu_just_selected = 0;

/**
 * cli_custom_getc - Readline character input hook that resets state after menu selection
 * @stream: Input FILE stream passed by readline
 *
 * Clears rl_last_func when an item was just chosen from the interactive
 * completion menu so a subsequent Tab press starts a fresh completion cycle,
 * then delegates character reading to rl_getc().
 *
 * Return: Next character read from @stream, or EOF
 */
int cli_custom_getc(FILE *stream)
{
    if (g_menu_just_selected)
    {
        rl_last_func         = NULL;
        g_menu_just_selected = 0;
    }
    return rl_getc(stream);
}

/**
 * menu_readchar - Read a single raw byte from standard input
 *
 * Performs an unbuffered 1-byte read() on STDIN_FILENO while the terminal is in
 * raw mode during interactive completion menu navigation.
 *
 * Return: Unsigned byte value (0..255) on success, or -1 on EOF/error
 */
static int menu_readchar(void)
{
    unsigned char ch;
    ssize_t       n = read(STDIN_FILENO, &ch, 1);
    return (n == 1) ? (int) ch : -1;
}

/**
 * menu_stdin_wait_ms - Wait up to a timeout for readable input on STDIN_FILENO
 * @ms: Timeout in milliseconds
 *
 * Uses select() on STDIN_FILENO to distinguish a bare Escape key press from a
 * multi-byte ANSI cursor movement sequence inside the completion menu.
 *
 * Return: 1 if input becomes available before the timeout expires, 0 otherwise
 */
static int menu_stdin_wait_ms(int ms)
{
    fd_set         fds;
    struct timeval tv;

    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    tv.tv_sec  = 0;
    tv.tv_usec = ms * 1000;
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

/**
 * @brief Deduplicate `matches[1..num_matches]` in place and return the unique count.
 */
static int dedup_completion_matches(char **matches, int num_matches)
{
    int uniq_count = 0;
    for (int i = 1; i <= num_matches; i++)
    {
        int is_dup = 0;
        for (int j = 1; j <= uniq_count; j++)
        {
            if (strcmp(matches[i], matches[j]) == 0)
            {
                is_dup = 1;
                break;
            }
        }
        if (!is_dup)
        {
            uniq_count++;
            if (uniq_count != i)
            {
                matches[uniq_count] = matches[i];
            }
        }
        else
        {
            free(matches[i]);
            matches[i] = NULL;
        }
    }
    matches[uniq_count + 1] = NULL;
    return uniq_count;
}

/**
 * @brief Replace the token being completed in `rl_line_buffer` with @cand.
 */
static void insert_selected_candidate(const char *cand)
{
    int del_from = cli_comp_start;
    int del_to   = rl_point;
    if (del_to < del_from)
    {
        del_to = del_from;
    }
    rl_delete_text(del_from, del_to);
    rl_point = del_from;
    rl_insert_text(cand);

    size_t clen = strlen(cand);
    if (clen > 0 && cand[clen - 1] != '.' && cand[clen - 1] != '/')
    {
        rl_insert_text(" ");
    }
}

/**
 * @brief Render the visible rows and status bar of the completion grid.
 */
static void draw_menu_grid(char **matches,
                           int    num_matches,
                           int    ncols,
                           int    col_width,
                           int    total_rows,
                           int    visible_rows,
                           int    selected,
                           int   *first_visible_row)
{
    int sel_row = selected / ncols;
    if (sel_row < *first_visible_row)
    {
        *first_visible_row = sel_row;
    }
    if (sel_row >= *first_visible_row + visible_rows)
    {
        *first_visible_row = sel_row - visible_rows + 1;
    }

    for (int vr = 0; vr < visible_rows; vr++)
    {
        int r = *first_visible_row + vr;
        printf("\033[2K\r");
        if (r < total_rows)
        {
            for (int c = 0; c < ncols; c++)
            {
                int idx = r * ncols + c;
                if (idx >= num_matches)
                {
                    break;
                }
                const char *cand = matches[idx + 1];
                if (idx == selected)
                {
                    printf("\033[7;1;36m %-*s\033[0m", col_width - 1, cand);
                }
                else
                {
                    printf(" %-*s", col_width - 1, cand);
                }
            }
        }
        printf("\n");
    }

    printf("\033[2K\r");
    int has_more_above = (*first_visible_row > 0);
    int has_more_below = (*first_visible_row + visible_rows < total_rows);
    printf("\033[36m\xe2\x94\x80\xe2\x94\x80\033[0m \033[1m[%d/%d]\033[0m %s%s\033[2m"
           "\xe2\x97\x84\xe2\x96\xb2\xe2\x96\xbc\xe2\x96\xba/Tab: navigate "
           "\xe2\x80\xa2 Enter: select \xe2\x80\xa2 Esc: cancel\033[0m "
           "\033[36m\xe2\x94\x80\xe2\x94\x80\033[0m",
           selected + 1, num_matches, has_more_above ? "\xe2\x96\xb2 " : "",
           has_more_below ? "\xe2\x96\xbc " : "");
    fflush(stdout);

    printf("\033[%dA\r", visible_rows);
    fflush(stdout);
}

/**
 * @brief Handle an escape sequence inside the interactive completion menu.
 *
 * @return 1 if selection changed (needs redraw), 0 if ignored, -1 if bare ESC (cancel)
 */
static int handle_menu_escape_seq(int *selected,
                                  int  num_matches,
                                  int  ncols,
                                  int  total_rows,
                                  int  visible_rows)
{
    if (!menu_stdin_wait_ms(50))
    {
        return -1;
    }
    int b1 = menu_readchar();
    if (b1 != '[' && b1 != 'O')
    {
        return 0;
    }
    int b2 = menu_readchar();
    if (b2 == 'A') /* Up */
    {
        *selected -= ncols;
        if (*selected < 0)
        {
            *selected += total_rows * ncols;
            if (*selected >= num_matches)
            {
                *selected -= ncols;
            }
        }
        return 1;
    }
    if (b2 == 'B') /* Down */
    {
        *selected += ncols;
        if (*selected >= num_matches)
        {
            *selected %= ncols;
        }
        return 1;
    }
    if (b2 == 'C') /* Right */
    {
        *selected = (*selected + 1) % num_matches;
        return 1;
    }
    if (b2 == 'D' || b2 == 'Z') /* Left or Shift-Tab */
    {
        *selected = (*selected - 1 + num_matches) % num_matches;
        return 1;
    }
    if (b2 == '5') /* PgUp */
    {
        menu_readchar();
        *selected -= (ncols * visible_rows);
        if (*selected < 0)
        {
            *selected = 0;
        }
        return 1;
    }
    if (b2 == '6') /* PgDn */
    {
        menu_readchar();
        *selected += (ncols * visible_rows);
        if (*selected >= num_matches)
        {
            *selected = num_matches - 1;
        }
        return 1;
    }
    return 0;
}

/**
 * cli_menu_display_matches - Interactive Zsh-style menu completion display hook
 * @matches:     Array of completion match strings (1-indexed, NULL-terminated)
 * @num_matches: Number of candidate strings in @matches
 * @max_length:  Maximum string length among the candidates
 *
 * Installed as rl_completion_display_matches_hook. Deduplicates matches, auto-inserts
 * if only one unique candidate remains, or enters a raw-terminal interactive grid
 * allowing arrow/Tab navigation and Enter selection before restoring the prompt.
 *
 * Return: void
 */
void cli_menu_display_matches(char **matches, int num_matches, int max_length)
{
    if (matches == NULL || num_matches <= 0)
    {
        return;
    }

    num_matches = dedup_completion_matches(matches, num_matches);
    if (num_matches <= 0)
    {
        return;
    }

    if (num_matches == 1)
    {
        insert_selected_candidate(matches[1]);
        if (ghost_chars_on_line > 0)
        {
            printf("\033[K");
            ghost_chars_on_line = 0;
        }
        set_pending_suggestion(NULL, 0);
        g_menu_just_selected = 1;
        rl_on_new_line();
        rl_forced_update_display();
        return;
    }

    if (ghost_chars_on_line > 0)
    {
        printf("\033[K");
        ghost_chars_on_line = 0;
    }
    set_pending_suggestion(NULL, 0);
    fflush(stdout);

    struct winsize ws;
    int            cols = 80;
    int            rows = 24;
    if (ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
    {
        cols = ws.ws_col;
        rows = ws.ws_row;
    }
    if (cols < 20 || rows < 5)
    {
        return;
    }

    int col_width = max_length + 2;
    if (col_width > cols - 2)
    {
        col_width = cols - 2;
    }
    if (col_width < 4)
    {
        col_width = 4;
    }

    int ncols = cols / col_width;
    if (ncols < 1)
    {
        ncols = 1;
    }
    int total_rows       = (num_matches + ncols - 1) / ncols;
    int max_visible_rows = 6;
    if (max_visible_rows > rows - 4)
    {
        max_visible_rows = rows - 4;
    }
    if (max_visible_rows < 1)
    {
        max_visible_rows = 1;
    }
    int visible_rows = (total_rows < max_visible_rows) ? total_rows : max_visible_rows;

    int selected          = 0;
    int first_visible_row = 0;
    int accepted          = 0;

    struct termios orig_termios;
    struct termios raw_termios;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_termios = orig_termios;
    raw_termios.c_lflag &= ~(ICANON | ECHO | ISIG);
    raw_termios.c_cc[VMIN]  = 1;
    raw_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios);

    printf("\033[?25l\n");
    for (int r = 0; r < visible_rows; r++)
    {
        printf("\033[2K\n");
    }
    printf("\033[2K\033[%dA\r", visible_rows);
    fflush(stdout);

    int redraw = 1;
    while (1)
    {
        if (redraw)
        {
            draw_menu_grid(matches, num_matches, ncols, col_width, total_rows, visible_rows,
                           selected, &first_visible_row);
            redraw = 0;
        }

        int c = menu_readchar();
        if (c == 27)
        {
            int esc_rc =
                handle_menu_escape_seq(&selected, num_matches, ncols, total_rows, visible_rows);
            if (esc_rc < 0)
            {
                accepted = 0;
                break;
            }
            if (esc_rc > 0)
            {
                redraw = 1;
            }
        }
        else if (c == '\t')
        {
            selected = (selected + 1) % num_matches;
            redraw   = 1;
        }
        else if (c == '\r' || c == '\n')
        {
            accepted = 1;
            break;
        }
        else if (c == 3 || c == 4 || c == 7)
        {
            accepted = 0;
            break;
        }
        else
        {
            accepted = 0;
            if (c > 0 && c != 27)
            {
                rl_stuff_char(c);
            }
            break;
        }
    } // while (1)

    for (int vr = 0; vr < visible_rows; vr++)
    {
        printf("\033[2K\n");
    }
    printf("\033[2K\033[%dA\r\033[2K\r", visible_rows + 1);
    fflush(stdout);

    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
    printf("\033[?25h");
    fflush(stdout);

    if (accepted && selected >= 0 && selected < num_matches)
    {
        insert_selected_candidate(matches[selected + 1]);
    }
    if (ghost_chars_on_line > 0)
    {
        ghost_chars_on_line = 0;
    }
    set_pending_suggestion(NULL, 0);
    g_menu_just_selected = 1;
    rl_on_new_line();
    rl_forced_update_display();
}

#else /* !USE_READLINE */

/**
 * cli_menu_display_matches - No-op completion display stub when readline is disabled
 * @matches:     Array of completion match strings (unused)
 * @num_matches: Number of candidate strings (unused)
 * @max_length:  Maximum candidate string length (unused)
 *
 * Provides link compatibility for non-readline builds.
 *
 * Return: void
 */
void cli_menu_display_matches(char **matches, int num_matches, int max_length)
{
    (void) matches;
    (void) num_matches;
    (void) max_length;
}

/**
 * cli_custom_getc - Fallback character input wrapper when readline is disabled
 * @stream: Input FILE stream to read from
 *
 * Reads the next character directly via fgetc() in builds without readline.
 *
 * Return: Next character read from @stream, or EOF
 */
int cli_custom_getc(FILE *stream)
{
    return fgetc(stream);
}

#endif /* USE_READLINE */

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-stream-graph_tui.c
 * @brief Interactive terminal browser for stream dependency graphs.
 */

#include "milk-stream-graph.h"

#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

/* =========================================================
 * Terminal raw mode
 * ========================================================= */

static struct termios sg_orig_termios;
static int            sg_raw_active = 0;

/**
 * @brief Enter raw terminal mode for graph display.
 */
static void sg_raw_enter(void)
{
    if (sg_raw_active)
    {
        return;
    }
    tcgetattr(STDIN_FILENO, &sg_orig_termios);
    struct termios raw = sg_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    /* Hide cursor */
    printf("\033[?25l");
    fflush(stdout);
    sg_raw_active = 1;
}

/**
 * @brief Exit raw terminal mode for graph display.
 */
static void sg_raw_exit(void)
{
    if (!sg_raw_active)
    {
        return;
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &sg_orig_termios);
    /* Show cursor, reset attrs */
    printf("\033[?25h" SGC_RESET "\n");
    fflush(stdout);
    sg_raw_active = 0;
}

/**
 * @brief Signal handler for stream graph tool.
 */
static void sg_sighandler(int sig)
{
    if (sig == SIGINT)
    {
        ov_sigINT = 1;
    }
    else if (sig == SIGTERM)
    {
        ov_sigTERM = 1;
    }
}

/* =========================================================
 * Interactive mode
 * ========================================================= */

/**
 * sg_interactive - Run interactive terminal lineage explorer
 * @model:          Pointer to data model
 * @initial_stream: Starting stream name
 * @mode:           Initial graph traversal mode
 */
void sg_interactive(
    OV_MODEL   *model,
    const char *initial_stream,
    sg_mode_t   mode)
{
    sg_raw_enter();

    struct sigaction sa;
    sa.sa_handler = sg_sighandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    char current_stream[STRINGMAXLEN_IMAGE_NAME];
    strncpy(current_stream, initial_stream, sizeof(current_stream) - 1);
    current_stream[sizeof(current_stream) - 1] = '\0';

    int        sel       = 0;
    int        need_scan = 1;
    SG_LINEAGE lin;
    int        total_items = 0;

    while (!OV_SIG_ANY_SET())
    {
        if (need_scan)
        {
            sg_scan_model(model);
            need_scan = 0;
        }

        int si = ov_find_stream_by_name(model, current_stream);

        memset(&lin, 0, sizeof(lin));
        if (si >= 0)
        {
            sg_compute_lineage(model, si, mode, &lin);
        }

        total_items = lin.nb_ancestors + lin.nb_descendants + 1;
        if (sel >= total_items && total_items > 0)
        {
            sel = total_items - 1;
        }
        if (sel < 0)
        {
            sel = 0;
        }

        /* Render */
        printf("\033[2J\033[H");
        printf(SGC_BOLD SGC_HEADER " milk-stream-graph" SGC_RESET SGC_TEXT
                                   "  stream: " SGC_BOLD SGC_STREAM "%s" SGC_RESET SGC_TEXT
                                   "  mode: " SGC_FPS "%s" SGC_RESET "\n",
               current_stream, sg_mode_label(mode));
        printf(SGC_DIM " q:quit r:rescan t/i/f:mode"
                       " ENTER:re-root g:goto" SGC_RESET "\n\n");

        if (si < 0)
        {
            printf(SGC_LOOP "  Stream '%s' not found\n" SGC_RESET, current_stream);
        }
        else
        {
            int row = 0;

            for (int i = lin.nb_ancestors - 1; i >= 0; i--)
            {
                const SG_LINEAGE_ENTRY *e  = &lin.ancestors[i];
                const char             *sn = model->streams[e->stream_idx].name;

                if (row == sel)
                {
                    printf("\033[7m");
                }
                printf(" " SGC_DEPTH "-%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET,
                       e->depth, sn);
                if (e->is_loop)
                {
                    printf(" " SGC_LOOP "[LOOP]" SGC_RESET);
                }
                if (row == sel)
                {
                    printf("\033[27m");
                }
                printf("\n");
                row++;

                if (e->via_name[0] != '\0')
                {
                    printf("      " SGC_ARROW "│" SGC_RESET "\n");
                    printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
                }
            }

            if (row == sel)
            {
                printf("\033[7m");
            }
            printf(" " SGC_DEPTH " 0 " SGC_RESET " " SGC_BOLD SGC_STREAM "%s" SGC_RESET,
                   current_stream);
            if (row == sel)
            {
                printf("\033[27m");
            }
            printf("\n");
            row++;

            for (int i = 0; i < lin.nb_descendants; i++)
            {
                const SG_LINEAGE_ENTRY *e  = &lin.descendants[i];
                const char             *sn = model->streams[e->stream_idx].name;

                if (e->via_name[0] != '\0')
                {
                    printf("      " SGC_ARROW "│" SGC_RESET "\n");
                    printf("      " SGC_ARROW "▼ " SGC_PROC "[%s]" SGC_RESET "\n", e->via_name);
                }

                if (row == sel)
                {
                    printf("\033[7m");
                }
                printf(" " SGC_DEPTH "+%-2d" SGC_RESET " " SGC_STREAM "%s" SGC_RESET,
                       e->depth, sn);
                if (e->is_loop)
                {
                    printf(" " SGC_LOOP "[LOOP]" SGC_RESET);
                }
                if (row == sel)
                {
                    printf("\033[27m");
                }
                printf("\n");
                row++;
            }

            /* Cycle */
            if (lin.has_loop && lin.cycle_len > 0)
            {
                printf("\n" SGC_BOLD SGC_LOOP " Cycle:" SGC_RESET " ");
                for (int c = 0; c < lin.cycle_len; c++)
                {
                    if (c > 0)
                    {
                        printf(SGC_ARROW " -> " SGC_RESET);
                    }
                    printf(SGC_STREAM "%s" SGC_RESET, model->streams[lin.cycle_path[c]].name);
                }
                printf("\n");
            }
        }
        fflush(stdout);

        /* Wait for input */
        struct pollfd pfd;
        pfd.fd     = STDIN_FILENO;
        pfd.events = POLLIN;

        if (poll(&pfd, 1, 200) <= 0)
        {
            continue;
        }

        char buf[8];
        int  n = (int) read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0)
        {
            continue;
        }

        if (buf[0] == 'q')
        {
            break;
        }
        else if (buf[0] == 'r')
        {
            need_scan = 1;
        }
        else if (buf[0] == 't')
        {
            mode = SG_MODE_TRIGGER;
        }
        else if (buf[0] == 'i')
        {
            mode = SG_MODE_INPUT;
        }
        else if (buf[0] == 'f')
        {
            mode = SG_MODE_FULL;
        }
        else if (buf[0] == '\n' || buf[0] == '\r')
        {
            /* Re-root on selected stream */
            if (total_items > 0)
            {
                int idx = -1;
                if (sel < lin.nb_ancestors)
                {
                    idx = lin.ancestors[lin.nb_ancestors - 1 - sel].stream_idx;
                }
                else if (sel == lin.nb_ancestors)
                {
                    /* selected root */
                }
                else
                {
                    idx = lin.descendants[sel - lin.nb_ancestors - 1].stream_idx;
                }

                if (idx >= 0)
                {
                    strncpy(current_stream, model->streams[idx].name, sizeof(current_stream) - 1);
                    sel       = 0;
                    need_scan = 1;
                }
            }
        }
        else if (buf[0] == 'g')
        {
            /* Go to stream by name */
            sg_raw_exit();
            printf("Enter stream name: ");
            fflush(stdout);
            char name[STRINGMAXLEN_IMAGE_NAME];
            if (fgets(name, sizeof(name), stdin) != NULL)
            {
                /* Trim newline */
                char *nl = strchr(name, '\n');
                if (nl)
                {
                    *nl = '\0';
                }
                if (name[0] != '\0')
                {
                    strncpy(current_stream, name, sizeof(current_stream) - 1);
                    sel       = 0;
                    need_scan = 1;
                }
            }
            sg_raw_enter();
        }
        else if (n >= 3 && buf[0] == '\033' && buf[1] == '[')
        {
            /* Arrow keys */
            if (buf[2] == 'A') /* UP */
            {
                if (sel > 0)
                {
                    sel--;
                }
            }
            else if (buf[2] == 'B') /* DOWN */
            {
                if (sel < total_items - 1)
                {
                    sel++;
                }
            }
        }
    } /* main loop */

    sg_raw_exit();
}

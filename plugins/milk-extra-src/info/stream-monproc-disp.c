// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream-monproc-disp.c
 * @brief   Display stream monitor info (ncurses)
 */

#define NCURSES_WIDECHAR 1

#include <curses.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "CommandLineInterface/CLIcore.h"
#include "TUItools.h"
#include "stream_monproc.h"

// External initialization functions from CLIcore
extern errno_t CLI_startup();
extern errno_t setSHMdir();
extern errno_t CLI_data_init();

static uint16_t wrow, wcol;

static void cleanup_ncurses()
{
    endwin();
}

static void print_bar(int len, int color_pair)
{
    attron(COLOR_PAIR(color_pair));
    for (int i = 0; i < len; i++)
    {
        addch(' ');
    }
    attroff(COLOR_PAIR(color_pair));
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <stream_name>\n", argv[0]);
        return 1;
    }

    char *streamname = argv[1];

    // Basic CLI init for SHM paths
    strncpy(data.processname, argv[0], STRINGMAXLEN_PROCESSNAME - 1);
    CLI_startup();
    setSHMdir();
    CLI_data_init();

    // Connect to monitor SHM
    STREAM_MON_STRUCT *smon = stream_monitor_connect(streamname, 0);
    if (!smon)
    {
        fprintf(stderr, "Error connecting to monitor SHM for stream %s\n", streamname);
        return 1;
    }

    // Init ncurses
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);
    curs_set(0);
    start_color();
    use_default_colors();
    atexit(cleanup_ncurses);

    // Color pairs
    init_pair(1, COLOR_WHITE, COLOR_BLUE);   // Header
    init_pair(2, COLOR_BLACK, COLOR_GREEN);  // Active
    init_pair(3, COLOR_WHITE, COLOR_RED);    // Inactive
    init_pair(4, COLOR_CYAN, COLOR_BLACK);   // Info
    init_pair(5, COLOR_YELLOW, COLOR_BLACK); // Warning
    init_pair(6, COLOR_WHITE, COLOR_CYAN);   // Histogram bar

    int ch;
    int loop = 1;

    // Local buffer for thresholds reconstruction
    float local_thresholds[STREAM_MON_MAX_HIST_BINS + 1];

    while (loop)
    {
        getmaxyx(stdscr, wrow, wcol);
        erase();

        // --------------------------------------------------------------------
        // Header
        // --------------------------------------------------------------------
        attron(COLOR_PAIR(1));
        mvprintw(0, 0, " Stream Monitor Display: %s (PID %d) - Press 'q' to exit ", streamname,
                 getpid());
        mvprintw(0, wcol - 20, " Count: %lu ", smon->cnt);
        attroff(COLOR_PAIR(1));

        int row = 2;

        // --------------------------------------------------------------------
        // Flux History (Recent)
        // --------------------------------------------------------------------
        mvprintw(row++, 0, "Recent Flux (last %d frames):", smon->size > 50 ? 50 : smon->size);

        // Find min/max for scaling
        double min_flux = 1e30, max_flux = -1e30;
        int    hist_len = 50;
        if (hist_len > smon->size)
        {
            hist_len = smon->size;
        }

        uint32_t current_idx = smon->cindex;

        for (int i = 0; i < hist_len; i++)
        {
            int    idx = (current_idx - i + smon->size) % smon->size; // Wrap around
            double val = smon->flux[idx];
            if (val < min_flux)
            {
                min_flux = val;
            }
            if (val > max_flux)
            {
                max_flux = val;
            }
        }

        if (max_flux == min_flux)
        {
            max_flux = min_flux + 1.0;
        }

        int bar_h = 5;
        // Draw a graph area
        for (int i = 0; i < hist_len; i++)
        {
            int    idx    = (current_idx - i + smon->size) % smon->size;
            double val    = smon->flux[idx];
            int    height = (int) ((val - min_flux) / (max_flux - min_flux) * bar_h);
            if (height < 0)
            {
                height = 0;
            }
            if (height > bar_h)
            {
                height = bar_h;
            }

            for (int h = 0; h < height; h++)
            {
                mvaddch(row + bar_h - h, i, '|');
            }
        }
        mvprintw(row, hist_len + 2, "Max: %.2e", max_flux);
        mvprintw(row + bar_h, hist_len + 2, "Min: %.2e", min_flux);

        row += bar_h + 2;

        // --------------------------------------------------------------------
        // Histogram
        // --------------------------------------------------------------------

        // Get current histogram snapshot from SHM
        uint32_t *hist_counts = smon->hist_counts[smon->cindex];
        float     h_min       = smon->hist_min_buf[smon->cindex];
        float     h_max       = smon->hist_max_buf[smon->cindex];
        uint32_t  nbins       = smon->hist_nbins;

        if (nbins > STREAM_MON_MAX_HIST_BINS)
        {
            nbins = STREAM_MON_MAX_HIST_BINS;
        }

        // Reconstruct thresholds locally
        float step = (h_max - h_min) / nbins;
        for (int i = 0; i <= nbins; i++)
        {
            local_thresholds[i] = h_min + i * step;
        }

        int available_rows = wrow - row - 5;
        if (available_rows < 5)
        {
            available_rows = 5; // Minimum space
        }

        int bin_factor = 1;
        if (nbins > available_rows)
        {
            bin_factor = (nbins + available_rows - 1) / available_rows; // ceil
        }

        mvprintw(row++, 0, "Histogram (Horizontal, binning x%d): Range [%.2e, %.2e]", bin_factor,
                 h_min, h_max);

        // Compute max count for scaling (considering binning)
        uint32_t max_count = 0;
        for (uint32_t i = 0; i < nbins; i += bin_factor)
        {
            uint32_t sum = 0;
            for (int k = 0; k < bin_factor && (i + k) < nbins; k++)
            {
                sum += hist_counts[i + k];
            }
            if (sum > max_count)
            {
                max_count = sum;
            }
        }
        if (max_count == 0)
        {
            max_count = 1;
        }

        for (uint32_t i = 0; i < nbins; i += bin_factor)
        {
            uint32_t sum     = 0;
            int      end_idx = i + bin_factor;
            if (end_idx > nbins)
            {
                end_idx = nbins;
            }

            for (int k = i; k < end_idx; k++)
            {
                sum += hist_counts[k];
            }

            // Format: [Low - High] Count  |BAR.......|
            char  label[64];
            float lower = local_thresholds[i];
            float upper = local_thresholds[end_idx];

            snprintf(label, sizeof(label), "%.2e - %.2e", lower, upper);

            if (row < wrow - 1)
            { // Safety check
                mvprintw(row, 0, "% -25s %6u ", label, sum);

                int bar_width = (int) ((double) sum / max_count * (wcol - 35));
                if (bar_width > wcol - 35)
                {
                    bar_width = wcol - 35;
                }

                // Draw bar
                move(row, 33);
                print_bar(bar_width, 6);
                row++;
            }
        }

        // --------------------------------------------------------------------
        // Output Streams Status
        // --------------------------------------------------------------------
        int status_col = 60;
        int status_row = 3;
        mvprintw(status_row++, status_col, "Active Output Streams:");

        for (int p = 0; p < 10; p++)
        { // Check powers of 2 up to 512
            int         bin = 1 << p;
            char        fname[512];
            struct stat st;

            snprintf(fname, sizeof(fname), "%s/%s.tbin%d.im.shm", data.shmdir, streamname, bin);
            int exists = (stat(fname, &st) == 0);

            move(status_row + p, status_col);
            if (exists)
            {
                attron(COLOR_PAIR(2));
                printw(" [ON]  .tbin%d ", bin);
                attroff(COLOR_PAIR(2));
            }
            else
            {
                attron(COLOR_PAIR(3));
                printw(" [OFF] .tbin%d ", bin);
                attroff(COLOR_PAIR(3));
            }
        }

        refresh();

        ch = getch();
        if (ch == 'q' || ch == 'Q')
        {
            loop = 0;
        }

        usleep(100000); // 100ms update
    }

    stream_monitor_detach(smon);
    return 0;
}

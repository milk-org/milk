/**
 * @file    stream-monproc-disp.c
 * @brief   Display stream monitor info (ANSI)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>

#include <termios.h>
#include <sys/ioctl.h>
#include <signal.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkdata/milkdata.h"
#include "fps_shmdirname.h"
#include "stream_monproc.h"

static uint16_t wrow, wcol;

// Terminal state
static struct termios orig_termios;
static int            terminal_initialized = 0;

static void cleanup_terminal()
{
    if (terminal_initialized)
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        printf("\033[?1049l\033[?25h\033[0m\n"); // Exit alt screen, show cursor, reset attrs
        fflush(stdout);
        terminal_initialized = 0;
    }
}

static void crash_handler(int sig)
{
    cleanup_terminal();
    struct sigaction sa_dfl;
    sa_dfl.sa_handler = SIG_DFL;
    sigemptyset(&sa_dfl.sa_mask);
    sa_dfl.sa_flags = 0;
    sigaction(sig, &sa_dfl, NULL);
    raise(sig);
}

static void init_terminal()
{
    tcgetattr(STDIN_FILENO, &orig_termios);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    terminal_initialized = 1;

    printf("\033[?1049h\033[?25l"); // Alt screen, hide cursor
    fflush(stdout);
    atexit(cleanup_terminal);

    struct sigaction sa_crash;
    sa_crash.sa_handler = crash_handler;
    sigemptyset(&sa_crash.sa_mask);
    sa_crash.sa_flags = 0;
    sigaction(SIGSEGV, &sa_crash, NULL);
    sigaction(SIGBUS, &sa_crash, NULL);
    sigaction(SIGABRT, &sa_crash, NULL);
    sigaction(SIGTERM, &sa_crash, NULL);
    sigaction(SIGINT, &sa_crash, NULL);
}

static void print_bar(int len, int color_code)
{
    printf("\033[%dm", color_code);
    for (int i = 0; i < len; i++)
    {
        putchar(' ');
    }
    printf("\033[0m");
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <stream_name>\n", argv[0]);
        return 1;
    }

    char *streamname = argv[1];

    // Get SHM dir and populate milk_data.shmdir (dcshmdir) for stream_monproc
    char shmdir[256];
    function_parameter_struct_shmdirname(shmdir);
    strncpy(milk_data.shmdir, shmdir, sizeof(milk_data.shmdir) - 1);

    // Connect to monitor SHM
    STREAM_MON_STRUCT *smon = stream_monitor_connect(streamname, 0);
    if (!smon)
    {
        fprintf(stderr, "Error connecting to monitor SHM for stream %s\n", streamname);
        return 1;
    }

    // Init ANSI terminal
    init_terminal();

    int loop = 1;

    // Local buffer for thresholds reconstruction
    float local_thresholds[STREAM_MON_MAX_HIST_BINS + 1];

    while (loop)
    {
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1)
        {
            wrow = ws.ws_row;
            wcol = ws.ws_col;
        }
        else
        {
            wrow = 24;
            wcol = 80;
        }
        printf("\033[2J");

        // --------------------------------------------------------------------
        // Header
        // --------------------------------------------------------------------
        printf("\033[37;44m");
        printf("\033[%d;%dH Stream Monitor Display: %s (PID %d) - Press 'q' to exit ", 1, 1,
               streamname, getpid());
        printf("\033[%d;%dH Count: %lu ", 1, wcol - 20 + 1, smon->cnt);
        printf("\033[0m");

        int row = 2;

        // --------------------------------------------------------------------
        // Flux History (Recent)
        // --------------------------------------------------------------------
        printf("\033[%d;%dHRecent Flux (last %d frames):", (row++) + 1, 0 + 1,
               smon->size > 50 ? 50 : smon->size);

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
                printf("\033[%d;%dH|", (row + bar_h - h) + 1, (i) + 1);
            }
        }
        printf("\033[%d;%dHMax: %.2e", (row) + 1, (hist_len + 2) + 1, max_flux);
        printf("\033[%d;%dHMin: %.2e", (row + bar_h) + 1, (hist_len + 2) + 1, min_flux);

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

        printf("\033[%d;%dHHistogram (Horizontal, binning x%d): Range [%.2e, %.2e]", (row++) + 1,
               0 + 1, bin_factor, h_min, h_max);

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
                printf("\033[%d;%dH%-25s %6u ", (row) + 1, 1, label, sum);
                int bar_width = (int) ((double) sum / max_count * (wcol - 35));
                if (bar_width > wcol - 35)
                {
                    bar_width = wcol - 35;
                }
                printf("\033[%d;%dH", (row) + 1, 33 + 1);
                print_bar(bar_width, 46); // Cyan background

                row++;
            }
        }

        // --------------------------------------------------------------------
        // Output Streams Status
        // --------------------------------------------------------------------
        int status_col = 60;
        int status_row = 3;
        printf("\033[%d;%dHActive Output Streams:", (status_row++) + 1, (status_col) + 1);

        for (int p = 0; p < 10; p++)
        { // Check powers of 2 up to 512
            int         bin = 1 << p;
            char        fname[4096];
            struct stat st;

            snprintf(fname, sizeof(fname), "%s/%s.tbin%d.im.shm", milk_data.shmdir, streamname,
                     bin);
            int exists = (stat(fname, &st) == 0);

            printf("\033[%d;%dH", (status_row + p) + 1, (status_col) + 1);
            if (exists)
            {
                printf("\033[30;42m [ON]  .tbin%d \033[0m", bin);
            }
            else
            {
                printf("\033[37;41m [OFF] .tbin%d \033[0m", bin);
            }
        }

        fflush(stdout);

        char c;
        if (read(STDIN_FILENO, &c, 1) == 1)
        {
            if (c == 'q' || c == 'Q' || c == 3)
            {
                loop = 0;
            }
        }

        usleep(100000); // 100ms update
    }

    stream_monitor_detach(smon);
    return 0;
}

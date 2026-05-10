/**
 * @file    milk-shmimmon.c
 * @brief   Standalone shared-memory stream monitor
 *
 * Live ncurses TUI for a single ImageStreamIO stream.
 * Links only against ImageStreamIO and ncurses — no
 * CLIcore, no FPS, no milkinfo library required.
 *
 * Usage:
 *   milk-shmimmon [OPTIONS] <stream>
 *
 * Options:
 *   -h, --help          Full usage text
 *   -h1, --help-oneline One-line description
 *   -f <hz>             Refresh rate in Hz (default: 10)
 */

#include <math.h>
#include <poll.h>
#include <termios.h>

#include <errno.h>
#include <getopt.h>
#include <locale.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <libgen.h>

#include "milk_help.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"


/* ================================================================
 * Constants
 * ============================================================= */

#define SHMIMMON_DESCRIPTION \
    "monitor a shared-memory image stream"

#define SHMIMMON_DEFAULT_HZ  10.0f
#define SHMIMMON_HIST_NBINS  20
#define SHMIMMON_SMALL_NEL   25    /* show per-pixel values below this */
#define SHMIMMON_STREAM_MAXLEN 256


/* ================================================================
 * Types
 * ============================================================= */

/**
 * @brief Per-frame statistics accumulated in compute_stats().
 */
typedef struct
{
    float    minv;
    float    maxv;
    double   mean;
    double   total;
    double   rms;
    long     hist[SHMIMMON_HIST_NBINS];
} StreamStats;


/* ================================================================
 * Globals
 * ============================================================= */

static volatile int g_quit = 0;


/* ================================================================
 * Signal handler
 * ============================================================= */

static void handle_sigint(int sig)
{
    (void) sig;
    g_quit = 1;
}

/* ================================================================
 * Terminal handling
 * ============================================================= */

static struct termios orig_termios;

static void disable_raw_mode(void)
{
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    printf("\033[?25h"); /* Restore cursor */
    fflush(stdout);
}

static void enable_raw_mode(void)
{
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    printf("\033[?25l"); /* Hide cursor */
    fflush(stdout);
}


/* ================================================================
 * Helpers
 * ============================================================= */

/** @brief Return the $MILK_SHM_DIR env var, or /dev/shm if unset. */
static const char *get_shm_dir(void)
{
    const char *d = getenv("MILK_SHM_DIR");
    return (d != NULL) ? d : "/dev/shm";
}


/**
 * @brief Compute min/max/mean/rms and histogram for any datatype.
 *
 * Uses a generic double accumulator; typed pointers avoid repeated
 * union dereferences inside the loop.
 *
 * @param img   Pointer to the connected IMAGE.
 * @param stats Output statistics structure (zeroed by caller).
 */
static void compute_stats(
    IMAGE        *img,
    StreamStats  *stats)
{
    const uint8_t  dtype  = img->md->datatype;
    const uint64_t nel    = img->md->nelement;

    /* — first pass: min, max, sum — */
    double   sum  = 0.0;
    float    minv, maxv;
    uint64_t ii;

    /* Initialise from element [0] to avoid sentinel issues */
#define STAT_INIT(FIELD) \
    minv = (float)(img->array.FIELD[0]); \
    maxv = minv; \
    for (ii = 0; ii < nel; ii++) { \
        float v = (float)(img->array.FIELD[ii]); \
        if (v < minv) minv = v; \
        if (v > maxv) maxv = v; \
        sum += v; \
    }

    switch (dtype)
    {
    case _DATATYPE_FLOAT:          STAT_INIT(F);    break;
    case _DATATYPE_DOUBLE:         STAT_INIT(D);    break;
    case _DATATYPE_UINT8:          STAT_INIT(UI8);  break;
    case _DATATYPE_INT8:           STAT_INIT(SI8);  break;
    case _DATATYPE_UINT16:         STAT_INIT(UI16); break;
    case _DATATYPE_INT16:          STAT_INIT(SI16); break;
    case _DATATYPE_UINT32:         STAT_INIT(UI32); break;
    case _DATATYPE_INT32:          STAT_INIT(SI32); break;
    case _DATATYPE_UINT64:         STAT_INIT(UI64); break;
    case _DATATYPE_INT64:          STAT_INIT(SI64); break;
    default:
        /* Unsupported type — leave stats zeroed */
        return;
    }
#undef STAT_INIT

    stats->minv  = minv;
    stats->maxv  = maxv;
    stats->total = sum;
    stats->mean  = sum / (double) nel;

    /* — second pass: rms and histogram — */
    const float  range  = maxv - minv;
    const int    nbins  = SHMIMMON_HIST_NBINS;
    double       sumsq  = 0.0;

    for (int h = 0; h < nbins; h++)
    {
        stats->hist[h] = 0;
    }

#define STAT_RMS_HIST(FIELD) \
    for (ii = 0; ii < nel; ii++) { \
        double v  = (double)(img->array.FIELD[ii]); \
        double d  = v - stats->mean; \
        sumsq += d * d; \
        if (range > 0.0f) { \
            int h = (int)(nbins * (v - minv) / range); \
            if (h >= nbins) h = nbins - 1; \
            if (h >= 0) stats->hist[h]++; \
        } \
    }

    switch (dtype)
    {
    case _DATATYPE_FLOAT:          STAT_RMS_HIST(F);    break;
    case _DATATYPE_DOUBLE:         STAT_RMS_HIST(D);    break;
    case _DATATYPE_UINT8:          STAT_RMS_HIST(UI8);  break;
    case _DATATYPE_INT8:           STAT_RMS_HIST(SI8);  break;
    case _DATATYPE_UINT16:         STAT_RMS_HIST(UI16); break;
    case _DATATYPE_INT16:          STAT_RMS_HIST(SI16); break;
    case _DATATYPE_UINT32:         STAT_RMS_HIST(UI32); break;
    case _DATATYPE_INT32:          STAT_RMS_HIST(SI32); break;
    case _DATATYPE_UINT64:         STAT_RMS_HIST(UI64); break;
    case _DATATYPE_INT64:          STAT_RMS_HIST(SI64); break;
    default:
        break;
    }
#undef STAT_RMS_HIST

    stats->rms = (nel > 0) ? sqrt(sumsq / (double) nel) : 0.0;
}


/**
 * @brief Print individual pixel values for small streams.
 *
 * @param img  Connected IMAGE.
 */
static void print_pixel_values(IMAGE *img)
{
    const uint64_t nel   = img->md->nelement;
    const uint8_t  dtype = img->md->datatype;

#define PIXEL_PRINT(FMT, FIELD) \
    for (uint64_t ii = 0; ii < nel; ii++) { \
        printf("%4lu  " FMT "\n", ii, img->array.FIELD[ii]); \
    }

    switch (dtype)
    {
    case _DATATYPE_FLOAT:   PIXEL_PRINT("%f",   F);    break;
    case _DATATYPE_DOUBLE:  PIXEL_PRINT("%f",   D);    break;
    case _DATATYPE_UINT8:   PIXEL_PRINT("%5u",  UI8);  break;
    case _DATATYPE_INT8:    PIXEL_PRINT("%5d",  SI8);  break;
    case _DATATYPE_UINT16:  PIXEL_PRINT("%5u",  UI16); break;
    case _DATATYPE_INT16:   PIXEL_PRINT("%5d",  SI16); break;
    case _DATATYPE_UINT32:  PIXEL_PRINT("%5u",  UI32); break;
    case _DATATYPE_INT32:   PIXEL_PRINT("%5d",  SI32); break;
    case _DATATYPE_UINT64:  PIXEL_PRINT("%5lu", UI64); break;
    case _DATATYPE_INT64:   PIXEL_PRINT("%5ld", SI64); break;
    default:
        break;
    }
#undef PIXEL_PRINT
}


/* ================================================================
 * TUI rendering
 * ============================================================= */

/**
 * @brief Draw one full TUI frame.
 *
 * @param img        Connected IMAGE.
 * @param stats      Pre-computed statistics.
 * @param hz_meas    Measured update frequency in Hz.
 * @param stream     Stream name string.
 */
static void draw_frame(
    IMAGE             *img,
    const StreamStats *stats,
    double             hz_meas,
    float              hz_target,
    const char        *stream)
{
    const IMAGE_METADATA *md    = img->md;
    const uint64_t        nel   = md->nelement;
    int                   row   = 0;
    int                   maxcol;

    maxcol = 80;
    printf("\033[H\033[2J");

#define TPRINT(...) \
    do { \
        char _tbuf[512]; \
        snprintf(_tbuf, sizeof(_tbuf), __VA_ARGS__); \
        _tbuf[sizeof(_tbuf) - 1] = '\0'; \
        (void) row; \
        (void) maxcol; \
        (void) _tbuf; \
        _TPRINT_IMPL(_tbuf) \
    } while(0)

#define _TPRINT_IMPL(s) printf("%s", s); row++;

    /* — header — */
    {
        /* Build type + shape string */
        char shape[128];
        snprintf(shape, sizeof(shape),
                 "%s [%u",
                 ImageStreamIO_typename(md->datatype),
                 (unsigned) md->size[0]);
        for (int ax = 1; ax < md->naxis; ax++)
        {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), " x %u", (unsigned) md->size[ax]);
            strncat(shape, tmp, sizeof(shape) - strlen(shape) - 1);
        }
        strncat(shape, "]", sizeof(shape) - strlen(shape) - 1);

        TPRINT("Stream: %-20s  %s\n", stream, shape);
    }

    /* — write flag and status — */
    TPRINT("[write %d] [status %2d]\n", md->write, md->status);

    /* — frame counter and frequency — */
    TPRINT("[cnt0 %10u] [cnt1 %10u] [%5.1f Hz meas] [%5.1f Hz tgt]\n",
           (unsigned) md->cnt0,
           (unsigned) md->cnt1,
           hz_meas,
           hz_target);

    /* — semaphores — */
    {
        int nsem = (int) md->sem;

        char svals[1024];
        char rpids[2048];

        svals[0] = '\0';
        rpids[0] = '\0';
        
        pid_t write_pid = -1;

        for (int s = 0; s < nsem; s++)
        {
            int   sv = ImageStreamIO_semvalue(img, s);
            char  tmp[64];

            snprintf(tmp, sizeof(tmp), " %6d", sv);
            strncat(svals, tmp, sizeof(svals) - strlen(svals) - 1);

            if (img->semWritePID[s] > 0) {
                write_pid = img->semWritePID[s];
            }

            pid_t rpid = img->semReadPID[s];
            if (rpid > 0) {
                if (kill(rpid, 0) == 0) {
                    snprintf(tmp, sizeof(tmp), " \033[32m%6d\033[0m", (int)rpid);
                } else {
                    snprintf(tmp, sizeof(tmp), " \033[31m%6d\033[0m", (int)rpid);
                }
            } else {
                snprintf(tmp, sizeof(tmp), " %6d", (int)rpid);
            }
            strncat(rpids, tmp, sizeof(rpids) - strlen(rpids) - 1);
        }

        TPRINT("[%2d sems%s]\n", nsem, svals);
        if (write_pid > 0) {
            if (kill(write_pid, 0) == 0) {
                TPRINT("[  WRITE \033[32m%6d\033[0m]\n", (int)write_pid);
            } else {
                TPRINT("[  WRITE \033[31m%6d\033[0m]\n", (int)write_pid);
            }
        } else {
            TPRINT("[  WRITE       ]\n");
        }
        TPRINT("[   READ%s]\n", rpids);
    }

    /* — circular buffer — */
    {
        int semlogval = 0;
        if (img->semlog != NULL)
        {
            sem_getvalue(img->semlog, &semlogval);
        }
        TPRINT("[semlog %3d]  [circbuff %3u/%3u  %6lu]\n",
               semlogval,
               (unsigned) md->CBindex,
               (unsigned) md->CBsize,
               (unsigned long) md->CBcycle);
    }

    TPRINT("\n");

    /* — image statistics — */
    TPRINT("total   = %12.6g\n", stats->total);
    TPRINT("mean    = %12.6g\n", stats->mean);
    TPRINT("RMS     = %12.6g\n", stats->rms);
    TPRINT("min     = %12.6g\n", (double) stats->minv);
    TPRINT("max     = %12.6g\n", (double) stats->maxv);
    TPRINT("\n");

    /* — pixel values or histogram — */
    if (nel <= SHMIMMON_SMALL_NEL)
    {
        TPRINT("Pixel values:\n");
        print_pixel_values(img);
    }
    else
    {
        TPRINT("Histogram (%d bins):\n", SHMIMMON_HIST_NBINS);

        /* Find histogram peak for bar scaling */
        long hmax = 1;
        for (int h = 0; h < SHMIMMON_HIST_NBINS; h++)
        {
            if (stats->hist[h] > hmax)
            {
                hmax = stats->hist[h];
            }
        }

        const float range = stats->maxv - stats->minv;

        for (int h = 0; h < SHMIMMON_HIST_NBINS; h++)
        {
            float lo = stats->minv + range * (float) h / SHMIMMON_HIST_NBINS;
            float hi = stats->minv + range * (float)(h + 1) / SHMIMMON_HIST_NBINS;

            char label[64];
            snprintf(label, sizeof(label),
                     "[%12.4e - %12.4e] %7ld",
                     (double) lo,
                     (double) hi,
                     stats->hist[h]);

            int bar_max_w = maxcol - 2 - (int) strlen(label);
            if (bar_max_w < 0)
            {
                bar_max_w = 0;
            }

            int bar_w = (int)(stats->hist[h] * bar_max_w / hmax);

            char bar[1024];
            bar[0] = '\0';
            for (int b = 0; b < bar_w && b < (int)((sizeof(bar) / 3) - 1); b++)
            {
                strcat(bar, "\xE2\x96\x88");
            }

            int r, g, b_col;
            float t = (float)h / (SHMIMMON_HIST_NBINS - 1);
            if (t < 0.25f) {
                r = 0;
                g = (int)(255 * (t / 0.25f));
                b_col = 255;
            } else if (t < 0.5f) {
                r = 0;
                g = 255;
                b_col = (int)(255 * (1.0f - (t - 0.25f) / 0.25f));
            } else if (t < 0.75f) {
                r = (int)(255 * ((t - 0.5f) / 0.25f));
                g = 255;
                b_col = 0;
            } else {
                r = 255;
                g = (int)(255 * (1.0f - (t - 0.75f) / 0.25f));
                b_col = 0;
            }

            printf("%s \033[38;2;%d;%d;%dm%s\033[0m\n", label, r, g, b_col, bar);
            row++;
        } // for (h)
    }

    TPRINT("\n[q/x] quit   [SPACE] reset RMS filter   [+/-] adjust sample rate\n");

    fflush(stdout);

#undef TPRINT
#undef _TPRINT_IMPL
}


/* ================================================================
 * Main
 * ============================================================= */

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, SHMIMMON_DESCRIPTION, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s[OPTIONS]%s %s<stream>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  Monitor the content of a shared-memory image stream in a live ncurses TUI.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-f <hz>",
           mh_color ? MH_RST : "", "Refresh rate in Hz (default: 10)");

    milk_help_section("Keys (while running)", mh_color);
    printf("  %s%-15s%s %s\n",
           mh_color ? MH_OPT : "", "q / x",
           mh_color ? MH_RST : "", "Quit");
    printf("  %s%-15s%s %s\n",
           mh_color ? MH_OPT : "", "SPACE",
           mh_color ? MH_RST : "", "Reset RMS exponential filter");
    printf("  %s%-15s%s %s\n\n",
           mh_color ? MH_OPT : "", "+ / -",
           mh_color ? MH_RST : "", "Adjust sample rate");

    milk_help_section("Environment", mh_color);
    printf("  MILK_SHM_DIR  Path to shared memory (default: /dev/shm)\n\n");
}


int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");

    const char *stream  = NULL;
    float       hz      = SHMIMMON_DEFAULT_HZ;

    const char *progname = basename(argv[0]);

    int action = milk_help_init(argc, argv,
                                SHMIMMON_DESCRIPTION,
                                "Monitor the content of a shared-memory image stream in a live ncurses TUI.");
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        return 0;
    }

    /* — argument parsing — */
    static const struct option longopts[] = {
        { NULL,          0,                 NULL,  0  }
    };

    int opt;

    while ((opt = getopt_long(argc, argv, "+f:", longopts, NULL)) != -1)
    {
        switch (opt)
        {

        case 'f':
            hz = strtof(optarg, NULL);
            if (hz <= 0.0f)
            {
                fprintf(stderr, "Error: -f requires a positive Hz value.\n");
                return 1;
            }
            break;

        default:
            fprintf(stderr, "Unknown option. Run %s -h for help.\n", progname);
            return 1;
        }
    }

    /* Positional argument: stream name */
    if (optind < argc)
    {
        stream = argv[optind];
    }

    if (stream == NULL)
    {
        fprintf(stderr,
                "Error: missing stream name.\n"
                "Run %s -h for help.\n",
                progname);
        return 1;
    }

    /* — verify the SHM file exists — */
    {
        char shmpath[512];
        snprintf(shmpath, sizeof(shmpath),
                 "%s/%s.im.shm", get_shm_dir(), stream);

        if (access(shmpath, F_OK) != 0)
        {
            fprintf(stderr,
                    "Error: stream '%s' not found.\n"
                    "Expected: %s\n",
                    stream, shmpath);
            return 1;
        }
    }

    /* — connect to stream — */
    IMAGE img;
    memset(&img, 0, sizeof(img));

    if (ImageStreamIO_openIm(&img, stream) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "Error: cannot open stream '%s'.\n", stream);
        return 1;
    }

    /* — signal handling — */
    signal(SIGINT,  handle_sigint);
    signal(SIGTERM, handle_sigint);

    /* — terminal setup — */
    enable_raw_mode();

    /* — main loop — */
    long sleep_ns = (long)(1.0e9 / (double) hz);
    struct timespec sleep_ts = {
        .tv_sec  = sleep_ns / 1000000000L,
        .tv_nsec = sleep_ns % 1000000000L,
    };

    struct timespec t_prev;
    clock_gettime(CLOCK_MONOTONIC, &t_prev);
    uint64_t cnt_prev = img.md->cnt0;
    double   hz_meas  = 0.0;

    while (!g_quit)
    {
        /* — measure actual frame rate — */
        {
            struct timespec t_now;
            clock_gettime(CLOCK_MONOTONIC, &t_now);

            double dt = (double)(t_now.tv_sec  - t_prev.tv_sec) +
                        1.0e-9 * (double)(t_now.tv_nsec - t_prev.tv_nsec);

            uint64_t cnt_now  = img.md->cnt0;
            uint64_t cnt_diff = cnt_now - cnt_prev;

            if (dt > 0.0)
            {
                hz_meas = (double) cnt_diff / dt;
            }

            t_prev    = t_now;
            cnt_prev  = cnt_now;
        }

        /* — compute statistics — */
        StreamStats stats;
        memset(&stats, 0, sizeof(stats));
        compute_stats(&img, &stats);

        /* — draw — */
        draw_frame(&img, &stats, hz_meas, hz, stream);

        /* — handle keyboard input — */
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        if (poll(&pfd, 1, 0) > 0)
        {
            if (pfd.revents & POLLIN)
            {
                char c;
                if (read(STDIN_FILENO, &c, 1) == 1)
                {
                    if (c == 'q' || c == 'x')
                    {
                        g_quit = 1;
                    }
                    else if (c == '+')
                    {
                        hz *= 1.25f;
                        if (hz > 1000.0f) hz = 1000.0f;
                        sleep_ns = (long)(1.0e9 / (double) hz);
                        sleep_ts.tv_sec  = sleep_ns / 1000000000L;
                        sleep_ts.tv_nsec = sleep_ns % 1000000000L;
                    }
                    else if (c == '-')
                    {
                        hz /= 1.25f;
                        if (hz < 0.1f) hz = 0.1f;
                        sleep_ns = (long)(1.0e9 / (double) hz);
                        sleep_ts.tv_sec  = sleep_ns / 1000000000L;
                        sleep_ts.tv_nsec = sleep_ns % 1000000000L;
                    }
                    /* SPACE is nominally handled, but previous code didn't actually implement RMS reset logic inside the loop! */
                }
            }
        }

        /* — sleep until next update — */
        nanosleep(&sleep_ts, NULL);
    }

    /* disable_raw_mode() called via atexit */

    ImageStreamIO_closeIm(&img);

    return 0;
}

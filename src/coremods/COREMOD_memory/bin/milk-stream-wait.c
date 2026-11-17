// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk-stream-wait.c
 * @brief   Wait for shared-memory streams to exist, match dimensions, and advance frames
 */

#define _GNU_SOURCE

#include <errno.h>
#include <libgen.h>
#include <poll.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_help.h"

#define MSW_DESC "wait for shared-memory image streams and frame updates"
#define MSW_DESC_LONG                                                                 \
    "Block until each specified ImageStreamIO shared-memory stream exists with\n"     \
    "initialized metadata (write == 0), optionally matches expected dimensions\n"     \
    "(:WxH or :WxHxD), and optionally advances by +N live frames via POSIX\n"         \
    "semaphores.\n\n"                                                                 \
    "Uses Linux inotify on MILK_SHM_DIR for event-driven creation detection and\n"    \
    "sem_timedwait() for zero-polling frame synchronization.\n\n"                     \
    "Exit status: 0 satisfied, 1 dimension/open error, 2 usage error, 124 timeout,\n" \
    "130 interrupted (SIGINT/SIGTERM)."

typedef struct
{
    char     name[256];
    uint32_t dims[3];
    uint8_t  ndims;
    long     nframes;
} stream_wait_spec_t;

static volatile sig_atomic_t g_interrupted = 0;

static void on_signal(int sig)
{
    (void) sig;
    g_interrupted = 1;
}

static double monotonic_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static const char *resolve_shmdir(char *buf, size_t bufsz)
{
    char probe[1024];
    if (ImageStreamIO_filename(probe, sizeof(probe), "__probe__") == IMAGESTREAMIO_SUCCESS)
    {
        char *slash = strrchr(probe, '/');
        if (slash != NULL && slash != probe)
        {
            size_t len = (size_t) (slash - probe);
            if (len < bufsz)
            {
                memcpy(buf, probe, len);
                buf[len] = '\0';
                return buf;
            }
        }
    }
    const char *env = getenv("MILK_SHM_DIR");
    if (env != NULL && env[0] != '\0')
    {
        return env;
    }
    struct stat st;
    if (stat("/milk/shm", &st) == 0 && S_ISDIR(st.st_mode))
    {
        return "/milk/shm";
    }
    return "/dev/shm";
} // resolve_shmdir

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, MSW_DESC, mh_color);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s-t SEC%s] [%s-s SEM%s] [%s-q%s] %s<spec>%s [%s<spec> ...%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", MSW_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_ARG, "<spec>", 24),
           "[s:|s.]<stream>[:D0[xD1[xD2]]][+N]");
    printf("  %-24s %s\n", "", "Examples: dm10disp, dm10disp:50x50, ocam2d+1, aol7_im:120x120+4");
    printf("\n");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-t, --timeout SEC", 24),
           "Overall timeout in seconds, 0 = infinite (default: 10)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-s, --sem IDX", 24),
           "Semaphore index for +N frame wait (default: auto)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-q, --quiet", 24),
           "Suppress timeout diagnostic messages on stderr");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h, --help", 24), "Show this help and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h1, --help-oneline", 24),
           "Print one-line description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h2, --help-description", 24),
           "Print verbose description and exit");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "-hm, --help-mono", 24),
           "Full help without ANSI color");

    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s%s %sdm10disp:50x50 ocam2dsim+1%s\n", mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s$ %s%s%s %s-t 5.0%s %saol7_modevalDM+2%s\n\n", mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    const char *see_also[] = { "milk-stream-link:establish shared memory stream symlink",
                               "milk-fps-wait:wait for FPS unit process completion",
                               "milk-shmimmon:monitor shared memory stream activity" };
    milk_help_see_also(see_also, 3, mh_color);
} // print_help

static int parse_dims(const char *str, uint32_t dims[3], uint8_t *ndims_out)
{
    uint8_t     nd  = 0;
    const char *cur = str;

    while (*cur != '\0')
    {
        if (nd >= 3 || *cur < '1' || *cur > '9')
        {
            return -1;
        }
        char         *end = NULL;
        unsigned long val = strtoul(cur, &end, 10);
        if (end == cur || val == 0 || val > UINT32_MAX)
        {
            return -1;
        }
        dims[nd++] = (uint32_t) val;
        if (*end == '\0')
        {
            break;
        }
        if (*end != 'x' && *end != 'X')
        {
            return -1;
        }
        cur = end + 1;
        if (*cur == '\0')
        {
            return -1;
        }
    }
    *ndims_out = nd;
    return (nd > 0) ? 0 : -1;
} // parse_dims

static int parse_spec(const char *raw, stream_wait_spec_t *sp)
{
    memset(sp, 0, sizeof(*sp));
    if (strncmp(raw, "s:", 2) == 0 || strncmp(raw, "s.", 2) == 0)
    {
        raw += 2;
    }
    char buf[512];
    int  n = snprintf(buf, sizeof(buf), "%s", raw);
    if (n <= 0 || (size_t) n >= sizeof(buf))
    {
        return -1;
    }

    char *plus = strrchr(buf, '+');
    if (plus != NULL)
    {
        *plus     = '\0';
        char *end = NULL;
        if (plus[1] < '1' || plus[1] > '9')
        {
            return -1;
        }
        long nf = strtol(plus + 1, &end, 10);
        if (end == plus + 1 || *end != '\0' || nf < 1)
        {
            return -1;
        }
        sp->nframes = nf;
    }

    char *colon = strchr(buf, ':');
    if (colon != NULL)
    {
        *colon = '\0';
        if (parse_dims(colon + 1, sp->dims, &sp->ndims) != 0)
        {
            return -1;
        }
    }

    size_t namelen = strlen(buf);
    if (namelen == 0 || namelen >= sizeof(sp->name) || strchr(buf, '/') != NULL)
    {
        return -1;
    }
    memcpy(sp->name, buf, namelen + 1);
    return 0;
} // parse_spec

static int check_stop(double      t_start,
                      double      timeout_sec,
                      const char *prog,
                      const char *sname,
                      int         q)
{
    if (g_interrupted)
    {
        return 130;
    }
    if (timeout_sec > 0.0 && monotonic_now() - t_start >= timeout_sec)
    {
        if (!q)
        {
            fprintf(stderr, "%s: timed out after %.3gs waiting for stream '%s'\n", prog,
                    timeout_sec, sname);
        }
        return 124;
    }
    return 0;
} // check_stop

static int verify_dims(const IMAGE *img, const stream_wait_spec_t *sp, const char *prog)
{
    if (sp->ndims == 0)
    {
        return 0;
    }
    int mismatch = (img->md->naxis != sp->ndims);
    for (uint8_t k = 0; !mismatch && k < sp->ndims; k++)
    {
        if (img->md->size[k] != sp->dims[k])
        {
            mismatch = 1;
        }
    }
    if (!mismatch)
    {
        return 0;
    }

    char got[64]  = "";
    char want[64] = "";
    int  goff     = 0;
    int  woff     = 0;
    for (uint8_t k = 0; k < img->md->naxis && k < 3; k++)
    {
        goff += snprintf(got + goff, sizeof(got) - (size_t) goff, "%s%u", k ? "x" : "",
                         (unsigned) img->md->size[k]);
    }
    for (uint8_t k = 0; k < sp->ndims; k++)
    {
        woff += snprintf(want + woff, sizeof(want) - (size_t) woff, "%s%u", k ? "x" : "",
                         (unsigned) sp->dims[k]);
    }
    fprintf(stderr, "%s: stream '%s' size mismatch: expected %s, got %s\n", prog, sp->name, want,
            got);
    return 1;
} // verify_dims

static int wait_one_spec(const stream_wait_spec_t *sp,
                         const char               *shmdir,
                         int                       req_sem,
                         double                    t_start,
                         double                    timeout_sec,
                         int                       quiet,
                         const char               *prog)
{
    IMAGE img;
    char  shmpath[1024];
    int   has_path =
        (ImageStreamIO_filename(shmpath, sizeof(shmpath), sp->name) == IMAGESTREAMIO_SUCCESS);
    int ifd     = -1;
    int created = 0;
    int rc      = 0;

    memset(&img, 0, sizeof(img));
    for (;;)
    {
        if ((!has_path || access(shmpath, F_OK) == 0) &&
            ImageStreamIO_openIm(&img, sp->name) == IMAGESTREAMIO_SUCCESS)
        {
            if (img.md != NULL && img.md->write == 0 && img.md->naxis > 0)
            {
                break;
            }
            ImageStreamIO_closeIm(&img);
            memset(&img, 0, sizeof(img));
        }

        created = 1;
        rc      = check_stop(t_start, timeout_sec, prog, sp->name, quiet);
        if (rc != 0)
        {
            goto out;
        }

        if (ifd == -1)
        {
            ifd = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
            if (ifd >= 0 && inotify_add_watch(ifd, shmdir, IN_CREATE | IN_MOVED_TO | IN_ATTRIB) < 0)
            {
                close(ifd);
                ifd = -2;
            }
            /* Re-check immediately after registering watch to prevent TOCTOU missed create */
            continue;
        }

        int file_present = (has_path && access(shmpath, F_OK) == 0);
        if (ifd >= 0 && !file_present)
        {
            int wait_ms = 200;
            if (timeout_sec > 0.0)
            {
                double rem_ms = (timeout_sec - (monotonic_now() - t_start)) * 1000.0;
                if (rem_ms < 1.0)
                {
                    rem_ms = 1.0;
                }
                if (rem_ms < (double) wait_ms)
                {
                    wait_ms = (int) rem_ms + 1;
                }
            }
            struct pollfd pfd = { ifd, POLLIN, 0 };
            if (poll(&pfd, 1, wait_ms) > 0)
            {
                char    evbuf[4096];
                ssize_t rr = read(ifd, evbuf, sizeof(evbuf));
                (void) rr;
            }
        }
        else
        {
            struct timespec slp = { 0, file_present ? 2000000L : 10000000L };
            nanosleep(&slp, NULL);
        }
    } // for creation wait

    rc = verify_dims(&img, sp, prog);
    if (rc != 0 || sp->nframes <= 0)
    {
        ImageStreamIO_closeIm(&img);
        goto out;
    }

    {
        uint64_t target_cnt = (created ? 0 : img.md->cnt0) + (uint64_t) sp->nframes;
        int      semidx     = -1;
        int      owned_pid  = 0;

        if (img.md->sem > 0)
        {
            if (req_sem >= 0 && req_sem < img.md->sem)
            {
                semidx = req_sem;
            }
            else if (req_sem < 0)
            {
                semidx    = ImageStreamIO_getsemwaitindex(&img, 0);
                owned_pid = (semidx >= 0 && semidx < img.md->sem);
            }
        }

        if (semidx >= 0 && semidx < img.md->sem)
        {
            while (sem_trywait(img.semptr[semidx]) == 0)
            {
            }
        }

        while (img.md->cnt0 < target_cnt)
        {
            rc = check_stop(t_start, timeout_sec, prog, sp->name, quiet);
            if (rc != 0)
            {
                break;
            }
            if (semidx < 0 || semidx >= img.md->sem)
            {
                struct timespec slp = { 0, 1000000L };
                nanosleep(&slp, NULL);
                continue;
            }
            double step = 0.05;
            if (timeout_sec > 0.0)
            {
                double rem = timeout_sec - (monotonic_now() - t_start);
                if (rem < step)
                {
                    step = (rem < 1.0e-4) ? 1.0e-4 : rem;
                }
            }
            struct timespec absts;
            clock_gettime(CLOCK_REALTIME, &absts);
            long add_ns = (long) (step * 1.0e9);
            absts.tv_nsec += add_ns;
            if (absts.tv_nsec >= 1000000000L)
            {
                absts.tv_sec += 1;
                absts.tv_nsec -= 1000000000L;
            }
            sem_timedwait(img.semptr[semidx], &absts);
        } // while cnt0 < target_cnt

        if (owned_pid && img.semReadPID != NULL && img.semReadPID[semidx] == getpid())
        {
            img.semReadPID[semidx] = 0;
        }
    }
    ImageStreamIO_closeIm(&img);

out:
    if (ifd >= 0)
    {
        close(ifd);
    }
    return rc;
} // wait_one_spec

int main(int argc, char *argv[])
{
    char       *progbuf  = strdup(argv[0]);
    const char *progname = (progbuf != NULL) ? basename(progbuf) : argv[0];

    int action = milk_help_init(argc, argv, MSW_DESC, MSW_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        free(progbuf);
        return 0;
    }
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, action == MH_ACTION_HELP);
        free(progbuf);
        return 0;
    }

    double timeout_sec = 10.0;
    int    req_sem     = -1;
    int    quiet       = 0;
    int    opts_done   = 0;

    stream_wait_spec_t *specs  = (stream_wait_spec_t *) calloc((size_t) argc, sizeof(*specs));
    int                 nspecs = 0;
    if (specs == NULL)
    {
        fprintf(stderr, "%s: out of memory\n", progname);
        free(progbuf);
        return 1;
    }

    for (int ii = 1; ii < argc; ii++)
    {
        const char *arg = argv[ii];
        if (!opts_done && strcmp(arg, "--") == 0)
        {
            opts_done = 1;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-t") == 0 || strcmp(arg, "--timeout") == 0))
        {
            if (ii + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires an argument\n", progname, arg);
                free(specs);
                free(progbuf);
                return 2;
            }
            char  *end = NULL;
            double val = strtod(argv[++ii], &end);
            if (end == argv[ii] || *end != '\0' || val < 0.0)
            {
                fprintf(stderr, "%s: invalid timeout '%s'\n", progname, argv[ii]);
                free(specs);
                free(progbuf);
                return 2;
            }
            timeout_sec = val;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-s") == 0 || strcmp(arg, "--sem") == 0))
        {
            if (ii + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires an argument\n", progname, arg);
                free(specs);
                free(progbuf);
                return 2;
            }
            char *end = NULL;
            long  val = strtol(argv[++ii], &end, 10);
            if (end == argv[ii] || *end != '\0' || val < -1 || val > 1024)
            {
                fprintf(stderr, "%s: invalid semaphore index '%s'\n", progname, argv[ii]);
                free(specs);
                free(progbuf);
                return 2;
            }
            req_sem = (int) val;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0))
        {
            quiet = 1;
            continue;
        }
        if (!opts_done && arg[0] == '-')
        {
            fprintf(stderr, "%s: unknown option '%s'\n", progname, arg);
            free(specs);
            free(progbuf);
            return 2;
        }
        if (parse_spec(arg, &specs[nspecs]) != 0)
        {
            fprintf(stderr, "%s: invalid stream wait spec '%s'\n", progname, arg);
            free(specs);
            free(progbuf);
            return 2;
        }
        nspecs++;
    } // for ii

    if (nspecs == 0)
    {
        fprintf(stderr, "%s: missing <spec> argument\n", progname);
        free(specs);
        free(progbuf);
        return 2;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    ImageStreamIO_set_printError(NULL);

    char        shmdir_buf[1024];
    const char *shmdir  = resolve_shmdir(shmdir_buf, sizeof(shmdir_buf));
    double      t_start = monotonic_now();
    int         rc      = 0;

    for (int ii = 0; ii < nspecs && rc == 0; ii++)
    {
        rc = wait_one_spec(&specs[ii], shmdir, req_sem, t_start, timeout_sec, quiet, progname);
    }

    free(specs);
    free(progbuf);
    return rc;
} // main

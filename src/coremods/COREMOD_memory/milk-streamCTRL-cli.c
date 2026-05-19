/**
 * @file milk-streamCTRL-cli.c
 * @brief CLI stream monitor (standalone, no CLIcore)
 *
 * Continuously scans and displays shared memory
 * streams. Standalone replacement for the ncurses
 * milk-streamCTRL TUI when CLI mode is not
 * available.
 *
 * Only depends on ImageStreamIO and milkdata.
 */

#include <dirent.h>
#include <getopt.h>
#include <regex.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <libgen.h>

#include "milk_help.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"

/* ANSI color codes */
#define C_TITLE "\033[1;97m"
#define C_HDR   "\033[1;34m"
#define C_NAME  "\033[1;32m"
#define C_TYPE  "\033[1;33m"
#define C_SIZE  "\033[1m"
#define C_CNT   "\033[1;35m"
#define C_RATE  "\033[1;36m"
#define C_LINK  "\033[36m"
#define C_ERR   "\033[1;31m"
#define C_DIM   "\033[2m"
#define C_RST   "\033[0m"

#define MAX_STREAMS     500
#define MAX_SNAME_LEN   128
#define MAX_FNAME_LEN   512

typedef struct
{
    char     sname[MAX_SNAME_LEN];
    int      is_link;
    char     link_target[MAX_FNAME_LEN];
    int      link_broken;
    uint64_t cnt0_prev;
    uint64_t cnt0;
    double   rate;
    uint8_t  datatype;
    uint32_t size[3];
    uint8_t  naxis;
    int      open_ok;
} stream_entry;

static volatile int keep_running = 1;

static void sighandler(int sig)
{
    (void) sig;
    keep_running = 0;
}

static const char *get_shmdir(void)
{
    const char *d = getenv("MILK_SHM_DIR");
    if(d)
    {
        return d;
    }
    struct stat st;
    if(stat("/milk/shm", &st) == 0)
    {
        return "/milk/shm";
    }
    return "/dev/shm";
}

static int scan_streams(
    const char   *shmdir,
    stream_entry *entries,
    int          max_entries,
    regex_t      *regex,
    int          use_regex)
{
    DIR           *d = opendir(shmdir);
    int            n = 0;

    if(!d)
    {
        return 0;
    }
    struct dirent *de;
    while((de = readdir(d)) && n < max_entries)
    {
        char *dot = strstr(de->d_name, ".im.shm");
        if(!dot)
        {
            continue;
        }
        int suffix_pos = (int)(dot - de->d_name);
        if(suffix_pos !=
                (int)strlen(de->d_name) - 7)
        {
            continue;
        }

        char sname[MAX_SNAME_LEN];
        strncpy(sname, de->d_name, sizeof(sname) - 1);
        sname[sizeof(sname) - 1] = '\0';
        sname[suffix_pos] = '\0';

        if(use_regex &&
                regexec(regex, sname,
                        0, NULL, 0) != 0)
        {
            continue;
        }

        strncpy(entries[n].sname, sname, MAX_SNAME_LEN - 1);
        entries[n].is_link = 0;
        entries[n].open_ok = 0;

        /* Check symlink */
        char fullpath[MAX_FNAME_LEN];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", shmdir, de->d_name);

        struct stat lbuf;
        if(lstat(fullpath, &lbuf) == 0 &&
                S_ISLNK(lbuf.st_mode))
        {
            entries[n].is_link = 1;
            ssize_t len = readlink(fullpath, entries[n].link_target, MAX_FNAME_LEN - 1);
            if(len > 0)
            {
                entries[n].link_target[len] = '\0';
            }
            struct stat tbuf;
            entries[n].link_broken = (stat(fullpath, &tbuf) != 0);
        }

        if(!entries[n].is_link ||
                !entries[n].link_broken)
        {
            IMAGE im = {0};
            errno_t r = ImageStreamIO_read_sharedmem_image_toIMAGE(sname, &im);
            if(r == IMAGESTREAMIO_SUCCESS &&
                    im.md)
            {
                entries[n].open_ok  = 1;
                entries[n].cnt0     = im.md->cnt0;
                entries[n].datatype = im.md->datatype;
                entries[n].naxis    = im.md->naxis;
                for(int i = 0; i < 3; i++)
                {
                    entries[n].size[i] = im.md->size[i];
                }
                ImageStreamIO_closeIm(&im);
            }
        }
        n++;
    }
    closedir(d);
    return n;
}

static void print_header(void)
{
    printf(C_TITLE
           "%-28s %-10s %-18s "
           "%12s %10s" C_RST "\n", "Stream", "Type", "Size", "Cnt0", "Rate(Hz)");
    for(int i = 0; i < 82; i++)
    {
        putchar('-');
    }
    putchar('\n');
}

static void print_stream(
    const stream_entry *e)
{
    if(e->is_link)
    {
        if(e->link_broken)
        {
            printf(C_LINK "%-28s" C_RST
                   " " C_ERR "LINK -> %s (broken)" C_RST "\n", e->sname, e->link_target);
        }
        else
        {
            printf(C_LINK "%-28s" C_RST " LINK -> %s\n", e->sname, e->link_target);
        }
        return;
    }
    if(!e->open_ok)
    {
        printf(C_NAME "%-28s" C_RST " " C_ERR "OPEN_FAILED" C_RST "\n", e->sname);
        return;
    }

    const char *tstr = ImageStreamIO_typename(e->datatype);
    char sizestr[32];
    if(e->naxis == 1)
    {
        snprintf(sizestr, sizeof(sizestr), "%u", e->size[0]);
    }
    else if(e->naxis == 2)
    {
        snprintf(sizestr, sizeof(sizestr), "%ux%u", e->size[0], e->size[1]);
    }
    else
    {
        snprintf(sizestr, sizeof(sizestr), "%ux%ux%u", e->size[0], e->size[1], e->size[2]);
    }

    printf(C_NAME "%-28s" C_RST
           " " C_TYPE "%-10s" C_RST
           " " C_SIZE "%-18s" C_RST
           " " C_CNT "%12lu" C_RST,
           e->sname, tstr ? tstr : "???", sizestr, (unsigned long) e->cnt0);
    if(e->rate > 0.01)
    {
        printf(" " C_RATE "%10.1f" C_RST, e->rate);
    }
    else
    {
        printf("           ");
    }
    printf("\n");
}

static void print_help(
    const char *progname,
    int        mh_color)
{
    milk_help_banner(progname, "command-line interface for monitoring shared memory streams", mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sregex%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  Standalone CLI stream monitor. Continuously scans and displays shared memory\n"
           "  streams. Standalone replacement for the ncurses milk-streamCTRL TUI when\n"
           "  CLI mode is not available.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-n, --interval SEC",
           mh_color ? MH_RST : "", "Refresh interval (default 1.0)");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-1, --once", mh_color ? MH_RST : "", "Print once and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");

    const char *see_also[] =
    {
        "milk-streamCTRL:launch the stream control dashboard TUI",
        "milk-stream-info:inspect stream metadata and data",
        "milk-stream-list:list active shared memory streams"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

int main(
    int argc,
    char *argv[])
{
    const char *progname = basename(argv[0]);

    int action = milk_help_init(argc, argv,
                                "command-line interface for monitoring shared memory streams",
                                "Standalone CLI stream monitor. Continuously scans and displays shared memory\n"
                                "streams. Standalone replacement for the ncurses milk-streamCTRL TUI.");
    if(action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if(action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        return 0;
    }

    double interval = 1.0;
    int    once     = 0;

    static struct option long_opts[] =
    {
        {"interval", required_argument, 0, 'n'},
        {"once",     no_argument,       0, '1'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while((opt = getopt_long(
                     argc, argv, "n:1h",
                     long_opts, NULL)) != -1)
    {
        switch(opt)
        {
        case 'n': interval = atof(optarg);
            break;
        case '1': once = 1;
            break;
        case 'h': print_help(progname, 1);
            return 0;
        default: printf("\n\033[1;31mERROR\033[0m: Invalid option.\n\n");
            print_help(progname, 1);
            return 1;
        }
    }

    const char *pattern = NULL;
    regex_t     regex;
    int         use_regex = 0;
    if(optind < argc)
    {
        pattern = argv[optind];
        if(regcomp(&regex, pattern,
                   REG_EXTENDED |
                   REG_NOSUB) != 0)
        {
            fprintf(stderr, "Invalid regex: %s\n", pattern);
            return 1;
        }
        use_regex = 1;
    }

    // ImageStreamIO_set_verbosity(0);
    {
        struct sigaction sa;
        sa.sa_handler = sighandler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
    }

    const char   *shmdir = get_shmdir();
    stream_entry  entries[MAX_STREAMS];
    stream_entry  prev[MAX_STREAMS];
    int           n_prev = 0;

    while(keep_running)
    {
        int n = scan_streams(shmdir, entries, MAX_STREAMS, use_regex ? &regex : NULL, use_regex);

        /* Compute rates from prev cnt0 */
        for(int i = 0; i < n; i++)
        {
            entries[i].rate = 0.0;
            for(int j = 0; j < n_prev; j++)
            {
                if(strcmp(entries[i].sname,
                          prev[j].sname) == 0)
                {
                    int64_t dc = (int64_t) entries[i].cnt0 - (int64_t) prev[j].cnt0;
                    if(dc > 0 && interval > 0)
                    {
                        entries[i].rate = (double) dc / interval;
                    }
                    entries[i].cnt0_prev = prev[j].cnt0;
                    break;
                }
            }
        }

        /* Clear screen and print */
        if(!once)
        {
            printf("\033[2J\033[H");
        }
        printf(C_TITLE "milk-streamCTRL-cli"
               C_RST " | " C_DIM "%s" C_RST " | " C_DIM "%d streams" C_RST "\n\n", shmdir, n);
        print_header();
        for(int i = 0; i < n; i++)
        {
            print_stream(&entries[i]);
        }

        if(once)
        {
            break;
        }

        /* Save for rate calc */
        memcpy(prev, entries, sizeof(stream_entry) * n);
        n_prev = n;

        usleep((useconds_t)(interval * 1e6));
    }

    if(use_regex)
    {
        regfree(&regex);
    }
    return 0;
}

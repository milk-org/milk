// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <dirent.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <termios.h>
#include <regex.h>

#include "libmilkcommon/multiselect_parse.h"
#include "libmilkcommon/milkDebugTools.h"

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_help.h"

#define SR_DESC "remove a shared memory image stream"
#define SR_DESC_LONG                                            \
    "Remove one or more ImageStreamIO shared-memory streams.\n" \
    "Three modes:\n"                                            \
    "  (default)  Exact name match: remove the named stream.\n" \
    "  -r         Regex match: list matches then confirm.\n"    \
    "  (no args)  Interactive: list all streams, select to remove."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, SR_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sSTREAM_NAME%s ...]\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SR_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-r, --regex", mh_color ? MH_RST : "",
           "Treat STREAM_NAME as a regex (may match multiple)");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-f, --force", mh_color ? MH_RST : "",
           "Skip confirmation prompt (use with -r)");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "--ignore-missing", mh_color ? MH_RST : "",
           "Succeed without error when stream does not exist");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-q, --quiet", mh_color ? MH_RST : "",
           "Suppress informational stdout messages");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-v, --verbose", mh_color ? MH_RST : "",
           "Verbose output");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-stream-rm%s %sdm00disp%s\n", mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-stream-rm%s %s-r%s %s'dm.*'%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-stream-rm%s %s-r -f%s %s'dm.*'%s\n\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-stream-list:list active shared memory streams",
                               "milk-stream-info:inspect stream metadata and data" };
    milk_help_see_also(see_also, 2, mh_color);
}

/**
 * remove_single_stream() - Remove one stream
 * @shmdir: shared memory directory path
 * @sname:  stream name (without .im.shm suffix)
 * @verbose: print extra info if true
 *
 * Handles both symlinks and real SHM files.
 * Returns 0 on success, 1 on failure.
 */
static int remove_single_stream(const char *shmdir,
                                const char *sname,
                                int         verbose,
                                int         quiet,
                                int         ignore_missing)
{
    if (sname[0] == '\0' || strchr(sname, '/') != NULL)
    {
        fprintf(stderr, "ERROR: invalid stream name '%s'\n", sname);
        return 1;
    }

    char fullpath[512];

    snprintf(fullpath, sizeof(fullpath), "%s/%s.im.shm", shmdir, sname);

    /* Check if symlink or missing */
    struct stat lbuf;

    if (lstat(fullpath, &lbuf) != 0)
    {
        if (errno == ENOENT && ignore_missing)
        {
            return 0;
        }
        fprintf(stderr, "Error: cannot open stream '%s'.\n", sname);
        return 1;
    }

    if (S_ISLNK(lbuf.st_mode))
    {
        if (verbose)
        {
            printf("Removing symlink '%s'...\n", sname);
        }
        if (unlink(fullpath) != 0)
        {
            if (errno == ENOENT && ignore_missing)
            {
                return 0;
            }
            PRINT_ERROR("unlink: %s", strerror(errno));
            return 1;
        }
        if (!quiet)
        {
            printf("  Stream symlink '%s'"
                   " removed.\n",
                   sname);
        }
        return 0;
    }

    /* Open and destroy the stream */
    IMAGE   image = { 0 };
    errno_t ret   = ImageStreamIO_read_sharedmem_image_toIMAGE(sname, &image);

    if (ret != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr,
                "Error: cannot open stream"
                " '%s'.\n",
                sname);
        return 1;
    }

    if (verbose)
    {
        printf("Removing stream '%s'...\n", sname);
    }

    /* Destroy semaphores */
    for (int si = 0; si < image.md->sem; si++)
    {
        sem_destroy(image.semptr[si]);
    }

    /* Close (unmap + close fd) */
    ImageStreamIO_closeIm(&image);

    /* Unlink the SHM file */
    if (unlink(fullpath) != 0)
    {
        PRINT_ERROR("unlink: %s", strerror(errno));
        return 1;
    }

    if (!quiet)
    {
        printf("  Stream '%s' removed.\n", sname);
    }
    return 0;
}

/**
 * scan_streams() - Scan SHM directory for streams
 * @shmdir:     shared memory directory
 * @names:      output array of stream names
 * @capacity:   initial array capacity (doubled as needed)
 *
 * Returns number of streams found.
 * Caller must free each name and the array.
 */
static int scan_streams(const char *shmdir, char ***names, int capacity)
{
    DIR *d = opendir(shmdir);

    if (!d)
    {
        fprintf(stderr, "Error opening directory %s\n", shmdir);
        return -1;
    }

    int count = 0;

    *names = calloc(capacity, sizeof(char *));

    struct dirent *dir;

    while ((dir = readdir(d)) != NULL)
    {
        char *pch = strstr(dir->d_name, ".im.shm");
        if (!pch)
        {
            continue;
        }
        int suffix_pos = (int) (pch - dir->d_name);
        int name_len   = (int) strlen(dir->d_name);
        if (suffix_pos != name_len - 7)
        {
            continue;
        }

        char sname[256];

        snprintf(sname, sizeof(sname), "%.*s", suffix_pos, dir->d_name);

        if (count >= capacity)
        {
            int    new_capacity = capacity * 2;
            char **tmp_names    = realloc(*names, new_capacity * sizeof(char *));
            if (tmp_names == NULL)
            {
                for (int i = 0; i < count; i++)
                {
                    free((*names)[i]);
                }
                free(*names);
                *names = NULL;
                closedir(d);
                return -1;
            }
            *names   = tmp_names;
            capacity = new_capacity;
        }
        (*names)[count] = strdup(sname);
        count++;
    }
    closedir(d);

    return count;
}

/**
 * read_confirmation() - Ask user Y/N question
 *
 * Returns 1 if user confirms, 0 otherwise.
 */
static int read_confirmation(void)
{
    printf("  Proceed? [y/N] ");
    fflush(stdout);

    struct termios old_term;
    int            is_tty = isatty(STDIN_FILENO);

    if (is_tty)
    {
        tcgetattr(STDIN_FILENO, &old_term);
        struct termios t = old_term;

        t.c_lflag |= (ICANON | ECHO);
        t.c_iflag |= ICRNL;
        tcsetattr(STDIN_FILENO, TCSANOW, &t);
    }

    char linebuf[64];
    int  ok = (fgets(linebuf, sizeof(linebuf), stdin) != NULL);

    if (is_tty)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    }

    if (!ok)
    {
        return 0;
    }

    /* Strip trailing whitespace */
    {
        char *p = linebuf + strlen(linebuf);

        while (p > linebuf && (*(p - 1) == '\n' || *(p - 1) == '\r'))
        {
            *(--p) = '\0';
        }
    }

    return (linebuf[0] == 'y' || linebuf[0] == 'Y');
}

/**
 * read_line() - Read a line from stdin
 * @buf:  output buffer
 * @size: buffer size
 *
 * Handles terminal mode setup and strips
 * trailing whitespace.  Returns 1 on success.
 */
static int read_line(char *buf, int size)
{
    struct termios old_term;
    int            is_tty = isatty(STDIN_FILENO);

    if (is_tty)
    {
        tcgetattr(STDIN_FILENO, &old_term);
        struct termios t = old_term;

        t.c_lflag |= (ICANON | ECHO);
        t.c_iflag |= ICRNL;
        tcsetattr(STDIN_FILENO, TCSANOW, &t);
    }

    int ok = (fgets(buf, size, stdin) != NULL);

    if (is_tty)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    }

    if (!ok)
    {
        return 0;
    }

    /* Strip trailing whitespace */
    {
        char *p = buf + strlen(buf);

        while (p > buf && (*(p - 1) == '\n' || *(p - 1) == '\r'))
        {
            *(--p) = '\0';
        }
    }

    return 1;
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv, SR_DESC, SR_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    enum
    {
        OPT_IGNORE_MISSING = 1001
    };

    int verbose        = 0;
    int quiet          = 0;
    int use_regex      = 0;
    int force          = 0;
    int ignore_missing = 0;
    int opt;

    static struct option long_options[] = { { "verbose", no_argument, 0, 'v' },
                                            { "quiet", no_argument, 0, 'q' },
                                            { "regex", no_argument, 0, 'r' },
                                            { "force", no_argument, 0, 'f' },
                                            { "ignore-missing", no_argument, 0,
                                              OPT_IGNORE_MISSING },
                                            { "help", no_argument, 0, 'h' },
                                            { 0, 0, 0, 0 } };

    while ((opt = getopt_long(argc, argv, "vqrfh", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'v':
            verbose = 1;
            break;
        case 'q':
            quiet = 1;
            break;
        case 'r':
            use_regex = 1;
            break;
        case 'f':
            force = 1;
            break;
        case OPT_IGNORE_MISSING:
            ignore_missing = 1;
            break;
        case 'h':
            break; /* handled above */
        default:
            printf("\n\033[1;31mERROR\033[0m: Invalid option.\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    const char *pattern = NULL;

    if (optind < argc)
    {
        pattern = argv[optind];
    }

    /* Determine SHM directory */
    const char *shmdir = getenv("MILK_SHM_DIR");

    if (shmdir == NULL)
    {
        shmdir = "/milk/shm";
        struct stat st;

        if (stat(shmdir, &st) != 0)
        {
            shmdir = "/dev/shm";
        }
    }

    /* ------------------------------------------
     * Mode 1: Exact match (one or more names)
     * ----------------------------------------*/
    if (pattern != NULL && !use_regex)
    {
        int errors = 0;
        for (int arg_idx = optind; arg_idx < argc; arg_idx++)
        {
            errors += remove_single_stream(shmdir, argv[arg_idx], verbose, quiet, ignore_missing);
        }
        return (errors > 0) ? 1 : 0;
    }

    /* ------------------------------------------
     * Scan all streams for regex/interactive mode
     * ----------------------------------------*/
    {
        regex_t regex;
        int     regex_compiled = 0;

        if (use_regex && pattern != NULL)
        {
            int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
            if (ret != 0)
            {
                fprintf(stderr, "Error: invalid regex '%s'.\n", pattern);
                return 1;
            }
            regex_compiled = 1;
        }

        int    capacity  = 64;
        char **all_names = NULL;
        int    total     = scan_streams(shmdir, &all_names, capacity);

        if (total < 0)
        {
            if (regex_compiled)
            {
                regfree(&regex);
            }
            return 1;
        }

        if (total == 0)
        {
            if (regex_compiled)
            {
                regfree(&regex);
            }
            if (use_regex && pattern != NULL && ignore_missing)
            {
                free(all_names);
                return 0;
            }
            if (!quiet)
            {
                printf("No streams found.\n");
            }
            free(all_names);
            return 1;
        }

        /* ------------------------------------------
         * Mode 2: Regex match (with -r flag)
         * ----------------------------------------*/
        if (use_regex && pattern != NULL)
        {
            /* Filter matching streams */
            int    match_cap   = 64;
            int    match_count = 0;
            char **match_names = calloc(match_cap, sizeof(char *));

            for (int i = 0; i < total; i++)
            {
                if (regexec(&regex, all_names[i], 0, NULL, 0) == 0)
                {
                    if (match_count >= match_cap)
                    {
                        int    new_match_cap = match_cap * 2;
                        char **tmp_match     = realloc(match_names, new_match_cap * sizeof(char *));
                        if (tmp_match == NULL)
                        {
                            for (int j = 0; j < match_count; j++)
                            {
                                free(match_names[j]);
                            }
                            free(match_names);
                            for (int j = i; j < total; j++)
                            {
                                free(all_names[j]);
                            }
                            free(all_names);
                            regfree(&regex);
                            return 1;
                        }
                        match_names = tmp_match;
                        match_cap   = new_match_cap;
                    }
                    match_names[match_count] = all_names[i];
                    match_count++;
                }
                else
                {
                    free(all_names[i]);
                }
            }
            free(all_names);
            regfree(&regex);

            if (match_count == 0)
            {
                free(match_names);
                if (ignore_missing)
                {
                    return 0;
                }
                fprintf(stderr,
                        "No streams matching"
                        " '%s' found.\n",
                        pattern);
                return 1;
            }

            /* Show matches */
            if (!(quiet && force))
            {
                printf("\n  Streams matching '%s'"
                       " (%d):\n\n",
                       pattern, match_count);
                for (int i = 0; i < match_count; i++)
                {
                    printf("    %s\n", match_names[i]);
                }
                printf("\n");
            }

            /* Confirm unless -f */
            if (!force)
            {
                if (!read_confirmation())
                {
                    printf("Cancelled.\n");
                    for (int i = 0; i < match_count; i++)
                    {
                        free(match_names[i]);
                    }
                    free(match_names);
                    return 0;
                }
            }

            /* Remove all matched streams */
            int errors = 0;

            for (int i = 0; i < match_count; i++)
            {
                errors +=
                    remove_single_stream(shmdir, match_names[i], verbose, quiet, ignore_missing);
                free(match_names[i]);
            }
            free(match_names);

            if (errors > 0)
            {
                fprintf(stderr,
                        "%d stream(s) failed"
                        " to remove.\n",
                        errors);
                return 1;
            }
            return 0;
        }

        /* ------------------------------------------
         * Mode 3: Interactive selection (no pattern)
         * ----------------------------------------*/
        printf("\n  Streams:\n\n");
        for (int i = 0; i < total; i++)
        {
            printf("  %3d  %s\n", i + 1, all_names[i]);
        }

        printf("\n  Enter number(s) to remove"
               " (e.g. 1 3 5-7, 'all',"
               " 0 to cancel): ");
        fflush(stdout);

        char linebuf[512];

        if (!read_line(linebuf, sizeof(linebuf)))
        {
            printf("Cancelled.\n");
            for (int i = 0; i < total; i++)
            {
                free(all_names[i]);
            }
            free(all_names);
            return 0;
        }

        int *selected = calloc(total, sizeof(int));
        int  nsel     = parse_multiselect(linebuf, selected, total);

        if (nsel <= 0)
        {
            printf("Cancelled.\n");
            free(selected);
            for (int i = 0; i < total; i++)
            {
                free(all_names[i]);
            }
            free(all_names);
            return 0;
        }

        int errors = 0;

        for (int i = 0; i < total; i++)
        {
            if (selected[i])
            {
                errors +=
                    remove_single_stream(shmdir, all_names[i], verbose, quiet, ignore_missing);
            }
        }

        free(selected);
        for (int i = 0; i < total; i++)
        {
            free(all_names[i]);
        }
        free(all_names);

        if (errors > 0)
        {
            fprintf(stderr,
                    "%d stream(s) failed"
                    " to remove.\n",
                    errors);
            return 1;
        }
        return 0;
    }
}

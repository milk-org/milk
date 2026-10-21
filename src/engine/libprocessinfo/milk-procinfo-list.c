// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-procinfo-list.c
 * @brief List processinfo shared-memory entries
 *
 * Scans the SHM directory directly for proc.*.shm files.
 * Does NOT depend on processinfo.list.shm or milk-procCTRL-scan.
 * Status is determined by mmap-ing each file and checking
 * the live PID via kill(pid, 0).
 */

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <regex.h>
#include <getopt.h>
#include <sys/mman.h>

#include "processinfo.h"
#include "processinfo_procdirname.h"
#include "milkDebugTools.h"
#include "milk_help.h"

#define C_TITLE MH_TITLE
#define C_HDR MH_HDR
#define C_NAME MH_CMD
#define C_TYPE MH_NOTE
#define C_ERR MH_ERR
#define C_DIM MH_DIM
#define C_RST MH_RST

#define PIL_DESC "list processinfo shared-memory entries"
#define PIL_DESC_LONG                                                        \
    "Scan the processinfo directory (e.g. /dev/shm) for proc.*.shm files\n"  \
    "and print a summary table. Each row shows process name, PID, and\n"     \
    "status (RUNNING/STOPPED/CRASHED). Status is determined live from the\n" \
    "process file — does not require milk-procCTRL-scan to be running.\n"    \
    "An optional POSIX regex filters which process names are shown."

/**
 * @brief Print help message for milk-procinfo-list.
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, PIL_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sregex%s]\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", PIL_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-procinfo-list%s\n", mh_color ? MH_CMD : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-procinfo-list%s %smyproc.*%s\n\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-procinfo-info:inspect processinfo memory contents",
                               "milk-procinfo-rm:remove a processinfo instance",
                               "milk-procCTRL:launch the processinfo dashboard TUI" };
    milk_help_see_also(see_also, 3, mh_color);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv, PIL_DESC, PIL_DESC_LONG);
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

    int                  opt;
    static struct option long_options[] = { { "help", no_argument, 0, 'h' }, { 0, 0, 0, 0 } };
    while ((opt = getopt_long(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            break; /* handled above */
        default:
            printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    /* Optional regex filter */
    const char *pattern = NULL;
    regex_t     regex;
    int         use_regex = 0;

    if (optind < argc)
    {
        pattern = argv[optind];
        int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
        if (ret != 0)
        {
            char errbuf[128];
            regerror(ret, &regex, errbuf, sizeof(errbuf));
            fprintf(stderr, "Error: Invalid regular expression: %s\n", errbuf);
            return 1;
        }
        use_regex = 1;
    }

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    DIR *dir = opendir(procdname);
    if (dir == NULL)
    {
        PRINT_ERROR("opendir: %s", strerror(errno));
        if (use_regex)
        {
            regfree(&regex);
        }
        return 1;
    }

    printf(C_TITLE "%-30s %-10s %-10s" C_RST "\n", "Process Name", "PID", "Status");
    printf(C_DIM);
    for (int i = 0; i < 60; i++)
    {
        putchar('-');
    }
    printf(C_RST "\n");

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        const char *fname = entry->d_name;

        /* Match proc.NAME.PID.shm */
        if (strncmp(fname, "proc.", 5) != 0)
        {
            continue;
        }
        int flen = (int) strlen(fname);
        if (flen < 10 || strcmp(fname + flen - 4, ".shm") != 0)
        {
            continue;
        }

        /* Find the dot separating NAME and PID:
         * walk backward from before .shm to find last '.' */
        const char *q = fname + flen - 5; /* last char before .shm */
        while (q > fname + 5 && *q != '.')
        {
            q--;
        }
        if (*q != '.')
        {
            continue;
        }

        /* Parse PID */
        pid_t pid = (pid_t) atoi(q + 1);
        if (pid <= 0)
        {
            continue;
        }

        /* Extract process name: fname+5 .. q-1 */
        int pname_len = (int) (q - (fname + 5));
        if (pname_len <= 0 || pname_len >= STRINGMAXLEN_PROCESSINFO_NAME)
        {
            continue;
        }

        char pname[STRINGMAXLEN_PROCESSINFO_NAME];
        memcpy(pname, fname + 5, (size_t) pname_len);
        pname[pname_len] = '\0';

        /* Apply optional regex filter */
        if (use_regex && regexec(&regex, pname, 0, NULL, 0) != 0)
        {
            continue;
        }

        /* Map the PROCESSINFO struct to get loopstat */
        char fullpath[STRINGMAXLEN_FULLFILENAME];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, fname);

        int loopstat = -1; /* unknown */
        int fd       = open(fullpath, O_RDONLY);
        if (fd != -1)
        {
            PROCESSINFO *pinfo =
                (PROCESSINFO *) mmap(NULL, sizeof(PROCESSINFO), PROT_READ, MAP_SHARED, fd, 0);
            if (pinfo != MAP_FAILED)
            {
                loopstat = pinfo->loopstat;
                /* Prefer the name stored inside the struct */
                if (pinfo->name[0] != '\0')
                {
                    strncpy(pname, pinfo->name, sizeof(pname) - 1);
                    pname[sizeof(pname) - 1] = '\0';
                }
                munmap(pinfo, sizeof(PROCESSINFO));
            }
            close(fd);
        }

        /* Determine status from live PID check */
        const char *status_str;
        const char *pid_color = "";
        const char *pid_reset = "";

        int alive = (kill(pid, 0) == 0 || errno == EPERM);

        if (alive)
        {
            if (loopstat == PROCESSINFO_LOOPSTAT_STOP)
            {
                status_str = C_TYPE "STOPPED" C_RST;
            }
            else
            {
                status_str = C_NAME "RUNNING" C_RST;
                pid_color  = C_NAME;
                pid_reset  = C_RST;
            }
        }
        else
        {
            if (loopstat == PROCESSINFO_LOOPSTAT_STOP)
            {
                status_str = C_TYPE "STOPPED" C_RST;
            }
            else
            {
                status_str = C_ERR "CRASHED" C_RST;
            }
        }

        printf(C_NAME "%-30s" C_RST " %s%-10ld%s %s\n", pname, pid_color, (long) pid, pid_reset,
               status_str);
    }

    closedir(dir);
    if (use_regex)
    {
        regfree(&regex);
    }

    return 0;
}

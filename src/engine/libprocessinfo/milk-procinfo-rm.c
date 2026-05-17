/**
 * @file milk-procinfo-rm.c
 * @brief Milk procinfo rm module
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <regex.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_procdirname.h"
#include "milkDebugTools.h"
#include "milk_help.h"

#define PI_RM_DESC \
    "remove processinfo shared-memory entries matching a regex"
#define PI_RM_DESC_LONG \
    "Scan the processinfo directory (e.g. /dev/shm) and remove all\n" \
    "proc.<name>.<pid>.shm files whose base name matches the given\n" \
    "POSIX extended regular expression.\n" \
    "If --clean-dead is used, removes only entries whose PID is no\n" \
    "longer alive or whose loopstat is CRASHED/STOPPED."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, PI_RM_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<regex>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", PI_RM_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-c, --clean-dead",
           mh_color ? MH_RST : "", "Remove all CRASHED or STOPPED entries");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-v, --verbose",
           mh_color ? MH_RST : "", "Verbose output");
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
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-procinfo-rm%s %smyproc%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = {
        "milk-procinfo-list:list active processinfo instances",
        "milk-procinfo-info:inspect processinfo memory contents"
    };
    milk_help_see_also(see_also, 2, mh_color);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                PI_RM_DESC, PI_RM_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
        return 0;
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int verbose = 0;
    int clean_dead = 0;
    int opt;

    static struct option long_options[] = {
        {"clean-dead", no_argument,       0, 'c'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "cvh",
                              long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'c': clean_dead = 1; break;
            case 'v': verbose = 1; break;
            case 'h': break; /* handled above */
            case '?':
            default:
                printf("\n\033[1;31mERROR\033[0m: Invalid option.\n\n");
                print_help(argv[0], 1);
                return 1;
        }
    }

    const char *pattern;
    if (optind >= argc)
    {
        if (clean_dead) {
            pattern = ".*";
        } else {
            printf("\n\033[1;31mERROR\033[0m Missing process name.\n");
            print_help(argv[0], 1);
            return 1;
        }
    }
    else
    {
        pattern = argv[optind];
    }
    regex_t regex;
    int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
    if (ret != 0) {
        char error_msg[128];
        regerror(ret, &regex, error_msg, sizeof(error_msg));
        printf("\n\033[1;31mERROR\033[0m Invalid regular expression. %s\n", error_msg);
        print_help(argv[0], 1);
        return 1;
    }

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    if (verbose) {
        printf("Scanning directory '%s' to remove processes matching '%s'...\n", procdname, pattern);
    }

    DIR *dir = opendir(procdname);
    if (!dir) {
        PRINT_ERROR("opendir: %s", strerror(errno));
        return 1;
    }

    struct dirent *entry;

    int removed_count = 0;
    int skipped_alive = 0;

    while ((entry = readdir(dir)) != NULL)
    {
        if (strncmp(entry->d_name, "proc.", 5) != 0)
        {
            continue;
        }
        if (strstr(entry->d_name, ".shm") == NULL)
        {
            continue;
        }

        /* Extract pname from proc.PNAME.XXXXXX.shm */
        char ext_pname[256];
        strncpy(ext_pname, entry->d_name + 5,
                sizeof(ext_pname) - 1);
        ext_pname[sizeof(ext_pname) - 1] = '\0';
        char *dot = strchr(ext_pname, '.');
        if (dot)
        {
            *dot = '\0';
        }

        if (regexec(&regex, ext_pname, 0, NULL, 0) != 0)
        {
            continue;
        }

        /* Match found — open and map to inspect */
        char fullpath[STRINGMAXLEN_FULLFILENAME + 256];
        snprintf(fullpath, sizeof(fullpath),
                 "%s/%s", procdname, entry->d_name);

        pid_t pid       = 0;
        int   loopstat  = -1;
        int   pid_alive = 0;

        int fd = open(fullpath, O_RDONLY);
        if (fd != -1)
        {
            PROCESSINFO *pinfo =
                (PROCESSINFO *) mmap(
                    NULL,
                    sizeof(PROCESSINFO),
                    PROT_READ,
                    MAP_SHARED,
                    fd,
                    0);
            if (pinfo != MAP_FAILED)
            {
                pid      = pinfo->PID;
                loopstat = pinfo->loopstat;
                /* alive = kill succeeds, or EPERM (process
                 * exists but we lack permission to signal) */
                pid_alive =
                    (kill(pid, 0) == 0 || errno == EPERM);
                munmap(pinfo, sizeof(PROCESSINFO));
            }
            close(fd);
        }

        /* Liveness guard: always block removal of alive procs */
        if (pid_alive)
        {
            fprintf(stderr,
                    "Skipping %s — PID %ld is still alive\n",
                    fullpath, (long) pid);
            skipped_alive++;
            continue;
        }

        /* --clean-dead: additionally require crashed/stopped */
        if (clean_dead)
        {
            if (loopstat != PROCESSINFO_LOOPSTAT_CRASHED &&
                loopstat != PROCESSINFO_LOOPSTAT_STOP)
            {
                if (verbose)
                {
                    printf("Skipping %s — not crashed/stopped\n",
                           fullpath);
                }
                continue;
            }
        }

        if (verbose)
        {
            printf("Removing %s\n", fullpath);
        }
        if (unlink(fullpath) == 0)
        {
            removed_count++;
        }
        else
        {
            PRINT_ERROR("unlink: %s", strerror(errno));
        }
    }
    closedir(dir);

    printf("Removed %d shared memory segment(s)"
           " matching '%s'",
           removed_count, pattern);
    if (skipped_alive > 0)
    {
        printf(" (%d skipped — PID still alive)",
               skipped_alive);
    }
    printf(".\n");



    regfree(&regex);

    return 0;
}

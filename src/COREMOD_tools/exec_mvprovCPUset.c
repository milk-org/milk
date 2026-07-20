// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#define _GNU_SOURCE // Needed for get/set resuid

#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/milkDebugTools.h"

DATA data = { 0 }; // Lot of macros rely on it

// Real UID of the invoking user (saved at startup, never changes)
static uid_t STARTUP_RUID = (uid_t) -1;
// Effective UID granted by the SUID bit (0 = root when compiled SUID root)
static uid_t STARTUP_EUID = (uid_t) -1;

// Restore effective UID to root (requires SUID-root binary).
// Returns 0 on success, -1 on failure.
int upscale_privileges()
{
    if (setresuid(STARTUP_EUID, STARTUP_EUID, STARTUP_EUID) != 0)
    {
        PRINT_ERROR("seteuid to root failed");
        return -1;
    }
    return 0;
}

// Drop effective UID back to the invoking user.
// Returns 0 on success, -1 on failure.
int downscale_privileges()
{
    if (setresuid(STARTUP_RUID, STARTUP_RUID, STARTUP_EUID) != 0)
    {
        PRINT_ERROR("seteuid to user failed");
        return -1;
    }
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-p <rtprio>] [-t <tsetspec>] [-c <csetname>] <pid>\n"
            "  -p <rtprio>   Set SCHED_FIFO real-time priority for <pid>\n"
            "  -t <tsetspec> Assign CPU affinity (taskset) to <pid>\n"
            "  -c <csetname> Move <pid> to named cpuset\n"
            "  <pid>         Target process ID (required positional, must be last)\n",
            prog);
}

int main(int argc, char **argv)
{
    STARTUP_RUID = getuid();  // unprivileged invoking user
    STARTUP_EUID = geteuid(); // root (0) when binary is SUID root

    // Drop to user immediately; escalate only for the privileged operations.
    if (seteuid(STARTUP_RUID) != 0)
    {
        PRINT_ERROR("initial seteuid drop failed");
        return EXIT_FAILURE;
    }

    int  rtprio        = -1;
    char tsetspec[256] = "";
    char csetname[256] = "";

    int opt;
    while ((opt = getopt(argc, argv, "p:t:c:")) != -1)
    {
        switch (opt)
        {
        case 'p':
            rtprio = atoi(optarg);
            break;
        case 't':
            strncpy(tsetspec, optarg, sizeof(tsetspec) - 1);
            break;
        case 'c':
            strncpy(csetname, optarg, sizeof(csetname) - 1);
            break;
        default:
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (optind >= argc)
    {
        fprintf(stderr, "Error: PID argument required\n");
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    pid_t pid = (pid_t) atoi(argv[optind]);

    if (upscale_privileges() != 0)
    {
        return EXIT_FAILURE;
    }

    int ret = EXIT_SUCCESS;

    if (csetname[0] != '\0')
    {
        // cgroups v1: write each TID individually — "tasks" accepts only one TID
        // at a time and does not move sibling threads automatically.
        // cgroups v2: write the TGID once to "cgroup.procs" — the kernel moves
        // all threads atomically, equivalent to cset proc --threads.
        char taskfile[512];
        snprintf(taskfile, sizeof(taskfile), "/sys/fs/cgroup/cpuset/%s/tasks", csetname);
        FILE *f = fopen(taskfile, "w");
        if (f != NULL)
        {
            // v1 path: iterate /proc/<pid>/task/ to move every thread
            char taskdir[64];
            snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", (int) pid);
            DIR *d = opendir(taskdir);
            if (d != NULL)
            {
                struct dirent *ent;
                while ((ent = readdir(d)) != NULL)
                {
                    if (ent->d_name[0] == '.')
                    {
                        continue;
                    }
                    fprintf(f, "%s\n", ent->d_name);
                }
                closedir(d);
            }
            else
            {
                // /proc/<pid>/task not readable — fall back to main thread only
                fprintf(f, "%d\n", (int) pid);
            }
            fclose(f);
        }
        else
        {
            // v2 path: writing TGID to cgroup.procs moves all threads atomically
            snprintf(taskfile, sizeof(taskfile), "/sys/fs/cgroup/%s/cgroup.procs", csetname);
            f = fopen(taskfile, "w");
            if (f != NULL)
            {
                fprintf(f, "%d\n", (int) pid);
                fclose(f);
            }
            else
            {
                PRINT_ERROR("cannot open cpuset tasks file for '%s': %s", csetname,
                            strerror(errno));
                ret = EXIT_FAILURE;
            }
        }
    }

    if (tsetspec[0] != '\0')
    {
        // taskset -pc accepts a CPU list (e.g. "0-3", "1,3") or hex mask.
        EXECUTE_SYSTEM_COMMAND("taskset -pc %s %d", tsetspec, (int) pid);
    }

    if (rtprio >= 0)
    {
        struct sched_param param;
        param.sched_priority = rtprio;
        if (sched_setscheduler(pid, (rtprio > 0) ? SCHED_FIFO : SCHED_OTHER, &param) != 0)
        {
            PRINT_ERROR("sched_setscheduler %s prio %d for pid %d: %s",
                        (rtprio > 0) ? "SCHED_FIFO" : "SCHED_OTHER", rtprio, (int) pid,
                        strerror(errno));
            ret = EXIT_FAILURE;
        }
    }

    downscale_privileges();
    return ret;
}

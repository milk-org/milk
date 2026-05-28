/**
 * @file    milk-makecsetandrt.c
 * @brief   Move a PID to a cpuset and assign RT priority
 *
 * Supports both cgroup v1 (cpuset) and cgroup v2 unified hierarchy.
 * When installed SUID root, allows unprivileged users to move
 * processes to cpusets and assign SCHED_FIFO real-time priorities.
 *
 * cgroup v1: /sys/fs/cgroup/cpuset/<name>/tasks
 *   Each thread ID is written individually (v1 requires it).
 * cgroup v2: /sys/fs/cgroup/<name>/cgroup.procs
 *   Writing the TGID moves all threads atomically.
 * v1 is tried first; v2 is the fallback.
 *
 * Usage:
 *   milk-makecsetandrt [-p <rtprio>] [-t <tsetspec>] \
 *                      [-c <csetname>] <pid>
 */

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "milk_help.h"

/* ANSI sequences used in status output */
#define ANSI_GRN "\033[1;32m"
#define ANSI_RED "\033[1;31m"
#define ANSI_BLD "\033[1m"
#define ANSI_RST "\033[0m"

#define MCSR_DESC "move PID to cpuset and assign real-time priority"

#define MCSR_DESC_LONG                                            \
    "Move a process to a named cpuset (cgroup v1 or v2) and\n"    \
    "optionally assign a SCHED_FIFO real-time priority and/or\n"  \
    "set CPU affinity via taskset.\n"                             \
    "\n"                                                          \
    "cgroup v1: /sys/fs/cgroup/cpuset/<name>/tasks\n"             \
    "  All thread IDs are written individually.\n"                \
    "cgroup v2: /sys/fs/cgroup/<name>/cgroup.procs\n"             \
    "  TGID written once; kernel moves all threads atomically.\n" \
    "v1 is tried first; v2 is the fallback.\n"                    \
    "\n"                                                          \
    "RT scheduling is applied via sched_setscheduler(2).\n"       \
    "CPU affinity is applied via sched_setaffinity(2).\n"         \
    "\n"                                                          \
    "Install SUID root to allow unprivileged callers to raise\n"  \
    "scheduling priority (requires CAP_SYS_NICE or root)."

/* ------------------------------------------------------------------ */
/* Privilege management (SUID-root pattern)                           */
/* ------------------------------------------------------------------ */

/* Real and effective UIDs saved at startup; never changed after that */
static uid_t s_ruid = (uid_t) -1;
static uid_t s_euid = (uid_t) -1;

/**
 * upscale_privileges() - Restore effective UID to root
 * Requires the binary to be installed SUID root.
 */
static int upscale_privileges(void)
{
    if (setresuid(s_euid, s_euid, s_euid) != 0)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": seteuid to root failed: %s\n",
                strerror(errno));
        return -1;
    }
    return 0;
}

/**
 * downscale_privileges() - Drop effective UID back to invoking user
 */
static int downscale_privileges(void)
{
    if (setresuid(s_ruid, s_ruid, s_euid) != 0)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": seteuid to user failed: %s\n",
                strerror(errno));
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* print_help()                                                        */
/* ------------------------------------------------------------------ */

/**
 * print_help() - Print full usage, options, and examples
 * @progname: argv[0]
 * @mh_color: non-zero for ANSI color output
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, MCSR_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s-p <rtprio>%s] [%s-t <tsetspec>%s]"
           " [%s-c <csetname>%s] %s<pid>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", MCSR_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-p <rtprio>", mh_color ? MH_RST : "",
           "SCHED_FIFO priority 1-99 (0 for SCHED_OTHER)");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-t <tsetspec>", mh_color ? MH_RST : "",
           "CPU affinity list (e.g. '0-3', '1,3')");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-c <csetname>", mh_color ? MH_RST : "",
           "cpuset/cgroup name (v1 or v2)");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Arguments", mh_color);
    printf("  %s%-20s%s %s\n\n", mh_color ? MH_ARG : "", "<pid>", mh_color ? MH_RST : "",
           "Target process ID (required, must be last)");
    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s %s-c milk -p 80%s %s33654%s\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ %s%s %s-p 50 -t 2-5%s %s12345%s\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ %s%s %s-c realtime -p 50 -t 1,3%s %s12345%s\n\n", mh_color ? MH_CMD : "",
           progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
}

/* ------------------------------------------------------------------ */
/* move_to_cset()                                                      */
/* ------------------------------------------------------------------ */

/**
 * move_to_cset() - Move a process to a named cpuset (v1 or v2)
 * @pid:      process ID to move
 * @csetname: cpuset/cgroup name
 *
 * Tries cgroup v1 first (/sys/fs/cgroup/cpuset/<name>/tasks),
 * iterating /proc/<pid>/task/ to write each TID individually.
 * Falls back to cgroup v2 (/sys/fs/cgroup/<name>/cgroup.procs),
 * writing the TGID once (kernel moves all threads atomically).
 *
 * Returns 0 on success, 1 on error.
 */
static int move_to_cset(pid_t pid, const char *csetname)
{
    char taskfile[512];

    /* Try cgroup v1 first */
    snprintf(taskfile, sizeof(taskfile), "/sys/fs/cgroup/cpuset/%s/tasks", csetname);
    FILE *f = fopen(taskfile, "w");
    if (f != NULL)
    {
        char taskdir[128];
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
            /* /proc/<pid>/task not readable -- fall back to main PID */
            fprintf(f, "%d\n", (int) pid);
        }
        fclose(f);
        printf("  moved PID %d to cgroup v1 cpuset '%s'\n", (int) pid, csetname);
        return 0;
    }

    /* Fall back to cgroup v2 -- TGID moves all threads atomically */
    snprintf(taskfile, sizeof(taskfile), "/sys/fs/cgroup/%s/cgroup.procs", csetname);
    f = fopen(taskfile, "w");
    if (f != NULL)
    {
        fprintf(f, "%d\n", (int) pid);
        fclose(f);
        printf("  moved PID %d to cgroup v2 '%s'\n", (int) pid, csetname);
        return 0;
    }

    fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": cannot open cpuset tasks file for '%s': %s\n",
            csetname, strerror(errno));
    return 1;
}

/* ------------------------------------------------------------------ */
/* set_cpu_affinity()                                                  */
/* ------------------------------------------------------------------ */

/**
 * parse_cpulist() - Parse a CPU list string into a cpu_set_t
 * @spec: CPU list (e.g. "0", "0-3", "1,3", "0-3,5,7-9")
 * @set:  output cpu_set_t, zeroed and populated on success
 *
 * Returns 0 on success, -1 on parse error.
 */
static int parse_cpulist(const char *spec, cpu_set_t *set)
{
    CPU_ZERO(set);

    char buf[256];
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *saveptr;
    char *token = strtok_r(buf, ",", &saveptr);
    while (token != NULL)
    {
        char *dash = strchr(token, '-');
        if (dash != NULL)
        {
            *dash = '\0';
            char *ep1, *ep2;
            long  lo = strtol(token, &ep1, 10);
            long  hi = strtol(dash + 1, &ep2, 10);
            if (*ep1 != '\0' || *ep2 != '\0' || lo < 0 || hi < lo || hi >= CPU_SETSIZE)
            {
                return -1;
            }
            for (long i = lo; i <= hi; i++)
            {
                CPU_SET((int) i, set);
            }
        }
        else
        {
            char *ep;
            long  cpu = strtol(token, &ep, 10);
            if (*ep != '\0' || cpu < 0 || cpu >= CPU_SETSIZE)
            {
                return -1;
            }
            CPU_SET((int) cpu, set);
        }
        token = strtok_r(NULL, ",", &saveptr);
    }
    return 0;
}

/**
 * set_cpu_affinity() - Set CPU affinity of all threads of a PID
 * @pid:      process ID
 * @tsetspec: CPU list string (e.g. "0-3", "1,3")
 *
 * Parses @tsetspec into a cpu_set_t and calls sched_setaffinity(2)
 * on every thread in /proc/<pid>/task/.
 *
 * Returns 0 on success, 1 on error.
 */
static int set_cpu_affinity(pid_t pid, const char *tsetspec)
{
    cpu_set_t set;
    if (parse_cpulist(tsetspec, &set) != 0)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": invalid CPU list '%s'\n", tsetspec);
        return 1;
    }

    char taskdir[128];
    snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", (int) pid);
    DIR *d = opendir(taskdir);
    if (d == NULL)
    {
        /* /proc/<pid>/task not readable -- apply to main thread only */
        if (sched_setaffinity(pid, sizeof(set), &set) != 0)
        {
            fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": sched_setaffinity for PID %d: %s\n",
                    (int) pid, strerror(errno));
            return 1;
        }
        printf("  CPU affinity '%s' set for PID %d\n", tsetspec, (int) pid);
        return 0;
    }

    int            errors = 0, done = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL)
    {
        if (ent->d_name[0] == '.')
        {
            continue;
        }
        char *endp;
        long  tid_l = strtol(ent->d_name, &endp, 10);
        if (*endp != '\0' || tid_l <= 0)
        {
            continue;
        }
        pid_t tid = (pid_t) tid_l;
        if (sched_setaffinity(tid, sizeof(set), &set) != 0)
        {
            fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": sched_setaffinity tid %d: %s\n", (int) tid,
                    strerror(errno));
            errors++;
        }
        else
        {
            done++;
        }
    }
    closedir(d);

    printf("  CPU affinity '%s' set on %d thread(s) for PID %d"
           " (%d error(s))\n",
           tsetspec, done, (int) pid, errors);
    return errors > 0 ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* set_rt_priority()                                                   */
/* ------------------------------------------------------------------ */

/**
 * set_rt_priority() - Assign RT scheduling priority to a process
 * @pid:    process ID
 * @rtprio: SCHED_FIFO priority 1-99 (0 for SCHED_OTHER)
 *
 * Uses sched_setscheduler(2) directly -- no chrt subprocess.
 * Requires CAP_SYS_NICE or root.
 *
 * Returns 0 on success, 1 on error.
 */
static int set_rt_priority(pid_t pid, int rtprio)
{
    struct sched_param param;
    param.sched_priority = rtprio;

    int policy = (rtprio > 0) ? SCHED_FIFO : SCHED_OTHER;
    if (sched_setscheduler(pid, policy, &param) != 0)
    {
        if (errno == EPERM)
        {
            fprintf(stderr,
                    ANSI_RED "ERROR" ANSI_RST ": SCHED_%s prio %d for PID %d:"
                             " permission denied"
                             " (needs CAP_SYS_NICE or root).\n",
                    (rtprio > 0) ? "FIFO" : "OTHER", rtprio, (int) pid);
        }
        else
        {
            fprintf(stderr,
                    ANSI_RED "ERROR" ANSI_RST ": sched_setscheduler SCHED_%s prio %d"
                             " for PID %d: %s\n",
                    (rtprio > 0) ? "FIFO" : "OTHER", rtprio, (int) pid, strerror(errno));
        }
        return 1;
    }

    printf("  SCHED_%s prio=%d set for PID %d\n", (rtprio > 0) ? "FIFO" : "OTHER", rtprio,
           (int) pid);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main()                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    /* Save UIDs before any seteuid() calls */
    s_ruid = getuid();
    s_euid = geteuid();

    /* Drop to invoking user immediately; escalate only for operations */
    if (seteuid(s_ruid) != 0)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": initial seteuid drop failed: %s\n",
                strerror(errno));
        return EXIT_FAILURE;
    }

    /* Handle -h / -h1 / -hm before getopt sees the flags */
    int action = milk_help_init(argc, argv, MCSR_DESC, MCSR_DESC_LONG);
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

    /* Parse options */
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
            print_help(argv[0], 1);
            return EXIT_FAILURE;
        }
    }

    if (optind >= argc)
    {
        fprintf(stderr, "\n" ANSI_RED "ERROR" ANSI_RST ": PID argument required.\n\n");
        print_help(argv[0], 1);
        return EXIT_FAILURE;
    }

    char *endptr;
    long  pid_l = strtol(argv[optind], &endptr, 10);
    if (*endptr != '\0' || pid_l <= 0)
    {
        fprintf(stderr,
                "\n" ANSI_RED "ERROR" ANSI_RST ": invalid PID '%s'"
                " -- must be a positive integer.\n\n",
                argv[optind]);
        return EXIT_FAILURE;
    }
    pid_t pid = (pid_t) pid_l;

    printf("milk-makecsetandrt: PID=%d cgroup=%s tset=%s prio=%d\n", (int) pid,
           csetname[0] ? csetname : "(none)", tsetspec[0] ? tsetspec : "(none)", rtprio);

    /* Escalate privileges for the privileged operations */
    if (upscale_privileges() != 0)
    {
        return EXIT_FAILURE;
    }

    int ret = EXIT_SUCCESS;

    if (csetname[0] != '\0')
    {
        if (move_to_cset(pid, csetname) != 0)
        {
            ret = EXIT_FAILURE;
        }
    }

    if (tsetspec[0] != '\0')
    {
        if (set_cpu_affinity(pid, tsetspec) != 0)
        {
            ret = EXIT_FAILURE;
        }
    }

    if (rtprio >= 0)
    {
        if (set_rt_priority(pid, rtprio) != 0)
        {
            ret = EXIT_FAILURE;
        }
    }

    downscale_privileges();
    return ret;
}

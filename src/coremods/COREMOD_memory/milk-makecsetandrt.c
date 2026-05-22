/**
 * @file    milk-makecsetandrt.c
 * @brief   Move a PID to a cgroup v2 cpuset and assign RT priority
 *
 * Replaces the legacy milk-makecsetandrt bash script (which used the
 * deprecated cset/cpuset v1 tool). This implementation writes directly
 * to the cgroup v2 unified hierarchy at /sys/fs/cgroup/ and uses the
 * sched_setscheduler(2) syscall for RT priority — no subprocesses.
 *
 * Cgroup v2 cpuset workflow:
 *   1. A named sub-cgroup must already exist under /sys/fs/cgroup/:
 *      /sys/fs/cgroup/<cgroupname>/
 *   2. The cpuset controller must be delegated at root level and
 *      cpuset.cpus must be set by the administrator beforehand.
 *   3. This tool writes all thread IDs of <PID> to:
 *      /sys/fs/cgroup/<cgroupname>/cgroup.threads
 *
 * Usage:
 *   milk-makecsetandrt <PID> <cgroupname> <prio>
 *
 *   PID        : process ID to move (integer > 0)
 *   cgroupname : cgroup v2 sub-cgroup name under /sys/fs/cgroup/
 *                (e.g. "milk", "realtime") — or "NULL" to skip
 *   prio       : SCHED_FIFO RT priority 1-99 (0 to skip)
 */

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "milk_help.h"

#define CGROOT "/sys/fs/cgroup"

/* ANSI sequences used in status output */
#define ANSI_GRN "\033[1;32m"
#define ANSI_RED "\033[1;31m"
#define ANSI_YLW "\033[1;33m"
#define ANSI_BLD "\033[1m"
#define ANSI_RST "\033[0m"

#define MCSR_DESC "move PID to cgroup v2 cpuset and assign real-time priority"

/* ------------------------------------------------------------------ */
/* Long description (used by milk_help_init and -h/--help)            */
/* ------------------------------------------------------------------ */
#define MCSR_DESC_LONG                                               \
    "Move a process (all its threads) to a named cgroup v2 cpuset\n" \
    "and optionally assign a SCHED_FIFO real-time priority.\n"       \
    "\n"                                                             \
    "The cgroup sub-directory must already exist under\n"            \
    "/sys/fs/cgroup/<cgroupname>/ and must have cpuset.cpus\n"       \
    "configured (done once by the system administrator).\n"          \
    "\n"                                                             \
    "All threads of <PID> are enumerated from /proc/<PID>/task/\n"   \
    "and written to /sys/fs/cgroup/<cgroupname>/cgroup.threads.\n"   \
    "RT priority is applied via sched_setscheduler(2) directly\n"    \
    "(no chrt subprocess). Requires CAP_SYS_NICE or root for\n"      \
    "raising scheduling priority.\n"                                 \
    "\n"                                                             \
    "Pass 'NULL' for <cgroupname> to skip CPU set migration.\n"      \
    "Pass 0 for <prio> to skip priority assignment."

/* ------------------------------------------------------------------ */
/* print_setup_commands() — emit the one-time admin setup block       */
/* ------------------------------------------------------------------ */

/**
 * print_setup_commands() - Print the one-time cgroup admin setup block
 * @cgname:   cgroup name (used to build the mkdir path)
 * @cpulist:  example CPU list string (NULL → "<cpulist>")
 * @color:    non-zero for ANSI color output
 *
 * Prints the exact shell commands needed to create and configure
 * the cgroup before milk-makecsetandrt can be used.
 */
static void print_setup_commands(const char *cgname, const char *cpulist, int color)
{
    const char *cg = (cgname && strcmp(cgname, "NULL") != 0) ? cgname : "milk";
    const char *cl = cpulist ? cpulist : "<cpulist>";

    if (color)
    {
        printf("%sOne-time cgroup v2 setup (run as root):%s\n\n", ANSI_BLD, ANSI_RST);
        printf("  %smkdir%s          /sys/fs/cgroup/%s\n", ANSI_GRN, ANSI_RST, cg);
        printf("  %secho%s '+cpuset'  > /sys/fs/cgroup/cgroup.subtree_control\n", ANSI_GRN,
               ANSI_RST);
        printf("  %secho%s '+cpuset'  > /sys/fs/cgroup/%s/cgroup.subtree_control\n", ANSI_GRN,
               ANSI_RST, cg);
        printf("  %secho%s '%s%s%s'    > /sys/fs/cgroup/%s/cpuset.cpus\n", ANSI_GRN, ANSI_RST,
               ANSI_YLW, cl, ANSI_RST, cg);
        printf("  %secho%s '0'        > /sys/fs/cgroup/%s/cpuset.mems\n\n", ANSI_GRN, ANSI_RST, cg);
    }
    else
    {
        printf("One-time cgroup v2 setup (run as root):\n\n");
        printf("  mkdir          /sys/fs/cgroup/%s\n", cg);
        printf("  echo '+cpuset' > /sys/fs/cgroup/cgroup.subtree_control\n");
        printf("  echo '+cpuset' > /sys/fs/cgroup/%s/cgroup.subtree_control\n", cg);
        printf("  echo '%s'    > /sys/fs/cgroup/%s/cpuset.cpus\n", cl, cg);
        printf("  echo '0'       > /sys/fs/cgroup/%s/cpuset.mems\n\n", cg);
    }
}

/* ------------------------------------------------------------------ */
/* print_help()                                                        */
/* ------------------------------------------------------------------ */

/**
 * print_help() - Print full usage, options, and prerequisite setup
 * @progname: argv[0]
 * @mh_color: non-zero for ANSI color output
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, MCSR_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s<PID>%s %s<cgroupname>%s %s<prio>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", MCSR_DESC_LONG);
    milk_help_section("Arguments", mh_color);
    printf("  %s%-20s%s %s\n", mh_color ? MH_ARG : "", "<PID>", mh_color ? MH_RST : "",
           "Process ID to move (integer > 0)");
    printf("  %s%-20s%s %s\n", mh_color ? MH_ARG : "", "<cgroupname>", mh_color ? MH_RST : "",
           "cgroup v2 name under /sys/fs/cgroup/ (or 'NULL' to skip)");
    printf("  %s%-20s%s %s\n\n", mh_color ? MH_ARG : "", "<prio>", mh_color ? MH_RST : "",
           "SCHED_FIFO priority 1-99 (0 to skip)");
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-makecsetandrt%s %s33654 milk 80%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-makecsetandrt%s %s12345 realtime 50%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-makecsetandrt%s %s12345 NULL 0%s\n\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Prerequisites", mh_color);
    print_setup_commands("milk", "2-5", mh_color);
    const char *see_also[] = { "milk-fps-set:set an FPS parameter value" };
    milk_help_see_also(see_also, 1, mh_color);
}

/* ------------------------------------------------------------------ */
/* check_cgroup_setup()                                                */
/* ------------------------------------------------------------------ */

/* Status codes returned by check_cgroup_setup */
#define CGCHECK_OK 0      /* all steps pass */
#define CGCHECK_NOMOUNT 1 /* cgroup v2 not mounted */
#define CGCHECK_NOCTRL 2  /* cpuset controller not available at root */
#define CGCHECK_NODIR 3   /* cgroup sub-directory does not exist */
#define CGCHECK_NOCPUS 4  /* cpuset.cpus is empty / not configured */

/**
 * check_cgroup_setup() - Probe cgroup v2 readiness and report status
 * @cgname: cgroup name under /sys/fs/cgroup/ (must not be "NULL")
 * @color:  non-zero for ANSI color output
 *
 * Checks four prerequisites in order:
 *   1. /sys/fs/cgroup is a cgroup v2 mount
 *   2. cpuset controller listed in /sys/fs/cgroup/cgroup.controllers
 *   3. /sys/fs/cgroup/<cgname>/ directory exists
 *   4. /sys/fs/cgroup/<cgname>/cpuset.cpus is non-empty
 *
 * Prints a ✓ / ✗ line for each step. Returns CGCHECK_OK only when all
 * four pass; returns the first failing CGCHECK_* constant otherwise.
 */
static int check_cgroup_setup(const char *cgname, int color)
{
    const char *ok  = color ? ANSI_GRN "✓" ANSI_RST : "[OK]";
    const char *err = color ? ANSI_RED "✗" ANSI_RST : "[!!]";

    printf("\n%sCgroup v2 readiness check for '%s':%s\n", color ? ANSI_BLD : "", cgname,
           color ? ANSI_RST : "");

    /* Step 1: cgroup v2 mount */
    {
        struct stat st;
        if (stat(CGROOT "/cgroup.controllers", &st) != 0)
        {
            printf("  %s cgroup v2 not mounted at " CGROOT "\n", err);
            return CGCHECK_NOMOUNT;
        }
        printf("  %s cgroup v2 mounted at " CGROOT "\n", ok);
    }

    /* Step 2: cpuset controller available at root */
    {
        FILE *fp = fopen(CGROOT "/cgroup.controllers", "r");
        if (fp == NULL)
        {
            printf("  %s cannot read " CGROOT "/cgroup.controllers\n", err);
            return CGCHECK_NOCTRL;
        }

        char line[256] = "";
        if (fgets(line, sizeof(line), fp) == NULL)
        {
            line[0] = '\0';
        }
        fclose(fp);

        if (strstr(line, "cpuset") == NULL)
        {
            printf("  %s 'cpuset' controller not available at root"
                   " (controllers: %s)\n",
                   err, line);
            return CGCHECK_NOCTRL;
        }
        printf("  %s cpuset controller available\n", ok);
    }

    /* Step 3: sub-cgroup directory exists */
    {
        char cgpath[512];
        snprintf(cgpath, sizeof(cgpath), CGROOT "/%s", cgname);

        struct stat st;
        if (stat(cgpath, &st) != 0 || !S_ISDIR(st.st_mode))
        {
            printf("  %s cgroup directory '%s' not found\n", err, cgpath);
            return CGCHECK_NODIR;
        }
        printf("  %s cgroup directory " CGROOT "/%s exists\n", ok, cgname);
    }

    /* Step 4: cpuset.cpus is non-empty */
    {
        char cpus_path[512];
        snprintf(cpus_path, sizeof(cpus_path), CGROOT "/%s/cpuset.cpus", cgname);

        FILE *fp = fopen(cpus_path, "r");
        if (fp == NULL)
        {
            printf("  %s cannot read '%s': %s\n", err, cpus_path, strerror(errno));
            printf("      (cpuset controller may not be delegated"
                   " to this sub-cgroup)\n");
            return CGCHECK_NOCPUS;
        }

        char cpus[64] = "";
        if (fgets(cpus, sizeof(cpus), fp) == NULL)
        {
            cpus[0] = '\0';
        }
        fclose(fp);

        cpus[strcspn(cpus, "\r\n")] = '\0';

        if (cpus[0] == '\0')
        {
            printf("  %s cpuset.cpus is empty — not configured\n", err);
            return CGCHECK_NOCPUS;
        }
        printf("  %s cpuset.cpus = %s\n", ok, cpus);
    }

    printf("\n");
    return CGCHECK_OK;
}

/* ------------------------------------------------------------------ */
/* move_to_cgroup()                                                    */
/* ------------------------------------------------------------------ */

/**
 * move_thread_to_cgroup() - Write a single TID into cgroup.threads
 * @threads_file: path to cgroup.threads
 * @tid:          thread ID to write
 *
 * Returns 0 on success, 1 on error.
 */
static int move_thread_to_cgroup(const char *threads_file, pid_t tid)
{
    FILE *fp = fopen(threads_file, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "  " ANSI_RED "ERROR" ANSI_RST ": cannot write tid %d to '%s': %s\n",
                (int) tid, threads_file, strerror(errno));
        return 1;
    }
    fprintf(fp, "%d\n", (int) tid);
    fclose(fp);
    return 0;
}

/**
 * move_to_cgroup() - Move all threads of a PID to a cgroup v2 cpuset
 * @pid:    process ID
 * @cgname: cgroup name under /sys/fs/cgroup/ (skipped if "NULL")
 *
 * Enumerates /proc/<pid>/task/ to find all thread IDs and writes each
 * to /sys/fs/cgroup/<cgname>/cgroup.threads.
 *
 * Returns 0 on success, 1 on error.
 */
static int move_to_cgroup(pid_t pid, const char *cgname)
{
    if (strcmp(cgname, "NULL") == 0)
    {
        return 0;
    }

    char threads_file[512];
    snprintf(threads_file, sizeof(threads_file), CGROOT "/%s/cgroup.threads", cgname);

    /* Enumerate all threads of PID via /proc/<pid>/task/ */
    char taskdir[128];
    snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", (int) pid);

    DIR *d = opendir(taskdir);
    if (d == NULL)
    {
        fprintf(stderr,
                ANSI_RED "ERROR" ANSI_RST ": cannot open '%s': %s\n"
                         "  PID %d may not exist.\n",
                taskdir, strerror(errno), (int) pid);
        return 1;
    }

    int            errors = 0;
    int            moved  = 0;
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
        printf("  moving tid %d -> " CGROOT "/%s\n", (int) tid, cgname);

        if (move_thread_to_cgroup(threads_file, tid) != 0)
        {
            errors++;
        }
        else
        {
            moved++;
        }
    }
    closedir(d);

    if (moved == 0 && errors == 0)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": no threads found for PID %d.\n", (int) pid);
        return 1;
    }

    printf("  moved %d thread(s) to cgroup '%s' (%d error(s))\n", moved, cgname, errors);
    return errors > 0 ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* set_rt_priority()                                                   */
/* ------------------------------------------------------------------ */

/**
 * set_rt_priority() - Assign SCHED_FIFO RT priority to all threads of a PID
 * @pid:    process ID
 * @rtprio: SCHED_FIFO priority 1-99 (0 to skip)
 *
 * Uses sched_setscheduler(2) on every TID in /proc/<pid>/task/ —
 * no chrt subprocess.
 *
 * Returns 0 on success, 1 if any thread failed.
 */
static int set_rt_priority(pid_t pid, int rtprio)
{
    if (rtprio <= 0)
    {
        return 0;
    }

    if (rtprio > 99)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": RT priority %d out of range (1-99).\n",
                rtprio);
        return 1;
    }

    struct sched_param sp;
    sp.sched_priority = rtprio;

    char taskdir[128];
    snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", (int) pid);

    DIR *d = opendir(taskdir);
    if (d == NULL)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": cannot open '%s': %s\n", taskdir,
                strerror(errno));
        return 1;
    }

    int            errors = 0;
    int            done   = 0;
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

        if (sched_setscheduler(tid, SCHED_FIFO, &sp) != 0)
        {
            if (errno == EPERM)
            {
                fprintf(stderr,
                        ANSI_RED "ERROR" ANSI_RST ": SCHED_FIFO prio %d on tid %d:"
                                 " permission denied"
                                 " (needs CAP_SYS_NICE or root).\n",
                        rtprio, (int) tid);
            }
            else
            {
                fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": sched_setscheduler tid %d: %s\n",
                        (int) tid, strerror(errno));
            }
            errors++;
        }
        else
        {
            printf("  tid %d SCHED_FIFO prio=%d set\n", (int) tid, rtprio);
            done++;
        }
    }
    closedir(d);

    if (done == 0 && errors == 0)
    {
        fprintf(stderr, ANSI_RED "ERROR" ANSI_RST ": no threads found for PID %d.\n", (int) pid);
        return 1;
    }

    printf("  set SCHED_FIFO prio=%d on %d thread(s) (%d error(s))\n", rtprio, done, errors);
    return errors > 0 ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* main()                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
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

    if (argc != 4)
    {
        fprintf(stderr, "\n" ANSI_RED "ERROR" ANSI_RST ": expected 3 arguments, got %d.\n\n",
                argc - 1);
        print_help(argv[0], 1);
        return 1;
    }

    /* Parse PID */
    char *endptr;
    long  pid_l = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || pid_l <= 0)
    {
        fprintf(stderr,
                "\n" ANSI_RED "ERROR" ANSI_RST ": invalid PID '%s'"
                " — must be a positive integer.\n\n",
                argv[1]);
        return 1;
    }
    pid_t pid = (pid_t) pid_l;

    /* Parse RT priority */
    long prio_l = strtol(argv[3], &endptr, 10);
    if (*endptr != '\0' || prio_l < 0 || prio_l > 99)
    {
        fprintf(stderr,
                "\n" ANSI_RED "ERROR" ANSI_RST ": invalid priority '%s'"
                " — must be 0-99.\n\n",
                argv[3]);
        return 1;
    }
    int rtprio = (int) prio_l;

    const char *cgname = argv[2];

    printf("milk-makecsetandrt: PID=%d cgroup=%s prio=%d\n", (int) pid, cgname, rtprio);

    /*
     * If a cgroup name was given (not "NULL"), run the readiness
     * check before attempting any writes. If the setup is incomplete,
     * print the exact commands needed and exit immediately so the
     * caller gets actionable instructions rather than a cryptic error.
     */
    if (strcmp(cgname, "NULL") != 0)
    {
        int status = check_cgroup_setup(cgname, 1);

        if (status != CGCHECK_OK)
        {
            printf("\n" ANSI_RED "SETUP INCOMPLETE" ANSI_RST " — run the following as root:\n\n");
            print_setup_commands(cgname, NULL, 1);
            return 1;
        }
    }

    int errors = 0;
    errors += move_to_cgroup(pid, cgname);
    errors += set_rt_priority(pid, rtprio);

    return errors > 0 ? 1 : 0;
}

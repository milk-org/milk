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
 *   2. The cpuset controller must be delegated to that cgroup
 *      (cpuset.cpus must be set by the administrator beforehand).
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

#define MCSR_DESC \
    "move PID to cgroup v2 cpuset and assign real-time priority"

#define MCSR_DESC_LONG \
    "Move a process (all its threads) to a named cgroup v2 cpuset\n" \
    "and optionally assign a SCHED_FIFO real-time priority.\n" \
    "\n" \
    "The cgroup sub-directory must already exist under\n" \
    "/sys/fs/cgroup/<cgroupname>/ and must have cpuset.cpus\n" \
    "configured (done once by the system administrator).\n" \
    "\n" \
    "All threads of <PID> are enumerated from /proc/<PID>/task/\n" \
    "and written to /sys/fs/cgroup/<cgroupname>/cgroup.threads.\n" \
    "RT priority is applied via sched_setscheduler(2) directly\n" \
    "(no chrt subprocess). Requires CAP_SYS_NICE or root for\n" \
    "raising scheduling priority.\n" \
    "\n" \
    "Pass 'NULL' for <cgroupname> to skip CPU set migration.\n" \
    "Pass 0 for <prio> to skip priority assignment."

/**
 * print_help() - Print usage and option help
 * @progname: argv[0]
 * @mh_color: non-zero for ANSI color output
 */
static void print_help(
    const char *progname,
    int         mh_color)
{
    milk_help_banner(progname, MCSR_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s<PID>%s %s<cgroupname>%s %s<prio>%s\n\n",
           mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", MCSR_DESC_LONG);
    milk_help_section("Arguments", mh_color);
    printf("  %s%-20s%s %s\n",
           mh_color ? MH_ARG : "", "<PID>",
           mh_color ? MH_RST : "",
           "Process ID to move (integer > 0)");
    printf("  %s%-20s%s %s\n",
           mh_color ? MH_ARG : "", "<cgroupname>",
           mh_color ? MH_RST : "",
           "cgroup v2 name under /sys/fs/cgroup/ (or 'NULL' to skip)");
    printf("  %s%-20s%s %s\n\n",
           mh_color ? MH_ARG : "", "<prio>",
           mh_color ? MH_RST : "",
           "SCHED_FIFO priority 1-99 (0 to skip)");
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "",
           "One-line description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-makecsetandrt%s %s33654 milk 80%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-makecsetandrt%s %s12345 realtime 50%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-makecsetandrt%s %s12345 NULL 0%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Prerequisites", mh_color);
    printf("  Create the cgroup once (as root):\n"
           "    mkdir /sys/fs/cgroup/milk\n"
           "    echo '+cpuset' > /sys/fs/cgroup/cgroup.subtree_control\n"
           "    echo '2-5'    > /sys/fs/cgroup/milk/cpuset.cpus\n"
           "    echo '0'      > /sys/fs/cgroup/milk/cpuset.mems\n\n");
    const char *see_also[] = { "milk-fps-set" };
    milk_help_see_also(see_also, 1, mh_color);
}

/**
 * cgroup_exists() - Check that cgroup dir exists and has cgroup.threads
 * @cgpath: absolute path to the cgroup directory
 *
 * Returns 1 if the cgroup is usable, 0 otherwise.
 */
static int cgroup_exists(const char *cgpath)
{
    struct stat st;

    if (stat(cgpath, &st) != 0 || !S_ISDIR(st.st_mode)) {
        return 0;
    }

    char threads_file[512];
    snprintf(threads_file, sizeof(threads_file),
             "%s/cgroup.threads", cgpath);

    if (stat(threads_file, &st) != 0) {
        return 0;
    }
    return 1;
}

/**
 * move_thread_to_cgroup() - Write a single TID into cgroup.threads
 * @threads_file: path to cgroup.threads
 * @tid:          thread ID to write
 *
 * Returns 0 on success, 1 on error.
 */
static int move_thread_to_cgroup(
    const char *threads_file,
    pid_t       tid)
{
    FILE *fp = fopen(threads_file, "w");
    if (fp == NULL) {
        fprintf(stderr,
                "  \033[1;31mERROR\033[0m: cannot write tid %d"
                " to '%s': %s\n",
                (int)tid, threads_file, strerror(errno));
        return 1;
    }
    fprintf(fp, "%d\n", (int)tid);
    fclose(fp);
    return 0;
}

/**
 * move_to_cgroup() - Move all threads of a PID to a cgroup v2 cpuset
 * @pid:      process ID
 * @cgname:   cgroup name under /sys/fs/cgroup/ (skipped if "NULL")
 *
 * Enumerates /proc/<pid>/task/ to find all thread IDs and writes each
 * to /sys/fs/cgroup/<cgname>/cgroup.threads.
 *
 * Returns 0 on success, 1 on error.
 */
static int move_to_cgroup(pid_t pid, const char *cgname)
{
    if (strcmp(cgname, "NULL") == 0) {
        return 0;
    }

    /* Build cgroup path */
    char cgpath[512];
    snprintf(cgpath, sizeof(cgpath), "%s/%s", CGROOT, cgname);

    if (!cgroup_exists(cgpath)) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: cgroup '%s' does not exist"
                " or is not accessible.\n"
                "  Create it first (as root):\n"
                "    mkdir %s\n"
                "    echo '+cpuset' > %s/cgroup.subtree_control\n"
                "    echo '<cpulist>' > %s/cpuset.cpus\n"
                "    echo '0' > %s/cpuset.mems\n",
                cgpath,
                cgpath, CGROOT, cgpath, cgpath);
        return 1;
    }

    /* Verify cpuset.cpus is configured */
    char cpus_file[512];
    snprintf(cpus_file, sizeof(cpus_file), "%s/cpuset.cpus", cgpath);
    {
        FILE *fp = fopen(cpus_file, "r");
        if (fp == NULL) {
            fprintf(stderr,
                    "\033[1;31mERROR\033[0m: '%s' not readable: %s\n"
                    "  The cpuset controller may not be enabled"
                    " in this cgroup.\n",
                    cpus_file, strerror(errno));
            return 1;
        }

        char cpus[64] = "(empty)";
        if (fgets(cpus, sizeof(cpus), fp) != NULL) {
            cpus[strcspn(cpus, "\r\n")] = '\0';
        }
        fclose(fp);
        printf("  cgroup '%s' cpuset.cpus = %s\n", cgname, cpus);
    }

    char threads_file[512];
    snprintf(threads_file, sizeof(threads_file),
             "%s/cgroup.threads", cgpath);

    /* Enumerate all threads of PID via /proc/<pid>/task/ */
    char taskdir[128];
    snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", (int)pid);

    DIR *d = opendir(taskdir);
    if (d == NULL) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: cannot open '%s': %s\n"
                "  PID %d may not exist.\n",
                taskdir, strerror(errno), (int)pid);
        return 1;
    }

    int errors = 0;
    int moved  = 0;
    struct dirent *ent;

    while ((ent = readdir(d)) != NULL) {
        /* Skip . and .. */
        if (ent->d_name[0] == '.') {
            continue;
        }

        char *endp;
        long tid_l = strtol(ent->d_name, &endp, 10);
        if (*endp != '\0' || tid_l <= 0) {
            continue;
        }

        pid_t tid = (pid_t)tid_l;
        printf("  moving tid %d -> %s\n", (int)tid, cgpath);

        if (move_thread_to_cgroup(threads_file, tid) != 0) {
            errors++;
        } else {
            moved++;
        }
    }
    closedir(d);

    if (moved == 0 && errors == 0) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: no threads found for"
                " PID %d.\n", (int)pid);
        return 1;
    }

    printf("  moved %d thread(s) to cgroup '%s' (%d error(s))\n",
           moved, cgname, errors);
    return errors > 0 ? 1 : 0;
}

/**
 * set_rt_priority() - Assign SCHED_FIFO RT priority to all threads of a PID
 * @pid:    process ID
 * @rtprio: SCHED_FIFO priority 1-99 (0 to skip)
 *
 * Uses sched_setscheduler(2) directly on every TID found in
 * /proc/<pid>/task/ — no chrt subprocess.
 *
 * Returns 0 on success, 1 if any thread failed.
 */
static int set_rt_priority(pid_t pid, int rtprio)
{
    if (rtprio <= 0) {
        return 0;
    }

    if (rtprio > 99) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: RT priority %d out"
                " of range (1-99).\n", rtprio);
        return 1;
    }

    struct sched_param sp;
    sp.sched_priority = rtprio;

    /* Enumerate all threads */
    char taskdir[128];
    snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", (int)pid);

    DIR *d = opendir(taskdir);
    if (d == NULL) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: cannot open '%s': %s\n",
                taskdir, strerror(errno));
        return 1;
    }

    int errors = 0;
    int done   = 0;
    struct dirent *ent;

    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') {
            continue;
        }

        char *endp;
        long tid_l = strtol(ent->d_name, &endp, 10);
        if (*endp != '\0' || tid_l <= 0) {
            continue;
        }

        pid_t tid = (pid_t)tid_l;

        if (sched_setscheduler(tid, SCHED_FIFO, &sp) != 0) {
            if (errno == EPERM) {
                fprintf(stderr,
                        "\033[1;31mERROR\033[0m: SCHED_FIFO prio %d"
                        " on tid %d: permission denied"
                        " (needs CAP_SYS_NICE or root).\n",
                        rtprio, (int)tid);
            } else {
                fprintf(stderr,
                        "\033[1;31mERROR\033[0m: sched_setscheduler"
                        " tid %d: %s\n",
                        (int)tid, strerror(errno));
            }
            errors++;
        } else {
            printf("  tid %d SCHED_FIFO prio=%d set\n",
                   (int)tid, rtprio);
            done++;
        }
    }
    closedir(d);

    if (done == 0 && errors == 0) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: no threads found for"
                " PID %d.\n", (int)pid);
        return 1;
    }

    printf("  set SCHED_FIFO prio=%d on %d thread(s) (%d error(s))\n",
           rtprio, done, errors);
    return errors > 0 ? 1 : 0;
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(
        argc, argv, MCSR_DESC, MCSR_DESC_LONG);

    if (action == MH_ACTION_H1 ||
        action == MH_ACTION_H2) {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);

    if (action == MH_ACTION_HELP ||
        action == MH_ACTION_MONO) {
        print_help(argv[0], mh_color);
        return 0;
    }

    if (argc != 4) {
        fprintf(stderr,
                "\n\033[1;31mERROR\033[0m: expected 3 arguments"
                ", got %d.\n\n", argc - 1);
        print_help(argv[0], 1);
        return 1;
    }

    /* Parse PID */
    char *endptr;
    long pid_l = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || pid_l <= 0) {
        fprintf(stderr,
                "\n\033[1;31mERROR\033[0m: invalid PID '%s'"
                " — must be a positive integer.\n\n", argv[1]);
        return 1;
    }
    pid_t pid = (pid_t)pid_l;

    /* Parse RT priority */
    long prio_l = strtol(argv[3], &endptr, 10);
    if (*endptr != '\0' || prio_l < 0 || prio_l > 99) {
        fprintf(stderr,
                "\n\033[1;31mERROR\033[0m: invalid priority '%s'"
                " — must be 0-99.\n\n", argv[3]);
        return 1;
    }
    int rtprio = (int)prio_l;

    const char *cgname = argv[2];

    printf("milk-makecsetandrt: PID=%d cgroup=%s prio=%d\n",
           (int)pid, cgname, rtprio);

    int errors = 0;
    errors += move_to_cgroup(pid, cgname);
    errors += set_rt_priority(pid, rtprio);

    return errors > 0 ? 1 : 0;
}

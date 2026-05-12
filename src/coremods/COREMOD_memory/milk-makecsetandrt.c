/**
 * @file    milk-makecsetandrt.c
 * @brief   Move a PID to a CPU set and assign RT priority
 *
 * Replaces the bash script milk-makecsetandrt.
 * Uses the same underlying tools (cset + chrt) as
 * COREMOD_TOOLS_mvProcCPUsetExt in mvprocCPUset.c.
 *
 * Usage:
 *   milk-makecsetandrt <PID> <cpuset> <prio>
 *
 *   PID    : process ID to move (integer > 0)
 *   cpuset : CPU set name (e.g. "realtime", "NULL" to skip)
 *   prio   : RT priority 1-99 (0 to skip)
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "milk_help.h"

#define MCSR_DESC \
    "move PID to CPU set and assign real-time priority"

#define MCSR_DESC_LONG \
    "Move a process (identified by PID) to a named CPU set\n" \
    "and optionally assign a SCHED_FIFO real-time priority.\n" \
    "\n" \
    "Uses 'cset proc' for CPU set migration and 'chrt' for\n" \
    "RT priority assignment. The cset tool must be installed\n" \
    "(package: cpuset). Elevated privileges (CAP_SYS_NICE\n" \
    "or sudo) are required when raising priority.\n" \
    "\n" \
    "Pass 'NULL' for <cpuset> to skip CPU set migration.\n" \
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
    printf("  %s%s%s %s<PID>%s %s<cpuset>%s %s<prio>%s\n\n",
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
           mh_color ? MH_ARG : "", "<cpuset>",
           mh_color ? MH_RST : "",
           "CPU set name (or 'NULL' to skip)");
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
    printf("  %s$ milk-makecsetandrt%s %s33654 ircam0_edt 80%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-makecsetandrt%s %s12345 NULL 0%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-fps-set" };
    milk_help_see_also(see_also, 1, mh_color);
}

/**
 * move_to_cpuset() - Move a PID to a named CPU set
 * @pid:     process ID to move
 * @csetname: CPU set name (skipped if "NULL")
 *
 * Invokes: cset proc --threads --force -m -p <pid> -t <csetname>
 * Returns 0 on success, 1 on error.
 */
static int move_to_cpuset(pid_t pid, const char *csetname)
{
    if (strcmp(csetname, "NULL") == 0) {
        return 0;
    }

    /* Check if cset is available */
    if (system("which cset > /dev/null 2>&1") != 0) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: 'cset' command not found. "
                "Install the 'cpuset' package.\n");
        return 1;
    }

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "cset proc --threads --force -m -p %d -t %s",
             (int)pid, csetname);
    printf("Executing: %s\n", cmd);

    int ret = system(cmd);
    if (ret != 0) {
        if (ret == 512) {
            fprintf(stderr,
                    "\033[1;31mERROR\033[0m: cset-proc error 512"
                    " — CPU set '%s' does not exist.\n",
                    csetname);
        } else {
            fprintf(stderr,
                    "\033[1;31mERROR\033[0m: cset-proc"
                    " returned %d.\n", ret);
        }
        return 1;
    }
    return 0;
}

/**
 * set_rt_priority() - Assign SCHED_FIFO RT priority to a PID
 * @pid:    process ID
 * @rtprio: priority 1-99 (0 to skip)
 *
 * Invokes: chrt -f -p <prio> <pid>
 * Returns 0 on success, 1 on error.
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

    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "chrt -f -p %d %d",
             rtprio, (int)pid);
    printf("Executing: %s\n", cmd);

    int ret = system(cmd);
    if (ret != 0) {
        if (errno == EPERM) {
            fprintf(stderr,
                    "\033[1;31mERROR\033[0m: Permission denied"
                    " — requires CAP_SYS_NICE or sudo.\n");
        } else {
            fprintf(stderr,
                    "\033[1;31mERROR\033[0m: chrt returned %d.\n",
                    ret);
        }
        return 1;
    }
    return 0;
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

    const char *csetname = argv[2];

    printf("milk-makecsetandrt: PID=%d cpuset=%s prio=%d\n",
           (int)pid, csetname, rtprio);

    int errors = 0;
    errors += move_to_cpuset(pid, csetname);
    errors += set_rt_priority(pid, rtprio);

    return errors > 0 ? 1 : 0;
}

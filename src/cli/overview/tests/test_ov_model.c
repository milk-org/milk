// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_model: milk-CTRL data-layer checks that need no shared memory. */

#include "overview_data_internal.h"

#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

/* A live process owned by another user (kill() -> EPERM) is alive, not dead (links-4) */
static void test_pid_liveness(void)
{
    CHECK(pid_get_status(getpid()) == OV_PID_ALIVE);
    if (geteuid() != 0)
    {
        CHECK(pid_get_status(1) != OV_PID_DEAD);
    }
}

/* /proc/<pid>/stat parsing survives a comm field with spaces (gap-11) */
static void test_stat_with_spaces(void)
{
    uint64_t ut = 0, st = 0;
    pid_t    pid = fork();
    if (pid == 0)
    {
        prctl(PR_SET_NAME, "a b c", 0, 0, 0);
        pause();
        _exit(0);
    }
    usleep(50000);
    CHECK(pid_get_cpu_ticks(pid, &ut, &st) == 0);
    CHECK(pid_check_zombie(pid) == 0);
    kill(pid, SIGKILL);
    int zombie = 0;
    for (int i = 0; i < 100 && !zombie; i++)
    {
        usleep(10000);
        zombie = pid_check_zombie(pid);
    }
    CHECK(zombie == 1);
    waitpid(pid, NULL, 0);
}

/* A PID shared by dead registry entries and a live one resolves to the live, else newest (links-11) */
static void test_find_proc_by_pid(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    m.nb_procs = 3;
    for (int i = 0; i < 3; i++)
    {
        m.procs[i].valid = 1;
        m.procs[i].PID   = 4242;
    }
    m.procs[0].createtime = 10.0;
    m.procs[1].createtime = 30.0;
    m.procs[2].createtime = 20.0;
    CHECK(ov_find_proc_by_pid(&m, 4242) == 1);
    m.procs[2].active = 1;
    CHECK(ov_find_proc_by_pid(&m, 4242) == 2);
    CHECK(ov_find_proc_by_pid(&m, 7) == -1);
}

int main(void)
{
    test_pid_liveness();
    test_stat_with_spaces();
    test_find_proc_by_pid();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}

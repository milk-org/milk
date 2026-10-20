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

int main(void)
{
    test_pid_liveness();
    test_stat_with_spaces();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}

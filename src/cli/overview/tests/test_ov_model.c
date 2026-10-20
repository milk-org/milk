// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_model: milk-CTRL data-layer checks that need no shared memory. */

#include "overview_data_internal.h"

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

int main(void)
{
    test_pid_liveness();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}

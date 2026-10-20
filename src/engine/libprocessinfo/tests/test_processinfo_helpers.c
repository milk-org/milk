// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest processinfo_unit_helpers: shared helpers of libprocessinfo (labels). */

#include <stdio.h>
#include <string.h>

#include "processinfo.h"
#include "processinfo_labels.h"
#include "processtools_trigger.h"

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

/* Every value has a short (at most 4 chars) and a long name; out of range is UNKNOWN */
static void test_labels(void)
{
    for (int s = PROCESSINFO_LOOPSTAT_INIT; s <= PROCESSINFO_LOOPSTAT_CRASHED; s++)
    {
        CHECK(strlen(processinfo_loopstat_name(s, 0)) <= 4);
        CHECK(strcmp(processinfo_loopstat_name(s, 1), "UNKNOWN") != 0);
    }
    CHECK(strcmp(processinfo_loopstat_name(PROCESSINFO_LOOPSTAT_ACTIVE, 0), "RUN") == 0);
    CHECK(strcmp(processinfo_loopstat_name(PROCESSINFO_LOOPSTAT_ERROR, 1), "ERROR") == 0);
    CHECK(strcmp(processinfo_loopstat_name(-1, 1), "UNKNOWN") == 0);
    CHECK(strcmp(processinfo_loopstat_name(7, 0), "??") == 0);

    for (int c = PROCESSINFO_CTRLVAL_RUN; c <= PROCESSINFO_CTRLVAL_EXIT; c++)
    {
        CHECK(strcmp(processinfo_ctrlval_name(c), "UNKNOWN") != 0);
    }
    CHECK(strcmp(processinfo_ctrlval_name(PROCESSINFO_CTRLVAL_INCR), "STEP") == 0);
    CHECK(strcmp(processinfo_ctrlval_name(9), "UNKNOWN") == 0);

    for (int m = PROCESSINFO_TRIGGERMODE_IMMEDIATE; m <= PROCESSINFO_TRIGGERMODE_CNT2; m++)
    {
        CHECK(strlen(processinfo_triggermode_name(m, 0)) == 3);
        CHECK(strcmp(processinfo_triggermode_name(m, 1), "UNKNOWN") != 0);
    }
    CHECK(strcmp(processinfo_triggermode_name(PROCESSINFO_TRIGGERMODE_CNT0, 1), "CNT0") == 0);
    CHECK(strcmp(processinfo_triggermode_name(99, 1), "UNKNOWN") == 0);
}

int main(void)
{
    test_labels();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}

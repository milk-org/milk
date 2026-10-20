// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest processinfo_unit_helpers: shared helpers of libprocessinfo (labels, timing medians, exec_end). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "processinfo.h"
#include "processinfo_labels.h"
#include "processinfo_timing.h"
#include "processtools.h"
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

static PROCESSINFO pi; /* large: static */

/* Fill @n iterations ending at ring slot @last: period @per_ns, execution @exec_ns, the last one under way */
static void fill_ring(int n, int last, int64_t per_ns, int64_t exec_ns)
{
    memset(&pi, 0, sizeof(pi));
    pi.timerindex = last;
    for (int k = 0; k < n; k++)
    {
        int     s = (last - k + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;
        int64_t t =
            1000000000LL * 1000 + (int64_t) (n - 1 - k) * per_ns + (k % 3) * 1000; /* jitter */
        pi.texecstart[s].tv_sec  = t / 1000000000LL;
        pi.texecstart[s].tv_nsec = t % 1000000000LL;
        if (k > 0)
        {
            int64_t e              = t + exec_ns;
            pi.texecend[s].tv_sec  = e / 1000000000LL;
            pi.texecend[s].tv_nsec = e % 1000000000LL;
        }
    }
}

/* Medians on full, partial (not wrapped) and empty rings; across the ring's wrap point */
static void test_timing_median(void)
{
    int64_t it, ex;
    fill_ring(PROCESSINFO_NBtimer, 37, 1000000, 300000);
    CHECK(processinfo_timing_median(&pi, &it, &ex) == PROCESSINFO_NBtimer - 1);
    CHECK(llabs(it - 1000000) <= 2000 && llabs(ex - 300000) <= 2000);

    fill_ring(11, 10, 2000000, 500000); /* 10 completed iterations, rest of the ring unset */
    CHECK(processinfo_timing_median(&pi, &it, &ex) == 10);
    CHECK(llabs(it - 2000000) <= 2000 && llabs(ex - 500000) <= 2000);

    fill_ring(5, 2, 3000000, 1000000); /* wraps through slot 0 */
    CHECK(processinfo_timing_median(&pi, &it, &ex) == 4);
    CHECK(llabs(it - 3000000) <= 2000 && llabs(ex - 1000000) <= 2000);

    memset(&pi, 0, sizeof(pi));
    CHECK(processinfo_timing_median(&pi, &it, &ex) == 0 && it == -1 && ex == -1);
}

/* exec_end measures across a second boundary (the seconds term used texecend twice) */
static void test_exec_end_limit(void)
{
    memset(&pi, 0, sizeof(pi));
    pi.MeasureTiming       = 1;
    pi.dtexec_limit_enable = 1;
    pi.dtexec_limit_value  = 1500000000L; /* 1.5 s */
    clock_gettime(CLOCK_REALTIME, &pi.texecstart[0]);
    pi.texecstart[0].tv_sec -= 2; /* started 2 s ago */
    processinfo_exec_end(&pi);
    CHECK(pi.dtexec_limit_cnt == 1);
}

int main(void)
{
    test_labels();
    test_timing_median();
    test_exec_end_limit();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}

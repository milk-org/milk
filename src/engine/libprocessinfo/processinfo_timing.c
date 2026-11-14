// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_timing.c
 * @brief Median iteration and execution times from the texecstart/texecend ring, computed by any reader
 *        (milk-CTRL, procCTRL-scan) without writing to the process's shared memory.
 */

#include <stdint.h>
#include <time.h>

#include "processinfo.h"
#include "processinfo_timing.h"

/* t1 - t0 in ns; -1 when either is unset or the order is reversed (sample being written) */
static int64_t dt_ns(const struct timespec *t0, const struct timespec *t1)
{
    if ((t0->tv_sec == 0 && t0->tv_nsec == 0) || (t1->tv_sec == 0 && t1->tv_nsec == 0))
    {
        return -1;
    }
    int64_t d =
        (int64_t) (t1->tv_sec - t0->tv_sec) * 1000000000LL + (int64_t) (t1->tv_nsec - t0->tv_nsec);
    return (d >= 0) ? d : -1;
}

/* k-th smallest of @a[0..n-1] (reorders @a); Hoare selection */
static int64_t select_kth(int64_t *a, int n, int k)
{
    int lo = 0, hi = n - 1;
    while (lo < hi)
    {
        int64_t pivot = a[(lo + hi) / 2];
        int     i = lo, j = hi;
        while (i <= j)
        {
            while (a[i] < pivot)
            {
                i++;
            }
            while (a[j] > pivot)
            {
                j--;
            }
            if (i <= j)
            {
                int64_t t = a[i];
                a[i++]    = a[j];
                a[j--]    = t;
            }
        }
        if (k <= j)
        {
            hi = j;
        }
        else if (k >= i)
        {
            lo = i;
        }
        else
        {
            break;
        }
    }
    return a[k];
}

/**
 * processinfo_timing_median - median time between iteration starts (@iter_ns) and of execution (@exec_ns) over
 * the completed iterations in @pinfo's timing ring; samples not yet written are skipped. Either output is -1
 * when it has no sample. Return: number of execution samples used, 0 if none.
 */
int processinfo_timing_median(const PROCESSINFO *pinfo, int64_t *iter_ns, int64_t *exec_ns)
{
    int64_t it[PROCESSINFO_NBtimer], ex[PROCESSINFO_NBtimer];
    int     ni = 0, ne = 0;
    int     last = pinfo->timerindex;
    last         = (last < 0 || last >= PROCESSINFO_NBtimer) ? 0 : last;

    /* iteration last is under way: its start counts for the interval, not for execution */
    for (int k = 0; k < PROCESSINFO_NBtimer - 1; k++)
    {
        int     t1 = (last - k + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;
        int     t0 = (t1 - 1 + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;
        int64_t di = dt_ns(&pinfo->texecstart[t0], &pinfo->texecstart[t1]);
        int64_t de = dt_ns(&pinfo->texecstart[t0], &pinfo->texecend[t0]);
        if (di >= 0)
        {
            it[ni++] = di;
        }
        if (de >= 0 && (di < 0 || de <= di))
        {
            ex[ne++] = de;
        }
    }
    *iter_ns = (ni > 0) ? select_kth(it, ni, ni / 2) : -1;
    *exec_ns = (ne > 0) ? select_kth(ex, ne, ne / 2) : -1;
    return ne;
}

int processinfo_timing_stats(const PROCESSINFO *pinfo,
                             int64_t           *iter_med_ns,
                             int64_t           *iter_max_ns,
                             int64_t           *exec_med_ns,
                             int64_t           *exec_max_ns,
                             int64_t           *jitter_ns)
{
    int64_t it[PROCESSINFO_NBtimer], ex[PROCESSINFO_NBtimer];
    int     ni = 0, ne = 0;
    int64_t imax = -1, emax = -1;
    int     last = pinfo->timerindex;
    last         = (last < 0 || last >= PROCESSINFO_NBtimer) ? 0 : last;

    for (int k = 0; k < PROCESSINFO_NBtimer - 1; k++)
    {
        int     t1 = (last - k + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;
        int     t0 = (t1 - 1 + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;
        int64_t di = dt_ns(&pinfo->texecstart[t0], &pinfo->texecstart[t1]);
        int64_t de = dt_ns(&pinfo->texecstart[t0], &pinfo->texecend[t0]);
        if (di >= 0)
        {
            it[ni++] = di;
            imax     = (di > imax) ? di : imax;
        }
        if (de >= 0 && (di < 0 || de <= di))
        {
            ex[ne++] = de;
            emax     = (de > emax) ? de : emax;
        }
    }
    int64_t imed = (ni > 0) ? select_kth(it, ni, ni / 2) : -1;
    int64_t emed = (ne > 0) ? select_kth(ex, ne, ne / 2) : -1;
    int64_t jit  = -1;
    for (int i = 0; i < ni; i++)
    {
        int64_t d = (it[i] > imed) ? it[i] - imed : imed - it[i];
        jit       = (d > jit) ? d : jit;
    }
    if (iter_med_ns != NULL)
    {
        *iter_med_ns = imed;
    }
    if (iter_max_ns != NULL)
    {
        *iter_max_ns = imax;
    }
    if (exec_med_ns != NULL)
    {
        *exec_med_ns = emed;
    }
    if (exec_max_ns != NULL)
    {
        *exec_max_ns = emax;
    }
    if (jitter_ns != NULL)
    {
        *jitter_ns = jit;
    }
    return ne;
}

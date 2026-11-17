// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_omp_team.c
 * @brief   OpenMP team sizing (milk_omp_team.h); the P-core count comes from milk_syscalls.h
 */

#include "milk_omp_team.h"

#include <stdlib.h>

#include "milk_compiler.h"
#include "milk_syscalls.h"

static int g_omp_force_serial = 0;

void milk_omp_force_serial(void)
{
    __atomic_store_n(&g_omp_force_serial, 1, __ATOMIC_RELEASE);
}

/**
 * milk_omp_team_size - Compute a size-aware OpenMP thread count capped at physical P-cores
 * @nel:        Total number of elements (work items) in the parallel loop
 * @per_thread: Minimum number of elements required per OpenMP thread (0 defaults to 262144)
 *
 * Returns 1 immediately when `nel < 2 * per_thread` to avoid OpenMP fork/join overhead on small
 * and medium arrays, and always after `milk_omp_force_serial()`. For larger arrays, caps the
 * thread count at the number of distinct physical P-cores in the process's CPU affinity mask,
 * lowered by `OMP_NUM_THREADS` or replaced by `MILK_CALC_THREADS` when set.
 *
 * Return: Number of OpenMP threads to use ($\ge 1$)
 */
int milk_omp_team_size(uint64_t nel, uint64_t per_thread)
{
    if (__atomic_load_n(&g_omp_force_serial, __ATOMIC_ACQUIRE))
    {
        return 1;
    }
    if (per_thread == 0)
    {
        per_thread = 262144;
    }
    if (nel < per_thread * 2)
    {
        return 1;
    }

    static int hw_max = 0;
    int        max_t  = __atomic_load_n(&hw_max, __ATOMIC_ACQUIRE);
    if (UNLIKELY(max_t <= 0))
    {
        max_t = milkos_pcore_count();
        __atomic_store_n(&hw_max, max_t, __ATOMIC_RELEASE);
    }

    const char *env = getenv("MILK_CALC_THREADS");
    if (env != NULL && *env != '\0')
    {
        int req = atoi(env);
        if (req > 0)
        {
            max_t = req; /* explicit: may exceed the physical-core count */
        }
    }
    else if ((env = getenv("OMP_NUM_THREADS")) != NULL && *env != '\0')
    {
        int req = atoi(env);
        if (req > 0 && req < max_t)
        {
            max_t = req; /* a user cap lowers the team, never widens it past the cores */
        }
    }

    uint64_t want = nel / per_thread;
    if (want < 1)
    {
        return 1;
    }
    if (want > (uint64_t) max_t)
    {
        return max_t;
    }
    return (int) want;
}

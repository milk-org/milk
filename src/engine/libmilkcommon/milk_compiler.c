// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_compiler.c
 * @brief   Runtime compiler/OpenMP helpers and build tag symbol
 */

#define _GNU_SOURCE
#include "milk_compiler.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

LIB_MILK_EMBED_BUILD_TAG();

/**
 * parse_cpulist - Parse a Linux sysfs CPU list file (e.g. "0-15\n") into @out
 * @path: Sysfs file path
 * @out:  Output CPU affinity set
 *
 * Return: 0 when at least one CPU was parsed, -1 otherwise
 */
static int parse_cpulist(const char *path, cpu_set_t *out)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return -1;
    }
    char buf[256];
    if (fgets(buf, sizeof(buf), fp) == NULL)
    {
        fclose(fp);
        return -1;
    }
    fclose(fp);

    CPU_ZERO(out);
    const char *p = buf;
    while (*p != '\0' && *p != '\n')
    {
        char *end = NULL;
        long  lo  = strtol(p, &end, 10);
        if (end == p || lo < 0 || lo >= CPU_SETSIZE)
        {
            break;
        }
        long hi = lo;
        p       = end;
        if (*p == '-')
        {
            p++;
            hi = strtol(p, &end, 10);
            if (end == p || hi < lo)
            {
                break;
            }
            if (hi >= CPU_SETSIZE)
            {
                hi = CPU_SETSIZE - 1;
            }
            p = end;
        }
        for (long c = lo; c <= hi; c++)
        {
            CPU_SET((int) c, out);
        }
        if (*p == ',')
        {
            p++;
        }
    }
    return (CPU_COUNT(out) > 0) ? 0 : -1;
}

/**
 * read_sysfs_int - Read a single decimal integer from @path, or @defval on failure
 * @path:   Sysfs file path
 * @defval: Fallback integer value
 *
 * Return: Parsed integer or @defval
 */
static int read_sysfs_int(const char *path, int defval)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return defval;
    }
    int val = defval;
    if (fscanf(fp, "%d", &val) != 1)
    {
        val = defval;
    }
    fclose(fp);
    return val;
}

/**
 * detect_pcore_team_size - Detect distinct physical P-cores in current CPU affinity
 *
 * Intersects `sched_getaffinity()` with `/sys/devices/cpu_core/cpus` when present (so hybrid
 * E-cores are not counted) and counts distinct `(physical_package_id, core_id)` pairs. The
 * process affinity mask is only read, never changed.
 *
 * Return: Number of physical P-cores available ($\ge 1$)
 */
static int detect_pcore_team_size(void)
{
    cpu_set_t aff;
    CPU_ZERO(&aff);
    if (sched_getaffinity(0, sizeof(aff), &aff) != 0 || CPU_COUNT(&aff) <= 0)
    {
        return 1;
    }

    cpu_set_t cand = aff;
    cpu_set_t pcores;
    if (parse_cpulist("/sys/devices/cpu_core/cpus", &pcores) == 0)
    {
        cpu_set_t inter;
        CPU_AND(&inter, &aff, &pcores);
        int n_inter = CPU_COUNT(&inter);
        if (n_inter > 0)
        {
            cand = inter;
        }
    }

    struct
    {
        int pkg;
        int core;
    } seen[256];
    int nseen = 0;

    for (int cpu = 0; cpu < CPU_SETSIZE && nseen < 256; cpu++)
    {
        if (!CPU_ISSET(cpu, &cand))
        {
            continue;
        }
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        int core = read_sysfs_int(path, cpu);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
                 cpu);
        int pkg = read_sysfs_int(path, 0);

        int dup = 0;
        for (int i = 0; i < nseen; i++)
        {
            if (seen[i].pkg == pkg && seen[i].core == core)
            {
                dup = 1;
                break;
            }
        }
        if (!dup)
        {
            seen[nseen].pkg  = pkg;
            seen[nseen].core = core;
            nseen++;
        }
    }

    return (nseen > 0) ? nseen : CPU_COUNT(&cand);
}

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
        max_t = detect_pcore_team_size();
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

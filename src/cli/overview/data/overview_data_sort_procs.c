// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include <string.h>
#include <stdlib.h>

extern int    ov_sort_dir_mul;
extern int8_t g_sort_depths[OV_MAX_NODES];

/* ----- Process comparators ----- */

/**
 * sort_proc_by_name - Compare processes alphabetically by name
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_name(const void *a, const void *b)
{
    return ov_sort_dir_mul * strcmp(((const OV_PROC *) a)->name, ((const OV_PROC *) b)->name);
}

/**
 * sort_proc_by_pid - Compare processes by process ID
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_pid(const void *a, const void *b)
{
    pid_t pa = ((const OV_PROC *) a)->PID;
    pid_t pb = ((const OV_PROC *) b)->PID;
    if (pa < pb)
    {
        return -ov_sort_dir_mul;
    }
    if (pa > pb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_stat - Compare processes by loop status
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_stat(const void *a, const void *b)
{
    int sa = ((const OV_PROC *) a)->loopstat;
    int sb = ((const OV_PROC *) b)->loopstat;
    if (sa < sb)
    {
        return -ov_sort_dir_mul;
    }
    if (sa > sb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_hz - Compare processes by loop frequency
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_hz(const void *a, const void *b)
{
    double ha = ((const OV_PROC *) a)->loop_hz;
    double hb = ((const OV_PROC *) b)->loop_hz;
    if (ha < hb)
    {
        return -ov_sort_dir_mul;
    }
    if (ha > hb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_mem - Compare processes by resident memory usage
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_mem(const void *a, const void *b)
{
    int64_t ma = ((const OV_PROC *) a)->mem_rss_kb;
    int64_t mb = ((const OV_PROC *) b)->mem_rss_kb;
    if (ma < mb)
    {
        return -ov_sort_dir_mul;
    }
    if (ma > mb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_ancestry - Compare processes by graph lineage depth
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_ancestry(const void *a, const void *b)
{
    const OV_PROC *pa = (const OV_PROC *) a;
    const OV_PROC *pb = (const OV_PROC *) b;
    int8_t         da =
        (pa->node_idx >= 0 && pa->node_idx < OV_MAX_NODES) ? g_sort_depths[pa->node_idx] : 127;
    int8_t db =
        (pb->node_idx >= 0 && pb->node_idx < OV_MAX_NODES) ? g_sort_depths[pb->node_idx] : 127;

    if (da == 127 && db != 127)
    {
        return 1;
    }
    if (db == 127 && da != 127)
    {
        return -1;
    }

    if (da != db)
    {
        return ov_sort_dir_mul * (da - db);
    }
    return sort_proc_by_name(a, b);
}

/**
 * sort_proc_by_prio - Compare processes by real-time priority
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_prio(const void *a, const void *b)
{
    int pa = ((const OV_PROC *) a)->rt_priority;
    int pb = ((const OV_PROC *) b)->rt_priority;
    if (pa < pb)
    {
        return -ov_sort_dir_mul;
    }
    if (pa > pb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_uptime - Compare processes by start time / uptime
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_uptime(const void *a, const void *b)
{
    int64_t ua = ((const OV_PROC *) a)->start_time_sec;
    int64_t ub = ((const OV_PROC *) b)->start_time_sec;
    if (ua < ub)
    {
        return -ov_sort_dir_mul;
    }
    if (ua > ub)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_cpu - Compare processes by CPU utilization percentage
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_cpu(const void *a, const void *b)
{
    float ca = ((const OV_PROC *) a)->cpu_used;
    float cb = ((const OV_PROC *) b)->cpu_used;
    if (ca < cb)
    {
        return -ov_sort_dir_mul;
    }
    if (ca > cb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_loopcnt - Compare processes by iteration count
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_loopcnt(const void *a, const void *b)
{
    int64_t la = ((const OV_PROC *) a)->loopcnt;
    int64_t lb = ((const OV_PROC *) b)->loopcnt;
    if (la < lb)
    {
        return -ov_sort_dir_mul;
    }
    if (la > lb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_proc_by_duty - Compare processes by execution duty cycle
 * @a: Pointer to first OV_PROC
 * @b: Pointer to second OV_PROC
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_proc_by_duty(const void *a, const void *b)
{
    const OV_PROC *pa = (const OV_PROC *) a;
    const OV_PROC *pb = (const OV_PROC *) b;
    double         da = (pa->dtmedian_iter_ns > 0)
                            ? (double) pa->dtmedian_exec_ns / (double) pa->dtmedian_iter_ns
                            : 0.0;
    double         db = (pb->dtmedian_iter_ns > 0)
                            ? (double) pb->dtmedian_exec_ns / (double) pb->dtmedian_iter_ns
                            : 0.0;
    if (da < db)
    {
        return -ov_sort_dir_mul;
    }
    if (da > db)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/** Number of sortable proc columns. */
#define OV_PROC_SORT_NCOL 10

/**
 * ov_sort_procs - Sort processes array in model according to selected key and direction
 * @model: Pointer to data model
 * @key:   Column sort key index
 * @dir:   Sort direction (0 for asc, 1 for desc)
 */
void ov_sort_procs(OV_MODEL *model, int key, int dir)
{
    if (model->nb_procs < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:
        cmp = sort_proc_by_pid;
        break;
    case 2:
        cmp = sort_proc_by_stat;
        break;
    case 3:
        cmp = sort_proc_by_hz;
        break;
    case 4:
        cmp = sort_proc_by_mem;
        break;
    case 5:
        cmp = sort_proc_by_ancestry;
        break;
    case 6:
        cmp = sort_proc_by_prio;
        break;
    case 7:
        cmp = sort_proc_by_uptime;
        break;
    case 8:
        cmp = sort_proc_by_cpu;
        break;
    case 9:
        cmp = sort_proc_by_loopcnt;
        break;
    case 10:
        cmp = sort_proc_by_duty;
        break;
    default:
        cmp = sort_proc_by_name;
        break;
    }
    qsort(model->procs, (size_t) model->nb_procs, sizeof(OV_PROC), cmp);
}

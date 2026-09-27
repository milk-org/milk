// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include <string.h>
#include <stdlib.h>

extern int    ov_sort_dir_mul;
extern int8_t g_sort_depths[OV_MAX_NODES];

/* ----- FPS comparators ----- */

/**
 * sort_fps_by_name - Compare FPS entries alphabetically by name
 * @a: Pointer to first OV_FPS
 * @b: Pointer to second OV_FPS
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_fps_by_name(const void *a, const void *b)
{
    return ov_sort_dir_mul * strcmp(((const OV_FPS *) a)->name, ((const OV_FPS *) b)->name);
}

/**
 * sort_fps_by_cpid - Compare FPS entries by config process ID
 * @a: Pointer to first OV_FPS
 * @b: Pointer to second OV_FPS
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_fps_by_cpid(const void *a, const void *b)
{
    pid_t pa = ((const OV_FPS *) a)->confpid;
    pid_t pb = ((const OV_FPS *) b)->confpid;
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
 * sort_fps_by_rpid - Compare FPS entries by run process ID
 * @a: Pointer to first OV_FPS
 * @b: Pointer to second OV_FPS
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_fps_by_rpid(const void *a, const void *b)
{
    pid_t pa = ((const OV_FPS *) a)->runpid;
    pid_t pb = ((const OV_FPS *) b)->runpid;
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
 * sort_fps_by_mem - Compare FPS entries by memory usage
 * @a: Pointer to first OV_FPS
 * @b: Pointer to second OV_FPS
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_fps_by_mem(const void *a, const void *b)
{
    int64_t ma = ((const OV_FPS *) a)->mem_rss_kb;
    int64_t mb = ((const OV_FPS *) b)->mem_rss_kb;
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
 * sort_fps_by_ancestry - Compare FPS entries by graph lineage depth
 * @a: Pointer to first OV_FPS
 * @b: Pointer to second OV_FPS
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_fps_by_ancestry(const void *a, const void *b)
{
    const OV_FPS *fa = (const OV_FPS *) a;
    const OV_FPS *fb = (const OV_FPS *) b;
    int8_t        da =
        (fa->node_idx >= 0 && fa->node_idx < OV_MAX_NODES) ? g_sort_depths[fa->node_idx] : 127;
    int8_t db =
        (fb->node_idx >= 0 && fb->node_idx < OV_MAX_NODES) ? g_sort_depths[fb->node_idx] : 127;

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
    return sort_fps_by_name(a, b);
}

/**
 * sort_fps_by_tmux - Compare FPS entries by tmux session status
 * @a: Pointer to first OV_FPS
 * @b: Pointer to second OV_FPS
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_fps_by_tmux(const void *a, const void *b)
{
    int ta = ((const OV_FPS *) a)->tmux_flags;
    int tb = ((const OV_FPS *) b)->tmux_flags;
    if (ta < tb)
    {
        return -ov_sort_dir_mul;
    }
    if (ta > tb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_fps_by_nstreams - Compare FPS entries by number of configured streams
 * @a: Pointer to first OV_FPS
 * @b: Pointer to second OV_FPS
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_fps_by_nstreams(const void *a, const void *b)
{
    int sa = ((const OV_FPS *) a)->nb_stream_params;
    int sb = ((const OV_FPS *) b)->nb_stream_params;
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

/** Number of sortable FPS columns. */
#define OV_FPS_SORT_NCOL 6

/**
 * ov_sort_fps - Sort FPS array in model according to selected key and direction
 * @model: Pointer to data model
 * @key:   Column sort key index
 * @dir:   Sort direction (0 for asc, 1 for desc)
 */
void ov_sort_fps(OV_MODEL *model, int key, int dir)
{
    if (model->nb_fps < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:
        cmp = sort_fps_by_cpid;
        break;
    case 2:
        cmp = sort_fps_by_mem;
        break;
    case 3:
        cmp = sort_fps_by_ancestry;
        break;
    case 4:
        cmp = sort_fps_by_rpid;
        break;
    case 5:
        cmp = sort_fps_by_tmux;
        break;
    case 6:
        cmp = sort_fps_by_nstreams;
        break;
    default:
        cmp = sort_fps_by_name;
        break;
    }
    qsort(model->fps, (size_t) model->nb_fps, sizeof(OV_FPS), cmp);
}

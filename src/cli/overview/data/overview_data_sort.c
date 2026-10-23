// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"

/* =========================================================
 * Sorting helpers
 *
 * Key assignments match visual column order
 * (left-to-right) so that ] cycles naturally.
 *
 * Streams: 0=NAME 1=TYP 2=SIZE 3=Hz
 *          4=INODE 5=COUNT
 * Procs:   0=NAME 1=PID 2=STAT 3=Hz 4=MEM
 *          5=ANCESTRY 6=PRIO 7=UPTIME
 *          8=CPU% 9=LOOPCNT 10=DUTY
 * FPS:     0=NAME 1=CPID 2=MEM 3=ANCESTRY
 *          4=RPID 5=TMX 6=STR
 * ========================================================= */

/**
 * Sort direction multiplier: +1 for ascending,
 * -1 for descending. Set before each qsort call.
 */
int ov_sort_dir_mul = 1;

/**
 * Cache of topological node depths used for ancestry sorting.
 */
int8_t g_sort_depths[OV_MAX_NODES];

/**
 * ov_sort_set_depths - Set node graph depth cache for topological sorting
 * @depths: Array of depth values indexed by node index
 */
void ov_sort_set_depths(const int8_t *depths)
{
    memcpy(g_sort_depths, depths, sizeof(g_sort_depths));
}

/* Index sort context: the UI thread sorts one panel at a time */
static const char *s_base;
static size_t      s_elsz;
static int (*s_cmp)(const void *, const void *);
static int (*s_tie)(const void *, const void *);

/** cmp_index - compare entries *a and *b through s_cmp, then s_tie, then their index (total order). */
static int cmp_index(const void *a, const void *b)
{
    int         ia = *(const int *) a, ib = *(const int *) b;
    const void *ea = s_base + (size_t) ia * s_elsz, *eb = s_base + (size_t) ib * s_elsz;
    int         r = s_cmp(ea, eb);
    if (r == 0)
    {
        r = s_tie(ea, eb);
    }
    return (r != 0) ? r : (ia > ib) - (ia < ib);
}

/**
 * ov_sort_index - fill @order with 0..@n-1 sorted by @cmp over the @elsz-byte entries at @base, ties by
 * @tie then index. The entries are not moved.
 */
void ov_sort_index(const void *base,
                   size_t      elsz,
                   int         n,
                   int (*cmp)(const void *, const void *),
                   int (*tie)(const void *, const void *),
                   int *order)
{
    for (int i = 0; i < n; i++)
    {
        order[i] = i;
    }
    s_base = (const char *) base;
    s_elsz = elsz;
    s_cmp  = cmp;
    s_tie  = tie;
    if (n > 1)
    {
        qsort(order, (size_t) n, sizeof(int), cmp_index);
    }
}

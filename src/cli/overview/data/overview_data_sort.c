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

/* =========================================================
 * Persistent sort ordering caches (Freeze mode)
 * ========================================================= */
static char g_stream_order[OV_MAX_STREAMS][80];
static int  g_nb_stream_order = 0;

static char g_proc_order[OV_MAX_PROCS][80];
static int  g_nb_proc_order = 0;

static char g_fps_order[OV_MAX_FPS][80];
static int  g_nb_fps_order = 0;

/**
 * get_stream_rank - compute display rank for a stream.
 * @name: Stream name.
 *
 * Return: Stored index rank, or 999999 if not found.
 */
static int get_stream_rank(const char *name)
{
    for (int i = 0; i < g_nb_stream_order; i++)
    {
        if (strncmp(g_stream_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

/**
 * get_proc_rank - compute display rank for a process.
 * @name: Process name.
 *
 * Return: Stored index rank, or 999999 if not found.
 */
static int get_proc_rank(const char *name)
{
    for (int i = 0; i < g_nb_proc_order; i++)
    {
        if (strncmp(g_proc_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

/**
 * get_fps_rank - compute display rank for an FPS instance.
 * @name: FPS name.
 *
 * Return: Stored index rank, or 999999 if not found.
 */
static int get_fps_rank(const char *name)
{
    for (int i = 0; i < g_nb_fps_order; i++)
    {
        if (strncmp(g_fps_order[i], name, 80) == 0)
        {
            return i;
        }
    }
    return 999999;
}

typedef struct
{
    int         rank;
    int         orig_idx;
    const char *name;
} ov_sort_rank_entry_t;

/**
 * sort_entry_by_rank - Compare snapshot rank entries by preserved visual rank
 * @a: Pointer to first ov_sort_rank_entry_t
 * @b: Pointer to second ov_sort_rank_entry_t
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_entry_by_rank(const void *a, const void *b)
{
    const ov_sort_rank_entry_t *ea = (const ov_sort_rank_entry_t *) a;
    const ov_sort_rank_entry_t *eb = (const ov_sort_rank_entry_t *) b;
    if (ea->rank != eb->rank)
    {
        return ea->rank - eb->rank;
    }
    return strcmp(ea->name, eb->name);
}

/**
 * ov_sort_freeze_snapshot - record current visual item order for freeze mode.
 * @mm: Current data model snapshot.
 */
void ov_sort_freeze_snapshot(const OV_MODEL *mm)
{
    g_nb_stream_order = mm->nb_streams;
    for (int i = 0; i < mm->nb_streams; i++)
    {
        strncpy(g_stream_order[i], mm->streams[i].name, 79);
        g_stream_order[i][79] = '\0';
    }

    g_nb_proc_order = mm->nb_procs;
    for (int i = 0; i < mm->nb_procs; i++)
    {
        strncpy(g_proc_order[i], mm->procs[i].name, 79);
        g_proc_order[i][79] = '\0';
    }

    g_nb_fps_order = mm->nb_fps;
    for (int i = 0; i < mm->nb_fps; i++)
    {
        strncpy(g_fps_order[i], mm->fps[i].name, 79);
        g_fps_order[i][79] = '\0';
    }
}

/**
 * ov_sort_apply_ranks - sort model items according to captured freeze order.
 * @mm: Data model snapshot to reorder.
 */
void ov_sort_apply_ranks(OV_MODEL *mm)
{
    if (g_nb_stream_order > 0 && mm->nb_streams > 1)
    {
        static ov_sort_rank_entry_t entries[OV_MAX_STREAMS];
        for (int i = 0; i < mm->nb_streams; i++)
        {
            entries[i].orig_idx = i;
            entries[i].name     = mm->streams[i].name;
            entries[i].rank     = get_stream_rank(mm->streams[i].name);
        }
        qsort(entries, (size_t) mm->nb_streams, sizeof(ov_sort_rank_entry_t), sort_entry_by_rank);

        static OV_STREAM temp_streams[OV_MAX_STREAMS];
        memcpy(temp_streams, mm->streams, (size_t) mm->nb_streams * sizeof(OV_STREAM));
        for (int i = 0; i < mm->nb_streams; i++)
        {
            mm->streams[i] = temp_streams[entries[i].orig_idx];
        }
    }

    if (g_nb_proc_order > 0 && mm->nb_procs > 1)
    {
        static ov_sort_rank_entry_t entries[OV_MAX_PROCS];
        for (int i = 0; i < mm->nb_procs; i++)
        {
            entries[i].orig_idx = i;
            entries[i].name     = mm->procs[i].name;
            entries[i].rank     = get_proc_rank(mm->procs[i].name);
        }
        qsort(entries, (size_t) mm->nb_procs, sizeof(ov_sort_rank_entry_t), sort_entry_by_rank);

        static OV_PROC temp_procs[OV_MAX_PROCS];
        memcpy(temp_procs, mm->procs, (size_t) mm->nb_procs * sizeof(OV_PROC));
        for (int i = 0; i < mm->nb_procs; i++)
        {
            mm->procs[i] = temp_procs[entries[i].orig_idx];
        }
    }

    if (g_nb_fps_order > 0 && mm->nb_fps > 1)
    {
        static ov_sort_rank_entry_t entries[OV_MAX_FPS];
        for (int i = 0; i < mm->nb_fps; i++)
        {
            entries[i].orig_idx = i;
            entries[i].name     = mm->fps[i].name;
            entries[i].rank     = get_fps_rank(mm->fps[i].name);
        }
        qsort(entries, (size_t) mm->nb_fps, sizeof(ov_sort_rank_entry_t), sort_entry_by_rank);

        static OV_FPS temp_fps[OV_MAX_FPS];
        memcpy(temp_fps, mm->fps, (size_t) mm->nb_fps * sizeof(OV_FPS));
        for (int i = 0; i < mm->nb_fps; i++)
        {
            mm->fps[i] = temp_fps[entries[i].orig_idx];
        }
    }
}

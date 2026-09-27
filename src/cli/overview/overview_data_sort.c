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
static int ov_sort_dir_mul = 1;

/**
 * Cache of topological node depths used for ancestry sorting.
 */
static int8_t g_sort_depths[OV_MAX_NODES];

/**
 * ov_sort_set_depths - Set node graph depth cache for topological sorting
 * @depths: Array of depth values indexed by node index
 */
void ov_sort_set_depths(const int8_t *depths)
{
    memcpy(g_sort_depths, depths, sizeof(g_sort_depths));
}

/* ----- Stream comparators ----- */

/**
 * sort_stream_by_name - Compare streams alphabetically by name
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_name(const void *a, const void *b)
{
    return ov_sort_dir_mul * strcmp(((const OV_STREAM *) a)->name, ((const OV_STREAM *) b)->name);
}

/**
 * sort_stream_by_type - Compare streams by data type code
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_type(const void *a, const void *b)
{
    int ta = ((const OV_STREAM *) a)->datatype;
    int tb = ((const OV_STREAM *) b)->datatype;
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
 * sort_stream_by_size - Compare streams by total element count
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_size(const void *a, const void *b)
{
    const OV_STREAM *sa = (const OV_STREAM *) a;
    const OV_STREAM *sb = (const OV_STREAM *) b;
    uint64_t         na = sa->nelement;
    uint64_t         nb = sb->nelement;
    if (na < nb)
    {
        return -ov_sort_dir_mul;
    }
    if (na > nb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_stream_by_hz - Compare streams by update frequency
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_hz(const void *a, const void *b)
{
    double ha = ((const OV_STREAM *) a)->update_hz;
    double hb = ((const OV_STREAM *) b)->update_hz;
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
 * dtype_bytes - bytes per element for a datatype.
 */
static int dtype_bytes(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
    case _DATATYPE_INT8:
        return 1;
    case _DATATYPE_UINT16:
    case _DATATYPE_INT16:
        return 2;
    case _DATATYPE_UINT32:
    case _DATATYPE_INT32:
    case _DATATYPE_FLOAT:
        return 4;
    case _DATATYPE_UINT64:
    case _DATATYPE_INT64:
    case _DATATYPE_DOUBLE:
        return 8;
    default:
        return 1;
    }
}

/**
 * sort_stream_by_throughput - Compare streams by calculated data throughput
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_throughput(const void *a, const void *b)
{
    const OV_STREAM *sa = (const OV_STREAM *) a;
    const OV_STREAM *sb = (const OV_STREAM *) b;
    double           ta = sa->update_hz * (double) sa->nelement * dtype_bytes(sa->datatype);
    double           tb = sb->update_hz * (double) sb->nelement * dtype_bytes(sb->datatype);
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
 * sort_stream_by_inode - Compare streams by shared memory file inode
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_inode(const void *a, const void *b)
{
    ino_t ia = ((const OV_STREAM *) a)->inode;
    ino_t ib = ((const OV_STREAM *) b)->inode;
    if (ia < ib)
    {
        return -ov_sort_dir_mul;
    }
    if (ia > ib)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * sort_stream_by_count - Compare streams by write counter cnt0
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_count(const void *a, const void *b)
{
    uint64_t ca = ((const OV_STREAM *) a)->cnt0;
    uint64_t cb = ((const OV_STREAM *) b)->cnt0;
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
 * sort_stream_by_ancestry - Compare streams by graph lineage depth
 * @a: Pointer to first OV_STREAM
 * @b: Pointer to second OV_STREAM
 *
 * Return: Negative, zero, or positive comparison result.
 */
static int sort_stream_by_ancestry(const void *a, const void *b)
{
    const OV_STREAM *sa = (const OV_STREAM *) a;
    const OV_STREAM *sb = (const OV_STREAM *) b;
    int8_t           da =
        (sa->node_idx >= 0 && sa->node_idx < OV_MAX_NODES) ? g_sort_depths[sa->node_idx] : 127;
    int8_t db =
        (sb->node_idx >= 0 && sb->node_idx < OV_MAX_NODES) ? g_sort_depths[sb->node_idx] : 127;

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
    return sort_stream_by_name(a, b);
}

/** Number of sortable stream columns. */
#define OV_STREAM_SORT_NCOL 7

/**
 * ov_sort_streams - Sort streams array in model according to selected key and direction
 * @model: Pointer to data model
 * @key:   Column sort key index
 * @dir:   Sort direction (0 for asc, 1 for desc)
 */
void ov_sort_streams(
    OV_MODEL *model,
    int       key,
    int       dir)
{
    if (model->nb_streams < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:
        cmp = sort_stream_by_type;
        break;
    case 2:
        cmp = sort_stream_by_size;
        break;
    case 3:
        cmp = sort_stream_by_hz;
        break;
    case 4:
        cmp = sort_stream_by_throughput;
        break;
    case 5:
        cmp = sort_stream_by_inode;
        break;
    case 6:
        cmp = sort_stream_by_count;
        break;
    case 7:
        cmp = sort_stream_by_ancestry;
        break;
    default:
        cmp = sort_stream_by_name;
        break;
    }
    qsort(model->streams, (size_t) model->nb_streams, sizeof(OV_STREAM), cmp);
}


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
void ov_sort_procs(
    OV_MODEL *model,
    int       key,
    int       dir)
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
void ov_sort_fps(
    OV_MODEL *model,
    int       key,
    int       dir)
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

#include "overview_data_internal.h"

/* =========================================================
 * Sorting helpers
 *
 * Key assignments match visual column order
 * (left-to-right) so that ] cycles naturally.
 *
 * Streams: 0=NAME 1=TYP 2=SIZE 3=Hz
 *          4=INODE 5=COUNT
 * Procs:   0=NAME 1=PID 2=STAT 3=Hz
 * FPS:     0=NAME 1=C(alive)
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
 * @brief Compute graph depth for topological sorting.
 */
void ov_sort_set_depths(const int8_t *depths)
{
    memcpy(g_sort_depths, depths, sizeof(g_sort_depths));
}

/* ----- Stream comparators ----- */

static int sort_stream_by_name(
    const void *a,
    const void *b)
{
    return ov_sort_dir_mul * strcmp(((const OV_STREAM *) a)->name, ((const OV_STREAM *) b)->name);
}

/**
 * @brief Sort streams by data type.
 */
static int sort_stream_by_type(
    const void *a,
    const void *b)
{
    int ta = ((const OV_STREAM *) a)->datatype;
    int tb = ((const OV_STREAM *) b)->datatype;
    if(ta < tb)
    {
        return -ov_sort_dir_mul;
    }
    if(ta > tb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

/**
 * @brief Sort streams by total element count.
 */
static int sort_stream_by_size(
    const void *a,
    const void *b)
{
    const OV_STREAM *sa = (const OV_STREAM *) a;
    const OV_STREAM *sb = (const OV_STREAM *) b;
    uint64_t na = sa->nelement;
    uint64_t nb = sb->nelement;
    if(na < nb)
    {
        return -ov_sort_dir_mul;
    }
    if(na > nb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_stream_by_hz(
    const void *a,
    const void *b)
{
    double ha = ((const OV_STREAM *) a)->update_hz;
    double hb = ((const OV_STREAM *) b)->update_hz;
    if(ha < hb)
    {
        return -ov_sort_dir_mul;
    }
    if(ha > hb)
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
    switch(dt)
    {
    case _DATATYPE_UINT8: case _DATATYPE_INT8: return 1;
    case _DATATYPE_UINT16: case _DATATYPE_INT16: return 2;
    case _DATATYPE_UINT32: case _DATATYPE_INT32: case _DATATYPE_FLOAT: return 4;
    case _DATATYPE_UINT64: case _DATATYPE_INT64: case _DATATYPE_DOUBLE: return 8;
    default: return 1;
    }
}

static int sort_stream_by_throughput(
    const void *a,
    const void *b)
{
    const OV_STREAM *sa = (const OV_STREAM *) a;
    const OV_STREAM *sb = (const OV_STREAM *) b;
    double ta = sa->update_hz * (double) sa->nelement * dtype_bytes(sa->datatype);
    double tb = sb->update_hz * (double) sb->nelement * dtype_bytes(sb->datatype);
    if(ta < tb)
    {
        return -ov_sort_dir_mul;
    }
    if(ta > tb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_stream_by_inode(
    const void *a,
    const void *b)
{
    ino_t ia = ((const OV_STREAM *) a)->inode;
    ino_t ib = ((const OV_STREAM *) b)->inode;
    if(ia < ib)
    {
        return -ov_sort_dir_mul;
    }
    if(ia > ib)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_stream_by_count(
    const void *a,
    const void *b)
{
    uint64_t ca = ((const OV_STREAM *) a)->cnt0;
    uint64_t cb = ((const OV_STREAM *) b)->cnt0;
    if(ca < cb)
    {
        return -ov_sort_dir_mul;
    }
    if(ca > cb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_stream_by_ancestry(
    const void *a,
    const void *b)
{
    const OV_STREAM *sa = (const OV_STREAM *) a;
    const OV_STREAM *sb = (const OV_STREAM *) b;
    int8_t da = (sa->node_idx >= 0 && sa->node_idx < OV_MAX_NODES) ? g_sort_depths[sa->node_idx] : 127;
    int8_t db = (sb->node_idx >= 0 && sb->node_idx < OV_MAX_NODES) ? g_sort_depths[sb->node_idx] : 127;

    if(da == 127 && db != 127)
    {
        return 1;
    }
    if(db == 127 && da != 127)
    {
        return -1;
    }

    if(da != db)
    {
        return ov_sort_dir_mul * (da - db);
    }
    return sort_stream_by_name(a, b);
}

/** Number of sortable stream columns. */
#define OV_STREAM_SORT_NCOL 7

void ov_sort_streams(
    OV_MODEL *model,
    int      key,
    int      dir)
{
    if(model->nb_streams < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch(key)
    {
    case 1: cmp = sort_stream_by_type;
        break;
    case 2: cmp = sort_stream_by_size;
        break;
    case 3: cmp = sort_stream_by_hz;
        break;
    case 4: cmp = sort_stream_by_throughput;
        break;
    case 5: cmp = sort_stream_by_inode;
        break;
    case 6: cmp = sort_stream_by_count;
        break;
    case 7: cmp = sort_stream_by_ancestry;
        break;
    default: cmp = sort_stream_by_name;
        break;
    }
    qsort(model->streams, (size_t) model->nb_streams, sizeof(OV_STREAM), cmp);
}


/* ----- Process comparators ----- */

static int sort_proc_by_name(
    const void *a,
    const void *b)
{
    return ov_sort_dir_mul * strcmp(((const OV_PROC *) a)->name, ((const OV_PROC *) b)->name);
}

static int sort_proc_by_pid(
    const void *a,
    const void *b)
{
    pid_t pa = ((const OV_PROC *) a)->PID;
    pid_t pb = ((const OV_PROC *) b)->PID;
    if(pa < pb)
    {
        return -ov_sort_dir_mul;
    }
    if(pa > pb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_proc_by_stat(
    const void *a,
    const void *b)
{
    int sa = ((const OV_PROC *) a)->loopstat;
    int sb = ((const OV_PROC *) b)->loopstat;
    if(sa < sb)
    {
        return -ov_sort_dir_mul;
    }
    if(sa > sb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_proc_by_hz(
    const void *a,
    const void *b)
{
    double ha = ((const OV_PROC *) a)->loop_hz;
    double hb = ((const OV_PROC *) b)->loop_hz;
    if(ha < hb)
    {
        return -ov_sort_dir_mul;
    }
    if(ha > hb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_proc_by_mem(
    const void *a,
    const void *b)
{
    int64_t ma = ((const OV_PROC *) a)->mem_rss_kb;
    int64_t mb = ((const OV_PROC *) b)->mem_rss_kb;
    if(ma < mb)
    {
        return -ov_sort_dir_mul;
    }
    if(ma > mb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_proc_by_ancestry(
    const void *a,
    const void *b)
{
    const OV_PROC *pa = (const OV_PROC *) a;
    const OV_PROC *pb = (const OV_PROC *) b;
    int8_t da = (pa->node_idx >= 0 && pa->node_idx < OV_MAX_NODES) ? g_sort_depths[pa->node_idx] : 127;
    int8_t db = (pb->node_idx >= 0 && pb->node_idx < OV_MAX_NODES) ? g_sort_depths[pb->node_idx] : 127;

    if(da == 127 && db != 127)
    {
        return 1;
    }
    if(db == 127 && da != 127)
    {
        return -1;
    }

    if(da != db)
    {
        return ov_sort_dir_mul * (da - db);
    }
    return sort_proc_by_name(a, b);
}

/** Number of sortable proc columns. */
#define OV_PROC_SORT_NCOL 5

void ov_sort_procs(
    OV_MODEL *model,
    int      key,
    int      dir)
{
    if(model->nb_procs < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch(key)
    {
    case 1: cmp = sort_proc_by_pid;
        break;
    case 2: cmp = sort_proc_by_stat;
        break;
    case 3: cmp = sort_proc_by_hz;
        break;
    case 4: cmp = sort_proc_by_mem;
        break;
    case 5: cmp = sort_proc_by_ancestry;
        break;
    default: cmp = sort_proc_by_name;
        break;
    }
    qsort(model->procs, (size_t) model->nb_procs, sizeof(OV_PROC), cmp);
}


/* ----- FPS comparators ----- */

static int sort_fps_by_name(
    const void *a,
    const void *b)
{
    return ov_sort_dir_mul * strcmp(((const OV_FPS *) a)->name, ((const OV_FPS *) b)->name);
}

static int sort_fps_by_cpid(
    const void *a,
    const void *b)
{
    pid_t pa = ((const OV_FPS *) a)->confpid;
    pid_t pb = ((const OV_FPS *) b)->confpid;
    if(pa < pb)
    {
        return -ov_sort_dir_mul;
    }
    if(pa > pb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_fps_by_rpid(
    const void *a,
    const void *b)
{
    pid_t pa = ((const OV_FPS *) a)->runpid;
    pid_t pb = ((const OV_FPS *) b)->runpid;
    if(pa < pb)
    {
        return -ov_sort_dir_mul;
    }
    if(pa > pb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_fps_by_mem(
    const void *a,
    const void *b)
{
    int64_t ma = ((const OV_FPS *) a)->mem_rss_kb;
    int64_t mb = ((const OV_FPS *) b)->mem_rss_kb;
    if(ma < mb)
    {
        return -ov_sort_dir_mul;
    }
    if(ma > mb)
    {
        return ov_sort_dir_mul;
    }
    return 0;
}

static int sort_fps_by_ancestry(
    const void *a,
    const void *b)
{
    const OV_FPS *fa = (const OV_FPS *) a;
    const OV_FPS *fb = (const OV_FPS *) b;
    int8_t da = (fa->node_idx >= 0 && fa->node_idx < OV_MAX_NODES) ? g_sort_depths[fa->node_idx] : 127;
    int8_t db = (fb->node_idx >= 0 && fb->node_idx < OV_MAX_NODES) ? g_sort_depths[fb->node_idx] : 127;

    if(da == 127 && db != 127)
    {
        return 1;
    }
    if(db == 127 && da != 127)
    {
        return -1;
    }

    if(da != db)
    {
        return ov_sort_dir_mul * (da - db);
    }
    return sort_fps_by_name(a, b);
}

/** Number of sortable FPS columns. */
#define OV_FPS_SORT_NCOL 4

void ov_sort_fps(
    OV_MODEL *model,
    int      key,
    int      dir)
{
    if(model->nb_fps < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch(key)
    {
    case 1: cmp = sort_fps_by_cpid;
        break;
    case 2: cmp = sort_fps_by_mem;
        break;
    case 3: cmp = sort_fps_by_ancestry;
        break;
    case 4: cmp = sort_fps_by_rpid;
        break;
    default: cmp = sort_fps_by_name;
        break;
    }
    qsort(model->fps, (size_t) model->nb_fps, sizeof(OV_FPS), cmp);
}

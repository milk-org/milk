// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include <string.h>
#include <stdlib.h>

extern int    ov_sort_dir_mul;
extern int8_t g_sort_depths[OV_MAX_NODES];

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
void ov_sort_streams(OV_MODEL *model, int key, int dir)
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

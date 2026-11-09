// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_stream_props.h
 * @brief The single list of stream and local-image property names (`s.<name>.<prop>`,
 *        `l.<name>.<prop>`)
 *
 * Purpose:     the listing, completion and help show, from one table, the names that milkprop
 *              serves.
 * Does:        names each property, with the alias also accepted and a one-line summary.
 * Used by:     milk_shm_prop_stream.c (`s.<name>.list`), the server's milk_server_complete.c
 *              (completion of `s.<name>.`), CLIcore_help_topics.c (`@help streams`).
 * See also:    docs/guide/scripting.md, "Reading" (the list for users).
 * Constraints: the table only names the properties: each one needs its code in
 *              milk_shm_prop_get_stream() (`json` is built in milk_shm_prop.c).
 */

#ifndef MILK_STREAM_PROPS_H
#define MILK_STREAM_PROPS_H

#include <stddef.h>

/**
 * struct milk_stream_prop_t - One stream property
 * @name:    Canonical name (what listings, completion and help show)
 * @alias:   Older or shorter name that is also accepted, or NULL
 * @summary: One-line description
 */
typedef struct
{
    const char *name;
    const char *alias;
    const char *summary;
} milk_stream_prop_t;

static const milk_stream_prop_t milk_stream_props[] = {
    { "xsize", "size0", "size along x" },
    { "ysize", "size1", "size along y (1 for a 1-D image)" },
    { "zsize", "size2", "size along z (1 below 3-D)" },
    { "naxis", NULL, "number of axes" },
    { "nelement", "nelem", "number of pixels" },
    { "dtype", NULL, "datatype: FLOAT, DOUBLE, UINT16, ..." },
    { "typename", NULL, "datatype: FLT32, FLT64, UI16, ..." },
    { "type", "typeid", "datatype code" },
    { "cnt0", NULL, "frame counter" },
    { "cnt1", NULL, "last slice written, in a rolling buffer" },
    { "sem", NULL, "number of semaphores" },
    { "pid", NULL, "process that created the stream" },
    { "owner_pid", "ownerPID", "process that owns the stream" },
    { "shared", NULL, "1 for a shared-memory stream, 0 for a local image" },
    { "NBkw", NULL, "number of keyword slots" },
    { "location", NULL, "-1 in CPU memory, else the GPU index" },
    { "loc", "storage", "SHM, RAM or GPU<n>" },
    { "inode", NULL, "inode of the stream file" },
    { "age", NULL, "seconds since the last write" },
    { "fps", NULL, "average frame rate since creation" },
    { "summary", NULL, "name, size, datatype and cnt0 on one line" },
    { "min", NULL, "smallest pixel, NaN skipped (as imin)" },
    { "max", NULL, "largest pixel, NaN skipped (as imax)" },
    { "mean", NULL, "mean pixel (as imean)" },
    { "std", NULL, "standard deviation (as istd)" },
    { "var", NULL, "variance (as ivar)" },
    { "tot", NULL, "sum of the pixels (as itot)" },
    { "count", NULL, "number of non-zero pixels (as icount)" },
    { "stats", NULL, "all the statistics, plus torn (1 when every read overlapped a write)" },
    { "kw.*", NULL, "keyword names; kw.<KEY> (or k.<KEY>) reads or @set writes one" },
    { "kv", NULL, "the metadata above (no statistics) as name<TAB>value lines" },
    { "json", NULL, "metadata and keywords as JSON" },
    { "list", "*", "these names" },
    { NULL, NULL, NULL }
};

#endif /* MILK_STREAM_PROPS_H */

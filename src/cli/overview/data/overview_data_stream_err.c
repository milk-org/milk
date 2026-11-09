// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_stream_err.c
 * @brief   Directory entries of the stream scan: symbolic links resolved (libmilkdata), failures
 *          explained.
 *
 * Purpose:     every *.im.shm file of the stream directory is listed: one that cannot be used
 *              shows the reason, and a symbolic link is followed to its stream.
 * Does:        ov_stream_dirent() gives the inode and link target of a directory entry (a regular
 *              file costs no syscall; other entries go through milk_stream_classify()) and reports
 *              a dangling link; the target's inode keys the cache, so a recreated target is
 *              noticed. ov_stream_map_err() turns a failed mapping into an ov_stream_err_t;
 *              ov_stream_err_label() names it in a few words ("cannot open", "not a stream
 *              file", "ImageStreamIO version", "damaged or truncated", "dangling link").
 * Used by:     overview_data_scan_streams.c (ov_stream_dirent, ov_stream_map_err; declared in
 *              overview_data_internal.h); the stream list, stream DETAILS and the F3 inspector
 *              (ov_stream_err_label: render/overview_columns_streams.c,
 *              overview_render_detail_stream.c, overview_render_insp.c; declared in
 *              overview_data.h). milk-fps-info also compiles this file.
 * Uses:        libmilkdata (milkdata_streamlist.h: milk_stream_classify).
 * See also:    docs/guide/milk-CTRL.md, "Streams view (F3)".
 */

#include "overview_data_internal.h"
#include "milkdata_streamlist.h"

#include <dirent.h>
#include <sys/stat.h>

/** ov_stream_err_label - why a listed stream cannot be used, in a few words ("" when it can). */
const char *ov_stream_err_label(int err)
{
    static const char *const l[] = { "",
                                     "cannot open",
                                     "not a stream file",
                                     "ImageStreamIO version",
                                     "damaged or truncated",
                                     "dangling link" };
    return (err > 0 && err <= OV_SERR_DANGLING) ? l[err] : "";
}

/**
 * ov_stream_dirent - inode and link of directory entry @ep in @dir: a link is followed to its target (whose inode
 * keys the cache, so a recreated target is noticed). Return: 0, OV_SERR_DANGLING, or -1 to skip the entry.
 */
int ov_stream_dirent(const char          *dir,
                     const struct dirent *ep,
                     ino_t               *ino,
                     int                 *is_link,
                     char                *target,
                     size_t               n)
{
    *is_link  = 0;
    target[0] = '\0';
    *ino      = ep->d_ino;
    if (ep->d_type == DT_REG && *ino != 0)
    {
        return 0; /* the common case: no syscall */
    }
    MILK_STREAM_ENT e;
    if (milk_stream_classify(dir, ep->d_name, &e) != 0)
    {
        return -1;
    }
    *is_link = e.is_link;
    snprintf(target, n, "%s", e.target);
    *ino = e.broken ? ep->d_ino : e.inode;
    return (e.is_link && e.broken) ? OV_SERR_DANGLING : 0;
}

/** ov_stream_map_err - ov_stream_err_t for a failed mapping: @rc from ov_shm_map, else @version_ok. */
int ov_stream_map_err(int rc, int version_ok)
{
    return (rc == -1)                ? OV_SERR_OPEN
           : (rc == -2)              ? OV_SERR_FILE
           : (rc != 0 || version_ok) ? OV_SERR_DAMAGED
                                     : OV_SERR_VERSION;
}

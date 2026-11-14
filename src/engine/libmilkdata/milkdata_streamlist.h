// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkdata_streamlist.h
 * @brief List the image streams (*.im.shm) of a shared-memory directory, symbolic links resolved
 */

#ifndef MILKDATA_STREAMLIST_H
#define MILKDATA_STREAMLIST_H

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define MILK_STREAM_NAME_MAX 256

/* One *.im.shm entry of a directory */
typedef struct
{
    char name[MILK_STREAM_NAME_MAX]; /* stream name: file name without .im.shm */
    char target
        [MILK_STREAM_NAME_MAX]; /* link: target stream name (basename, no .im.shm); "" otherwise */
    ino_t inode;                /* inode of the file, of the target for a link; 0 when broken */
    int   is_link;
    int   broken; /* dangling or unreadable link, or not a regular file */
} MILK_STREAM_ENT;

const char *milk_stream_dir(char *buf, size_t n);
int         milk_stream_classify(const char *dir, const char *fname, MILK_STREAM_ENT *e);
int         milk_stream_list(const char *dir, const char *filter, MILK_STREAM_ENT *out, int cap);

#ifdef __cplusplus
}
#endif

#endif /* MILKDATA_STREAMLIST_H */

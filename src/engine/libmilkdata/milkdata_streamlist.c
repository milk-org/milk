// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkdata_streamlist.c
 * @brief List the image streams of a shared-memory directory: one readdir, lstat + stat + readlink per
 *        symbolic link. Nothing is mapped.
 */

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milkdata_streamlist.h"

#define SHM_SUFFIX ".im.shm"
#define SHM_SUFFIX_LEN 7

/** milk_stream_dir - the stream directory as ImageStreamIO picks it: $MILK_SHM_DIR, /milk/shm, else /tmp. */
const char *milk_stream_dir(char *buf, size_t n)
{
    const char *env     = getenv("MILK_SHM_DIR");
    const char *cand[3] = { env, "/milk/shm", "/tmp" };
    struct stat st;
    for (int i = 0; i < 3; i++)
    {
        if (cand[i] != NULL && cand[i][0] != '\0' && stat(cand[i], &st) == 0 && S_ISDIR(st.st_mode))
        {
            snprintf(buf, n, "%s", cand[i]);
            return buf;
        }
    }
    snprintf(buf, n, "/tmp");
    return buf;
}

/* Copy @fname without its .im.shm suffix into @out; -1 if not a stream file name or too long */
static int stream_name(const char *fname, char *out)
{
    size_t len = strlen(fname);
    if (len <= SHM_SUFFIX_LEN || strcmp(fname + len - SHM_SUFFIX_LEN, SHM_SUFFIX) != 0 ||
        len - SHM_SUFFIX_LEN >= MILK_STREAM_NAME_MAX)
    {
        return -1;
    }
    memcpy(out, fname, len - SHM_SUFFIX_LEN);
    out[len - SHM_SUFFIX_LEN] = '\0';
    return 0;
}

/**
 * milk_stream_classify - fill @e for file @fname of @dir: stream name, link target, inode (the target's for a
 * link), broken flag. Return: 0, or -1 when @fname is not a stream file name or cannot be examined.
 */
int milk_stream_classify(const char *dir, const char *fname, MILK_STREAM_ENT *e)
{
    char        path[4096];
    struct stat st;
    memset(e, 0, sizeof(*e));
    if (stream_name(fname, e->name) != 0 ||
        snprintf(path, sizeof(path), "%s/%s", dir, fname) >= (int) sizeof(path) ||
        lstat(path, &st) != 0)
    {
        return -1;
    }
    e->is_link = S_ISLNK(st.st_mode);
    if (e->is_link)
    {
        char    lnk[4096];
        ssize_t n = readlink(path, lnk, sizeof(lnk) - 1);
        if (n > 0)
        {
            lnk[n]         = '\0';
            const char *bn = strrchr(lnk, '/');
            bn             = (bn != NULL) ? bn + 1 : lnk;
            if (stream_name(bn, e->target) != 0)
            {
                snprintf(e->target, sizeof(e->target), "%.*s", MILK_STREAM_NAME_MAX - 1, bn);
            }
        }
        if (stat(path, &st) != 0) /* follows the link */
        {
            e->broken = 1;
            return 0;
        }
    }
    e->broken = !S_ISREG(st.st_mode);
    e->inode  = e->broken ? 0 : st.st_ino;
    return 0;
}

/**
 * milk_stream_list - the streams of @dir (NULL: milk_stream_dir) whose name contains @filter (NULL or "": all),
 * at most @cap of them into @out, broken links included (flagged). Return: number of matching entries, which
 * may exceed @cap; -1 if @dir cannot be read.
 */
int milk_stream_list(const char *dir, const char *filter, MILK_STREAM_ENT *out, int cap)
{
    char           dbuf[1024];
    struct dirent *de;
    int            n = 0;
    dir              = (dir != NULL) ? dir : milk_stream_dir(dbuf, sizeof(dbuf));
    DIR *d           = opendir(dir);
    if (d == NULL)
    {
        return -1;
    }
    while ((de = readdir(d)) != NULL)
    {
        MILK_STREAM_ENT e;
        if (milk_stream_classify(dir, de->d_name, &e) != 0 ||
            (filter != NULL && filter[0] != '\0' && strstr(e.name, filter) == NULL))
        {
            continue;
        }
        if (n < cap)
        {
            out[n] = e;
        }
        n++;
    }
    closedir(d);
    return n;
}

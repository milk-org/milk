// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkdata_unit_streamlist: regular, linked, dangling and long (240-char) names, filter and cap. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milkdata_streamlist.h"

static int failures = 0;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

static char dir[] = "/tmp/milk-slist-XXXXXX";

static void touch(const char *name)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    CHECK(f != NULL);
    if (f != NULL)
    {
        fclose(f);
    }
}

static const MILK_STREAM_ENT *find(const MILK_STREAM_ENT *e, int n, const char *name)
{
    for (int i = 0; i < n; i++)
    {
        if (strcmp(e[i].name, name) == 0)
        {
            return &e[i];
        }
    }
    return NULL;
}

int main(void)
{
    char            path[512], lpath[512], longname[400];
    MILK_STREAM_ENT e[16];
    CHECK(mkdtemp(dir) != NULL);
    touch("wfs.im.shm");
    touch("dm.im.shm");
    touch("notes.txt");
    touch(".im.shm");
    memset(longname, 'x', 240);
    strcpy(longname + 240, ".im.shm");
    touch(longname);
    snprintf(path, sizeof(path), "%s/wfs.im.shm", dir);
    snprintf(lpath, sizeof(lpath), "%s/alias.im.shm", dir);
    CHECK(symlink(path, lpath) == 0);
    snprintf(lpath, sizeof(lpath), "%s/gone.im.shm", dir);
    CHECK(symlink("/nonexistent/x.im.shm", lpath) == 0);

    int n = milk_stream_list(dir, NULL, e, 16);
    CHECK(n == 5); /* wfs, dm, alias, gone and the long one: not notes.txt or the bare suffix */
    longname[240] = '\0';
    CHECK(find(e, n, longname) != NULL);
    longname[240]            = '.';
    const MILK_STREAM_ENT *w = find(e, n, "wfs"), *a = find(e, n, "alias"), *g = find(e, n, "gone");
    CHECK(w != NULL && !w->is_link && !w->broken && w->inode != 0 && w->target[0] == '\0');
    CHECK(a != NULL && a->is_link && !a->broken && strcmp(a->target, "wfs") == 0 &&
          a->inode == w->inode);
    CHECK(g != NULL && g->is_link && g->broken && g->inode == 0);

    CHECK(milk_stream_list(dir, "fs", e, 16) == 1 && strcmp(e[0].name, "wfs") == 0);
    CHECK(milk_stream_list(dir, "", e, 2) == 5); /* count beyond the cap */
    CHECK(milk_stream_list("/nonexistent-dir", NULL, e, 16) == -1);

    setenv("MILK_SHM_DIR", dir, 1);
    CHECK(milk_stream_list(NULL, "dm", e, 16) == 1);
    char buf[256];
    setenv("MILK_SHM_DIR", "/nonexistent-dir", 1);
    CHECK(strcmp(milk_stream_dir(buf, sizeof(buf)), "/nonexistent-dir") != 0);

    const char *rm[] = { "wfs.im.shm", "dm.im.shm",    "notes.txt",  ".im.shm",
                         longname,     "alias.im.shm", "gone.im.shm" };
    for (size_t i = 0; i < sizeof(rm) / sizeof(rm[0]); i++)
    {
        snprintf(path, sizeof(path), "%s/%s", dir, rm[i]);
        CHECK(unlink(path) == 0);
    }
    CHECK(rmdir(dir) == 0);
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}

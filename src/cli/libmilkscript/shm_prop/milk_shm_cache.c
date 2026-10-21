// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_cache.c
 * @brief Cache of mapped shared-memory streams and FPS (see milk_shm_cache.h).
 */

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_shmdirname.h"
#include "milk_shm_cache.h"
#include "milk_shm_prop_internal.h"

#define SHM_CACHE_NSTREAM 32
#define SHM_CACHE_NFPS 16

typedef struct
{
    int           used;
    char          path[STRINGMAXLEN_DIRNAME + STRINGMAXLEN_IMAGE_NAME + 16]; /* backing file */
    ino_t         ino;                                                       /* inode we mapped */
    unsigned long tick;                                                      /* last use (LRU) */
} ent_hdr_t;

typedef struct
{
    ent_hdr_t h;
    IMAGE     im;
} stream_ent_t;

typedef struct
{
    ent_hdr_t h;
    FPS       fps;
} fps_ent_t;

static stream_ent_t  g_stream[SHM_CACHE_NSTREAM];
static fps_ent_t     g_fps[SHM_CACHE_NFPS];
static unsigned long g_tick = 0;

/** pick_slot() - index of a free slot, else of the least recently used one (@n entries @stride apart). */
static int pick_slot(const ent_hdr_t *first, int n, size_t stride)
{
    int           best  = 0;
    unsigned long btick = ~0UL;
    for (int i = 0; i < n; i++)
    {
        const ent_hdr_t *e = (const ent_hdr_t *) ((const char *) first + (size_t) i * stride);
        if (!e->used)
        {
            return i;
        }
        if (e->tick < btick)
        {
            btick = e->tick;
            best  = i;
        }
    }
    return best;
}

IMAGE *milk_shm_cache_stream(const char *name)
{
    char path[sizeof(g_stream[0].h.path)];
    snprintf(path, sizeof(path), "%s/%s.im.shm", milk_shm_prop_shmdir(), name);
    struct stat st;
    int         exists = (stat(path, &st) == 0);

    for (int i = 0; i < SHM_CACHE_NSTREAM; i++)
    {
        stream_ent_t *e = &g_stream[i];
        if (e->h.used && strcmp(e->h.path, path) == 0)
        {
            if (exists && e->h.ino == st.st_ino)
            {
                e->h.tick = ++g_tick;
                return &e->im;
            }
            ImageStreamIO_closeIm(&e->im); /* re-created or removed */
            e->h.used = 0;
            break;
        }
    }
    if (!exists)
    {
        return NULL;
    }

    stream_ent_t *e = &g_stream[pick_slot(&g_stream[0].h, SHM_CACHE_NSTREAM, sizeof(g_stream[0]))];
    if (e->h.used)
    {
        ImageStreamIO_closeIm(&e->im);
        e->h.used = 0;
    }
    memset(&e->im, 0, sizeof(e->im));
    if (ImageStreamIO_openIm(&e->im, name) != IMAGESTREAMIO_SUCCESS || e->im.md == NULL)
    {
        return NULL;
    }
    snprintf(e->h.path, sizeof(e->h.path), "%s", path);
    e->h.ino  = e->im.md->inode; /* the file we mapped, even if it was replaced since stat() */
    e->h.tick = ++g_tick;
    e->h.used = 1;
    return &e->im;
}

FPS *milk_shm_cache_fps(const char *name)
{
    char shmdname[STRINGMAXLEN_DIRNAME];
    function_parameter_struct_shmdirname(shmdname);
    char path[sizeof(g_fps[0].h.path)];
    snprintf(path, sizeof(path), "%s/%s.fps.shm", shmdname, name);
    struct stat st;
    int         exists = (stat(path, &st) == 0);

    for (int i = 0; i < SHM_CACHE_NFPS; i++)
    {
        fps_ent_t *e = &g_fps[i];
        if (e->h.used && strcmp(e->h.path, path) == 0)
        {
            if (exists && e->h.ino == st.st_ino)
            {
                e->h.tick = ++g_tick;
                return &e->fps;
            }
            fps_disconnect(&e->fps);
            e->h.used = 0;
            break;
        }
    }
    if (!exists)
    {
        return NULL;
    }

    fps_ent_t *e = &g_fps[pick_slot(&g_fps[0].h, SHM_CACHE_NFPS, sizeof(g_fps[0]))];
    if (e->h.used)
    {
        fps_disconnect(&e->fps);
        e->h.used = 0;
    }
    memset(&e->fps, 0, sizeof(e->fps));
    if (fps_connect(name, &e->fps, FPSCONNECT_SIMPLE) == -1 || e->fps.md == NULL ||
        e->fps.parray == NULL)
    {
        return NULL;
    }
    struct stat fst;
    snprintf(e->h.path, sizeof(e->h.path), "%s", path);
    e->h.ino  = (fstat(e->fps.SMfd, &fst) == 0) ? fst.st_ino : st.st_ino;
    e->h.tick = ++g_tick;
    e->h.used = 1;
    return &e->fps;
}

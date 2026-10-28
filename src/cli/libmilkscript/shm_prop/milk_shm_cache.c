// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_cache.c
 * @brief Cache of mapped shared-memory streams and FPS (see milk_shm_cache.h).
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_shmdirname.h"
#include "milk_shm_cache.h"
#include "milk_shm_prop_internal.h"

#define SHM_CACHE_NSTREAM 64
#define SHM_CACHE_NFPS 16

typedef struct
{
    int           used;
    int           pinned;                                                    /* active pin count */
    char          path[STRINGMAXLEN_DIRNAME + STRINGMAXLEN_IMAGE_NAME + 16]; /* backing file */
    ino_t         ino;                                                       /* inode we mapped */
    unsigned long tick;                                                      /* last use (LRU) */
} ent_hdr_t;

typedef struct
{
    ent_hdr_t h;
    int       semindex; /* cached semaphore slot (-1 if none) */
    pid_t     sem_pid;  /* PID that claimed semindex */
    IMAGE     im;
} stream_ent_t;

typedef struct
{
    ent_hdr_t h;
    FPS       fps;
} fps_ent_t;

static pthread_mutex_t g_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static stream_ent_t    g_stream[SHM_CACHE_NSTREAM];
static fps_ent_t       g_fps[SHM_CACHE_NFPS];
static unsigned long   g_tick              = 0;
static int             g_atexit_registered = 0;
static __thread int    g_cache_full        = 0; /* the last lookup found every stream slot pinned */

int milk_shm_cache_full(void)
{
    return g_cache_full;
}

int milk_shm_sem_claim(IMAGE *im)
{
    if (im == NULL || im->md == NULL || im->semReadPID == NULL)
    {
        return -1;
    }
    pid_t me = getpid();
    for (int i = 0; i < im->md->sem; i++)
    {
        pid_t cur = __atomic_load_n(&im->semReadPID[i], __ATOMIC_ACQUIRE);
        if (cur == me || (cur != 0 && getpgid(cur) >= 0)) /* ours already, or a live reader's */
        {
            continue;
        }
        if (__atomic_compare_exchange_n(&im->semReadPID[i], &cur, me, 0, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE))
        {
            return i;
        }
    }
    return -1;
}

void milk_shm_sem_release(IMAGE *im, int slot)
{
    if (im == NULL || im->md == NULL || im->semReadPID == NULL || slot < 0 || slot >= im->md->sem)
    {
        return;
    }
    pid_t me = getpid();
    __atomic_compare_exchange_n(&im->semReadPID[slot], &me, 0, 0, __ATOMIC_ACQ_REL,
                                __ATOMIC_ACQUIRE);
}

/** release_stream_sem() - release the cached semaphore slot on @e if owned by this process. */
static void release_stream_sem(stream_ent_t *e)
{
    pid_t me = getpid();
    if (e->semindex >= 0 && e->sem_pid == me && e->im.md != NULL && e->semindex < e->im.md->sem &&
        e->im.semReadPID != NULL && e->im.semReadPID[e->semindex] == me)
    {
        e->im.semReadPID[e->semindex] = 0;
    }
    e->semindex = -1;
    e->sem_pid  = 0;
}

/** cache_fork_prepare() / cache_fork_parent() / cache_fork_child() - keep g_cache_lock usable across a
 * fork(): a server child (`--async`, `@replay`) must not inherit it locked by a thread it does not have. */
static void cache_fork_prepare(void)
{
    pthread_mutex_lock(&g_cache_lock);
}

static void cache_fork_parent(void)
{
    pthread_mutex_unlock(&g_cache_lock);
}

static void cache_fork_child(void)
{
    pthread_mutex_init(&g_cache_lock, NULL);
}

/** shm_cache_atexit() - release any semaphore slots held by this process on exit. */
static void shm_cache_atexit(void)
{
    pthread_mutex_lock(&g_cache_lock);
    for (int i = 0; i < SHM_CACHE_NSTREAM; i++)
    {
        if (g_stream[i].h.used)
        {
            release_stream_sem(&g_stream[i]);
        }
    }
    pthread_mutex_unlock(&g_cache_lock);
}

/** pick_slot() - index of a free slot, else of the least recently used unpinned one (-1 if all pinned). */
static int pick_slot(const ent_hdr_t *first, int n, size_t stride)
{
    int           best  = -1;
    unsigned long btick = ~0UL;
    for (int i = 0; i < n; i++)
    {
        const ent_hdr_t *e = (const ent_hdr_t *) ((const char *) first + (size_t) i * stride);
        if (!e->used)
        {
            return i;
        }
        if (e->pinned == 0 && e->tick < btick)
        {
            btick = e->tick;
            best  = i;
        }
    }
    return best;
}

/** lookup_or_open_stream_locked() - find or map stream @name under g_cache_lock. */
static stream_ent_t *lookup_or_open_stream_locked(const char *name)
{
    g_cache_full = 0;
    if (name == NULL || name[0] == '\0')
    {
        return NULL;
    }
    if (!g_atexit_registered)
    {
        atexit(shm_cache_atexit);
        pthread_atfork(cache_fork_prepare, cache_fork_parent, cache_fork_child);
        g_atexit_registered = 1;
    }

    char path[sizeof(g_stream[0].h.path)];
    snprintf(path, sizeof(path), "%s/%s.im.shm", milk_shm_prop_shmdir(), name);
    struct stat st;
    int         exists = (stat(path, &st) == 0 && st.st_size >= (off_t) sizeof(IMAGE_METADATA));

    for (int i = 0; i < SHM_CACHE_NSTREAM; i++)
    {
        stream_ent_t *e = &g_stream[i];
        if (e->h.used && strcmp(e->h.path, path) == 0)
        {
            if (exists && e->h.ino == st.st_ino)
            {
                e->h.tick = ++g_tick;
                return e;
            }
            if (e->h.pinned > 0)
            {
                /* Detach stale entry so it closes when the last pin drops */
                e->h.path[0] = '\0';
                continue;
            }
            release_stream_sem(e);
            ImageStreamIO_closeIm(&e->im); /* re-created or removed */
            e->h.used = 0;
            break;
        }
    }
    if (!exists)
    {
        return NULL;
    }

    int slot = pick_slot(&g_stream[0].h, SHM_CACHE_NSTREAM, sizeof(g_stream[0]));
    if (slot < 0)
    {
        g_cache_full = 1;
        return NULL;
    }
    stream_ent_t *e = &g_stream[slot];
    if (e->h.used)
    {
        release_stream_sem(e);
        ImageStreamIO_closeIm(&e->im);
        e->h.used = 0;
    }
    memset(&e->im, 0, sizeof(e->im));
    if (ImageStreamIO_openIm(&e->im, name) != IMAGESTREAMIO_SUCCESS || e->im.md == NULL)
    {
        return NULL;
    }
    snprintf(e->h.path, sizeof(e->h.path), "%s", path);
    e->h.ino    = (e->im.md->inode != 0) ? e->im.md->inode : st.st_ino;
    e->h.tick   = ++g_tick;
    e->h.pinned = 0;
    e->h.used   = 1;
    e->semindex = -1;
    e->sem_pid  = 0;
    return e;
}

IMAGE *milk_shm_cache_stream(const char *name)
{
    pthread_mutex_lock(&g_cache_lock);
    stream_ent_t *e  = lookup_or_open_stream_locked(name);
    IMAGE        *im = (e != NULL) ? &e->im : NULL;
    pthread_mutex_unlock(&g_cache_lock);
    return im;
}

IMAGE *milk_shm_cache_pin_stream(const char *name, int req_sem, int *out_semindex)
{
    pthread_mutex_lock(&g_cache_lock);
    stream_ent_t *e = lookup_or_open_stream_locked(name);
    if (e == NULL)
    {
        pthread_mutex_unlock(&g_cache_lock);
        return NULL;
    }
    e->h.pinned++;
    int semindex = req_sem;
    if (req_sem < 0)
    {
        pid_t me        = getpid();
        int   sem_valid = (e->semindex >= 0 && e->sem_pid == me && e->im.md != NULL &&
                           e->semindex < e->im.md->sem && e->im.semReadPID != NULL &&
                           e->im.semReadPID[e->semindex] == me);
        if (e->h.pinned == 1)
        {
            if (!sem_valid)
            {
                e->semindex = milk_shm_sem_claim(&e->im);
                e->sem_pid  = (e->semindex >= 0) ? me : 0;
            }
            semindex = e->semindex;
        }
        else
        {
            semindex =
                milk_shm_sem_claim(&e->im); /* a second waiter never shares the first's slot */
        }
    }
    if (out_semindex != NULL)
    {
        *out_semindex = semindex;
    }
    IMAGE *im = &e->im;
    pthread_mutex_unlock(&g_cache_lock);
    return im;
}

void milk_shm_cache_unpin_stream(IMAGE *im, int req_sem, int semindex)
{
    if (im == NULL)
    {
        return;
    }
    pthread_mutex_lock(&g_cache_lock);
    for (int i = 0; i < SHM_CACHE_NSTREAM; i++)
    {
        stream_ent_t *e = &g_stream[i];
        if (e->h.used && &e->im == im)
        {
            if (req_sem < 0 && semindex >= 0 && semindex != e->semindex && e->im.md != NULL &&
                semindex < e->im.md->sem && e->im.semReadPID != NULL &&
                e->im.semReadPID[semindex] == getpid())
            {
                e->im.semReadPID[semindex] = 0;
            }
            if (e->h.pinned > 0)
            {
                e->h.pinned--;
            }
            if (e->h.pinned == 0)
            {
                /* the mapping stays; the semaphore slot goes back so idle shells do not hold the
                 * stream's few slots (a C reader needs one; a waiter re-acquires in a 10-entry scan) */
                release_stream_sem(e);
                if (e->h.path[0] == '\0')
                {
                    ImageStreamIO_closeIm(&e->im);
                    e->h.used = 0;
                }
            }
            break;
        }
    }
    pthread_mutex_unlock(&g_cache_lock);
}

FPS *milk_shm_cache_fps(const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return NULL;
    }
    pthread_mutex_lock(&g_cache_lock);
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
                pthread_mutex_unlock(&g_cache_lock);
                return &e->fps;
            }
            fps_disconnect(&e->fps);
            e->h.used = 0;
            break;
        }
    }
    if (!exists)
    {
        pthread_mutex_unlock(&g_cache_lock);
        return NULL;
    }

    int slot = pick_slot(&g_fps[0].h, SHM_CACHE_NFPS, sizeof(g_fps[0]));
    if (slot < 0)
    {
        pthread_mutex_unlock(&g_cache_lock);
        return NULL;
    }
    fps_ent_t *e = &g_fps[slot];
    if (e->h.used)
    {
        fps_disconnect(&e->fps);
        e->h.used = 0;
    }
    memset(&e->fps, 0, sizeof(e->fps));
    if (fps_connect(name, &e->fps, FPSCONNECT_SIMPLE) == -1 || e->fps.md == NULL ||
        e->fps.parray == NULL)
    {
        pthread_mutex_unlock(&g_cache_lock);
        return NULL;
    }
    struct stat fst;
    snprintf(e->h.path, sizeof(e->h.path), "%s", path);
    e->h.ino    = (fstat(e->fps.SMfd, &fst) == 0) ? fst.st_ino : st.st_ino;
    e->h.tick   = ++g_tick;
    e->h.pinned = 0;
    e->h.used   = 1;
    FPS *res    = &e->fps;
    pthread_mutex_unlock(&g_cache_lock);
    return res;
}

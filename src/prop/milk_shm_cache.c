// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_cache.c
 * @brief The cache of mapped streams and FPSs, and semaphore slot ownership (see
 *        milk_shm_cache.h).
 *
 * Two fixed tables of slots (64 streams, 16 FPSs) share one mutex. A slot keeps the path and
 * inode of its file: a lookup stat()s the file, reuses the slot when the inode matches, and else
 * unmaps it (a pinned slot is detached, and unmapped at its last unpin) and maps the file again
 * into a free or least-recently-used unpinned slot. A semaphore slot belongs to the PID written
 * in semReadPID[]: it is claimed with a compare-and-exchange, taken over from a dead process,
 * and never one this process already holds. A stream's first waiter keeps its slot in the cache
 * entry until the last unpin; an atexit() handler gives back the slots the cache still holds,
 * and pthread_atfork() handlers keep the mutex usable in a forked child.
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

/* ============================================================================
 * Cache Data Structures and Process-Wide State
 * ============================================================================ */

/**
 * struct ent_hdr_t - Common metadata header for stream and FPS cache entries
 * @used:   Non-zero when this slot holds an active shared-memory mapping
 * @pinned: Active reference/pin count preventing LRU eviction (0 when unpinned)
 * @path:   Full filesystem path of the mapped `.im.shm` or `.fps.shm` file
 *          (set to `""` when detached after a recreation while still pinned)
 * @ino:    Filesystem inode (`st_ino`) of the backing file at the time it was mapped
 * @tick:   Monotonic access timestamp (`g_tick`) used for LRU replacement
 */
typedef struct
{
    int           used;
    int           pinned;
    char          path[STRINGMAXLEN_DIRNAME + STRINGMAXLEN_IMAGE_NAME + 16];
    ino_t         ino;
    unsigned long tick;
} ent_hdr_t;

/**
 * struct stream_ent_t - Cache slot for a mapped ImageStreamIO shared-memory stream
 * @h:        Common cache entry header (must be the first member for `pick_slot()`)
 * @semindex: Cached reader semaphore slot index in `im.semReadPID[]` (-1 if none)
 * @sem_pid:  Process ID (`getpid()`) that claimed @semindex
 * @im:       Mapped ImageStreamIO `IMAGE` structure
 */
typedef struct
{
    ent_hdr_t h;
    int       semindex;
    pid_t     sem_pid;
    IMAGE     im;
} stream_ent_t;

/**
 * struct fps_ent_t - Cache slot for a connected Function Parameter Structure (FPS)
 * @h:   Common cache entry header (must be the first member for `pick_slot()`)
 * @fps: Connected `FPS` handle
 */
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

/* Thread-local flag set when the last stream lookup found all 64 slots pinned */
static __thread int g_cache_full = 0;

/* ============================================================================
 * Semaphore Slot Claim / Release and Cache Status
 * ============================================================================ */

/**
 * milk_shm_cache_full - Query whether the calling thread's last stream lookup exhausted the cache
 *
 * Return: 1 if the most recent stream lookup on this thread failed because all
 *         `SHM_CACHE_NSTREAM` (64) slots were pinned, 0 otherwise
 */
int milk_shm_cache_full(void)
{
    return g_cache_full;
}

/**
 * milk_shm_sem_claim - Atomically claim a free reader semaphore slot on stream @im
 * @im: Mapped ImageStreamIO `IMAGE` whose `semReadPID` array will be scanned
 *
 * Scans `im->semReadPID[0 .. im->md->sem - 1]` for an unclaimed slot (`0`) or a
 * slot owned by a dead process (`getpgid(cur) < 0`), skipping any slot already
 * owned by this process (`cur == getpid()`) so that concurrent waiters within
 * the same process never contend on the same semaphore. Claims the slot via
 * `__atomic_compare_exchange_n()`.
 *
 * Return: Claimed semaphore index (`0 .. im->md->sem - 1`), or -1 if none available
 */
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
        /* Skip slots already owned by this process or by another live reader process */
        if (cur == me || (cur != 0 && getpgid(cur) >= 0))
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

/**
 * milk_shm_sem_release - Atomically release reader semaphore @slot on stream @im
 * @im:   Mapped ImageStreamIO `IMAGE`
 * @slot: Semaphore index previously obtained from `milk_shm_sem_claim()`
 *
 * Clears `im->semReadPID[slot]` back to `0` only if it is still owned by the
 * calling process (`getpid()`).
 *
 * Return: void
 */
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

/**
 * release_stream_sem - Release the cached semaphore slot on stream entry @e if owned by us
 * @e: Stream cache entry whose cached semaphore slot should be released
 *
 * Context: Caller must hold `g_cache_lock`.
 *
 * Return: void
 */
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

/* ============================================================================
 * Fork Safety, Exit Cleanup, and LRU Slot Selection
 * ============================================================================ */

/**
 * cache_fork_prepare - `pthread_atfork` prepare handler acquiring `g_cache_lock` before `fork()`
 *
 * Ensures that no other thread is midway through mutating the cache when a child
 * process is forked (e.g. for `--async` or `@replay`).
 *
 * Return: void
 */
static void cache_fork_prepare(void)
{
    pthread_mutex_lock(&g_cache_lock);
}

/**
 * cache_fork_parent - `pthread_atfork` parent handler releasing `g_cache_lock` after `fork()`
 *
 * Return: void
 */
static void cache_fork_parent(void)
{
    pthread_mutex_unlock(&g_cache_lock);
}

/**
 * cache_fork_child - `pthread_atfork` child handler reinitializing `g_cache_lock` in the child
 *
 * Return: void
 */
static void cache_fork_child(void)
{
    pthread_mutex_init(&g_cache_lock, NULL);
}

/**
 * shm_cache_atexit - `atexit()` handler releasing all semaphore slots held by this process
 *
 * Iterates over all active stream cache entries under `g_cache_lock` and clears
 * any `semReadPID` entries owned by this process so readers do not leak scarce
 * semaphore slots after process exit.
 *
 * Return: void
 */
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

/**
 * pick_slot - Select a free cache slot or the least-recently-used unpinned slot
 * @first:  Pointer to the `ent_hdr_t` header of element 0 in the cache array
 * @n:      Total number of slots in the cache array
 * @stride: Byte stride (`sizeof(stream_ent_t)` or `sizeof(fps_ent_t)`) between entries
 *
 * Context: Caller must hold `g_cache_lock`.
 *
 * Return: Slot index in `[0, n - 1]`, or -1 if every slot in the table is currently pinned
 */
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

/* ============================================================================
 * Stream and FPS Cache Lookup / Pinning API
 * ============================================================================ */

/**
 * lookup_or_open_stream_locked - Find a valid cached stream or map `@name` from `/dev/shm`
 * @name: Bare stream name (without directory prefix or `.im.shm` suffix)
 *
 * Checks whether `@name` is already mapped in `g_stream[]` with a matching inode.
 * If the backing file was recreated (`st_ino` changed) or removed:
 * - Unpinned stale entries are closed immediately.
 * - Pinned stale entries have `h.path` cleared (`""`) so they are ignored by
 *   subsequent lookups and closed when the last pin is released.
 *
 * If no valid cached mapping is found and the `.im.shm` file exists, evicts an
 * unpinned LRU slot via `pick_slot()` and maps the stream with
 * `ImageStreamIO_openIm()`.
 *
 * Context: Caller must hold `g_cache_lock`.
 *
 * Return: Pointer to the valid `stream_ent_t`, or NULL if missing or cache is full
 */
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
            /* Stream was recreated or removed on disk; close stale mapping */
            ImageStreamIO_closeIm(&e->im);
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

/**
 * milk_shm_cache_stream - Look up or map shared-memory stream @name without pinning
 * @name: Bare stream name
 *
 * Suitable for short-lived metadata/pixel reads that complete before another
 * 64 distinct streams are opened.
 *
 * Return: Pointer to the cached `IMAGE`, or NULL if the stream does not exist
 */
IMAGE *milk_shm_cache_stream(const char *name)
{
    pthread_mutex_lock(&g_cache_lock);
    stream_ent_t *e  = lookup_or_open_stream_locked(name);
    IMAGE        *im = (e != NULL) ? &e->im : NULL;
    pthread_mutex_unlock(&g_cache_lock);
    return im;
}

/**
 * milk_shm_cache_pin_stream - Map stream @name, increment its pin count, and claim a semaphore
 * @name:         Bare stream name
 * @req_sem:      Explicit semaphore index to use (`>= 0`), or `-1` to auto-claim a slot
 * @out_semindex: Optional output receiving the resolved/claimed semaphore slot index
 *
 * Prevents the stream's cache entry from being evicted across blocking waits.
 * When `@req_sem < 0`, the first pinner caches its claimed semaphore index in
 * `e->semindex`; any concurrent additional pinner on the same stream claims a
 * distinct semaphore slot via `milk_shm_sem_claim()` so two waiters never steal
 * posts from each other.
 *
 * Return: Pointer to the pinned `IMAGE`, or NULL if not found or all cache slots are pinned
 */
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
            /* A second concurrent waiter never shares the first waiter's semaphore slot */
            semindex = milk_shm_sem_claim(&e->im);
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

/**
 * milk_shm_cache_unpin_stream - Decrement the pin count on @im and release its semaphore slot
 * @im:       Pinned `IMAGE` pointer previously returned by `milk_shm_cache_pin_stream()`
 * @req_sem:  The `@req_sem` argument originally passed to `milk_shm_cache_pin_stream()`
 * @semindex: The semaphore index returned in `@out_semindex` by `milk_shm_cache_pin_stream()`
 *
 * Releases any secondary auto-claimed semaphore slot immediately, and when the
 * pin count reaches 0, releases the primary semaphore slot back to the stream so
 * idle processes do not hold scarce reader semaphore slots. If the entry was
 * detached (`h.path[0] == '\0'`) while pinned because the backing file was
 * recreated, closes the stale mapping when the last pin drops.
 *
 * Return: void
 */
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
                /*
                 * Keep the shared-memory mapping cached, but return the semaphore slot
                 * so idle shells do not hold the stream's few reader slots.
                 */
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

/**
 * milk_shm_cache_fps - Look up or connect to Function Parameter Structure (FPS) @name
 * @name: Bare FPS name (without directory prefix or `.fps.shm` suffix)
 *
 * Checks `g_fps[]` under `g_cache_lock` for a cached connection whose backing
 * `.fps.shm` file still exists with the same inode (`st_ino`). Reconnects or
 * evicts the least-recently-used slot via `pick_slot()` when needed.
 *
 * Return: Pointer to the cached `FPS` structure, or NULL if `@name` does not exist
 */
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

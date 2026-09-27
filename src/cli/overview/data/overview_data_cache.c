#include <pthread.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "overview_data_internal.h"

#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "fps_types.h"
#include "fps_paramvalue.h"
#include "fps_printparameter_valuestring.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_save2disk.h"

/* =========================================================
 * Persistent SHM mapping caches
 *
 * These caches keep SHM file descriptors and mappings
 * alive across scan ticks to avoid the overhead of
 * open/mmap/munmap/close on every cycle.
 *
 * Staleness: each tick checks stat() inode; a mismatch
 * or missing file triggers remap/eviction.
 * ========================================================= */

/* --- Stream cache --- */


ov_stream_cache_t s_scache[OV_MAX_STREAMS];
int               s_scache_nb    = 0;
pthread_mutex_t   s_scache_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * scache_find - find stream in cache by name.
 *
 * Return: cache index, or -1 if not found.
 */
int scache_find(const char *name)
{
    for (int i = 0; i < s_scache_nb; i++)
    {
        if (strcmp(s_scache[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * scache_evict_locked - close mapping and compact array (caller holds s_scache_mutex).
 */
void scache_evict_locked(int ci)
{
    ImageStreamIO_closeIm(&s_scache[ci].img);
    s_scache_nb--;
    if (ci < s_scache_nb)
    {
        s_scache[ci] = s_scache[s_scache_nb];
    }
}

/**
 * scache_evict - close mapping and compact array.
 */
void scache_evict(int ci)
{
    pthread_mutex_lock(&s_scache_mutex);
    scache_evict_locked(ci);
    pthread_mutex_unlock(&s_scache_mutex);
}

/**
 * scache_evict_by_name - find and evict stream by name under lock.
 */
void scache_evict_by_name(const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return;
    }
    pthread_mutex_lock(&s_scache_mutex);
    int ci = scache_find(name);
    if (ci >= 0)
    {
        scache_evict_locked(ci);
    }
    pthread_mutex_unlock(&s_scache_mutex);
}

/* --- FPS cache --- */

pthread_mutex_t      s_fcache_mutex = PTHREAD_MUTEX_INITIALIZER;
static OV_FPS_PARAMS s_active_fps_params;

ov_fps_cache_t s_fcache[OV_MAX_FPS];
int            s_fcache_nb = 0;

/**
 * @brief Look up an FPS entry in the connection cache.
 */
int fcache_find(const char *name)
{
    for (int i = 0; i < s_fcache_nb; i++)
    {
        if (strcmp(s_fcache[i].fname, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Evict and disconnect an FPS cache entry (caller must hold s_fcache_mutex).
 */
void fcache_evict_locked(int ci)
{
    fps_disconnect(&s_fcache[ci].fps);
    s_fcache_nb--;
    if (ci < s_fcache_nb)
    {
        s_fcache[ci] = s_fcache[s_fcache_nb];
    }
}

/**
 * @brief Evict and disconnect an FPS cache entry.
 */
void fcache_evict(int ci)
{
    pthread_mutex_lock(&s_fcache_mutex);
    fcache_evict_locked(ci);
    pthread_mutex_unlock(&s_fcache_mutex);
}

/* --- Proc cache --- */


ov_proc_cache_t s_pcache[OV_MAX_PROCS];
int             s_pcache_nb = 0;


/**
 * @brief Look up a process by PID in the cache.
 */
int pcache_find_pid(pid_t pid)
{
    for (int i = 0; i < s_pcache_nb; i++)
    {
        if (s_pcache[i].pid == pid)
        {
            return i;
        }
    }
    return -1;
}

/**
 * pcache_evict - Evict a cached processinfo mapping by index
 * @ci: Cache slot index to evict
 *
 * Unmaps shared memory, closes file descriptor, and replaces the slot
 * with the last cache entry to keep cache contiguous.
 */
void pcache_evict(int ci)
{
    munmap(s_pcache[ci].pinfo, sizeof(PROCESSINFO));
    close(s_pcache[ci].fd);
    s_pcache_nb--;
    if (ci < s_pcache_nb)
    {
        s_pcache[ci] = s_pcache[s_pcache_nb];
    }
}

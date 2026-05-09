#include "overview_data_internal.h"

/* =========================================================
 * Persistent SHM mapping caches
 *
 * These caches keep SHM file descriptors and mappings
 * alive across scan ticks to avoid the overhead of
 * open/mmap/munmap/close on every cycle.
 *
 * Thread safety: the cache is accessed only by the
 * background scan thread, so no locking is required.
 *
 * Staleness: each tick checks stat() inode; a mismatch
 * or missing file triggers remap/eviction.
 * ========================================================= */

/* --- Stream cache --- */



ov_stream_cache_t s_scache[OV_MAX_STREAMS];
int               s_scache_nb = 0;

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
 * scache_evict - close mapping and compact array.
 */
void scache_evict(int ci)
{
    ImageStreamIO_closeIm(&s_scache[ci].img);
    s_scache_nb--;
    if (ci < s_scache_nb)
    {
        s_scache[ci] = s_scache[s_scache_nb];
    }
}

/* --- FPS cache --- */



ov_fps_cache_t s_fcache[OV_MAX_FPS];
int            s_fcache_nb = 0;

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

void fcache_evict(int ci)
{
    fps_disconnect(
        &s_fcache[ci].fps);
    s_fcache_nb--;
    if (ci < s_fcache_nb)
    {
        s_fcache[ci] = s_fcache[s_fcache_nb];
    }
}

/* --- Proc cache --- */



ov_proc_cache_t s_pcache[OV_MAX_PROCS];
int             s_pcache_nb = 0;

/* Processinfo list mapping (single, persistent) */
PROCESSINFOLIST *s_pilist     = NULL;
int              s_pilist_fd  = -1;

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



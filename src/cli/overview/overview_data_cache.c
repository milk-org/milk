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
 * @brief Evict and disconnect an FPS cache entry.
 */
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

/* =========================================================
 * FPS cache public accessors
 * ========================================================= */

/**
 * ov_fcache_get_fps - return raw FPS pointer by name.
 *
 * Returns the memory-mapped FPS struct from the cache,
 * or NULL if the FPS is not currently cached.
 */
FPS *ov_fcache_get_fps(const char *name)
{
    int ci = fcache_find(name);
    if (ci < 0)
    {
        return NULL;
    }
    return &s_fcache[ci].fps;
}

/**
 * ov_fcache_get_param_index - map display index to
 *     raw FPS parameter array index.
 *
 * @fps_name: FPS name to look up in cache
 * @disp_idx: display parameter index (0..nb_disp_params-1)
 *
 * Return: raw parray index, or -1 on error.
 */
int ov_fcache_get_param_index(
    const char *fps_name,
    int        disp_idx)
{
    int ci = fcache_find(fps_name);
    if (ci < 0
        || disp_idx < 0
        || disp_idx >= s_fcache[ci].dparam_nb)
    {
        return -1;
    }
    return s_fcache[ci].dparam_idx[disp_idx];
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"


/* =========================================================
 * Cache cleanup (called from ov_scan_stop)
 * ========================================================= */

/**
 * ov_scan_cache_cleanup - Release and unmap all cached streams and FPS descriptors
 * Called during worker shutdown to close opened ImageStreamIO images, disconnect
 * active FPS mappings, and free cached processinfo entries.
 */

struct timespec s_shm_mtime   = { 0, 0 };
struct timespec s_fps_mtime   = { 0, 0 };
struct timespec s_proc_mtime  = { 0, 0 };
double          s_scan_dt_sec = 0.0;

void ov_scan_cache_cleanup(void)
{
    /* Close all cached stream mappings */
    pthread_mutex_lock(&s_scache_mutex);
    for (int i = s_scache_nb - 1; i >= 0; i--)
    {
        ImageStreamIO_closeIm(&s_scache[i].img);
    }
    s_scache_nb = 0;
    pthread_mutex_unlock(&s_scache_mutex);

    /* Disconnect all cached FPS mappings */
    pthread_mutex_lock(&s_fcache_mutex);
    for (int i = s_fcache_nb - 1; i >= 0; i--)
    {
        fcache_evict_locked(i);
    }
    pthread_mutex_unlock(&s_fcache_mutex);

    /* Unmap all cached proc mappings */
    for (int i = s_pcache_nb - 1; i >= 0; i--)
    {
        pcache_evict(i);
    }
    ov_shm_unmap(&s_plist_map);

    /* Reset directory mtime trackers */
    memset(&s_shm_mtime, 0, sizeof(s_shm_mtime));
    memset(&s_fps_mtime, 0, sizeof(s_fps_mtime));
    memset(&s_proc_mtime, 0, sizeof(s_proc_mtime));

    /* Reset PID liveness cache */
    pid_cache_reset();

    /* Reset graph and rate state */
    s_scan_dt_sec = 0.0;
}


/* =========================================================
 * Post-scan enrichment: FPS Hz sparklines (run PID to proc) and process uptime
 * ========================================================= */

/**
 * get_boot_time - read system boot time in seconds.
 *
 * Return: seconds since epoch of system boot, or 0 on error.
 */
static int64_t get_boot_time(void)
{
    static int64_t cached_boot = 0;
    if (cached_boot > 0)
    {
        return cached_boot;
    }
    FILE *fp = fopen("/proc/stat", "r");
    if (fp == NULL)
    {
        return 0;
    }
    char line[256];
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (strncmp(line, "btime ", 6) == 0)
        {
            cached_boot = (int64_t) atoll(line + 6);
            break;
        }
    }
    fclose(fp);
    return cached_boot;
}

/**
 * pid_get_start_time - get process start time.
 * @pid: process ID
 *
 * Return: epoch seconds of process start, or 0 on error.
 */
static int64_t pid_get_start_time(pid_t pid)
{
    char        buf[1024];
    const char *p = pid_stat_skip(pid_stat_fields(pid, buf, sizeof(buf)), 19); /* field 22 */
    if (p == NULL)
    {
        return 0;
    }
    int64_t starttime = (int64_t) atoll(p);
    long    clk       = sysconf(_SC_CLK_TCK);
    if (clk <= 0)
    {
        clk = 100;
    }
    int64_t boot = get_boot_time();
    return boot + starttime / clk;
}


/**
 * ov_post_scan_enrich - FPS run-process Hz history and process uptime.
 * @model: Pointer to data model
 *
 * Per-item history (stale count, new-item age, rates) lives in the scan caches.
 */
void ov_post_scan_enrich(OV_MODEL *model)
{
    int64_t now_epoch = (int64_t) time(NULL);

    /* --- FPS Hz history: run-process rate, kept in the FPS cache across scans --- */
    pthread_mutex_lock(&s_fcache_mutex);
    for (int fi = 0; fi < model->nb_fps; fi++)
    {
        OV_FPS *f  = &model->fps[fi];
        int     pi = ov_find_proc_by_pid(model, f->runpid);
        int     ci = fcache_find(f->name);
        if (ci < 0)
        {
            continue;
        }
        ov_fps_cache_t *ce = &s_fcache[ci];
        ce->hz_hist[ce->hz_hist_idx % OV_SPARKLINE_LEN] =
            (pi >= 0) ? (float) model->procs[pi].loop_hz : 0.0f;
        ce->hz_hist_idx++;
        memcpy(f->hz_hist, ce->hz_hist, sizeof(f->hz_hist));
        f->hz_hist_idx = ce->hz_hist_idx;
    }
    pthread_mutex_unlock(&s_fcache_mutex);

    /* --- #7: Process uptime --- */
    for (int pi = 0; pi < model->nb_procs; pi++)
    {
        OV_PROC *p = &model->procs[pi];
        if (p->PID > 0)
        {
            int ci = pcache_find(p->pindex, p->PID);
            if (ci >= 0 && s_pcache[ci].has_start_time)
            {
                if (s_pcache[ci].start_time > 0)
                {
                    p->start_time_sec = now_epoch - s_pcache[ci].start_time;
                }
            }
            else
            {
                int64_t st = pid_get_start_time(p->PID);
                if (ci >= 0)
                {
                    s_pcache[ci].start_time     = st;
                    s_pcache[ci].has_start_time = 1;
                }
                if (st > 0)
                {
                    p->start_time_sec = now_epoch - st;
                }
            }
        }
    }
}

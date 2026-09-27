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
    for (int i = s_scache_nb - 1; i >= 0; i--)
    {
        ImageStreamIO_closeIm(&s_scache[i].img);
    }
    s_scache_nb = 0;

    /* Disconnect all cached FPS mappings */
    pthread_mutex_lock(&s_fcache_mutex);
    for (int i = s_fcache_nb - 1; i >= 0; i--)
    {
        fps_disconnect(&s_fcache[i].fps);
    }
    s_fcache_nb = 0;
    pthread_mutex_unlock(&s_fcache_mutex);

    /* Unmap all cached proc mappings */
    for (int i = s_pcache_nb - 1; i >= 0; i--)
    {
        munmap(s_pcache[i].pinfo, sizeof(PROCESSINFO));
        close(s_pcache[i].fd);
    }
    s_pcache_nb = 0;

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
 * Post-scan enrichment
 *
 * After all scanners and graph build, this function adds:
 *  - FPS Hz sparklines (cross-ref RPID to proc)
 *  - Process uptime (from /proc/PID/stat)
 *  - Stale process detection (alive but counter stuck)
 *  - New-item flash (items not in previous scan)
 * ========================================================= */

/* Previous scan name sets for new-item detection */
static char s_prev_stream_names[OV_MAX_STREAMS][40];
static int  s_prev_nb_streams = 0;
static char s_prev_fps_names[OV_MAX_FPS][STRINGMAXLEN_FPS_NAME];
static int  s_prev_nb_fps = 0;
static char s_prev_proc_names[OV_MAX_PROCS][48];
static int  s_prev_nb_procs = 0;

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
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat", (int) pid);
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return 0;
    }
    /* Field 22 is starttime (in clock ticks).
     * Skip past the comm field (in parens) first. */
    char buf[1024];
    if (fgets(buf, sizeof(buf), fp) == NULL)
    {
        fclose(fp);
        return 0;
    }
    fclose(fp);

    /* Find closing paren of comm field */
    char *cp = strrchr(buf, ')');
    if (cp == NULL)
    {
        return 0;
    }
    /* Skip fields 3..21 (19 fields after ')') */
    char *p = cp + 2;
    for (int ff = 0; ff < 19; ff++)
    {
        while (*p == ' ')
        {
            p++;
        }
        while (*p != ' ' && *p != '\0')
        {
            p++;
        }
    }
    while (*p == ' ')
    {
        p++;
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
 * name_in_list - check if name exists in a name array.
 * @name:  name to search for
 * @list:  array of name strings
 * @count: number of entries in list
 * @width: sizeof each entry (stride)
 *
 * Return: 1 if found, 0 otherwise.
 */
static int name_in_list(const char *name, const char *list, int count, int width)
{
    for (int ii = 0; ii < count; ii++)
    {
        if (strcmp(name, list + ii * width) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * ov_post_scan_enrich - Enrich model entries with historical sparklines and stale flags
 * @model: Pointer to data model
 *
 * Updates sparklines, calculates uptime, detects stale streams/processes, and
 * flags newly appeared items for UI highlight effects.
 */
void ov_post_scan_enrich(OV_MODEL *model)
{
    int64_t now_epoch = (int64_t) time(NULL);

    /* --- #6: FPS sparklines (cross-ref RPID to procs) --- */
    for (int fi = 0; fi < model->nb_fps; fi++)
    {
        OV_FPS *f  = &model->fps[fi];
        float   hz = 0.0f;
        if (f->runpid > 0)
        {
            for (int pi = 0; pi < model->nb_procs; pi++)
            {
                if (model->procs[pi].PID == f->runpid)
                {
                    hz = (float) model->procs[pi].loop_hz;
                    break;
                }
            }
        }
        int idx         = f->hz_hist_idx % OV_SPARKLINE_LEN;
        f->hz_hist[idx] = hz;
        f->hz_hist_idx++;
    }

    /* --- #7: Process uptime --- */
    for (int pi = 0; pi < model->nb_procs; pi++)
    {
        OV_PROC *p = &model->procs[pi];
        if (p->PID > 0)
        {
            int ci = pcache_find_pid(p->PID);
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

    /* --- #15: Stale process detection --- */
    for (int pi = 0; pi < model->nb_procs; pi++)
    {
        OV_PROC *p = &model->procs[pi];
        if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE && !p->cnt_active)
        {
            p->stale_count++;
        }
        else
        {
            p->stale_count = 0;
        }
    }

    /* --- #16: New-item flash --- */
    /* Streams */
    for (int si = 0; si < model->nb_streams; si++)
    {
        OV_STREAM *s = &model->streams[si];
        if (s_prev_nb_streams > 0 &&
            !name_in_list(s->name, (const char *) s_prev_stream_names, s_prev_nb_streams, 40))
        {
            s->is_new = 4;
        }
    }
    /* FPS */
    for (int fi = 0; fi < model->nb_fps; fi++)
    {
        OV_FPS *f = &model->fps[fi];
        if (s_prev_nb_fps > 0 && !name_in_list(f->name, (const char *) s_prev_fps_names,
                                               s_prev_nb_fps, STRINGMAXLEN_FPS_NAME))
        {
            f->is_new = 4;
        }
    }
    /* Procs */
    for (int pi = 0; pi < model->nb_procs; pi++)
    {
        OV_PROC *p = &model->procs[pi];
        /* Build unique key: name + PID */
        char key[48];
        snprintf(key, sizeof(key), "%s/%d", p->name, (int) p->PID);
        if (s_prev_nb_procs > 0 &&
            !name_in_list(key, (const char *) s_prev_proc_names, s_prev_nb_procs, 48))
        {
            p->is_new = 4;
        }
    }

    /* Save current names for next scan */
    s_prev_nb_streams = model->nb_streams;
    for (int si = 0; si < model->nb_streams && si < OV_MAX_STREAMS; si++)
    {
        strncpy(s_prev_stream_names[si], model->streams[si].name, 39);
    }
    s_prev_nb_fps = model->nb_fps;
    for (int fi = 0; fi < model->nb_fps && fi < OV_MAX_FPS; fi++)
    {
        strncpy(s_prev_fps_names[fi], model->fps[fi].name, STRINGMAXLEN_FPS_NAME - 1);
    }
    s_prev_nb_procs = model->nb_procs;
    for (int pi = 0; pi < model->nb_procs && pi < OV_MAX_PROCS; pi++)
    {
        snprintf(s_prev_proc_names[pi], sizeof(s_prev_proc_names[pi]), "%s/%d",
                 model->procs[pi].name, (int) model->procs[pi].PID);
    }
}

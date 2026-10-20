// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"

#include <limits.h>

/**
 * plist_map_ro - (re)map <procdir>/processinfo.list.shm read-only; NULL when it does not exist.
 * A viewer never creates the registry.
 */
static const PROCESSINFOLIST *plist_map_ro(const char *procdir)
{
    char        path[PATH_MAX];
    struct stat st;
    snprintf(path, sizeof(path), "%s/processinfo.list.shm", procdir);
    if (stat(path, &st) != 0)
    {
        ov_shm_unmap(&s_plist_map);
        return NULL;
    }
    if (!ov_shm_same_file(&s_plist_map, &st))
    {
        ov_shm_unmap(&s_plist_map);
        ov_shm_map(&s_plist_map, path, 0, sizeof(PROCESSINFOLIST));
    }
    return (const PROCESSINFOLIST *) s_plist_map.addr;
}

/**
 * pcache_get - cached read-only map of one registry entry's proc file, remapped when @recheck finds
 * it removed or re-created. A failed map is kept as a negative entry (pinfo NULL) so files of dead
 * entries are not re-opened every scan. Return: cache index, or -1 when the cache is full.
 */
static int pcache_get(long pindex, pid_t pid, const char *fpath, int recheck)
{
    int ci        = pcache_find(pindex, pid);
    int seen_dead = 0;
    if (ci >= 0 && recheck)
    {
        struct stat st;
        int         present = (stat(fpath, &st) == 0);
        int         stale =
            s_pcache[ci].pinfo ? (!present || !ov_shm_same_file(&s_pcache[ci].map, &st)) : present;
        if (stale)
        {
            seen_dead = s_pcache[ci].seen_dead;
            pcache_evict(ci);
            ci = -1;
        }
    }
    if (ci < 0)
    {
        if (s_pcache_nb >= OV_MAX_PROCS)
        {
            return -1;
        }
        ci = s_pcache_nb++;
        memset(&s_pcache[ci], 0, sizeof(ov_proc_cache_t));
        s_pcache[ci].pid       = pid;
        s_pcache[ci].pindex    = pindex;
        s_pcache[ci].seen_dead = seen_dead;
        if (ov_shm_map(&s_pcache[ci].map, fpath, 0, sizeof(PROCESSINFO)) == 0)
        {
            s_pcache[ci].pinfo = (const PROCESSINFO *) s_pcache[ci].map.addr;
        }
    }
    s_pcache[ci].in_use = 1;
    return ci;
}

/**
 * @brief Scan processinfo for active processes, read-only.
 *
 * Registry states 2/3 written by others are honoured. An entry still marked active (1) whose PID
 * is gone is shown as STOPPED (clean exit: file removed or loopstat STOP) or CRASHED, computed
 * here and never written back; the CONTROL-mode cleanup ('C') reaps such slots on request.
 */
void ov_scan_procs(OV_MODEL *model)
{
    const char *shmdir = ov_get_procdir();

    /* Mark all proc cache entries as not-in-use */
    for (int i = 0; i < s_pcache_nb; i++)
    {
        s_pcache[i].in_use = 0;
    }

    const PROCESSINFOLIST *pl  = plist_map_ro(shmdir);
    int                    idx = 0;

    /* Proc files are re-checked only when the directory changed */
    int         recheck = 0;
    struct stat dst;
    if (stat(shmdir, &dst) == 0 &&
        (dst.st_mtim.tv_sec != s_proc_mtime.tv_sec || dst.st_mtim.tv_nsec != s_proc_mtime.tv_nsec))
    {
        s_proc_mtime = dst.st_mtim;
        recheck      = 1;
    }

    for (long i = 0; pl != NULL && i < PROCESSINFOLISTSIZE && idx < OV_MAX_PROCS; i++)
    {
        int state = pl->active[i];
        if (state == 0)
        {
            continue;
        }

        pid_t pid = pl->PIDarray[i];
        if (pid <= 0)
        {
            continue;
        }

        char pname[STRINGMAXLEN_PROCESSINFO_NAME];
        strncpy(pname, pl->pnamearray[i], sizeof(pname) - 1);
        pname[sizeof(pname) - 1] = '\0';

        char fpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/proc.%s.%06d.shm", shmdir, pname, (int) pid);

        int                ci    = pcache_get(i, pid, fpath, recheck);
        const PROCESSINFO *pinfo = (ci >= 0) ? s_pcache[ci].pinfo : NULL;
        int alive = (state == 1) && !(ci >= 0 && s_pcache[ci].seen_dead) && pid_is_alive(pid);

        if (state == 1 && !alive)
        {
            if (ci >= 0)
            {
                s_pcache[ci].seen_dead = 1;
            }
            state = (pinfo == NULL || pinfo->loopstat == PROCESSINFO_LOOPSTAT_STOP) ? 2 : 3;
        }
        if (!alive)
        {
            pinfo = NULL; /* dead entries show zeros and their STOP/CRASHED label */
        }

        OV_PROC *p = &model->procs[idx];
        memset(p, 0, sizeof(OV_PROC));

        if (pinfo != NULL && pinfo->name[0] != '\0')
        {
            strncpy(p->name, pinfo->name, sizeof(p->name) - 1);
        }
        else
        {
            strncpy(p->name, pname, sizeof(p->name) - 1);
        }

        p->PID    = pid;
        p->valid  = 1;
        p->active = (state == 1 && alive);

        if (pinfo != NULL)
        {
            p->loopstat = pinfo->loopstat;
        }
        else if (alive)
        {
            /* live PID: proc file absent (STOP) or present but not mappable (ERROR) */
            p->loopstat =
                (access(fpath, F_OK) == 0) ? PROCESSINFO_LOOPSTAT_ERROR : PROCESSINFO_LOOPSTAT_STOP;
        }
        else
        {
            p->loopstat = (state == 2) ? PROCESSINFO_LOOPSTAT_STOP : PROCESSINFO_LOOPSTAT_CRASHED;
        }

        if (pinfo != NULL)
        {
            p->CTRLval    = pinfo->CTRLval;
            p->loopcnt    = pinfo->loopcnt;
            p->mem_rss_kb = pid_get_rss_kb(pinfo->PID);

            p->dtmedian_iter_ns = pinfo->dtmedian_iter_ns;
            p->dtmedian_exec_ns = pinfo->dtmedian_exec_ns;
            if (p->dtmedian_iter_ns > 0)
            {
                p->loop_hz = 1.0e9 / (double) p->dtmedian_iter_ns;
            }

            strncpy(p->trigstreamname, pinfo->triggerstreamname, sizeof(p->trigstreamname) - 1);
            p->triggermode         = pinfo->triggermode;
            p->triggersem          = pinfo->triggersem;
            p->triggermissed       = pinfo->triggermissedframe;
            p->triggermissed_cumul = pinfo->triggermissedframe_cumul;
            p->MeasureTiming       = pinfo->MeasureTiming;
            p->rt_priority         = pinfo->RT_priority;
            strncpy(p->statusmsg, pinfo->statusmsg, sizeof(p->statusmsg) - 1);

            if (ci >= 0)
            {
                ov_proc_cache_t *ce = &s_pcache[ci];

                if (p->loop_hz < 0.1 && ce->has_prev_loop && s_scan_dt_sec > 0.01)
                {
                    int64_t dlc = p->loopcnt - ce->prev_loopcnt;
                    if (dlc > 0)
                    {
                        p->loop_hz = (double) dlc / s_scan_dt_sec;
                    }
                }
                p->cnt_active     = (ce->has_prev_loop && p->loopcnt != ce->prev_loopcnt);
                ce->prev_loopcnt  = p->loopcnt;
                ce->has_prev_loop = 1;

                uint64_t ut = 0, st = 0;
                if (pid_get_cpu_ticks(pid, &ut, &st) == 0)
                {
                    if (ce->has_prev_cpu && s_scan_dt_sec > 0.01)
                    {
                        long     clk    = sysconf(_SC_CLK_TCK);
                        uint64_t dticks = (ut - ce->prev_utime) + (st - ce->prev_stime);
                        ce->cpu_pct =
                            (float) ((double) dticks / ((double) clk * s_scan_dt_sec) * 100.0);
                    }
                    ce->prev_utime   = ut;
                    ce->prev_stime   = st;
                    ce->has_prev_cpu = 1;
                }
                p->cpu_used = ce->cpu_pct;
            }
        }
        else
        {
            p->CTRLval             = 0;
            p->loopcnt             = 0;
            p->mem_rss_kb          = 0;
            p->loop_hz             = 0.0;
            p->trigstreamname[0]   = '\0';
            p->triggermode         = 0;
            p->triggersem          = 0;
            p->triggermissed       = 0;
            p->triggermissed_cumul = 0;
            p->MeasureTiming       = 0;
            p->rt_priority         = 0;
            p->cpu_used            = 0.0f;
            p->statusmsg[0]        = '\0';
        }

        p->node_idx = -1;
        idx++;
    }

    model->nb_procs = idx;

    for (int i = s_pcache_nb - 1; i >= 0; i--)
    {
        if (!s_pcache[i].in_use)
        {
            pcache_evict(i);
        }
    }
}

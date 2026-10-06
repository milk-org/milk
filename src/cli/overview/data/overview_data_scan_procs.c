// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"


/**
 * @brief Scan processinfo for active processes.
 */
void ov_scan_procs(OV_MODEL *model)
{
    if (pinfolist == NULL)
    {
        long pindex_unused;
        processinfo_shm_list_create(&pindex_unused);
    }

    if (pinfolist == NULL)
    {
        model->nb_procs = 0;
        return;
    }

    char shmdir[OV_SHMDIR_MAXLEN];
    processinfo_procdirname(shmdir);

    /* Mark all proc cache entries as not-in-use */
    for (int i = 0; i < s_pcache_nb; i++)
    {
        s_pcache[i].in_use = 0;
    }

    int idx = 0;

    for (long i = 0; i < PROCESSINFOLISTSIZE && idx < OV_MAX_PROCS; i++)
    {
        if (pinfolist->active[i] == 0)
        {
            continue;
        }

        pid_t pid = pinfolist->PIDarray[i];
        if (pid <= 0)
        {
            continue;
        }

        char pname[STRINGMAXLEN_PROCESSINFO_NAME];
        strncpy(pname, pinfolist->pnamearray[i], sizeof(pname) - 1);
        pname[sizeof(pname) - 1] = '\0';

        char fpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/proc.%s.%06d.shm", shmdir, pname, (int) pid);

        int alive = 0;
        if (pinfolist->active[i] == 1)
        {
            alive = pid_is_alive(pid);
        }

        /* Transition state if active but died */
        if (pinfolist->active[i] == 1 && !alive)
        {
            int          fd;
            PROCESSINFO *pinfo_tmp = processinfo_shm_link(fpath, &fd);
            if (pinfo_tmp != (PROCESSINFO *) MAP_FAILED)
            {
                if (pinfo_tmp->loopstat == PROCESSINFO_LOOPSTAT_STOP)
                {
                    pinfolist->active[i] = 2; /* STOPPED */
                }
                else
                {
                    pinfolist->active[i] = 3; /* CRASHED */
                }
                processinfo_shm_close(pinfo_tmp, fd);
            }
            else
            {
                pinfolist->active[i] = 2; /* default to STOPPED if file removed */
            }
        }

        PROCESSINFO *pinfo = NULL;
        int          ci    = pcache_find_pid(pid);

        if (pinfolist->active[i] == 1 && alive)
        {
            if (ci >= 0)
            {
                s_pcache[ci].in_use = 1;
                pinfo               = s_pcache[ci].pinfo;
            }
            else
            {
                int pfd = open(fpath, O_RDONLY);
                if (pfd != -1)
                {
                    PROCESSINFO *pm = (PROCESSINFO *) mmap(NULL, sizeof(PROCESSINFO), PROT_READ,
                                                           MAP_SHARED, pfd, 0);
                    if (pm != MAP_FAILED)
                    {
                        if (s_pcache_nb >= OV_MAX_PROCS)
                        {
                            munmap(pm, sizeof(PROCESSINFO));
                            close(pfd);
                            continue;
                        }

                        ci = s_pcache_nb;
                        memset(&s_pcache[ci], 0, sizeof(ov_proc_cache_t));
                        s_pcache[ci].pid    = pid;
                        s_pcache[ci].pinfo  = pm;
                        s_pcache[ci].fd     = pfd;
                        s_pcache[ci].in_use = 1;
                        s_pcache_nb++;
                        pinfo = pm;
                    }
                    else
                    {
                        close(pfd);
                    }
                }
            }
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
        p->active = (pinfolist->active[i] == 1 && alive);

        if (pinfo != NULL)
        {
            p->loopstat = pinfo->loopstat;
        }
        else
        {
            p->loopstat = (pinfolist->active[i] == 2) ? PROCESSINFO_LOOPSTAT_STOP
                                                      : PROCESSINFO_LOOPSTAT_CRASHED;
        }

        if (!alive)
        {
            p->loopstat = (pinfo != NULL && pinfo->loopstat == PROCESSINFO_LOOPSTAT_STOP)
                              ? PROCESSINFO_LOOPSTAT_STOP
                              : ((pinfolist->active[i] == 2) ? PROCESSINFO_LOOPSTAT_STOP
                                                             : PROCESSINFO_LOOPSTAT_CRASHED);
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

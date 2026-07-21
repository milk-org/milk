// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file procCTRL_TUI_sort.c
 * @brief Sorting comparator for procCTRL TUI
 *
 * Contains the proc_comp() function used by
 * processinfo_CTRLscreen() to sort processes.
 *
 * @see procCTRL_TUI.c for the main TUI loop.
 */

#include <string.h>

#include <processtools.h>
#include "processinfo_scan_shm.h"
#include "procCTRL_TUI.h"

int              sort_ctx_m         = 0;
int              sort_ctx_col       = 0;
int              sort_ctx_dir       = 0;
PROCSCAN_SHM    *sort_ctx_scan_shm  = NULL;
PROCESSINFOLIST *sort_ctx_pinfolist = NULL;

/**
 * @brief Compare two process entries for sorting.
 *
 * Sort key depends on the current column selection.
 */
int proc_comp(const void *a, const void *b)
{
    int idx1 = *(const int *) a;
    int idx2 = *(const int *) b;
    int res  = 0;

    if (!sort_ctx_scan_shm || !sort_ctx_pinfolist)
    {
        return 0;
    }

    PROCESSINFODISP *p1 = &sort_ctx_scan_shm->pinfodisp[idx1];
    PROCESSINFODISP *p2 = &sort_ctx_scan_shm->pinfodisp[idx2];

    switch (sort_ctx_col)
    {
    case 1: // idx
        res = (idx1 < idx2) ? -1 : (idx1 > idx2) ? 1 : 0;
        break;
    case 2: // status
        res = (sort_ctx_pinfolist->active[idx1] < sort_ctx_pinfolist->active[idx2])   ? -1
              : (sort_ctx_pinfolist->active[idx1] > sort_ctx_pinfolist->active[idx2]) ? 1
                                                                                      : 0;
        break;
    case 3: // pid
        res = (sort_ctx_pinfolist->PIDarray[idx1] < sort_ctx_pinfolist->PIDarray[idx2])   ? -1
              : (sort_ctx_pinfolist->PIDarray[idx1] > sort_ctx_pinfolist->PIDarray[idx2]) ? 1
                                                                                          : 0;
        break;
    default:
        if (sort_ctx_m == PROCCTRL_DISPLAYMODE_CTRL)
        {
            switch (sort_ctx_col)
            {
            case 4: // tstart
                res =
                    (sort_ctx_pinfolist->createtime[idx1] < sort_ctx_pinfolist->createtime[idx2])
                        ? -1
                    : (sort_ctx_pinfolist->createtime[idx1] > sort_ctx_pinfolist->createtime[idx2])
                        ? 1
                        : 0;
                break;
            case 5: // pname
                res = strcmp(sort_ctx_pinfolist->pnamearray[idx1],
                             sort_ctx_pinfolist->pnamearray[idx2]);
                break;
            case 6: // state (ctrlval)
            {
                int v1 = p1->loopstat;
                int v2 = p2->loopstat;
                res    = (v1 < v2) ? -1 : (v1 > v2) ? 1 : 0;
            }
            break;
            case 7: // lcnt
                res = (p1->loopcnt < p2->loopcnt) ? -1 : (p1->loopcnt > p2->loopcnt) ? 1 : 0;
                break;
            case 8: // msg
                res = strcmp(p1->statusmsg, p2->statusmsg);
                break;
            }
        }
        else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_RESOURCES)
        {
            switch (sort_ctx_col)
            {
            case 4: // pname
                res = strcmp(sort_ctx_pinfolist->pnamearray[idx1],
                             sort_ctx_pinfolist->pnamearray[idx2]);
                break;
            case 5: // prio
                res = (p1->rt_priority < p2->rt_priority)   ? -1
                      : (p1->rt_priority > p2->rt_priority) ? 1
                                                            : 0;
                break;
            case 6: // cpu
                res = (p1->subprocCPUloadarray_timeaveraged[0] <
                       p2->subprocCPUloadarray_timeaveraged[0])
                          ? -1
                      : (p1->subprocCPUloadarray_timeaveraged[0] >
                         p2->subprocCPUloadarray_timeaveraged[0])
                          ? 1
                          : 0;
                break;
            case 7: // mem
                res = (p1->VmRSSarray[0] < p2->VmRSSarray[0])   ? -1
                      : (p1->VmRSSarray[0] > p2->VmRSSarray[0]) ? 1
                                                                : 0;
                break;
            case 8: // thr
                res = (p1->threads < p2->threads) ? -1 : (p1->threads > p2->threads) ? 1 : 0;
                break;
            }
        }
        else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_TRIGGER)
        {
            switch (sort_ctx_col)
            {
            case 4: // pname
                res = strcmp(sort_ctx_pinfolist->pnamearray[idx1],
                             sort_ctx_pinfolist->pnamearray[idx2]);
                break;
            case 5: // tmode
                res = (p1->triggermode < p2->triggermode)   ? -1
                      : (p1->triggermode > p2->triggermode) ? 1
                                                            : 0;
                break;
            case 6: // tstream
                res = strcmp(p1->triggerstreamname, p2->triggerstreamname);
                break;
            case 7: // tsem
                res = (p1->triggersem < p2->triggersem)   ? -1
                      : (p1->triggersem > p2->triggersem) ? 1
                                                          : 0;
                break;
            case 8: // tcnt
                res = (p1->triggerstreamcnt < p2->triggerstreamcnt)   ? -1
                      : (p1->triggerstreamcnt > p2->triggerstreamcnt) ? 1
                                                                      : 0;
                break;
            case 9: // tmiss
                res = (p1->triggermissedframe_cumul < p2->triggermissedframe_cumul)   ? -1
                      : (p1->triggermissedframe_cumul > p2->triggermissedframe_cumul) ? 1
                                                                                      : 0;
                break;
            }
        }
        else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_TIMING)
        {
            switch (sort_ctx_col)
            {
            case 4: // pname
                res = strcmp(sort_ctx_pinfolist->pnamearray[idx1],
                             sort_ctx_pinfolist->pnamearray[idx2]);
                break;
            case 5: // freq
            {
                double f1 = (p1->dtmedian_iter_ns > 0) ? 1.0e9 / p1->dtmedian_iter_ns : 0;
                double f2 = (p2->dtmedian_iter_ns > 0) ? 1.0e9 / p2->dtmedian_iter_ns : 0;
                res       = (f1 < f2) ? -1 : (f1 > f2) ? 1 : 0;
            }
            break;
            case 6: // exec
                res = (p1->dtmedian_exec_ns < p2->dtmedian_exec_ns)   ? -1
                      : (p1->dtmedian_exec_ns > p2->dtmedian_exec_ns) ? 1
                                                                      : 0;
                break;
            case 7: // over
            {
                double o1 = (p1->dtmedian_iter_ns > 0)
                                ? 100.0 * p1->dtmedian_exec_ns / p1->dtmedian_iter_ns
                                : 0;
                double o2 = (p2->dtmedian_iter_ns > 0)
                                ? 100.0 * p2->dtmedian_exec_ns / p2->dtmedian_iter_ns
                                : 0;
                res       = (o1 < o2) ? -1 : (o1 > o2) ? 1 : 0;
            }
            break;
            }
        }
        else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_PROCINFO)
        {
            switch (sort_ctx_col)
            {
            case 4: // pname
                res = strcmp(sort_ctx_pinfolist->pnamearray[idx1],
                             sort_ctx_pinfolist->pnamearray[idx2]);
                break;
            case 5: // RT
                res = (p1->rt_priority < p2->rt_priority)   ? -1
                      : (p1->rt_priority > p2->rt_priority) ? 1
                                                            : 0;
                break;
            case 6: // loopMax
                res = (p1->loopcntMax < p2->loopcntMax)   ? -1
                      : (p1->loopcntMax > p2->loopcntMax) ? 1
                                                          : 0;
                break;
            case 7: // trig
                res = (p1->triggermode < p2->triggermode)   ? -1
                      : (p1->triggermode > p2->triggermode) ? 1
                                                            : 0;
                break;
            case 8: // timeout
            {
                double t1 = p1->triggertimeout.tv_sec + 1e-9 * p1->triggertimeout.tv_nsec;
                double t2 = p2->triggertimeout.tv_sec + 1e-9 * p2->triggertimeout.tv_nsec;
                res       = (t1 < t2) ? -1 : (t1 > t2) ? 1 : 0;
            }
            break;
            case 9: // timing
                res = (p1->MeasureTiming < p2->MeasureTiming)   ? -1
                      : (p1->MeasureTiming > p2->MeasureTiming) ? 1
                                                                : 0;
                break;
            }
        }
        break;
    }

    if (sort_ctx_dir == 1)
    {
        res = -res;
    }
    return res;
}

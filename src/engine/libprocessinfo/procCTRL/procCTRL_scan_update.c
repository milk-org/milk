/**
 * @file procCTRL_scan_update.c
 * @brief Scan update functions for procCTRL scanner
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <sys/mman.h>

#include "processinfo.h"
#include "processinfo_shm_link.h"
#include "processinfo_shm_close.h"
#include "procCTRL_PIDcollectSystemInfo.h"
#include "quicksort.h"
#include "procCTRL_scan_internal.h"

void scan_update_pinfolist_status(const char *procdname, PROCSCAN_SHM *scan_shm, int *active_count, int *stopped_count, int *crashed_count, double *timearray, long *indexarray, int *listcnt) {
    if (pinfolist == NULL) return;
    
    for (long i = 0; i < PROCESSINFOLISTSIZE; i++) {
        if (pinfolist->active[i] == 1) {
            if (kill(pinfolist->PIDarray[i], 0) == -1 && errno == ESRCH) {
                char SM_fname[STRINGMAXLEN_FULLFILENAME];
                snprintf(SM_fname, sizeof(SM_fname), "%s/proc.%s.%06d.shm", procdname, pinfolist->pnamearray[i], pinfolist->PIDarray[i]);
                int fd;
                PROCESSINFO *pinfo = processinfo_shm_link(SM_fname, &fd);
                if (pinfo != (PROCESSINFO *)MAP_FAILED) {
                    if (pinfo->loopstat == 3) pinfolist->active[i] = 2; // STOPPED
                    else pinfolist->active[i] = 3; // CRASHED
                    
                    // FINAL SNAPSHOT
                    scan_shm->pinfodisp[i].loopcnt = pinfo->loopcnt;
                    strncpy(scan_shm->pinfodisp[i].statusmsg, pinfo->statusmsg, 199);
                    strncpy(scan_shm->pinfodisp[i].name, pinfo->name, 39);

                    processinfo_shm_close(pinfo, fd);
                } else {
                    pinfolist->active[i] = 2; 
                }
            }
        }

        if (pinfolist->active[i] != 0) {
            if (pinfolist->active[i] == 1) (*active_count)++;
            else if (pinfolist->active[i] == 2) (*stopped_count)++;
            else if (pinfolist->active[i] == 3) (*crashed_count)++;

            if (*listcnt < PROCESSINFOLISTSIZE) {
                indexarray[*listcnt] = i;
                timearray[*listcnt] = -1.0 * pinfolist->createtime[i];
                (*listcnt)++;
            }
        }
    }
}

void scan_update_process_details(const char *procdname, PROCSCAN_SHM *scan_shm, int *serviced_count) {
    for (long i = 0; i < PROCESSINFOLISTSIZE; i++) {
        if (pinfolist->active[i] == 1) {
            
            scan_shm->pinfodisp[i].pindex = i;
            PIDcollectSystemInfo(&scan_shm->pinfodisp[i], 0);

            if (scan_shm->request_scan[i] == 1) {
                if (pinfo_mappings[i] == NULL) {
                    char SM_fname[STRINGMAXLEN_FULLFILENAME];
                    snprintf(SM_fname, sizeof(SM_fname), "%s/proc.%s.%06d.shm", procdname, pinfolist->pnamearray[i], pinfolist->PIDarray[i]);
                    pinfo_mappings[i] = processinfo_shm_link(SM_fname, &pinfo_fds[i]);
                    if (pinfo_mappings[i] == (PROCESSINFO *)MAP_FAILED) pinfo_mappings[i] = NULL;
                }

                if (pinfo_mappings[i] != NULL) {
                    PROCESSINFO *pinfo = pinfo_mappings[i];
                    PROCESSINFODISP *pdisp = &scan_shm->pinfodisp[i];
                    
                    pdisp->PID = pinfolist->PIDarray[i];
                    pdisp->loopcnt = pinfo->loopcnt;
                    pdisp->loopcntMax = pinfo->loopcntMax;
                    pdisp->loopstat = pinfo->loopstat;
                    pdisp->rt_priority = pinfo->RT_priority;
                    strncpy(pdisp->statusmsg, pinfo->statusmsg, 199);
                    strncpy(pdisp->name, pinfo->name, 39);
                    
                    pdisp->triggermode = pinfo->triggermode;
                    strncpy(pdisp->triggerstreamname, pinfo->triggerstreamname, 79);
                    pdisp->triggersem = pinfo->triggersem;
                    pdisp->triggerstreamcnt = pinfo->triggerstreamcnt;
                    pdisp->triggertimeout = pinfo->triggertimeout;
                    pdisp->triggermissedframe = pinfo->triggermissedframe;
                    pdisp->triggermissedframe_cumul = pinfo->triggermissedframe_cumul;
                    pdisp->MeasureTiming = pinfo->MeasureTiming;
                    
                    if (pinfo->MeasureTiming != 0) {
                        long dtiter_array[PROCESSINFO_NBtimer - 1];
                        long dtexec_array[PROCESSINFO_NBtimer - 1];
                        
                        for(int tindex = 0; tindex < PROCESSINFO_NBtimer - 1; tindex++) {
                            int ti1 = (pinfo->timerindex - tindex + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;
                            int ti0 = (ti1 - 1 + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;

                            dtiter_array[tindex] = (pinfo->texecstart[ti1].tv_nsec - pinfo->texecstart[ti0].tv_nsec) + 
                                                   1000000000L * (pinfo->texecstart[ti1].tv_sec - pinfo->texecstart[ti0].tv_sec);

                            dtexec_array[tindex] = (pinfo->texecend[ti0].tv_nsec - pinfo->texecstart[ti0].tv_nsec) + 
                                                   1000000000L * (pinfo->texecend[ti0].tv_sec - pinfo->texecstart[ti0].tv_sec);
                        }
                        
                        quick_sort_long(dtiter_array, PROCESSINFO_NBtimer - 1);
                        quick_sort_long(dtexec_array, PROCESSINFO_NBtimer - 1);
                        
                        pinfo->dtmedian_iter_ns = dtiter_array[(PROCESSINFO_NBtimer - 1) / 2];
                        pinfo->dtmedian_exec_ns = dtexec_array[(PROCESSINFO_NBtimer - 1) / 2];
                    }

                    pdisp->dtmedian_iter_ns = pinfo->dtmedian_iter_ns;
                    pdisp->dtmedian_exec_ns = pinfo->dtmedian_exec_ns;
                }
                
                scan_shm->request_scan[i] = 0;
                (*serviced_count)++;
            } else {
                if (pinfolist->active[i] != 1 && pinfo_mappings[i] != NULL) {
                    processinfo_shm_close(pinfo_mappings[i], pinfo_fds[i]);
                    pinfo_mappings[i] = NULL;
                }
            }
        }
    }
}

/**
 * @file procCTRL_scan_rebuild.c
 * @brief Process list rebuild module for procCTRL scanner
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <signal.h>

#include "processinfo.h"
#include "processinfo_shm_link.h"
#include "processinfo_shm_close.h"
#include "procCTRL_scan_internal.h"

void rebuild_process_list(const char *procdname, PROCSCAN_SHM *scan_shm) {
    printf("Rebuilding process list from %s...\n", procdname);
    for (long i = 0; i < PROCESSINFOLISTSIZE; i++) {
        pinfolist->active[i] = 0;
    }
    DIR *d = opendir(procdname);
    if (d) {
        long pindex = 0;
        struct dirent *dir;
        while ((dir = readdir(d)) != NULL) {
            if (strncmp(dir->d_name, "proc.", 5) == 0 && strstr(dir->d_name, ".shm")) {
                char workbuf[512];
                strncpy(workbuf, dir->d_name, 511);
                workbuf[511] = '\0';
                
                char fullpath[STRINGMAXLEN_FULLFILENAME];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, dir->d_name);

                char *shm_ext = strstr(workbuf, ".shm");
                if (shm_ext) *shm_ext = '\0';
                char *pid_str = strrchr(workbuf, '.');
                if (pid_str) {
                    *pid_str = '\0';
                    pid_str++;
                    int pid = atoi(pid_str);
                    char *name = workbuf + 5;
                    
                    if (pindex < PROCESSINFOLISTSIZE) {
                        int fd;
                        PROCESSINFO *pinfo = processinfo_shm_link(fullpath, &fd);
                        if (pinfo != (PROCESSINFO *)MAP_FAILED) {
                            // Check if still alive
                            if (kill(pid, 0) == 0) {
                                pinfolist->active[pindex] = 1;
                            } else {
                                if (pinfo->loopstat == 3) pinfolist->active[pindex] = 2; // STOPPED
                                else pinfolist->active[pindex] = 3; // CRASHED
                            }
                            
                            pinfolist->PIDarray[pindex] = pid;
                            strncpy(pinfolist->pnamearray[pindex], name, STRINGMAXLEN_PROCESSINFO_NAME - 1);
                            pinfolist->createtime[pindex] = 1.0 * pinfo->createtime.tv_sec + 1.0e-9 * pinfo->createtime.tv_nsec;
                            
                            // Restore stats in scan_shm
                            scan_shm->pinfodisp[pindex].PID = pid;
                            scan_shm->pinfodisp[pindex].loopcnt = pinfo->loopcnt;
                            strncpy(scan_shm->pinfodisp[pindex].name, pinfo->name, 39);
                            strncpy(scan_shm->pinfodisp[pindex].statusmsg, pinfo->statusmsg, 199);

                            processinfo_shm_close(pinfo, fd);
                            pindex++;
                        }
                    }
                }
            }
        }
        closedir(d);
    }
}

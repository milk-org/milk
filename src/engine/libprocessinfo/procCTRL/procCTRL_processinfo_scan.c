/**
 * @file procCTRL_processinfo_scan.c
 * @brief Procctrl processinfo scan module
 */

#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#include "processinfo.h"
#include "processinfo_internal.h"
#include "milkDebugTools.h"
#include "processtools.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_link.h"
#include "processinfo_shm_close.h"
#include "procCTRL_PIDcollectSystemInfo.h"
#include "procCTRL_GetCPUloads.h"
#include "procCTRL_processinfo_scan.h"
#include "procCTRL_TUI.h"
#include "timeutils.h"
#include "quicksort.h"

// Perform one scan step (update data, sort list)
void processinfo_scan_step(PROCINFOPROC *pinfop)
{
    FILE *flog = NULL;
    if (strlen(procCTRL_logfile) > 0) flog = fopen(procCTRL_logfile, "a");

    if (flog) { fprintf(flog, "  scan_step: start\n"); fflush(flog); }

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    pinfop->scanPID = getpid();

    static int       firstIter = 1;
    static struct    timespec t0;
    struct timespec  t1;
    double           tdiffv;
    struct timespec  tdiff;

    clock_gettime(CLOCK_REALTIME, &t1);
    if(firstIter == 1)
    {
        tdiffv = 0.1;
        firstIter = 0;
    }
    else
    {
        tdiff = timespec_diff(t0, t1);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
    }
    clock_gettime(CLOCK_REALTIME, &t0);
    pinfop->dtscan = tdiffv;

    if (flog) { fprintf(flog, "  scan_step: loop over %d entries\n", PROCESSINFOLISTSIZE); fflush(flog); }

    if (pinfolist == NULL) {
        if (flog) { fprintf(flog, "  scan_step: ERROR pinfolist is NULL\n"); fclose(flog); }
        return;
    }

    for(long pindex = 0; pindex < PROCESSINFOLISTSIZE; pindex++)
    {
        char SM_fname[STRINGMAXLEN_FULLFILENAME];

        pinfop->PIDarray[pindex] = pinfolist->PIDarray[pindex];

        if(pinfolist->active[pindex] == 0)
        {
            pinfop->updatearray[pindex] = 0;
            if(pinfop->pinfommapped[pindex] == 1)
            {
                if (pinfop->pinfoarray[pindex] != NULL)
                    processinfo_shm_close(pinfop->pinfoarray[pindex],
                        pinfop->fdarray[pindex]);
                pinfop->pinfommapped[pindex] = 0;
                pinfop->pinfoarray[pindex] = NULL;
            }
        }
        else
        {
            if(pinfop->pinfommapped[pindex] == 0)
            {
                pinfop->updatearray[pindex] = 1;
            }
        }

        if(pinfop->updatearray[pindex] == 1)
        {
            WRITE_FULLFILENAME(SM_fname, "%s/proc.%s.%06d.shm", procdname, pinfolist->pnamearray[pindex], (int) pinfolist->PIDarray[pindex]);

            pinfop->pinfoarray[pindex] = processinfo_shm_link(SM_fname,
                &pinfop->fdarray[pindex]);
            
            if(pinfop->pinfoarray[pindex] == (PROCESSINFO *) MAP_FAILED)
            {
                pinfop->pinfommapped[pindex] = 0;
                pinfop->pinfoarray[pindex] = NULL;
            }
            else
            {
                pinfop->pinfommapped[pindex] = 1;
                pinfop->updatearray[pindex] = 0;
            }
        }
    }

    pinfop->loopcnt++;
    if (flog) { fprintf(flog, "  scan_step: done\n"); fclose(flog); }
}
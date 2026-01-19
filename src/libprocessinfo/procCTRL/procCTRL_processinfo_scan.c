#include <sys/stat.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "processinfo.h"
#include "processtools.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_link.h"
#include "processinfo_shm_close.h"
#include "procCTRL_PIDcollectSystemInfo.h"
#include "procCTRL_GetCPUloads.h"
#include "procCTRL_processinfo_scan.h"
#include "procCTRL_TUI.h"

extern PROCESSINFOLIST *pinfolist;

void *processinfo_scan(void *thptr)
{
    PROCINFOPROC *pinfop = (PROCINFOPROC *) thptr;
    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    pinfop->scanPID = getpid();

    while(pinfop->loop == 1)
    {
        pinfop->SCANBLOCK_requested = 1;
        while(pinfop->SCANBLOCK_OK == 0)
        {
            usleep(100);
            if(pinfop->loop == 0) pthread_exit(NULL);
        }
        pinfop->SCANBLOCK_requested = 0;

        pinfop->NBpindexActive = 0;
        if (pinfolist != NULL) {
            for (long i = 0; i < PROCESSINFOLISTSIZE; i++) {
                if (pinfolist->active[i] != 0) {
                    pinfop->pindexActive[pinfop->NBpindexActive++] = i;
                }
            }
        }

        pinfop->SCANBLOCK_OK = 0;

        // Connect to active SHMs
        for (int i = 0; i < pinfop->NBpindexActive; i++) {
            long idx = pinfop->pindexActive[i];
            if (pinfop->pinfommapped[idx] == 0) {
                char SM_fname[256];
                snprintf(SM_fname, 256, "%s/proc.%s.%06d.shm", procdname, pinfolist->pnamearray[idx], pinfolist->PIDarray[idx]);
                pinfop->pinfoarray[idx] = processinfo_shm_link(SM_fname, &pinfop->fdarray[idx]);
                if (pinfop->pinfoarray[idx] != MAP_FAILED) {
                    pinfop->pinfommapped[idx] = 1;
                }
            }
        }

        usleep(pinfop->twaitus);
    }
    return NULL;
}
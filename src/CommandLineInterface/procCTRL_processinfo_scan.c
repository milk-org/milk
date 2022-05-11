#include <sys/stat.h>
#include <pthread.h>
#include <sys/mman.h> // mmap()

#include <ncurses.h>

#include "CLIcore.h"
#include <processtools.h>

#include "CommandLineInterface/timeutils.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "procCTRL_PIDcollectSystemInfo.h"
#include "procCTRL_GetCPUloads.h"

#include "processinfo_procdirname.h"


#include "procCTRL_TUI.h"


extern PROCESSINFOLIST *pinfolist;


/**
 * ## Purpose
 *
 * Scan function for processinfo CTRL
 *
 * ## Description
 *
 * Runs in background loop as thread initiated by processinfo_CTRL
 *
 */
void *processinfo_scan(void *thptr)
{
    PROCINFOPROC *pinfop;

    pinfop = (PROCINFOPROC *) thptr;

    long pindex;
    long pindexdisp;

    pinfop->loopcnt = 0;

    // timing
    static int             firstIter = 1;
    static struct timespec t0;
    struct timespec        t1;
    double                 tdiffv;
    struct timespec        tdiff;

    char procdname[200];
    processinfo_procdirname(procdname);

    pinfop->scanPID = getpid();

    pinfop->scandebugline = __LINE__;



    long loopcnt = 0;
    while (pinfop->loop == 1)
    {

        DEBUG_TRACEPOINT(" ");

        pinfop->scandebugline = __LINE__;

        DEBUG_TRACEPOINT(" ");

        // timing measurement
        clock_gettime(CLOCK_REALTIME, &t1);
        if (firstIter == 1)
        {
            tdiffv    = 0.1;
            firstIter = 0;
        }
        else
        {
            tdiff  = timespec_diff(t0, t1);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        }
        clock_gettime(CLOCK_REALTIME, &t0);
        pinfop->dtscan = tdiffv;

        DEBUG_TRACEPOINT(" ");

        pinfop->scandebugline = __LINE__;

        pinfop->SCANBLOCK_requested = 1; // request scan
        //system("echo \"scanblock request write 1\" > steplog.sRQw1.txt");//TEST

        while (pinfop->SCANBLOCK_OK == 0) // wait for display to OK scan
        {
            //system("echo \"scanblock OK read 0\" > steplog.sOKr0.txt");//TEST
            usleep(100);
            pinfop->scandebugline = __LINE__;
            if (pinfop->loop == 0)
            {
                int line = __LINE__;
                pthread_exit(&line);
            }
        }
        pinfop->SCANBLOCK_requested =
            0; // acknowledge that request has been granted
        //system("echo \"scanblock request write 0\" > steplog.sRQw0.txt");//TEST

        DEBUG_TRACEPOINT(" ");

        // LOAD / UPDATE process information
        // This step re-mmaps pinfo and rebuilds list, so we need to ensure it is run exclusively of the dislpay
        //
        pinfop->scandebugline = __LINE__;

        DEBUG_TRACEPOINT(" ");

        for (pindex = 0; pindex < pinfop->NBpinfodisp; pindex++)
        {

            if (pinfop->loop == 1)
            {

                DEBUG_TRACEPOINT("pindex %ld / %ld",
                                 pindex,
                                 pinfop->NBpinfodisp);

                // shared memory file name
                char SM_fname
                    [STRINGMAXLEN_FULLFILENAME]; // shared memory file name
                struct stat file_stat;

                pinfop->scandebugline    = __LINE__;
                pinfop->PIDarray[pindex] = pinfolist->PIDarray[pindex];
                DEBUG_TRACEPOINT("pindex %ld / %ld",
                                 pindex,
                                 pinfop->NBpinfodisp);

                // SHOULD WE (RE)LOAD ?
                if (pinfolist->active[pindex] == 0) // inactive
                {
                    pinfop->updatearray[pindex] = 0;
                }

                if ((pinfolist->active[pindex] == 1) ||
                    (pinfolist->active[pindex] == 2)) // active or crashed
                {
                    pinfop->updatearray[pindex] = 1;
                }
                //    if(pinfolist->active[pindex] == 2) // mmap crashed, file may still be present
                //        updatearray[pindex] = 1;

                if (pinfolist->active[pindex] == 3) // file has gone away
                {
                    pinfop->updatearray[pindex] = 0;
                }

                DEBUG_TRACEPOINT(" ");

                pinfop->scandebugline = __LINE__;

                // check if process info file exists
                WRITE_FULLFILENAME(SM_fname,
                                   "%s/proc.%s.%06d.shm",
                                   procdname,
                                   pinfolist->pnamearray[pindex],
                                   (int) pinfolist->PIDarray[pindex]);

                // Does file exist ?
                if (stat(SM_fname, &file_stat) == -1 && errno == ENOENT)
                {
                    // if not, don't (re)load and remove from process info list
                    pinfolist->active[pindex]   = 0;
                    pinfop->updatearray[pindex] = 0;
                }

                DEBUG_TRACEPOINT(" ");

                if (pinfolist->active[pindex] == 1)
                {
                    // check if process still exists
                    struct stat sts;
                    char        procfname[STRINGMAXLEN_FULLFILENAME];

                    WRITE_FULLFILENAME(procfname,
                                       "/proc/%d",
                                       (int) pinfolist->PIDarray[pindex]);
                    if (stat(procfname, &sts) == -1 && errno == ENOENT)
                    {
                        // process doesn't exist -> flag as inactive
                        pinfolist->active[pindex] = 2;
                    }
                }



                DEBUG_TRACEPOINT(" ");

                pinfop->scandebugline = __LINE__;

                if ((pindex < pinfop->NBpinfodisp) &&
                    (pinfop->updatearray[pindex] == 1))
                {
                    // (RE)LOAD
                    //struct stat file_stat;




                    DEBUG_TRACEPOINT(" ");

                    // if already mmapped, first unmap
                    if (pinfop->pinfommapped[pindex] == 1)
                    {
                        processinfo_shm_close(pinfop->pinfoarray[pindex],
                                              pinfop->fdarray[pindex]);
                        pinfop->pinfommapped[pindex] = 0;
                    }

                    DEBUG_TRACEPOINT(" ");

                    // COLLECT INFORMATION FROM PROCESSINFO FILE
                    pinfop->pinfoarray[pindex] =
                        processinfo_shm_link(SM_fname,
                                             &pinfop->fdarray[pindex]);

                    if (pinfop->pinfoarray[pindex] == MAP_FAILED)
                    {
                        close(pinfop->fdarray[pindex]);
                        endwin();
                        fprintf(stderr,
                                "[%d] Error mapping file %s\n",
                                __LINE__,
                                SM_fname);
                        pinfolist->active[pindex]    = 3;
                        pinfop->pinfommapped[pindex] = 0;
                    }
                    else
                    {
                        pinfop->pinfommapped[pindex] = 1;
                        strncpy(pinfop->pinfodisp[pindex].name,
                                pinfop->pinfoarray[pindex]->name,
                                40 - 1);

                        struct tm *createtm;
                        createtm = gmtime(
                            &pinfop->pinfoarray[pindex]->createtime.tv_sec);
                        pinfop->pinfodisp[pindex].createtime_hr =
                            createtm->tm_hour;
                        pinfop->pinfodisp[pindex].createtime_min =
                            createtm->tm_min;
                        pinfop->pinfodisp[pindex].createtime_sec =
                            createtm->tm_sec;
                        pinfop->pinfodisp[pindex].createtime_ns =
                            pinfop->pinfoarray[pindex]->createtime.tv_nsec;

                        pinfop->pinfodisp[pindex].loopcnt =
                            pinfop->pinfoarray[pindex]->loopcnt;
                    }

                    DEBUG_TRACEPOINT(" ");

                    pinfop->pinfodisp[pindex].active =
                        pinfolist->active[pindex];
                    pinfop->pinfodisp[pindex].PID = pinfolist->PIDarray[pindex];

                    pinfop->pinfodisp[pindex].updatecnt++;

                    // pinfop->updatearray[pindex] == 0; // by default, no need to re-connect

                    DEBUG_TRACEPOINT(" ");
                }

                pinfop->scandebugline = __LINE__;
            }
            else
            {
                int line = __LINE__;
                pthread_exit(&line);
            }
        }


        /** ### Build a time-sorted list of processes
          *
          *
          *
          */
        int index;

        DEBUG_TRACEPOINT(" ");

        pinfop->NBpindexActive = 0;
        for (pindex = 0; pindex < PROCESSINFOLISTSIZE; pindex++)
            if (pinfolist->active[pindex] != 0)
            {
                pinfop->pindexActive[pinfop->NBpindexActive] = pindex;
                pinfop->NBpindexActive++;
            }



        if (pinfop->NBpindexActive > 0)
        {

            double *timearray;
            long   *indexarray;
            timearray =
                (double *) malloc(sizeof(double) * pinfop->NBpindexActive);
            if (timearray == NULL)
            {
                PRINT_ERROR("malloc returns NULL pointer");
                abort();
            }
            indexarray = (long *) malloc(sizeof(long) * pinfop->NBpindexActive);
            if (indexarray == NULL)
            {
                PRINT_ERROR("malloc returns NULL pointer");
                abort();
            }


            int listcnt = 0;
            for (index = 0; index < pinfop->NBpindexActive; index++)
            {
                pindex = pinfop->pindexActive[index];
                if (pinfop->pinfommapped[pindex] == 1)
                {
                    indexarray[index] = pindex;
                    // minus sign for most recent first
                    //TUI_printfw("index  %ld  ->  pindex  %ld\n", index, pindex);
                    timearray[index] =
                        -1.0 * pinfop->pinfoarray[pindex]->createtime.tv_sec -
                        1.0e-9 * pinfop->pinfoarray[pindex]->createtime.tv_nsec;
                    listcnt++;
                }
            }
            DEBUG_TRACEPOINT(" ");



            pinfop->NBpindexActive = listcnt;



            if (pinfop->NBpindexActive > 0)
            {
                quick_sort2l_double(timearray,
                                    indexarray,
                                    pinfop->NBpindexActive);
            }


            for (index = 0; index < pinfop->NBpindexActive; index++)
            {
                pinfop->sorted_pindex_time[index] = indexarray[index];
            }

            DEBUG_TRACEPOINT(" ");

            free(timearray);
            free(indexarray);
        }



        pinfop->scandebugline = __LINE__;

        pinfop->SCANBLOCK_OK = 0; // let display thread we're done
        //system("echo \"scanblock OK write 0\" > steplog.sOKw0.txt");//TEST

        pinfop->scandebugline = __LINE__;

        DEBUG_TRACEPOINT(" ");




        if (pinfop->DisplayMode ==
            PROCCTRL_DISPLAYMODE_RESOURCES) // only compute of displayed processes
        {
            DEBUG_TRACEPOINT(" ");
            pinfop->scandebugline = __LINE__;
            GetCPUloads(pinfop);
            pinfop->scandebugline = __LINE__;
            // collect required info for display
            for (pindexdisp = 0; pindexdisp < pinfop->NBpinfodisp; pindexdisp++)
            {
                if (pinfop->loop == 1)
                {
                    DEBUG_TRACEPOINT(" ");

                    if (pinfolist->active[pindexdisp] != 0)
                    {
                        pinfop->scandebugline = __LINE__;

                        if (pinfop->pinfodisp[pindexdisp].NBsubprocesses !=
                            0) // pinfop->pinfodisp[pindex].NBsubprocesses should never be zero - should be at least 1 (for main process)
                        {

                            int spindex; // sub process index, 0 for main

                            if (pinfop->psysinfostatus[pindexdisp] != -1)
                            {
                                for (spindex = 0;
                                     spindex < pinfop->pinfodisp[pindexdisp]
                                                   .NBsubprocesses;
                                     spindex++)
                                {
                                    // place info in subprocess arrays
                                    pinfop->pinfodisp[pindexdisp]
                                        .sampletimearray_prev[spindex] =
                                        pinfop->pinfodisp[pindexdisp]
                                            .sampletimearray[spindex];
                                    // Context Switches

                                    pinfop->pinfodisp[pindexdisp]
                                        .ctxtsw_voluntary_prev[spindex] =
                                        pinfop->pinfodisp[pindexdisp]
                                            .ctxtsw_voluntary[spindex];
                                    pinfop->pinfodisp[pindexdisp]
                                        .ctxtsw_nonvoluntary_prev[spindex] =
                                        pinfop->pinfodisp[pindexdisp]
                                            .ctxtsw_nonvoluntary[spindex];

                                    // CPU use
                                    pinfop->pinfodisp[pindexdisp]
                                        .cpuloadcntarray_prev[spindex] =
                                        pinfop->pinfodisp[pindexdisp]
                                            .cpuloadcntarray[spindex];
                                }
                            }

                            pinfop->scandebugline = __LINE__;

                            pinfop->psysinfostatus[pindex] =
                                PIDcollectSystemInfo(
                                    &(pinfop->pinfodisp[pindexdisp]),
                                    0);

                            if (pinfop->psysinfostatus[pindexdisp] != -1)
                            {
                                char cpuliststring[200];
                                char cpustring[16];

                                for (spindex = 0;
                                     spindex < pinfop->pinfodisp[pindexdisp]
                                                   .NBsubprocesses;
                                     spindex++)
                                {
                                    if (pinfop->pinfodisp[pindexdisp]
                                            .sampletimearray[spindex] !=
                                        pinfop->pinfodisp[pindexdisp]
                                            .sampletimearray_prev[spindex])
                                    {
                                        // get CPU and MEM load

                                        // THIS DOES NOT WORK ON TICKLESS KERNEL
                                        pinfop->pinfodisp[pindexdisp]
                                            .subprocCPUloadarray[spindex] =
                                            100.0 *
                                            ((1.0 *
                                                  pinfop->pinfodisp[pindexdisp]
                                                      .cpuloadcntarray
                                                          [spindex] -
                                              pinfop->pinfodisp[pindexdisp]
                                                  .cpuloadcntarray_prev
                                                      [spindex]) /
                                             sysconf(_SC_CLK_TCK)) /
                                            (pinfop->pinfodisp[pindexdisp]
                                                 .sampletimearray[spindex] -
                                             pinfop->pinfodisp[pindexdisp]
                                                 .sampletimearray_prev
                                                     [spindex]);

                                        pinfop->pinfodisp[pindexdisp]
                                            .subprocCPUloadarray_timeaveraged
                                                [spindex] =
                                            0.9 *
                                                pinfop->pinfodisp[pindexdisp]
                                                    .subprocCPUloadarray_timeaveraged
                                                        [spindex] +
                                            0.1 * pinfop->pinfodisp[pindexdisp]
                                                      .subprocCPUloadarray
                                                          [spindex];
                                    }
                                }

                                sprintf(
                                    cpuliststring,
                                    ",%s,",
                                    pinfop->pinfodisp[pindexdisp].cpusallowed);

                                pinfop->scandebugline = __LINE__;

                                int cpu;
                                for (cpu = 0; cpu < pinfop->NBcpus; cpu++)
                                {
                                    int cpuOK = 0;
                                    int cpumin, cpumax;

                                    sprintf(cpustring,
                                            ",%d,",
                                            pinfop->CPUids[cpu]);
                                    if (strstr(cpuliststring, cpustring) !=
                                        NULL)
                                    {
                                        cpuOK = 1;
                                    }

                                    for (cpumin = 0;
                                         cpumin <= pinfop->CPUids[cpu];
                                         cpumin++)
                                        for (cpumax = pinfop->CPUids[cpu];
                                             cpumax < pinfop->NBcpus;
                                             cpumax++)
                                        {
                                            sprintf(cpustring,
                                                    ",%d-%d,",
                                                    cpumin,
                                                    cpumax);
                                            if (strstr(cpuliststring,
                                                       cpustring) != NULL)
                                            {
                                                cpuOK = 1;
                                            }
                                        }
                                    pinfop->pinfodisp[pindexdisp]
                                        .cpuOKarray[cpu] = cpuOK;
                                }
                            }
                        }
                    }
                }
                else
                {
                    DEBUG_TRACEPOINT(" ");
                    int line = __LINE__;
                    pthread_exit(&line);
                }
            } // end of if(pinfop->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES)

            pinfop->scandebugline = __LINE__;

        } // end of DisplayMode PROCCTRL_DISPLAYMODE_RESOURCES

        DEBUG_TRACEPOINT(" ");

        pinfop->loopcnt++;

        int loopcntiter   = 0;
        int NBloopcntiter = 10;
        while ((pinfop->loop == 1) && (loopcntiter < NBloopcntiter))
        {
            usleep(pinfop->twaitus / NBloopcntiter);
            loopcntiter++;
        }

        if (pinfop->loop == 0)
        {
            int line = __LINE__;
            pthread_exit(&line);
        }
    }

    if (pinfop->loop == 0)
    {
        int line = __LINE__;
        pthread_exit(&line);
    }

    return NULL;
}

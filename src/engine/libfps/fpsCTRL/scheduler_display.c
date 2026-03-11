/**
 * @file scheduler_display.c
 * @brief Scheduler display module
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ncurses.h>
#include <time.h>

#include "fps.h"
#include "fps_internal.h"
#include "TUItools.h"
#include "scheduler_display.h"
#include "quicksort.h"

errno_t fpsCTRL_scheduler_display(
    FPSCTRL_TASK_ENTRY *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    int                 wrow,
    int                *wrowstart
)
{
    struct timespec tnow;
    struct timespec tdiff;

    clock_gettime(CLOCK_REALTIME, &tnow);

    // Sort entries from most recent to most ancient, using inputindex
    double *sort_evalarray = (double *) malloc(sizeof(double) * NB_FPSCTRL_TASK_MAX);
    long *sort_indexarray = (long *) malloc(sizeof(long) * NB_FPSCTRL_TASK_MAX);

    long sortcnt = 0;
    for(int fpscmdindex = 0; fpscmdindex < NB_FPSCTRL_TASK_MAX; fpscmdindex++)
    {
        if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_SHOW)
        {
            sort_evalarray[sortcnt] = -1.0 * fpsctrltasklist[fpscmdindex].inputindex;
            sort_indexarray[sortcnt] = fpscmdindex;
            sortcnt++;
        }
    }

    if(sortcnt > 0)
    {
        quick_sort2l(sort_evalarray, sort_indexarray, sortcnt);
    }
    free(sort_evalarray);

    TUI_printfw(" showing   %d / %d  tasks\n", wrow - 8, (int)sortcnt);

    for(int sortindex = 0; sortindex < sortcnt; sortindex++)
    {
        int fpscmdindex = sort_indexarray[sortindex];

        if(sortindex < wrow - 8)   // display
        {
            int attron2 = 0;
            int attrbold = 0;

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RUNNING)
            {
                attron2 = 1;
                screenprint_setcolor(COLOR_OK);
            }
            else if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE)
            {
                attrbold = 1;
                screenprint_setbold();
            }

            // measure age since submission
            tdiff =  timespec_diff(fpsctrltasklist[fpscmdindex].creationtime,
                tnow);
            double tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
            TUI_printfw("%6.2f s ", tdiffv);

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RUNNING)
            {
                tdiff =  timespec_diff(
                    fpsctrltasklist[fpscmdindex].activationtime,
                    tnow);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                TUI_printfw(" %6.2f s ", tdiffv);
            }
            else if(!(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE))
            {
                tdiff =  timespec_diff(fpsctrltasklist[fpscmdindex].activationtime,
                                       fpsctrltasklist[fpscmdindex].completiontime);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                screenprint_setcolor(COLOR_WARNING);
                TUI_printfw(" %6.2f s ", tdiffv);
                screenprint_unsetcolor(COLOR_WARNING);
            }
            else
            {
                TUI_printfw("          ");
            }

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE) TUI_printfw(">> ");
            else TUI_printfw("  ");

            if(fpsctrltasklist[fpscmdindex].flag & FPSTASK_FLAG_WAITONRUN) TUI_printfw("WR ");
            else TUI_printfw("   ");

            if(fpsctrltasklist[fpscmdindex].flag & FPSTASK_FLAG_WAITONCONF) TUI_printfw("WC ");
            else TUI_printfw("   ");

            TUI_printfw("[Q:%02d P:%02d] %4d",
                    fpsctrltasklist[fpscmdindex].queue,
                    fpsctrlqueuelist[fpsctrltasklist[fpscmdindex].queue].priority,
                    fpscmdindex);

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RECEIVED) TUI_printfw(" R");
            else TUI_printfw(" -");

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_CMDNOTFOUND)
            {
                screenprint_setcolor(COLOR_WARNING);
                TUI_printfw(" NOTCMD");
                screenprint_unsetcolor(COLOR_WARNING);
            }
            else if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_CMDFAIL)
            {
                screenprint_setcolor(COLOR_ERROR);
                TUI_printfw(" FAILED");
                screenprint_unsetcolor(COLOR_ERROR);
            }
            else if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_CMDOK)
            {
                screenprint_setcolor(COLOR_OK);
                TUI_printfw(" PROCOK");
                screenprint_unsetcolor(COLOR_OK);
            }
            else if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RECEIVED)
            {
                screenprint_setcolor(COLOR_OK);
                TUI_printfw(" RECVD ");
                screenprint_unsetcolor(COLOR_OK);
            }
            else if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_WAITING)
            {
                screenprint_setcolor(5);
                TUI_printfw("WAITING");
                screenprint_unsetcolor(5);
            }
            else
            {
                screenprint_setcolor(COLOR_WARNING);
                TUI_printfw(" ????  ");
                screenprint_unsetcolor(COLOR_WARNING);
            }

            TUI_printfw("  %s\n", fpsctrltasklist[fpscmdindex].cmdstring);

            if(attron2 == 1) screenprint_unsetcolor(COLOR_OK);
            if(attrbold == 1) screenprint_unsetbold();
        }
    }
    free(sort_indexarray);

    return RETURN_SUCCESS;
}

#include "CommandLineInterface/CLIcore.h"

#include "CommandLineInterface/timeutils.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "TUItools.h"



errno_t fpsCTRL_scheduler_display(
    FPSCTRL_TASK_ENTRY *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    int                 wrow,
    int                *firstrow
)
{

    struct timespec tnow;
    struct timespec tdiff;

    clock_gettime(CLOCK_REALTIME, &tnow);


    // Sort entries from most recent to most ancient, using inputindex
    DEBUG_TRACEPOINT(" ");

    double *sort_evalarray;
    sort_evalarray = (double *) malloc(sizeof(double) * NB_FPSCTRL_TASK_MAX);
    if(sort_evalarray == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    long *sort_indexarray;
    sort_indexarray = (long *) malloc(sizeof(long) * NB_FPSCTRL_TASK_MAX);
    if(sort_indexarray == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    long sortcnt = 0;
    for(int fpscmdindex = 0; fpscmdindex < NB_FPSCTRL_TASK_MAX; fpscmdindex++)
    {
        if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_SHOW)
        {
            sort_evalarray[sortcnt] =
                -1.0 * fpsctrltasklist[fpscmdindex].inputindex;
            sort_indexarray[sortcnt] = fpscmdindex;
            sortcnt++;
        }
    }
    DEBUG_TRACEPOINT(" ");
    if(sortcnt > 0)
    {
        quick_sort2l(sort_evalarray, sort_indexarray, sortcnt);
    }
    free(sort_evalarray);

    DEBUG_TRACEPOINT(" ");

    if(*firstrow < 0)
    {
        *firstrow = 0;
    }
    if(*firstrow > (sortcnt - (wrow - 8)))
    {
        *firstrow = sortcnt - (wrow - 8);
    }
    TUI_printfw(" showing   %5d / %5d  starting at %d", wrow - 8, sortcnt,
                *firstrow);
    TUI_newline();

    for(int sortindex = 0; sortindex < sortcnt; sortindex++)
    {

        DEBUG_TRACEPOINT("iteration %d / %ld", sortindex, sortcnt);

        int fpscmdindex = sort_indexarray[sortindex];

        DEBUG_TRACEPOINT("fpscmdindex = %d", fpscmdindex);

        if((sortindex - (*firstrow) < wrow - 8)
                && (sortindex >= *firstrow))    // display
        {
            int attron2  = 0;
            int attrbold = 0;

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RUNNING)
            {
                // task is running
                attron2 = 1;
                screenprint_setcolor(COLOR_OK);
            }
            else if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE)
            {
                // task is queued to run
                attrbold = 1;
                screenprint_setbold();
            }

            // measure age since submission
            tdiff = timespec_diff(fpsctrltasklist[fpscmdindex].creationtime, tnow);
            double tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
            TUI_printfw("%6.2f s ", tdiffv);

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RUNNING)
            {
                // run time (ongoing)
                tdiff = timespec_diff(fpsctrltasklist[fpscmdindex].activationtime, tnow);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                TUI_printfw(" %6.2f s ", tdiffv);
            }
            else if(!(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE))
            {
                // run time (past)
                tdiff = timespec_diff(fpsctrltasklist[fpscmdindex].activationtime,
                                      fpsctrltasklist[fpscmdindex].completiontime);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                screenprint_setcolor(COLOR_WARNING);
                TUI_printfw(" %6.2f s ", tdiffv);
                screenprint_unsetcolor(COLOR_WARNING);
                // age since completion
                tdiff = timespec_diff(fpsctrltasklist[fpscmdindex].completiontime, tnow);
                double tdiffv = tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
            }
            else
            {
                TUI_printfw("          ", tdiffv);
            }

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE)
            {
                TUI_printfw(">>");
            }
            else
            {
                TUI_printfw("  ");
            }

            if(fpsctrltasklist[fpscmdindex].flag & FPSTASK_FLAG_WAITONRUN)
            {
                TUI_printfw("WR ");
            }
            else
            {
                TUI_printfw("   ");
            }

            if(fpsctrltasklist[fpscmdindex].flag & FPSTASK_FLAG_WAITONCONF)
            {
                TUI_printfw("WC ");
            }
            else
            {
                TUI_printfw("   ");
            }

            TUI_printfw(
                "[Q:%02d P:%02d] %4d",
                fpsctrltasklist[fpscmdindex].queue,
                fpsctrlqueuelist[fpsctrltasklist[fpscmdindex].queue].priority,
                fpscmdindex);

            if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RECEIVED)
            {
                TUI_printfw(" R");
            }
            else
            {
                TUI_printfw(" -");
            }

            if(fpsctrltasklist[fpscmdindex].status &
                    FPSTASK_STATUS_CMDNOTFOUND)
            {
                screenprint_setcolor(3);
                TUI_printfw(" NOTCMD");
                screenprint_unsetcolor(3);
            }
            else if(fpsctrltasklist[fpscmdindex].status &
                    FPSTASK_STATUS_CMDFAIL)
            {
                screenprint_setcolor(4);
                TUI_printfw(" FAILED");
                if(fpsctrltasklist[fpscmdindex].status &
                        FPSTASK_STATUS_ERR_NBARG)
                {
                    TUI_printfw(" NBARG");
                }
                if(fpsctrltasklist[fpscmdindex].status &
                        FPSTASK_STATUS_ERR_ARGTYPE)
                {
                    TUI_printfw(" ARGTYPE");
                }
                if(fpsctrltasklist[fpscmdindex].status &
                        FPSTASK_STATUS_ERR_TYPECONV)
                {
                    TUI_printfw(" TYPECOV");
                }
                if(fpsctrltasklist[fpscmdindex].status &
                        FPSTASK_STATUS_ERR_NOFPS)
                {
                    TUI_printfw(" NOFPS");
                }
                screenprint_unsetcolor(4);
            }
            else if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_CMDOK)
            {
                screenprint_setcolor(2);
                TUI_printfw(" PROCOK");
                screenprint_unsetcolor(2);
            }
            else if(fpsctrltasklist[fpscmdindex].status &
                    FPSTASK_STATUS_RECEIVED)
            {
                screenprint_setcolor(2);
                TUI_printfw(" RECVD ");
                screenprint_unsetcolor(2);
            }
            else if(fpsctrltasklist[fpscmdindex].status &
                    FPSTASK_STATUS_WAITING)
            {
                screenprint_setcolor(5);
                TUI_printfw("WAITING");
                screenprint_unsetcolor(5);
            }
            else
            {
                screenprint_setcolor(3);
                TUI_printfw(" ????  ");
                screenprint_unsetcolor(3);
            }

            TUI_printfw("  %s", fpsctrltasklist[fpscmdindex].cmdstring);
            TUI_newline();

            if(attron2 == 1)
            {
                screenprint_unsetcolor(2);
            }
            if(attrbold == 1)
            {
                screenprint_unsetbold();
            }
        }
    }
    free(sort_indexarray);

    return RETURN_SUCCESS;
}

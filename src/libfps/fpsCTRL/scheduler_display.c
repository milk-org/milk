#include <stdio.h>
#include <string.h>
#include <ncurses.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_TUI_shim.h"
#include "scheduler_display.h"

errno_t fpsCTRL_scheduler_display(
    FPSCTRL_TASK_ENTRY *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    int                 wrow,
    int                *wrowstart
)
{
    // Simple display of the scheduler queues
    
    TUI_printfw("\nSCHEDULER QUEUES:\n");
    for(int q=0; q<NB_FPSCTRL_TASKQUEUE_MAX; q++) {
        if(fpsctrlqueuelist[q].priority > 0) {
            TUI_printfw("  Q%02d : Prio %d\n", q, fpsctrlqueuelist[q].priority);
        }
    }

    TUI_printfw("\nTASKS:\n");
    int count = 0;
    for(int i=0; i<NB_FPSCTRL_TASK_MAX; i++) {
        if(fpsctrltasklist[i].status & FPSTASK_STATUS_ACTIVE) {
            if (count < wrow - 10) { // Simple limit
                TUI_printfw("  [%03d] Q%02d Status %04llx : %s\n", 
                    i, fpsctrltasklist[i].queue, 
                    (unsigned long long)fpsctrltasklist[i].status, 
                    fpsctrltasklist[i].cmdstring);
                count++;
            }
        }
    }

    return RETURN_SUCCESS;
}
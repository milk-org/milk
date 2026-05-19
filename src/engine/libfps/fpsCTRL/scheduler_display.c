/**
 * @file scheduler_display.c
 * @brief Scheduler display module
 */


#include "fps.h"
#include "fpsCTRL_TUIcompat.h"
#include "engine/libfpsseq/fpsseq.h"
#include "quicksort.h"

/**
 * @brief Render the FPS scheduler queue display.
 *
 * Shows pending and active tasks in each priority
 * queue with timing information.
 */
errno_t fpsCTRL_scheduler_display(
    FPSCTRL_PROCESS_VARS *fpsCTRLvar,
    int wrow,
    int *wrowstart __attribute__((unused)))
{
    struct timespec tnow;
    struct timespec tdiff;

    clock_gettime(CLOCK_REALTIME, &tnow);

    if(fpsCTRLvar->milkseq_state == NULL)
    {
        char names[1][FPSSEQ_NAME_MAX];
        if(milkseq_list(names, 1) > 0)
        {
            fpsCTRLvar->milkseq_state = milkseq_connect(names[0]);
            if(fpsCTRLvar->milkseq_state)
            {
                strncpy(fpsCTRLvar->milkseq_name,
                        names[0],
                        sizeof(fpsCTRLvar->milkseq_name)
                        - 1);
                fpsCTRLvar->milkseq_name[
                sizeof(fpsCTRLvar->milkseq_name)
                - 1] = '\0';
            }
        }
    }

    if(fpsCTRLvar->milkseq_state == NULL)
    {
        TUI_printfw(" \n No sequencer connected.\n");
        return RETURN_SUCCESS;
    }

    MILKSEQ_STATE *state = (MILKSEQ_STATE *) fpsCTRLvar->milkseq_state;

    // Lazy-allocate persistent sort buffers
    long need_cap = (long) state->NBtasks_max;
    if(need_cap > fpsCTRLvar->sched_sort_cap)
    {
        free(fpsCTRLvar->sched_sort_eval);
        free(fpsCTRLvar->sched_sort_index);
        fpsCTRLvar->sched_sort_eval =
            (double *) malloc(sizeof(double) * need_cap);
        fpsCTRLvar->sched_sort_index =
            (long *) malloc(sizeof(long) * need_cap);
        fpsCTRLvar->sched_sort_cap = need_cap;
    }
    double *sort_evalarray = fpsCTRLvar->sched_sort_eval;
    long *sort_indexarray = fpsCTRLvar->sched_sort_index;

    if(sort_evalarray == NULL
            || sort_indexarray == NULL)
    {
        TUI_printfw(" \n Sort buffer alloc error.\n");
        return RETURN_FAILURE;
    }

    long sortcnt = 0;
    for(int fpscmd_idx = 0; fpscmd_idx < (int)state->NBtasks_max; fpscmd_idx++)
    {
        if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_SHOW)
        {
            sort_evalarray[sortcnt] = -1.0 * state->tasklist[fpscmd_idx].inputindex;
            sort_indexarray[sortcnt] = fpscmd_idx;
            sortcnt++;
        }
    }

    if(sortcnt > 0)
    {
        quick_sort2l(sort_evalarray, sort_indexarray, sortcnt);
    }

    TUI_printfw(" showing   %d / %d  tasks   [ Sequencer: %s ]\n", wrow - 8, (int)sortcnt, state->name);

    for(int sort_idx = 0; sort_idx < sortcnt; sort_idx++)
    {
        int fpscmd_idx = sort_indexarray[sort_idx];

        if(sort_idx < wrow - 8)   // display
        {
            int attron2 = 0;
            int attrbold = 0;

            if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_RUNNING)
            {
                attron2 = 1;
                screenprint_setcolor(COLOR_OK);
            }
            else if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_ACTIVE)
            {
                attrbold = 1;
                screenprint_setbold();
            }

            // measure age since submission
            tdiff =  timespec_diff(state->tasklist[fpscmd_idx].creationtime,
                                   tnow);
            double tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
            TUI_printfw("%6.2f s ", tdiffv);

            if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_RUNNING)
            {
                tdiff =  timespec_diff(
                             state->tasklist[fpscmd_idx].activationtime,
                             tnow);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                TUI_printfw(" %6.2f s ", tdiffv);
            }
            else if(!(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_ACTIVE))
            {
                tdiff =  timespec_diff(state->tasklist[fpscmd_idx].activationtime,
                                       state->tasklist[fpscmd_idx].completiontime);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                screenprint_setcolor(COLOR_WARNING);
                TUI_printfw(" %6.2f s ", tdiffv);
                screenprint_unsetcolor(COLOR_WARNING);
            }
            else
            {
                TUI_printfw("          ");
            }

            if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_ACTIVE)
            {
                TUI_printfw(">> ");
            }
            else
            {
                TUI_printfw("  ");
            }

            if(state->tasklist[fpscmd_idx].flag & FPSTASK_FLAG_WAITONRUN)
            {
                TUI_printfw("WR ");
            }
            else
            {
                TUI_printfw("   ");
            }

            if(state->tasklist[fpscmd_idx].flag & FPSTASK_FLAG_WAITONCONF)
            {
                TUI_printfw("WC ");
            }
            else
            {
                TUI_printfw("   ");
            }

            TUI_printfw("[Q:%02d P:%02d] %4d",
                        state->tasklist[fpscmd_idx].queue,
                        state->queuelist[state->tasklist[fpscmd_idx].queue].priority,
                        fpscmd_idx);

            if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_RECEIVED)
            {
                TUI_printfw(" R");
            }
            else
            {
                TUI_printfw(" -");
            }

            if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_CMDNOTFOUND)
            {
                screenprint_setcolor(COLOR_WARNING);
                TUI_printfw(" NOTCMD");
                screenprint_unsetcolor(COLOR_WARNING);
            }
            else if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_CMDFAIL)
            {
                screenprint_setcolor(COLOR_ERROR);
                TUI_printfw(" FAILED");
                screenprint_unsetcolor(COLOR_ERROR);
            }
            else if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_CMDOK)
            {
                screenprint_setcolor(COLOR_OK);
                TUI_printfw(" PROCOK");
                screenprint_unsetcolor(COLOR_OK);
            }
            else if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_RECEIVED)
            {
                screenprint_setcolor(COLOR_OK);
                TUI_printfw(" RECVD ");
                screenprint_unsetcolor(COLOR_OK);
            }
            else if(state->tasklist[fpscmd_idx].status & FPSTASK_STATUS_WAITING)
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

            TUI_printfw("  %s\n", state->tasklist[fpscmd_idx].cmdstring);

            if(attron2 == 1)
            {
                screenprint_unsetcolor(COLOR_OK);
            }
            if(attrbold == 1)
            {
                screenprint_unsetbold();
            }
        }
    }

    return RETURN_SUCCESS;
}

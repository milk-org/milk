/**
 * @file fpsseq_scheduler.c
 * @brief Task scheduling logic for the milk-seq FPS Sequencer
 *
 * Scans task queues, checks dependencies (WAITONRUN, WAITONCONF),
 * and dispatches commands to the execution engine.
 */

#include <limits.h>


#include "fpsseq.h"

#include "timeutils.h"

/**
 * milkseq_scheduler_step - Run one iteration of the task scheduler
 * @state:    Sequencer state mapped from SHM
 * @fps:      Array of all FPS entries
 * @keywnode: Keyword tree root
 * @vars:     TUI-level process variables (exitloop, logging)
 *
 * Three-phase scheduling algorithm:
 *   1. Scan each queue for its oldest active task. If the task is
 *      running, check dependency flags (WAITONRUN, WAITONCONF,
 *      WAITSEQ_IDLE) and mark it completed when all are met.
 *   2. Purge completed tasks when the task array exceeds the
 *      configured high-water mark (NB_FPSCTRL_TASK_PURGESIZE).
 *   3. Select the highest-priority queue with a ready task and
 *      dispatch it via milkseq_exec_cmd().
 *
 * Return: Number of tasks launched this step (0 or 1)
 */
int milkseq_scheduler_step(
    MILKSEQ_STATE        *state,
    FPS                  *fps,
    KEYWORD_TREE_NODE    *keywnode,
    FPSCTRL_PROCESS_VARS *vars)
{
    if (!state) return 0;

    int QUEUE_NOTASK = -1;
    int QUEUE_WAIT = -2;
    int QUEUE_SCANREADY = -3;

    int NBtaskLaunched = 0;
    int queue_nexttask[NB_FPSCTRL_TASKQUEUE_MAX];

    for (uint32_t qi = 0; qi < NB_FPSCTRL_TASKQUEUE_MAX; qi++) {
        queue_nexttask[qi] = QUEUE_SCANREADY;

        while (queue_nexttask[qi] == QUEUE_SCANREADY) {
            uint64_t inputindexmin = UINT_MAX;
            int cmdindexExec = -1;
            int cmdOK = 0;

            queue_nexttask[qi] = QUEUE_NOTASK;

            // Find oldest active task in this queue
            for (uint32_t cmdindex = 0; cmdindex < state->NBtasks_max && cmdindex < NB_FPSCTRL_TASK_MAX; cmdindex++) {
                if ((state->tasklist[cmdindex].status & FPSTASK_STATUS_ACTIVE) &&
                    (state->tasklist[cmdindex].queue == qi)) {
                    if (state->tasklist[cmdindex].inputindex < inputindexmin) {
                        inputindexmin = state->tasklist[cmdindex].inputindex;
                        cmdindexExec = cmdindex;
                        cmdOK = 1;
                    }
                }
            }

            if (cmdOK == 1) {
                if (!(state->tasklist[cmdindexExec].status & FPSTASK_STATUS_RUNNING)) {
                    queue_nexttask[qi] = cmdindexExec;
                } else {
                    // Check if running task completed
                    int task_completed = 1;

                    if (state->tasklist[cmdindexExec].flag & FPSTASK_FLAG_WAITONRUN) {
                        if (fps[state->tasklist[cmdindexExec].fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN) {
                            task_completed = 0;
                            queue_nexttask[qi] = QUEUE_WAIT;
                        }
                    }

                    if (state->tasklist[cmdindexExec].flag & MILKSEQ_TASKFLAG_WAITFPS_RUNNING) {
                        if (!(fps[state->tasklist[cmdindexExec].fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)) {
                            task_completed = 0;
                            queue_nexttask[qi] = QUEUE_WAIT;
                        }
                    }

                    if (state->tasklist[cmdindexExec].flag & MILKSEQ_TASKFLAG_WAITFPS_NORUN) {
                        if (fps[state->tasklist[cmdindexExec].fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN) {
                            task_completed = 0;
                            queue_nexttask[qi] = QUEUE_WAIT;
                        }
                    }

                    if (state->tasklist[cmdindexExec].flag & MILKSEQ_TASKFLAG_WAITSEQ_IDLE) {
                        char target_seq[FPSSEQ_NAME_MAX];
                        if (sscanf(state->tasklist[cmdindexExec].cmdstring, "wait_seq %63s idle", target_seq) == 1) {
                            MILKSEQ_STATE *tstate = milkseq_connect(target_seq);
                            if (tstate != NULL) {
                                if (tstate->NBtasks_active > 0 || (tstate->status & MILKSEQ_STATUS_RUNNING)) {
                                    task_completed = 0;
                                    queue_nexttask[qi] = QUEUE_WAIT;
                                }
                                milkseq_disconnect(tstate);
                            }
                        }
                    }

                    if (state->tasklist[cmdindexExec].flag & FPSTASK_FLAG_WAITONCONF) {
                        if (fps[state->tasklist[cmdindexExec].fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED) {
                            task_completed = 0;
                            queue_nexttask[qi] = QUEUE_WAIT;
                        }
                    }

                    if (task_completed == 1) {
                        state->tasklist[cmdindexExec].status &= ~FPSTASK_STATUS_RUNNING;
                        state->tasklist[cmdindexExec].status |= FPSTASK_STATUS_COMPLETED;
                        state->tasklist[cmdindexExec].status &= ~FPSTASK_STATUS_ACTIVE;

                        clock_gettime(CLOCK_MILK, &state->tasklist[cmdindexExec].completiontime);
                        queue_nexttask[qi] = QUEUE_SCANREADY;

                        if (state->NBtasks_active > 0) state->NBtasks_active--;
                        state->NBtasks_completed++;
                    }
                }
            }
        }
    }

    // Remove old tasks (purge)
    struct timespec tnow;
    double tnowd;
    clock_gettime(CLOCK_MILK, &tnow);
    tnowd = 1.0 * tnow.tv_sec + 1.0e-9 * tnow.tv_nsec;

    uint32_t taskcnt = state->NBtasks_max;
    // We want to keep at most NB_FPSCTRL_TASK_MAX - NB_FPSCTRL_TASK_PURGESIZE empty spots?? No, backwards.
    // The original logic checks if active tasks + completed tasks > size - purgesize, then deletes oldest.
    // Actually the original logic counted completed tasks, and if taskcnt > limit, zeroed out the status of the oldest.
    while (taskcnt > state->NBtasks_max - NB_FPSCTRL_TASK_PURGESIZE) {
        taskcnt = 0;
        double oldest_age = 0.0;
        long oldest_index = -1;

        for (uint32_t cmdindex = 0; cmdindex < state->NBtasks_max && cmdindex < NB_FPSCTRL_TASK_MAX; cmdindex++) {
            if (state->tasklist[cmdindex].status & FPSTASK_STATUS_COMPLETED) {
                double age = tnowd - (1.0 * state->tasklist[cmdindex].completiontime.tv_sec +
                                      1.0e-9 * state->tasklist[cmdindex].completiontime.tv_nsec);
                if (age > oldest_age) {
                    oldest_age = age;
                    oldest_index = cmdindex;
                }
                taskcnt++;
            }
        }
        if (taskcnt > state->NBtasks_max - NB_FPSCTRL_TASK_PURGESIZE && oldest_index != -1) {
            state->tasklist[oldest_index].status = 0; // mark empty
        } else {
            break;
        }
    }

    // Find highest priority queue with a ready task
    int nexttask_priority = -1;
    int nexttask_cmdindex = -1;
    for (uint32_t qi = 0; qi < NB_FPSCTRL_TASKQUEUE_MAX; qi++) {
        if (queue_nexttask[qi] != QUEUE_NOTASK && queue_nexttask[qi] != QUEUE_WAIT) {
            if (state->queuelist[qi].priority > nexttask_priority) {
                nexttask_priority = state->queuelist[qi].priority;
                nexttask_cmdindex = queue_nexttask[qi];
            }
        }
    }

    if (nexttask_cmdindex != -1 && nexttask_priority > 0) {
        int cmdindexExec = nexttask_cmdindex;
        uint64_t taskstatus = 0;

        state->tasklist[cmdindexExec].fpsindex = milkseq_exec_cmd(
            cmdindexExec, state, fps, keywnode, vars, &taskstatus);

        NBtaskLaunched++;
        state->tasklist[cmdindexExec].status |= taskstatus;
        clock_gettime(CLOCK_MILK, &state->tasklist[cmdindexExec].activationtime);

        state->tasklist[cmdindexExec].status |= FPSTASK_STATUS_RUNNING;
        state->tasklist[cmdindexExec].status &= ~FPSTASK_STATUS_WAITING;
    }

    return NBtaskLaunched;
}

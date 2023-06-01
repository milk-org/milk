/**
 * @file fps_process_fpsCMDarray.c
 */

#include <limits.h>

#include "CommandLineInterface/CLIcore.h"

#include "fps_processcmdline.h"

/** @brief Find the next task to execute
 *
 * Tasks are arranged in execution queues.
 * Each task belongs to a single queue.
 *
 * This function is run by functionparameter_CTRLscreen() at regular intervals to probe queues and run pending tasks.
 * If a task is found, it is executed by calling functionparameter_FPSprocess_cmdline()
 *
 * Each queue has a priority index.
 *
 * RULES :
 * - priorities are associated to queues, not individual tasks: changing a queue priority affects all tasks in the queue
 * - If queue priority = 0, no task is executed in the queue: it is paused
 * - Task order within a queue must be respected. Execution order is submission order (FIFO)
 * - Tasks can overlap if they belong to separate queues and have the same priority
 * - A running task waiting to be completed cannot block tasks in other queues
 * - If two tasks are ready with the same priority, the one in the lower queue will be launched
 *
 * CONVENTIONS AND GUIDELINES :
 * - queue #0 is the main queue
 * - Keep queue 0 priority at 10
 * - Do not pause queue 0
 * - Return to queue 0 when done working in other queues
 */

int function_parameter_process_fpsCMDarray(FPSCTRL_TASK_ENTRY *fpsctrltasklist,
        FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
        KEYWORD_TREE_NODE  *keywnode,
        FPSCTRL_PROCESS_VARS *fpsCTRLvar,
        FUNCTION_PARAMETER_STRUCT *fps)
{
    // queue has no task
    int QUEUE_NOTASK = -1;

    // queue has a running task, must waiting for completion
    int QUEUE_WAIT = -2;

    // queue is ready for next scan
    int QUEUE_SCANREADY = -3;

    // the scheduler handles multiple queues
    // in each queue, we look for a task to run, and run it if conditions are met

    int NBtaskLaunched = 0;

    // For each queue, lets find which task is ready
    // results are written in array
    // if no task ready in queue, value = QUEUE_NOTASK
    //
    int queue_nexttask[NB_FPSCTRL_TASKQUEUE_MAX];

    for(uint32_t qi = 0; qi < NB_FPSCTRL_TASKQUEUE_MAX; qi++)
    {
        queue_nexttask[qi] = QUEUE_SCANREADY;

        while(queue_nexttask[qi] == QUEUE_SCANREADY)
        {
            // find next command to execute
            uint64_t inputindexmin = UINT_MAX;
            int      cmdindexExec;
            int      cmdOK = 0;

            queue_nexttask[qi] = QUEUE_NOTASK;
            //
            // Find task with smallest inputindex within this queue
            // This is the one to be executed
            //
            for(int cmdindex = 0; cmdindex < NB_FPSCTRL_TASK_MAX; cmdindex++)
            {
                if((fpsctrltasklist[cmdindex].status &
                        FPSTASK_STATUS_ACTIVE) &&
                        (fpsctrltasklist[cmdindex].queue == qi))
                {
                    if(fpsctrltasklist[cmdindex].inputindex < inputindexmin)
                    {
                        inputindexmin = fpsctrltasklist[cmdindex].inputindex;
                        cmdindexExec  = cmdindex;
                        cmdOK         = 1;
                    }
                }
            }

            if(cmdOK == 1)  // A potential task to be executed has been found
            {
                if(!(fpsctrltasklist[cmdindexExec].status &
                        FPSTASK_STATUS_RUNNING)) // if task not running, launch it
                {
                    queue_nexttask[qi] = cmdindexExec;
                }
                else
                {
                    // if it's already running, lets check if it is completed
                    int task_completed = 1; // default

                    if(fpsctrltasklist[cmdindexExec].flag &
                            FPSTASK_FLAG_WAITONRUN) // are we waiting for run to be completed ?
                    {
                        if((fps[fpsctrltasklist[cmdindexExec].fpsindex]
                                .md->status &
                                FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN))
                        {
                            task_completed     = 0; // must wait
                            queue_nexttask[qi] = QUEUE_WAIT;
                        }
                    }

                    if(fpsctrltasklist[cmdindexExec].flag &
                            FPSTASK_FLAG_WAITONCONF) // are we waiting for conf update to be completed ?
                    {
                        if(fps[fpsctrltasklist[cmdindexExec].fpsindex]
                                .md->status &
                                FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED)
                        {
                            task_completed     = 0; // must wait
                            queue_nexttask[qi] = QUEUE_WAIT;
                        }
                    }

                    if(task_completed == 1)
                    {
                        // update status - no longer running
                        fpsctrltasklist[cmdindexExec].status &=
                            ~FPSTASK_STATUS_RUNNING;
                        fpsctrltasklist[cmdindexExec].status |=
                            FPSTASK_STATUS_COMPLETED;

                        //no longer active, remove it from list
                        fpsctrltasklist[cmdindexExec].status &=
                            ~FPSTASK_STATUS_ACTIVE;

                        //   fpsctrltasklist[cmdindexExec].status &= ~FPSTASK_STATUS_SHOW; // and stop displaying

                        clock_gettime(
                            CLOCK_MILK,
                            &fpsctrltasklist[cmdindexExec].completiontime);
                        queue_nexttask[qi] = QUEUE_SCANREADY;
                    }
                }
            } // end if(cmdOK==1)
        }     // end while QUEUE_SCANREADY
    }

    // Remove old tasks
    //
    double         *completion_age; // completion time
    long            oldest_index = 0;
    struct timespec tnow;
    double          tnowd;

    completion_age = (double *) malloc(sizeof(double) * NB_FPSCTRL_TASK_MAX);
    if(completion_age == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    clock_gettime(CLOCK_MILK, &tnow);
    tnowd = 1.0 * tnow.tv_sec + 1.0e-9 * tnow.tv_nsec;

    long taskcnt = NB_FPSCTRL_TASK_MAX;

    while(taskcnt > NB_FPSCTRL_TASK_MAX - NB_FPSCTRL_TASK_PURGESIZE)
    {
        taskcnt           = 0;
        double oldest_age = 0.0;
        for(int cmdindex = 0; cmdindex < NB_FPSCTRL_TASK_MAX; cmdindex++)
        {
            // how many tasks are candidates for removal (completed) ?
            if(fpsctrltasklist[cmdindex].status & FPSTASK_STATUS_COMPLETED)
            {

                completion_age[taskcnt] =
                    tnowd -
                    (1.0 * fpsctrltasklist[cmdindex].completiontime.tv_sec +
                     1.0e-9 * fpsctrltasklist[cmdindex].completiontime.tv_nsec);

                if(completion_age[taskcnt] > oldest_age)
                {
                    oldest_age   = completion_age[taskcnt];
                    oldest_index = cmdindex;
                }
                taskcnt++;
            }
        }
        if(taskcnt > NB_FPSCTRL_TASK_MAX - NB_FPSCTRL_TASK_PURGESIZE)
        {
            fpsctrltasklist[oldest_index].status = 0;
        }
    }

    free(completion_age);

    // find out which task to run among the ones pre-selected above

    int nexttask_priority = -1;
    int nexttask_cmdindex = -1;
    for(uint32_t qi = 0; qi < NB_FPSCTRL_TASKQUEUE_MAX; qi++)
    {
        if((queue_nexttask[qi] != QUEUE_NOTASK) &&
                (queue_nexttask[qi] != QUEUE_WAIT))
        {
            if(fpsctrlqueuelist[qi].priority > nexttask_priority)
            {
                nexttask_priority = fpsctrlqueuelist[qi].priority;
                nexttask_cmdindex = queue_nexttask[qi];
            }
        }
    }

    if(nexttask_cmdindex != -1)
    {
        if(nexttask_priority > 0)
        {
            // execute task
            int cmdindexExec = nexttask_cmdindex;

            uint64_t taskstatus = 0;

            fpsctrltasklist[cmdindexExec].fpsindex =
                functionparameter_FPSprocess_cmdline(
                    fpsctrltasklist[cmdindexExec].cmdstring,
                    fpsctrlqueuelist,
                    keywnode,
                    fpsCTRLvar,
                    fps,
                    &taskstatus);
            NBtaskLaunched++;

            // update status form cmdline interpreter
            fpsctrltasklist[cmdindexExec].status |= taskstatus;

            clock_gettime(CLOCK_MILK,
                          &fpsctrltasklist[cmdindexExec].activationtime);

            // update status to running
            fpsctrltasklist[cmdindexExec].status |= FPSTASK_STATUS_RUNNING;
            fpsctrltasklist[cmdindexExec].status &= ~FPSTASK_STATUS_WAITING;
        }
    }

    return NBtaskLaunched;
}

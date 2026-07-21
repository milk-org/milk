// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file fpsseq_fifo.c
 * @brief Reads incoming commands from FIFO into the task list
 *
 * Populates the sequence state task arrays from the named FIFO asynchronously.
 */

#include <errno.h>
#include <string.h>

#include "fpsseq.h"
#include "timeutils.h"

/**
 * milkseq_fifo_read - Read commands from the sequencer FIFO
 * @state:    Sequencer state (tasks appended to state->tasklist)
 * @fifo_fd:  File descriptor of the FIFO (must be O_NONBLOCK)
 *
 * Reads the FIFO byte-by-byte until EAGAIN, assembling complete
 * newline-terminated lines. Each line is either a meta-command
 * (taskcntzero, setqindex, setqprio, waitonrun/conf toggles)
 * that updates sequencer state directly, or a regular command
 * string that is enqueued as a new task in the first free slot.
 *
 * Return: Number of regular tasks enqueued (excludes meta-commands)
 */
int milkseq_fifo_read(MILKSEQ_STATE *state, int fifo_fd)
{
    if (!state || fifo_fd < 0)
    {
        return 0;
    }

    int  cmdcnt = 0;
    char buff[512];
    int  total_bytes = 0;
    int  bytes;
    char buf0[1];

    int lineOK = 1;

    while (lineOK == 1)
    {
        total_bytes = 0;
        lineOK      = 0;
        for (;;)
        {
            bytes = read(fifo_fd, buf0, 1);
            if (bytes > 0)
            {
                if (total_bytes < (int) sizeof(buff) - 1)
                {
                    buff[total_bytes] = buf0[0];
                    total_bytes++;
                }
            }
            else
            {
                if (errno == EWOULDBLOCK || errno == EAGAIN)
                {
                    break;
                }
                else
                {
                    return cmdcnt;
                }
            }

            if (buf0[0] == '\n')
            {
                buff[total_bytes - 1] = '\0';
                char *FPScmdline      = buff;

                // Find next free task index
                uint32_t cmdindex   = 0;
                int      cmdindexOK = 0;
                while (cmdindexOK == 0 && cmdindex < state->NBtasks_max)
                {
                    if (state->tasklist[cmdindex].status == 0)
                    {
                        cmdindexOK = 1;
                    }
                    else
                    {
                        cmdindex++;
                    }
                }

                if (cmdindex == state->NBtasks_max)
                {
                    printf("ERROR: fpscmdarray is full. Reached max tasks.\n");
                    // cannot accept more right now.
                    break;
                }

                int cmdFOUND = 0;

                if (FPScmdline[0] == '#' || FPScmdline[0] == ' ' || total_bytes < 2)
                {
                    cmdFOUND = 1;
                }

                if (cmdFOUND == 0 && strncmp(FPScmdline, "taskcntzero", 11) == 0)
                {
                    cmdFOUND                  = 1;
                    state->task_input_counter = 0;
                }

                if (cmdFOUND == 0 && strncmp(FPScmdline, "setqindex", 9) == 0)
                {
                    cmdFOUND = 1;
                    char stringtmp[200];
                    int  queue_index;
                    if (sscanf(FPScmdline, "%s %d", stringtmp, &queue_index) == 2)
                    {
                        if (queue_index > -1 && queue_index < NB_FPSCTRL_TASKQUEUE_MAX)
                        {
                            state->current_queue = queue_index;
                        }
                    }
                }

                if (cmdFOUND == 0 && strncmp(FPScmdline, "setqprio", 8) == 0)
                {
                    cmdFOUND = 1;
                    char stringtmp[200];
                    int  queue_priority;
                    if (sscanf(FPScmdline, "%s %d", stringtmp, &queue_priority) == 2)
                    {
                        if (queue_priority < 0)
                        {
                            queue_priority = 0;
                        }
                        state->queuelist[state->current_queue].priority = queue_priority;
                    }
                }

                if (cmdFOUND == 0 && strncmp(FPScmdline, "waitonrunON", 11) == 0)
                {
                    cmdFOUND                 = 1;
                    state->current_waitonrun = 1;
                }
                if (cmdFOUND == 0 && strncmp(FPScmdline, "waitonrunOFF", 12) == 0)
                {
                    cmdFOUND                 = 1;
                    state->current_waitonrun = 0;
                }
                if (cmdFOUND == 0 && strncmp(FPScmdline, "waitonconfON", 12) == 0)
                {
                    cmdFOUND                  = 1;
                    state->current_waitonconf = 1;
                }
                if (cmdFOUND == 0 && strncmp(FPScmdline, "waitonconfOFF", 13) == 0)
                {
                    cmdFOUND                  = 1;
                    state->current_waitonconf = 0;
                }

                // If not handled above, treat as normal task
                if (cmdFOUND == 0)
                {
                    strncpy(state->tasklist[cmdindex].cmdstring, FPScmdline,
                            STRINGMAXLEN_FPS_CMDLINE - 1);
                    state->tasklist[cmdindex].cmdstring[STRINGMAXLEN_FPS_CMDLINE - 1] = '\0';

                    state->tasklist[cmdindex].status = FPSTASK_STATUS_ACTIVE | FPSTASK_STATUS_SHOW;
                    state->tasklist[cmdindex].inputindex = state->task_input_counter;
                    state->tasklist[cmdindex].queue      = state->current_queue;
                    clock_gettime(CLOCK_MILK, &state->tasklist[cmdindex].creationtime);

                    state->tasklist[cmdindex].status |= FPSTASK_STATUS_WAITING;

                    if (state->current_waitonrun == 1)
                    {
                        state->tasklist[cmdindex].flag |= FPSTASK_FLAG_WAITONRUN;
                    }
                    else
                    {
                        state->tasklist[cmdindex].flag &= ~FPSTASK_FLAG_WAITONRUN;
                    }

                    if (state->current_waitonconf == 1)
                    {
                        state->tasklist[cmdindex].flag |= FPSTASK_FLAG_WAITONCONF;
                    }
                    else
                    {
                        state->tasklist[cmdindex].flag &= ~FPSTASK_FLAG_WAITONCONF;
                    }

                    state->task_input_counter++;
                    state->NBtasks_active++;
                    cmdcnt++;
                }

                lineOK = 1;
                break; // Break the char reading loop to start next line
            }
        }
    }

    return cmdcnt;
}

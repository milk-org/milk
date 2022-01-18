/**
 * @file    fps_read_fpsCMD_fifo.c
 */

#include "CommandLineInterface/CLIcore.h"

// fill up task list from fifo submissions

int functionparameter_read_fpsCMD_fifo(int fpsCTRLfifofd, FPSCTRL_TASK_ENTRY *fpsctrltasklist,
                                       FPSCTRL_TASK_QUEUE *fpsctrlqueuelist)
{
    int cmdcnt = 0;
    char *FPScmdline = NULL;
    char buff[200];
    int total_bytes = 0;
    int bytes;
    char buf0[1];

    // toggles
    static uint32_t queue = 0;
    static int waitonrun = 0;
    static int waitonconf = 0;

    static uint16_t cmdinputcnt = 0;

    int lineOK = 1; // keep reading

    DEBUG_TRACEPOINT(" ");

    while (lineOK == 1)
    {
        total_bytes = 0;
        lineOK = 0;
        for (;;)
        {
            bytes = read(fpsCTRLfifofd, buf0, 1); // read one char at a time
            DEBUG_TRACEPOINT("ERRROR: BUFFER OVERFLOW %d %d\n", bytes, total_bytes);
            if (bytes > 0)
            {
                buff[total_bytes] = buf0[0];
                total_bytes += (size_t)bytes;
            }
            else
            {
                if (errno == EWOULDBLOCK)
                {
                    break;
                }
                else // read 0 byte
                {
                    //perror("read 0 byte");
                    return cmdcnt;
                }
            }

            DEBUG_TRACEPOINT(" ");

            if (buf0[0] == '\n')
            {
                // reached end of line
                // -> process command
                //

                buff[total_bytes - 1] = '\0';
                FPScmdline = buff;

                // find next index
                int cmdindex = 0;
                int cmdindexOK = 0;
                while ((cmdindexOK == 0) && (cmdindex < NB_FPSCTRL_TASK_MAX))
                {
                    if (fpsctrltasklist[cmdindex].status == 0)
                    {
                        cmdindexOK = 1;
                    }
                    else
                    {
                        cmdindex++;
                    }
                }

                if (cmdindex == NB_FPSCTRL_TASK_MAX)
                {
                    printf("ERROR: fpscmdarray is full\n");
                    exit(0);
                }

                DEBUG_TRACEPOINT(" ");

                // Some commands affect how the task list is configured instead of being inserted as entries
                int cmdFOUND = 0;

                if ((FPScmdline[0] == '#') || (FPScmdline[0] == ' ') || (total_bytes < 2)) // disregard line
                {
                    cmdFOUND = 1;
                }

                // set wait on run ON
                if ((cmdFOUND == 0) && (strncmp(FPScmdline, "taskcntzero", strlen("taskcntzero")) == 0))
                {
                    cmdFOUND = 1;
                    cmdinputcnt = 0;
                }

                // Set queue index
                // entries will now be placed in queue specified by this command
                if ((cmdFOUND == 0) && (strncmp(FPScmdline, "setqindex", strlen("setqindex")) == 0))
                {
                    cmdFOUND = 1;
                    char stringtmp[200];
                    int queue_index;
                    sscanf(FPScmdline, "%s %d", stringtmp, &queue_index);

                    if ((queue_index > -1) && (queue_index < NB_FPSCTRL_TASKQUEUE_MAX))
                    {
                        queue = queue_index;
                    }
                }

                // Set queue priority
                if ((cmdFOUND == 0) && (strncmp(FPScmdline, "setqprio", strlen("setqprio")) == 0))
                {
                    cmdFOUND = 1;
                    char stringtmp[200];
                    int queue_priority;
                    sscanf(FPScmdline, "%s %d", stringtmp, &queue_priority);

                    if (queue_priority < 0)
                    {
                        queue_priority = 0;
                    }

                    fpsctrlqueuelist[queue].priority = queue_priority;
                }

                // set wait on run ON
                if ((cmdFOUND == 0) && (strncmp(FPScmdline, "waitonrunON", strlen("waitonrunON")) == 0))
                {
                    cmdFOUND = 1;
                    waitonrun = 1;
                }

                // set wait on run OFF
                if ((cmdFOUND == 0) && (strncmp(FPScmdline, "waitonrunOFF", strlen("waitonrunOFF")) == 0))
                {
                    cmdFOUND = 1;
                    waitonrun = 0;
                }

                // set wait on conf ON
                if ((cmdFOUND == 0) && (strncmp(FPScmdline, "waitonconfON", strlen("waitonconfON")) == 0))
                {
                    cmdFOUND = 1;
                    waitonconf = 1;
                }

                // set wait on conf OFF
                if ((cmdFOUND == 0) && (strncmp(FPScmdline, "waitonconfOFF", strlen("waitonconfOFF")) == 0))
                {
                    cmdFOUND = 1;
                    waitonconf = 0;
                }

                // set wait point for arbitrary FPS run to have finished

                DEBUG_TRACEPOINT(" ");

                // for all other commands, put in task list
                if (cmdFOUND == 0)
                {
                    strncpy(fpsctrltasklist[cmdindex].cmdstring, FPScmdline, STRINGMAXLEN_FPS_CMDLINE - 1);

                    fpsctrltasklist[cmdindex].status = FPSTASK_STATUS_ACTIVE | FPSTASK_STATUS_SHOW;
                    fpsctrltasklist[cmdindex].inputindex = cmdinputcnt;
                    fpsctrltasklist[cmdindex].queue = queue;
                    clock_gettime(CLOCK_REALTIME, &fpsctrltasklist[cmdindex].creationtime);

                    // waiting to be processed
                    fpsctrltasklist[cmdindex].status |= FPSTASK_STATUS_WAITING;

                    if (waitonrun == 1)
                    {
                        fpsctrltasklist[cmdindex].flag |= FPSTASK_FLAG_WAITONRUN;
                    }
                    else
                    {
                        fpsctrltasklist[cmdindex].flag &= ~FPSTASK_FLAG_WAITONRUN;
                    }

                    if (waitonconf == 1)
                    {
                        fpsctrltasklist[cmdindex].flag |= FPSTASK_FLAG_WAITONCONF;
                    }
                    else
                    {
                        fpsctrltasklist[cmdindex].flag &= ~FPSTASK_FLAG_WAITONCONF;
                    }

                    cmdinputcnt++;

                    cmdcnt++;
                }
                lineOK = 1;
                break;
            }
        }
    }

    DEBUG_TRACEPOINT(" ");

    return cmdcnt;
}

/**
 * @file fpsseq_types.h
 * @brief Core data structures for the milk-seq FPS Sequencer
 *
 * Extends the basic FPSCTRL_TASK_ENTRY and FPSCTRL_TASK_QUEUE types
 * to support multi-instance shared memory execution.
 */

#ifndef FPSSEQ_TYPES_H
#define FPSSEQ_TYPES_H

#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#include "fps_types.h"

#define FPSSEQ_NAME_MAX 64
#define FPSSEQ_FIFO_PATH_MAX 256
#define FPSSEQ_SCRIPT_PATH_MAX 512
#define FPSSEQ_ERROR_STR_MAX 512

/* Status flags for the sequencer itself */
#define MILKSEQ_STATUS_IDLE     0x0001
#define MILKSEQ_STATUS_RUNNING  0x0002
#define MILKSEQ_STATUS_ERROR    0x0004
#define MILKSEQ_STATUS_STOPPING 0x0008

/* Extensions to FPSTASK_FLAG_* for synchronization */
#define MILKSEQ_TASKFLAG_WAITFPS_RUNNING 0x00000100
#define MILKSEQ_TASKFLAG_WAITFPS_NORUN   0x00000200
#define MILKSEQ_TASKFLAG_WAITSEQ_IDLE    0x00000400

/* Extensions to FPSTASK_FLAG_* for error policies */
#define MILKSEQ_TASKFLAG_ONERROR_ABORT   0x00001000
#define MILKSEQ_TASKFLAG_ONERROR_SKIP    0x00002000
#define MILKSEQ_TASKFLAG_ONERROR_RETRY   0x00004000

/**
 * @brief Sequencer instance state, mapped into shared memory.
 *
 * One per running milk-seq instance.
 * SHM path: /dev/shm/milkseq.<name>.shm
 */
typedef struct {
    char     name[FPSSEQ_NAME_MAX];
    uint32_t status;
    pid_t    pid;
    struct timespec starttime;

    /* Task array (extracted from fpsCTRL) */
    uint32_t NBtasks_max;
    uint32_t NBtasks_active;
    uint32_t NBtasks_completed;
    uint64_t task_input_counter;
    FPSCTRL_TASK_ENTRY tasklist[NB_FPSCTRL_TASK_MAX];

    /* Queue array (extracted from fpsCTRL) */
    FPSCTRL_TASK_QUEUE queuelist[NB_FPSCTRL_TASKQUEUE_MAX];

    /* external IO */
    char fifo_path[FPSSEQ_FIFO_PATH_MAX];
    char script_path[FPSSEQ_SCRIPT_PATH_MAX];

    /* FIFO/Script parsing context */
    uint32_t current_queue;
    int32_t current_waitonrun;
    int32_t current_waitonconf;

    /* Error tracking */
    uint32_t error_count;
    char     last_error[FPSSEQ_ERROR_STR_MAX];

} MILKSEQ_STATE;

#endif // FPSSEQ_TYPES_H

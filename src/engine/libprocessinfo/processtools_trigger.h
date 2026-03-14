/**
 * @file    processtools_trigger.h
 *
 * @brief   Process triggering
 *
 *
 */

#ifndef _PROCESSTOOLS_TRIGGER_H
#define _PROCESSTOOLS_TRIGGER_H

// input stream triggering modes

// trigger immediately
#define PROCESSINFO_TRIGGERMODE_IMMEDIATE 0

// trigger when cnt0 increments
#define PROCESSINFO_TRIGGERMODE_CNT0 1

// trigger when cnt1 increments
#define PROCESSINFO_TRIGGERMODE_CNT1 2

// trigger when semaphore is posted
#define PROCESSINFO_TRIGGERMODE_SEMAPHORE 3

// trigger after a time delay
#define PROCESSINFO_TRIGGERMODE_DELAY 4

// trigger when semaphore is posted AND propagate the timeout (i.e. enter the execution anyway)
#define PROCESSINFO_TRIGGERMODE_SEMAPHORE_PROP_TIMEOUTS 5

// trigger when cnt0 < cnt2 (demand-driven / flow control)
#define PROCESSINFO_TRIGGERMODE_CNT2 6

// trigger is currently waiting for input
#define PROCESSINFO_TRIGGERSTATUS_WAITING 1
// trigger has been received and we're executing the loop
#define PROCESSINFO_TRIGGERSTATUS_RECEIVED 2
// trigger has not been received but we've skipped out of the wait into the execution of the loop
#define PROCESSINFO_TRIGGERSTATUS_TIMEDOUT 3

#include "processinfo.h"
#include "ImageStreamIO/ImageStruct.h"

errno_t processinfo_waitoninputstream_init(PROCESSINFO *processinfo,
        IMAGE        *image,
        int          triggermode,
        int          semindexrequested);

errno_t processinfo_waitoninputstream(PROCESSINFO *processinfo);

#define PROCINFO_TRIGGER_DELAYUS(delayus)                                      \
    do                                                                         \
    {                                                                          \
        processinfo_waitoninputstream_init(processinfo,                        \
                                           NULL,                               \
                                           PROCESSINFO_TRIGGERMODE_DELAY,      \
                                           -1);                                \
        processinfo->triggerdelay.tv_sec  = 0;                                 \
        processinfo->triggerdelay.tv_nsec = (long) ((delayus) *1000);          \
        while (processinfo->triggerdelay.tv_nsec > 1000000000)                 \
        {                                                                      \
            processinfo->triggerdelay.tv_nsec -= 1000000000;                   \
            processinfo->triggerdelay.tv_sec += 1;                             \
        }                                                                      \
    } while (0)

#endif

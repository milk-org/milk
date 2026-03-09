/**
 * @file processtools_trigger.c
 *
 *
 */

#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processtools_trigger.h"
#include "ImageStreamIO/ImageStreamIO.h"

#ifndef CLOCK_MILK
#define CLOCK_MILK CLOCK_REALTIME
#endif

/** @brief Set up input wait stream
 *
 * Specify stream on which the loop process will be triggering, and
 * what is the trigger mode.
 *
 * The actual trigger mode may be different from the requested trigger mode.
 *
 * The standard option should be tiggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE
 * and semindex = -1, which will automatically find a suitable semaphore
 *
 */
errno_t processinfo_waitoninputstream_init(
    PROCESSINFO *processinfo,
    IMAGE        *image,
    int          triggermode,
    int          semindexrequested
)
{
    DEBUG_TRACE_FSTART("%p %d %d", (void*)image, triggermode, semindexrequested);
    
    // Legacy support: triggerstreamID is not used internally by the library anymore
    // but we can set it to -1 to indicate unused.
    processinfo->triggerstreamID = -1;
    processinfo->trigger_image = image;

    if(image != NULL)
    {
        processinfo->triggerstreaminode = image->md[0].inode;
        strncpy(processinfo->triggerstreamname,
                image->md[0].name,
                STRINGMAXLEN_IMAGE_NAME);
    }
    else
    {
        // convention : stream name single space : inactive
        DEBUG_TRACEPOINT("Setting trigger stream name to single space");
        processinfo->triggerstreaminode = 0;
        strcpy(processinfo->triggerstreamname, " ");
    }

    processinfo->triggermissedframe_cumul = 0;
    processinfo->trigggertimeoutcnt       = 0;
    processinfo->triggerstatus            = 0;

    // Default timeout: 2 seconds
    processinfo->triggertimeout.tv_sec  = 2;
    processinfo->triggertimeout.tv_nsec = 0;

    // Set requested triggermode
    processinfo->triggermode = triggermode;

    // valid modes

    if(triggermode == PROCESSINFO_TRIGGERMODE_CNT0)
    {
        DEBUG_TRACEPOINT("trigger mode %d = cnt0", PROCESSINFO_TRIGGERMODE_CNT0);

        if(image == NULL)
        {
            PRINT_ERROR("missing trigger image");
            return RETURN_FAILURE;
        }
        // trigger on cnt0 increment
        processinfo->triggerstreamcnt = image->md[0].cnt0;
    }

    if(triggermode == PROCESSINFO_TRIGGERMODE_CNT1)
    {
        DEBUG_TRACEPOINT("trigger mode %d = cnt1", PROCESSINFO_TRIGGERMODE_CNT1);

        if(image == NULL)
        {
            PRINT_ERROR("missing trigger image");
            return RETURN_FAILURE;
        }
        // trigger on cnt1 increment
        processinfo->triggerstreamcnt = image->md[0].cnt1;
    }

    if(triggermode == PROCESSINFO_TRIGGERMODE_CNT2)
    {
        DEBUG_TRACEPOINT("trigger mode %d = cnt2", PROCESSINFO_TRIGGERMODE_CNT2);

        if(image == NULL)
        {
            PRINT_ERROR("missing trigger image");
            return RETURN_FAILURE;
        }
        // trigger on cnt0 < cnt2
        processinfo->triggerstreamcnt = image->md[0].cnt0;
    }

    if(triggermode == PROCESSINFO_TRIGGERMODE_IMMEDIATE)
    {
        DEBUG_TRACEPOINT("trigger mode %d = immediate",
                         PROCESSINFO_TRIGGERMODE_IMMEDIATE);
        // immmediate trigger
        processinfo->triggerstreamcnt = 0;
    }

    if(triggermode == PROCESSINFO_TRIGGERMODE_DELAY)
    {
        DEBUG_TRACEPOINT("trigger mode %d = time delay",
                         PROCESSINFO_TRIGGERMODE_DELAY);
        // time wait
        processinfo->triggerstreamcnt = 0;
    }

    // checking if semaphore trigger mode OK
    if(triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE ||
       triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE_PROP_TIMEOUTS)
    {
        DEBUG_TRACEPOINT("trigger mode %d = semaphore %d",
                         triggermode,
                         semindexrequested);
        if(semindexrequested < -1)
        {
            PRINT_ERROR("invalid semaphore index %d", semindexrequested);
            return RETURN_FAILURE;
        }
        if(image == NULL)
        {
            PRINT_ERROR("image not valid");
            return RETURN_FAILURE;
        }
        processinfo->triggersem =
            ImageStreamIO_getsemwaitindex(image, semindexrequested);
        if(processinfo->triggersem == -1)
        {
            // could not find available semaphore
            // fall back to CNT0 trigger mode
            processinfo->triggermode = PROCESSINFO_TRIGGERMODE_CNT0;
            processinfo->triggerstreamcnt = image->md[0].cnt0;
        }
        else
        {
            // register PID to stream
            image->semReadPID[processinfo->triggersem] = getpid();
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/** @brief Wait on a stream
 *
 */
errno_t processinfo_waitoninputstream(PROCESSINFO *processinfo)
{
    IMAGE *image = processinfo->trigger_image;
    processinfo->triggermissedframe = 0;

    if(processinfo->triggermode == PROCESSINFO_TRIGGERMODE_IMMEDIATE)
    {
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;
        // return immediately
        return RETURN_SUCCESS;
    }

    if(processinfo->triggermode == PROCESSINFO_TRIGGERMODE_CNT0)
    {
        if(image == NULL) return RETURN_FAILURE;
        // use cnt0
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        while(image->md[0].cnt0 == processinfo->triggerstreamcnt)
        {
            // test if new frame exists
            usleep(5);
        }
        processinfo->triggermissedframe =
            image->md[0].cnt0 -
            processinfo->triggerstreamcnt - 1;
        // update trigger counter
        processinfo->triggerstreamcnt = image->md[0].cnt0;

        processinfo->triggermissedframe_cumul +=
            processinfo->triggermissedframe;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if(processinfo->triggermode == PROCESSINFO_TRIGGERMODE_CNT1)
    {
        if(image == NULL) return RETURN_FAILURE;
        // use cnt1
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        while(image->md[0].cnt1 == processinfo->triggerstreamcnt)
        {
            // test if new frame exists
            usleep(5);
        }
        processinfo->triggermissedframe =
            image->md[0].cnt1 -
            processinfo->triggerstreamcnt - 1;
        // update trigger counter
        processinfo->triggerstreamcnt = image->md[0].cnt1;

        processinfo->triggermissedframe_cumul +=
            processinfo->triggermissedframe;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if(processinfo->triggermode == PROCESSINFO_TRIGGERMODE_CNT2)
    {
        if(image == NULL) return RETURN_FAILURE;
        // use cnt2
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        while(image->md[0].cnt0 >= image->md[0].cnt2)
        {
            // wait until we are allowed to proceed
            usleep(5);
        }
        processinfo->triggermissedframe = 0;

        // update trigger counter
        processinfo->triggerstreamcnt = image->md[0].cnt0;

        processinfo->triggermissedframe_cumul +=
            processinfo->triggermissedframe;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if(processinfo->triggermode == PROCESSINFO_TRIGGERMODE_DELAY)
    {
        // return after fixed delay
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        // Note: nanosleep adds a few x10us of latency on most systems
        nanosleep(&processinfo->triggerdelay, NULL);
        processinfo->triggerstreamcnt++;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if(processinfo->triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE ||
            processinfo->triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE_PROP_TIMEOUTS)
    {
        if(image == NULL) return RETURN_FAILURE;
        
        int semr;
        int tmpstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        DEBUG_TRACEPOINT("wait on semaphore");

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        // get current time
        struct timespec ts;
        if(clock_gettime(CLOCK_MILK, &ts) == -1)
        {
            perror("clock_gettime");
            exit(EXIT_FAILURE);
        }

        // is semaphore at zero ?
        DEBUG_TRACEPOINT("test sem status");
        semr = 0;
        while(semr == 0)
        {
            // this should only run once, returning semr = -1 with errno = EAGAIN
            // otherwise, we're potentially missing frames
            DEBUG_TRACEPOINT("sem_trywait");
            semr = ImageStreamIO_semtrywait(image, processinfo->triggersem);
            if(semr == 0)
            {
                processinfo->triggermissedframe++;
            }
        }

        // expected state: NBmissedframe = 0, semr = -1, errno = EAGAIN
        // missed frame state: NBmissedframe>0, semr = -1, errno = EAGAIN
        DEBUG_TRACEPOINT("triggermissedframe = %d",
                         processinfo->triggermissedframe);
        if(processinfo->triggermissedframe == 0)
        {
            DEBUG_TRACEPOINT("timedwait");
            // add timeout
            ts.tv_sec += processinfo->triggertimeout.tv_sec;
            ts.tv_nsec += processinfo->triggertimeout.tv_nsec;
            while(ts.tv_nsec > 1000000000)
            {
                ts.tv_nsec -= 1000000000;
                ts.tv_sec++;
            }

            semr = ImageStreamIO_semtimedwait(image, processinfo->triggersem, &ts);
            if(semr == -1)
            {
                if(errno == ETIMEDOUT)
                {
                    // timeout condition
                    processinfo->trigggertimeoutcnt++;
                    tmpstatus = PROCESSINFO_TRIGGERSTATUS_TIMEDOUT;
                }
            }
            else
            {
                tmpstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;
            }
        }

        processinfo->triggermissedframe_cumul +=
            processinfo->triggermissedframe;

        processinfo->triggerstatus = tmpstatus;

        return RETURN_SUCCESS;
    }

    return RETURN_FAILURE;
}
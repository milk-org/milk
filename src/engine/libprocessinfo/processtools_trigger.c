// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processtools_trigger.c
 *
 *
 */


#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <time.h>

#include "processinfo_internal.h"
#include "processinfo_WriteMessage.h"
#include "processtools_trigger.h"
#include "ImageStreamIO/ImageStreamIO.h"

#ifndef CLOCK_MILK
#    define CLOCK_MILK CLOCK_REALTIME
#endif

/**
 * trigger_timeout_recheck - After a semaphore timeout: remap a re-created trigger stream, report a dead owner
 * @processinfo: Process info whose trigger timed out
 * @image:       The trigger stream's IMAGE (the runtime's slot, remapped in place)
 *
 * A stream re-created by its owner has a new inode: the old mapping never sees a frame again, so it is
 * reopened in the same IMAGE and the semaphore slot re-picked. An owner that is gone is reported once
 * through the processinfo message, instead of silent 2 s timeouts forever.
 */
static void trigger_timeout_recheck(PROCESSINFO *processinfo, IMAGE *image)
{
    static pid_t reported_owner = 0;
    char         fname[1024];
    struct stat  st;

    if (image == NULL || image->md == NULL)
    {
        return;
    }
    if (ImageStreamIO_filename(fname, sizeof(fname), processinfo->triggerstreamname) ==
            IMAGESTREAMIO_SUCCESS &&
        stat(fname, &st) == 0 && st.st_ino != processinfo->triggerstreaminode)
    {
        IMAGE tmp;
        memset(&tmp, 0, sizeof(tmp));
        /* the same call the calculator's refresh uses: the slot stays a used, named table entry */
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(processinfo->triggerstreamname, &tmp) ==
                IMAGESTREAMIO_SUCCESS &&
            tmp.md != NULL)
        {
            ImageStreamIO_closeIm(image);
            *image                          = tmp;
            image->used                     = 1;
            processinfo->triggerstreaminode = image->md[0].inode;
            processinfo->triggerstreamcnt   = image->md[0].cnt0;
            processinfo->triggersem         = ImageStreamIO_getsemwaitindex(image, -1);
            reported_owner                  = 0;
            processinfo_WriteMessage(processinfo, "trigger stream re-created: remapped");
        }
        return;
    }
    pid_t owner = image->md[0].ownerPID;
    if (owner > 0 && owner != reported_owner && kill(owner, 0) == -1 && errno == ESRCH)
    {
        char msg[200];
        snprintf(msg, sizeof(msg), "trigger %s: owner %d gone", processinfo->triggerstreamname,
                 (int) owner);
        processinfo_WriteMessage(processinfo, msg);
        reported_owner = owner;
    }
}

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
errno_t processinfo_waitoninputstream_init(PROCESSINFO *processinfo,
                                           IMAGE       *image,
                                           int          triggermode,
                                           int          semindexrequested)
{
    DEBUG_TRACE_FSTART("%p %d %d", (void *) image, triggermode, semindexrequested);

    // Legacy support: triggerstreamID is not used internally by the library anymore
    // but we can set it to -1 to indicate unused.
    processinfo->triggerstreamID = -1;
    processinfo->trigger_image   = image;

    if (image != NULL)
    {
        processinfo->triggerstreaminode = image->md[0].inode;
        strncpy(processinfo->triggerstreamname, image->md[0].name, STRINGMAXLEN_IMAGE_NAME);
    }
    else
    {
        // convention : stream name single space : inactive
        DEBUG_TRACEPOINT("Setting trigger stream name to single space");
        processinfo->triggerstreaminode = 0;
        strncpy(processinfo->triggerstreamname, " ", STRINGMAXLEN_IMAGE_NAME - 1);
        processinfo->triggerstreamname[STRINGMAXLEN_IMAGE_NAME - 1] = '\0';
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

    if (triggermode == PROCESSINFO_TRIGGERMODE_CNT0)
    {
        DEBUG_TRACEPOINT("trigger mode %d = cnt0", PROCESSINFO_TRIGGERMODE_CNT0);

        if (image == NULL)
        {
            PRINT_ERROR("missing trigger image");
            return RETURN_FAILURE;
        }
        // trigger on cnt0 increment
        processinfo->triggerstreamcnt = image->md[0].cnt0;
    }

    if (triggermode == PROCESSINFO_TRIGGERMODE_CNT1)
    {
        DEBUG_TRACEPOINT("trigger mode %d = cnt1", PROCESSINFO_TRIGGERMODE_CNT1);

        if (image == NULL)
        {
            PRINT_ERROR("missing trigger image");
            return RETURN_FAILURE;
        }
        // trigger on cnt1 increment
        processinfo->triggerstreamcnt = image->md[0].cnt1;
    }

    if (triggermode == PROCESSINFO_TRIGGERMODE_CNT2)
    {
        DEBUG_TRACEPOINT("trigger mode %d = cnt2", PROCESSINFO_TRIGGERMODE_CNT2);

        if (image == NULL)
        {
            PRINT_ERROR("missing trigger image");
            return RETURN_FAILURE;
        }
        // trigger on cnt0 < cnt2
        processinfo->triggerstreamcnt = image->md[0].cnt0;
    }

    if (triggermode == PROCESSINFO_TRIGGERMODE_IMMEDIATE)
    {
        DEBUG_TRACEPOINT("trigger mode %d = immediate", PROCESSINFO_TRIGGERMODE_IMMEDIATE);
        // immmediate trigger
        processinfo->triggerstreamcnt = 0;
    }

    if (triggermode == PROCESSINFO_TRIGGERMODE_DELAY)
    {
        DEBUG_TRACEPOINT("trigger mode %d = time delay", PROCESSINFO_TRIGGERMODE_DELAY);
        // time wait
        processinfo->triggerstreamcnt = 0;
    }

    // checking if semaphore trigger mode OK
    if (triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE ||
        triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE_PROP_TIMEOUTS)
    {
        DEBUG_TRACEPOINT("trigger mode %d = semaphore %d", triggermode, semindexrequested);
        if (semindexrequested < -1)
        {
            PRINT_ERROR("invalid semaphore index %d", semindexrequested);
            return RETURN_FAILURE;
        }
        if (image == NULL)
        {
            PRINT_ERROR("image not valid");
            return RETURN_FAILURE;
        }
        processinfo->triggersem = ImageStreamIO_getsemwaitindex(image, semindexrequested);
        if (processinfo->triggersem == -1)
        {
            PRINT_ERROR("fallback to CNT0 triggermode (busywait)");
            processinfo->triggermode      = PROCESSINFO_TRIGGERMODE_CNT0;
            processinfo->triggerstreamcnt = image->md[0].cnt0;
        }
        else
        {
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
    IMAGE *image                    = processinfo->trigger_image;
    processinfo->triggermissedframe = 0;

    if (processinfo->triggermode == PROCESSINFO_TRIGGERMODE_IMMEDIATE)
    {
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;
        // return immediately
        return RETURN_SUCCESS;
    }

    if (processinfo->triggermode == PROCESSINFO_TRIGGERMODE_CNT0)
    {
        if (image == NULL)
        {
            return RETURN_FAILURE;
        }
        // use cnt0
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        while (image->md[0].cnt0 == processinfo->triggerstreamcnt)
        {
            // test if new frame exists
            usleep(5);
        }
        processinfo->triggermissedframe = image->md[0].cnt0 - processinfo->triggerstreamcnt - 1;
        // update trigger counter
        processinfo->triggerstreamcnt = image->md[0].cnt0;

        processinfo->triggermissedframe_cumul += processinfo->triggermissedframe;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if (processinfo->triggermode == PROCESSINFO_TRIGGERMODE_CNT1)
    {
        if (image == NULL)
        {
            return RETURN_FAILURE;
        }
        // use cnt1
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        while (image->md[0].cnt1 == processinfo->triggerstreamcnt)
        {
            // test if new frame exists
            usleep(5);
        }
        processinfo->triggermissedframe = image->md[0].cnt1 - processinfo->triggerstreamcnt - 1;
        // update trigger counter
        processinfo->triggerstreamcnt = image->md[0].cnt1;

        processinfo->triggermissedframe_cumul += processinfo->triggermissedframe;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if (processinfo->triggermode == PROCESSINFO_TRIGGERMODE_CNT2)
    {
        if (image == NULL)
        {
            return RETURN_FAILURE;
        }
        // use cnt2
        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        while (image->md[0].cnt0 >= image->md[0].cnt2)
        {
            // wait until we are allowed to proceed
            usleep(5);
        }
        processinfo->triggermissedframe = 0;

        // update trigger counter
        processinfo->triggerstreamcnt = image->md[0].cnt0;

        processinfo->triggermissedframe_cumul += processinfo->triggermissedframe;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if (processinfo->triggermode == PROCESSINFO_TRIGGERMODE_DELAY)
    {
        /* Return at the next tick of a fixed period, on an absolute deadline: a relative nanosleep
         * drifts by the loop's own run time and, under the default 50 us timer slack, cannot do
         * periods below ~60 us. One trigger loop per process: the deadline is process state. */
        static struct timespec deadline;
        static int             armed     = 0;
        static int             slack_set = 0;
        const struct timespec  d         = processinfo->triggerdelay;
        struct timespec        now;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (!slack_set && d.tv_sec == 0 && d.tv_nsec < 1000000L)
        {
            (void) prctl(PR_SET_TIMERSLACK, 1000UL); /* 1 us: sub-millisecond periods are meant */
            slack_set = 1;
        }
        if (!armed)
        {
            deadline = now;
            armed    = 1;
        }
        deadline.tv_sec += d.tv_sec;
        deadline.tv_nsec += d.tv_nsec;
        if (deadline.tv_nsec >= 1000000000L)
        {
            deadline.tv_sec++;
            deadline.tv_nsec -= 1000000000L;
        }
        /* fell behind by more than one period (a long iteration, a pause): resume from now, no burst */
        int64_t late_ns = (int64_t) (now.tv_sec - deadline.tv_sec) * 1000000000LL +
                          (now.tv_nsec - deadline.tv_nsec);
        if (late_ns > (int64_t) d.tv_sec * 1000000000LL + d.tv_nsec)
        {
            deadline = now;
        }
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL) == EINTR)
        {
        }
        processinfo->triggerstreamcnt++;

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        return RETURN_SUCCESS;
    }

    if (processinfo->triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE ||
        processinfo->triggermode == PROCESSINFO_TRIGGERMODE_SEMAPHORE_PROP_TIMEOUTS)
    {
        if (image == NULL)
        {
            return RETURN_FAILURE;
        }

        int semr;
        int tmpstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;

        DEBUG_TRACEPOINT("wait on semaphore");

        processinfo->triggerstatus = PROCESSINFO_TRIGGERSTATUS_WAITING;

        // get current time
        struct timespec ts;
        if (clock_gettime(CLOCK_MILK, &ts) == -1)
        {
            PRINT_ERROR("clock_gettime: %s", strerror(errno));
            exit(EXIT_FAILURE);
        }

        // is semaphore at zero ?
        DEBUG_TRACEPOINT("test sem status");
        semr = 0;
        while (semr == 0)
        {
            // this should only run once, returning semr = -1 with errno = EAGAIN
            // otherwise, we're potentially missing frames
            DEBUG_TRACEPOINT("sem_trywait");
            semr = ImageStreamIO_semtrywait(image, processinfo->triggersem);
            if (semr == 0)
            {
                processinfo->triggermissedframe++;
            }
        }

        // expected state: NBmissedframe = 0, semr = -1, errno = EAGAIN
        // missed frame state: NBmissedframe>0, semr = -1, errno = EAGAIN
        DEBUG_TRACEPOINT("triggermissedframe = %d", processinfo->triggermissedframe);
        if (processinfo->triggermissedframe == 0)
        {
            DEBUG_TRACEPOINT("timedwait");
            // add timeout
            long timeout_sec  = processinfo->triggertimeout.tv_sec;
            long timeout_nsec = processinfo->triggertimeout.tv_nsec;
            if (timeout_sec == 0 && timeout_nsec == 0)
            {
                timeout_sec = 2;
            }
            ts.tv_sec += timeout_sec;
            ts.tv_nsec += timeout_nsec;
            while (ts.tv_nsec > 1000000000)
            {
                ts.tv_nsec -= 1000000000;
                ts.tv_sec++;
            }

            semr = ImageStreamIO_semtimedwait(image, processinfo->triggersem, &ts);
            if (semr == -1)
            {
                if (errno == ETIMEDOUT)
                {
                    // timeout condition
                    processinfo->trigggertimeoutcnt++;
                    tmpstatus = PROCESSINFO_TRIGGERSTATUS_TIMEDOUT;
                    trigger_timeout_recheck(processinfo, image);
                }
            }
            else
            {
                tmpstatus = PROCESSINFO_TRIGGERSTATUS_RECEIVED;
            }
        }

        processinfo->triggermissedframe_cumul += processinfo->triggermissedframe;

        processinfo->triggerstatus = tmpstatus;

        return RETURN_SUCCESS;
    }

    return RETURN_FAILURE;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CLIcore.h"
#include <processtools.h>

/** @brief Update ouput stream at completion of processinfo-enabled loop
 * iteration
 *
 */

errno_t processinfo_update_output_stream_atime(PROCESSINFO     *processinfo,
                                               imageID          outstreamID,
                                               struct timespec *atime)
{
    if (data.image[outstreamID].md->shared == 1)
    {
        // Always update PID and timestamp, regardless of processinfo status
        struct timespec ts;
        if (clock_gettime(CLOCK_MILK, &ts) == -1)
        {
            perror("clock_gettime");
            exit(EXIT_FAILURE);
        }

        data.image[outstreamID].streamproctrace[0].procwrite_PID   = getpid();
        data.image[outstreamID].streamproctrace[0].ts_streamupdate = ts;

        DEBUG_TRACEPOINT(" ");

        if (processinfo != NULL)
        {
            imageID IDin = processinfo->triggerstreamID;
            DEBUG_TRACEPOINT("trigger IDin = %ld", IDin);

            if (IDin > -1)
            {
                int sptisize = data.image[IDin].md[0].NBproctrace - 1;

                // copy streamproctrace from input to output
                memcpy(&data.image[outstreamID].streamproctrace[1],
                       &data.image[IDin].streamproctrace[0], sizeof(STREAM_PROC_TRACE) * sptisize);
            }

            // write first streamproctrace entry
            DEBUG_TRACEPOINT("trigger info");
            data.image[outstreamID].streamproctrace[0].trigsemindex = processinfo->triggermode;

            data.image[outstreamID].streamproctrace[0].trigger_inode =
                processinfo->triggerstreaminode;

            data.image[outstreamID].streamproctrace[0].ts_procstart =
                processinfo->texecstart[processinfo->timerindex];

            data.image[outstreamID].streamproctrace[0].trigsemindex = processinfo->triggersem;

            data.image[outstreamID].streamproctrace[0].triggerstatus = processinfo->triggerstatus;

            if (IDin > -1)
            {
                data.image[outstreamID].streamproctrace[0].cnt0 = data.image[IDin].md[0].cnt0;
            }
        }

        DEBUG_TRACEPOINT(" ");
    }

    ImageStreamIO_UpdateIm_atime(&data.image[outstreamID], atime);

    return RETURN_SUCCESS;
}

errno_t processinfo_update_output_stream(PROCESSINFO *processinfo, imageID outstreamID)
{
    processinfo_update_output_stream_atime(processinfo, outstreamID, NULL);
}

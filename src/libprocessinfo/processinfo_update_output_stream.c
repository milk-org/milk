#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_update_output_stream.h"
#include "ImageStreamIO/ImageStreamIO.h"

#ifndef CLOCK_MILK
#define CLOCK_MILK CLOCK_REALTIME
#endif

/**
 * @brief Update output stream metadata and telemetry at the end of a loop iteration.
 */
errno_t processinfo_update_output_stream(
    PROCESSINFO *processinfo,
    IMAGE        *output_image,
    IMAGE        *input_image
)
{
    if(output_image == NULL)
    {
        return RETURN_FAILURE;
    }

    if(output_image->md->shared == 1)
    {
        // Always update PID and timestamp, regardless of processinfo status
        struct timespec ts;
        if(clock_gettime(CLOCK_MILK, &ts) == -1)
        {
            perror("clock_gettime");
            exit(EXIT_FAILURE);
        }

        output_image->streamproctrace[0].procwrite_PID = getpid();
        output_image->streamproctrace[0].ts_streamupdate = ts;

        DEBUG_TRACEPOINT(" ");

        if(processinfo != NULL)
        {
            // If input_image is NULL, try to use processinfo->trigger_image
            if(input_image == NULL)
            {
                if(processinfo->trigger_image != NULL)
                {
                    input_image = processinfo->trigger_image;
                }
            }

            if(input_image != NULL)
            {
                int sptisize = input_image->md[0].NBproctrace - 1;
                // Ensure we don't overflow output trace
                int sptosize = output_image->md[0].NBproctrace - 1;
                if(sptisize > sptosize)
                {
                    sptisize = sptosize;
                }

                if(sptisize > 0)
                {
                    // copy streamproctrace from input to output
                    memcpy(&output_image->streamproctrace[1],
                           &input_image->streamproctrace[0],
                           sizeof(STREAM_PROC_TRACE) * sptisize);
                }
            }

            // write first streamproctrace entry
            DEBUG_TRACEPOINT("trigger info");
            output_image->streamproctrace[0].trigsemindex =
                processinfo->triggermode;

            output_image->streamproctrace[0].trigger_inode =
                processinfo->triggerstreaminode;

            output_image->streamproctrace[0].ts_procstart =
                processinfo->texecstart[processinfo->timerindex];

            output_image->streamproctrace[0].trigsemindex =
                processinfo->triggersem;

            output_image->streamproctrace[0].triggerstatus =
                processinfo->triggerstatus;

            if(input_image != NULL)
            {
                output_image->streamproctrace[0].cnt0 =
                    input_image->md[0].cnt0;
            }
        }

        DEBUG_TRACEPOINT(" ");
    }

    ImageStreamIO_UpdateIm(output_image);

    return RETURN_SUCCESS;
}
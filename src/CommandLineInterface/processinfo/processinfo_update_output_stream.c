#include "CLIcore.h"
#include <processtools.h>




/** @brief Update ouput stream at completion of processinfo-enabled loop iteration
 *
 */

errno_t processinfo_update_output_stream(
    PROCESSINFO *processinfo,
    imageID      outstreamID
)
{
    if(data.image[outstreamID].md->shared == 1)
    {
        imageID IDin;

        DEBUG_TRACEPOINT(" ");

        if(processinfo != NULL)
        {
            IDin = processinfo->triggerstreamID;
            DEBUG_TRACEPOINT("trigger IDin = %ld", IDin);

            if(IDin > -1)
            {
                int sptisize = data.image[IDin].md[0].NBproctrace - 1;

                // copy streamproctrace from input to output
                memcpy(&data.image[outstreamID].streamproctrace[1],
                       &data.image[IDin].streamproctrace[0],
                       sizeof(STREAM_PROC_TRACE) * sptisize);
            }

            DEBUG_TRACEPOINT("timing");
            struct timespec ts;
            if(clock_gettime(CLOCK_MILK, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }

            // write first streamproctrace entry
            DEBUG_TRACEPOINT("trigger info");
            data.image[outstreamID].streamproctrace[0].trigsemindex =
                processinfo->triggermode;

            data.image[outstreamID].streamproctrace[0].procwrite_PID = getpid();

            data.image[outstreamID].streamproctrace[0].trigger_inode =
                processinfo->triggerstreaminode;

            data.image[outstreamID].streamproctrace[0].ts_procstart =
                processinfo->texecstart[processinfo->timerindex];

            data.image[outstreamID].streamproctrace[0].ts_streamupdate = ts;

            data.image[outstreamID].streamproctrace[0].trigsemindex =
                processinfo->triggersem;

            data.image[outstreamID].streamproctrace[0].triggerstatus =
                processinfo->triggerstatus;

            if(IDin > -1)
            {
                data.image[outstreamID].streamproctrace[0].cnt0 =
                    data.image[IDin].md[0].cnt0;
            }
        }

        DEBUG_TRACEPOINT(" ");
    }

    ImageStreamIO_UpdateIm(&data.image[outstreamID]);

    return RETURN_SUCCESS;
}

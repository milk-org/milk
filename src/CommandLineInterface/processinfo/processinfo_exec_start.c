#include "CLIcore.h"
#include <processtools.h>


int processinfo_exec_start(PROCESSINFO *processinfo)
{
    DEBUG_TRACEPOINT(" ");
    if(processinfo->MeasureTiming == 1)
    {

        processinfo->timerindex++;
        if(processinfo->timerindex == PROCESSINFO_NBtimer)
        {
            processinfo->timerindex = 0;
            processinfo->timingbuffercnt++;
        }

        clock_gettime(CLOCK_REALTIME,
                      &processinfo->texecstart[processinfo->timerindex]);

        if(processinfo->dtiter_limit_enable != 0)
        {
            long dtiter;
            int  timerindexlast;

            if(processinfo->timerindex == 0)
            {
                timerindexlast = PROCESSINFO_NBtimer - 1;
            }
            else
            {
                timerindexlast = processinfo->timerindex - 1;
            }

            dtiter = processinfo->texecstart[processinfo->timerindex].tv_nsec -
                     processinfo->texecstart[timerindexlast].tv_nsec;
            dtiter += 1000000000 *
                      (processinfo->texecstart[processinfo->timerindex].tv_sec -
                       processinfo->texecstart[timerindexlast].tv_sec);

            if(dtiter > processinfo->dtiter_limit_value)
            {
                char msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

                {
                    int slen =
                        snprintf(msgstring,
                                 STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                 "dtiter %4ld  %4d %6.1f us  > %6.1f us",
                                 processinfo->dtiter_limit_cnt,
                                 processinfo->timerindex,
                                 0.001 * dtiter,
                                 0.001 * processinfo->dtiter_limit_value);
                    if(slen < 1)
                    {
                        PRINT_ERROR("snprintf wrote <1 char");
                        abort(); // can't handle this error any other way
                    }
                    if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
                    {
                        PRINT_ERROR("snprintf string truncation");
                        abort(); // can't handle this error any other way
                    }
                }

                processinfo_WriteMessage(processinfo, msgstring);

                if(processinfo->dtiter_limit_enable ==
                        2) // pause process due to timing limit
                {
                    processinfo->CTRLval = 1;
                    sprintf(msgstring, "dtiter lim -> paused");
                    processinfo_WriteMessage(processinfo, msgstring);
                }
                processinfo->dtiter_limit_cnt++;
            }
        }
    }
    DEBUG_TRACEPOINT(" ");
    return 0;
}

#include "CLIcore.h"
#include <processtools.h>


int processinfo_exec_end(PROCESSINFO *processinfo)
{
    int loopOK = 1;

    DEBUG_TRACEPOINT("End of execution loop, measure timing = %d",
                     processinfo->MeasureTiming);
    if(processinfo->MeasureTiming == 1)
    {
        clock_gettime(CLOCK_MILK,
                      &processinfo->texecend[processinfo->timerindex]);

        if(processinfo->dtexec_limit_enable != 0)
        {
            long dtexec;

            dtexec = processinfo->texecend[processinfo->timerindex].tv_nsec -
                     processinfo->texecstart[processinfo->timerindex].tv_nsec;
            dtexec += 1000000000 *
                      (processinfo->texecend[processinfo->timerindex].tv_sec -
                       processinfo->texecend[processinfo->timerindex].tv_sec);

            if(dtexec > processinfo->dtexec_limit_value)
            {
                char msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

                {
                    int slen =
                        snprintf(msgstring,
                                 STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                 "dtexec %4ld  %4d %6.1f us  > %6.1f us",
                                 processinfo->dtexec_limit_cnt,
                                 processinfo->timerindex,
                                 0.001 * dtexec,
                                 0.001 * processinfo->dtexec_limit_value);
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

                if(processinfo->dtexec_limit_enable ==
                        2) // pause process due to timing limit
                {
                    processinfo->CTRLval = 1;
                    {
                        int slen = snprintf(msgstring,
                                            STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                            "dtexec lim -> paused");
                        if(slen < 1)
                        {
                            PRINT_ERROR("snprintf wrote <1 char");
                            abort(); // can't handle this error any other way
                        }
                        if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
                        {
                            PRINT_ERROR(
                                "snprintf string "
                                "truncation");
                            abort(); // can't handle this error any other way
                        }
                    }

                    processinfo_WriteMessage(processinfo, msgstring);
                }
                processinfo->dtexec_limit_cnt++;
            }
        }
    }
    DEBUG_TRACEPOINT("End of execution loop: check signals");
    loopOK = processinfo_ProcessSignals(processinfo);

    processinfo->loopcnt++;

    return loopOK; // returns 0 if signal stops loop
}

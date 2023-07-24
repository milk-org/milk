#include "CLIcore.h"
#include <processtools.h>


/**
 * @brief Return loop status
 *
 * 0 if loop should exit
 * 1 otherwise
 *
 * @param processinfo
 * @return int loop status
 */
int processinfo_loopstep(
    PROCESSINFO *processinfo
)
{
    int loopstatus = 1;

    while(processinfo->CTRLval == 1)  // pause
    {
        usleep(50);
    }
    if(processinfo->CTRLval == 2)  // single iteration
    {
        processinfo->CTRLval = 1;
    }
    if(processinfo->CTRLval == 3)  // exit loop
    {
        loopstatus = 0;
    }

    if(data.signal_INT == 1)  // CTRL-C
    {
        loopstatus = 0;
    }

    if(data.signal_HUP == 1)  // terminal has disappeared
    {
        loopstatus = 0;
    }

    if(processinfo->loopcntMax != -1)
        if(processinfo->loopcnt >= processinfo->loopcntMax - 1)
        {
            loopstatus = 0;
        }

    return loopstatus;
}

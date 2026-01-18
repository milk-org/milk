#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_loopstep.h"
#include "processinfo_signals.h"


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

    if(processinfo_signal_INT == 1)  // CTRL-C
    {
        loopstatus = 0;
    }

    if(processinfo_signal_HUP == 1)  // terminal has disappeared
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
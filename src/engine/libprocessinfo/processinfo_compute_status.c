/**
 * @file processinfo_compute_status.c
 * @brief Processinfo compute status module
 */

#include "processinfo.h"

int processinfo_compute_status(PROCESSINFO *processinfo)
{
    int processcompstatus = 1;

    if(processinfo->CTRLval == 5)
    {
        processcompstatus = 0;
    }

    return processcompstatus;
}

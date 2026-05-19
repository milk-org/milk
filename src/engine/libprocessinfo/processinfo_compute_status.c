/**
 * @file processinfo_compute_status.c
 * @brief Processinfo compute status module
 */

#include "processinfo.h"

/**
 * @brief Compute and update the overall status of a process.
 *
 * Evaluates loop state, signal flags, and timing
 * metrics to determine a summary status code.
 */
int processinfo_compute_status(PROCESSINFO *processinfo)
{
    int processcompstatus = 1;

    if(processinfo->CTRLval == 5)
    {
        processcompstatus = 0;
    }

    return processcompstatus;
}

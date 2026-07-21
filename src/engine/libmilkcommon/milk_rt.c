// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_rt.h"
#include "milkDebugTools.h"

#include "unistd.h" // getpid()

int milkrt_RTPrio(const int rtprio)
{
    if (rtprio <= 0)
    {
        PRINT_WARNING("Invoking RT prio with rtprio %d <= 0; skipping.", rtprio);
        return RETURN_SUCCESS;
    }

    EXECUTE_SYSTEM_COMMAND("milk-makecsetandrt -p %d %d", rtprio, getpid());

    return RETURN_SUCCESS;
}

int milkrt_Tset(const char *tsetspec)
{
    // Pass down to extended version and return retcode back up
    return milkrt_TsetExt(getpid(), tsetspec);
}

int milkrt_TsetExt(const int pid, const char *tsetspec)
{
    EXECUTE_SYSTEM_COMMAND("milk-makecsetandrt -t %s %d", tsetspec, pid);
    return 0;
}

int milkrt_CPUset(const char *csetname)
{
    // Pass down to extended version and return retcode back up
    return milkrt_CPUsetExt(getpid(), csetname, -1);
}

int milkrt_CPUsetExt(const int pid, const char *csetname, const int rtprio)
{
    if (rtprio > 0)
    {
        EXECUTE_SYSTEM_COMMAND("milk-makecsetandrt -c %s -p %d %d", csetname, rtprio, pid);
    }
    else
    {
        EXECUTE_SYSTEM_COMMAND("milk-makecsetandrt -c %s %d", csetname, pid);
    }

    return EXIT_SUCCESS;
}

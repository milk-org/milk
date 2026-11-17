// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_rt.h"
#include "milkDebugTools.h"
#include "milk_spawn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // getpid()

#define MILKRT_HELPER "milk-makecsetandrt"

/* argv[0] is filled in with the setuid helper of this install; the other arguments (cpuset names, CPU
 * lists from the FPS) never reach a shell. */
static void run_helper(const char *argv[])
{
    char path[4096];

    milkspawn_tool_path(MILKRT_HELPER, path, sizeof(path));
    argv[0] = path;
    int rc  = milkspawn_run(argv, STDOUT_FILENO, STDERR_FILENO);
    if (rc != 0)
    {
        PRINT_ERROR("%s failed with code %d", path, rc);
    }
}

int milkrt_RTPrio(const int rtprio)
{
    if (rtprio <= 0)
    {
        PRINT_WARNING("Invoking RT prio with rtprio %d <= 0; skipping.", rtprio);
        return RETURN_SUCCESS;
    }

    char prio[16];
    char pid[16];
    snprintf(prio, sizeof(prio), "%d", rtprio);
    snprintf(pid, sizeof(pid), "%d", (int) getpid());
    const char *argv[] = { NULL, "-p", prio, pid, NULL };
    run_helper(argv);

    return RETURN_SUCCESS;
}

int milkrt_Tset(const char *tsetspec)
{
    // Pass down to extended version and return retcode back up
    return milkrt_TsetExt(getpid(), tsetspec);
}

int milkrt_TsetExt(const int pid, const char *tsetspec)
{
    char pidstr[16];
    snprintf(pidstr, sizeof(pidstr), "%d", pid);
    const char *argv[] = { NULL, "-t", tsetspec, pidstr, NULL };
    run_helper(argv);
    return 0;
}

int milkrt_CPUset(const char *csetname)
{
    // Pass down to extended version and return retcode back up
    return milkrt_CPUsetExt(getpid(), csetname, -1);
}

int milkrt_CPUsetExt(const int pid, const char *csetname, const int rtprio)
{
    char pidstr[16];
    char prio[16];
    snprintf(pidstr, sizeof(pidstr), "%d", pid);
    snprintf(prio, sizeof(prio), "%d", rtprio);
    const char *with_prio[] = { NULL, "-c", csetname, "-p", prio, pidstr, NULL };
    const char *no_prio[]   = { NULL, "-c", csetname, pidstr, NULL };
    run_helper((rtprio > 0) ? with_prio : no_prio);

    return EXIT_SUCCESS;
}

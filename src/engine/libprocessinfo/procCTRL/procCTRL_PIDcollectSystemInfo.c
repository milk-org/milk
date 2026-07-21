// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file procCTRL_PIDcollectSystemInfo.c
 * @brief Procctrl pidcollectsysteminfo module
 */

#include <string.h>
#include <math.h>

#include "procCTRL_PIDcollectSystemInfo.h"

/**
 * @brief Collect system-level info for a PID.
 *
 * Reads /proc/<pid>/status and /proc/<pid>/stat
 * to gather memory usage, CPU time, and scheduling
 * data.
 */
int PIDcollectSystemInfo(PROCESSINFODISP *pinfodisp, int mode)
{
    char  procfname[200];
    FILE *fp;
    char  line[1024];

    (void) mode;

    if (pinfodisp->PID <= 0)
    {
        return -1;
    }

    // 1. Get info from /proc/PID/status
    snprintf(procfname, 200, "/proc/%d/status", (int) pinfodisp->PID);
    fp = fopen(procfname, "r");
    if (fp != NULL)
    {
        while (fgets(line, sizeof(line) - 1, fp) != NULL)
        {
            if (strncmp(line, "Threads:", 8) == 0)
            {
                sscanf(line + 8, "%d", &pinfodisp->threads);
            }
            if (strncmp(line, "VmRSS:", 6) == 0)
            {
                long vmrss;
                sscanf(line + 6, "%ld", &vmrss);
                pinfodisp->VmRSSarray[0] = vmrss;
            }
            if (strncmp(line, "Cpus_allowed_list:", 13) == 0)
            {
                sscanf(line + 13, "%s", pinfodisp->cpusallowed);
            }
            if (strncmp(line, "State:", 6) == 0)
            {
                sscanf(line + 6, " %c", &pinfodisp->state);
            }
        }
        fclose(fp);
    }

    // 2. Get timing info from /proc/PID/stat
    long utime = 0, stime = 0;
    snprintf(procfname, 200, "/proc/%d/stat", (int) pinfodisp->PID);
    fp = fopen(procfname, "r");
    if (fp != NULL)
    {
        if (fgets(line, sizeof(line) - 1, fp) != NULL)
        {
            char *p = strrchr(line, ')');
            if (p)
            {
                p += 2; // Skip ") "

                int   field = 0;
                char *token = strtok(p, " ");
                while (token != NULL)
                {
                    if (field == 11)
                    {
                        utime = atol(token);
                    }
                    if (field == 12)
                    {
                        stime = atol(token);
                        break;
                    }
                    token = strtok(NULL, " ");
                    field++;
                }
            }
        }
        fclose(fp);
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = (double) ts.tv_sec + 1.0e-9 * ts.tv_nsec;

    // Use local static tracking to avoid SHM interference
    static struct
    {
        pid_t     PID;
        long long prev_total_time;
        double    prev_sample_time;
    } pid_track[PROCESSINFOLISTSIZE];
    static int track_init = 0;
    if (!track_init)
    {
        for (int i = 0; i < PROCESSINFOLISTSIZE; i++)
        {
            pid_track[i].PID = 0;
        }
        track_init = 1;
    }

    // Find or assign slot for this PID (using pindex if available and stable)
    int slot = -1;
    if (pinfodisp->pindex >= 0 && pinfodisp->pindex < PROCESSINFOLISTSIZE)
    {
        slot = pinfodisp->pindex;
    }

    if (slot >= 0)
    {
        if (pid_track[slot].PID != pinfodisp->PID)
        {
            // New PID in this slot or first time
            pid_track[slot].PID                            = pinfodisp->PID;
            pid_track[slot].prev_total_time                = utime + stime;
            pid_track[slot].prev_sample_time               = now;
            pinfodisp->subprocCPUloadarray[0]              = 0.0;
            pinfodisp->subprocCPUloadarray_timeaveraged[0] = 0.0;
        }
        else
        {
            double dt = now - pid_track[slot].prev_sample_time;
            if (dt > 0.01)
            {
                long long total_time    = utime + stime;
                long long dcnt          = total_time - pid_track[slot].prev_total_time;
                long      ticks_per_sec = sysconf(_SC_CLK_TCK);

                float instantaneous_cpu           = 100.0 * (double) dcnt / (dt * ticks_per_sec);
                pinfodisp->subprocCPUloadarray[0] = instantaneous_cpu;

                // 1s averaged CPU usage using Exponential Moving Average
                float tau   = 1.0;
                float alpha = 1.0f - expf(-dt / tau);
                pinfodisp->subprocCPUloadarray_timeaveraged[0] =
                    (1.0 - alpha) * pinfodisp->subprocCPUloadarray_timeaveraged[0] +
                    alpha * instantaneous_cpu;

                pid_track[slot].prev_total_time  = total_time;
                pid_track[slot].prev_sample_time = now;
            }
        }
    }

    return 0;
}

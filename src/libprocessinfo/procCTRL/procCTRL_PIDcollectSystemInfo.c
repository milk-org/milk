#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <math.h>

#include "processinfo.h"
#include "procCTRL_PIDcollectSystemInfo.h"

int PIDcollectSystemInfo(PROCESSINFODISP *pinfodisp, int mode)
{
    char procfname[200];
    FILE *fp;
    char line[1024];

    (void)mode;

    if (pinfodisp->PID <= 0) return -1;

    // 1. Get info from /proc/PID/status
    snprintf(procfname, 200, "/proc/%d/status", (int)pinfodisp->PID);
    fp = fopen(procfname, "r");
    if (fp != NULL) {
        while (fgets(line, sizeof(line)-1, fp) != NULL) {
            if (strncmp(line, "Threads:", 8) == 0) {
                sscanf(line + 8, "%d", &pinfodisp->threads);
            }
            if (strncmp(line, "VmRSS:", 6) == 0) {
                long vmrss;
                sscanf(line + 6, "%ld", &vmrss);
                pinfodisp->VmRSSarray[0] = vmrss;
            }
            if (strncmp(line, "Cpus_allowed_list:", 13) == 0) {
                sscanf(line + 13, "%s", pinfodisp->cpusallowed);
            }
            if (strncmp(line, "State:", 6) == 0) {
                sscanf(line + 6, " %c", &pinfodisp->state);
            }
        }
        fclose(fp);
    }

    // 2. Get timing info from /proc/PID/stat
    snprintf(procfname, 200, "/proc/%d/stat", (int)pinfodisp->PID);
    fp = fopen(procfname, "r");
    if (fp != NULL) {
        if (fgets(line, sizeof(line)-1, fp) != NULL) {
            char *p = strrchr(line, ')');
            if (p) {
                p += 2; // Skip ") "
                long utime = 0, stime = 0;
                
                int field = 0;
                char *token = strtok(p, " ");
                while (token != NULL) {
                    if (field == 11) utime = atol(token);
                    if (field == 12) {
                        stime = atol(token);
                        break;
                    }
                    token = strtok(NULL, " ");
                    field++;
                }
                
                if (pinfodisp->sampletimearray[0] > 0) {
                    pinfodisp->cpuloadcntarray_prev[0] = pinfodisp->cpuloadcntarray[0];
                    pinfodisp->cpuloadcntarray[0] = utime + stime;
                } else {
                    // First time we see this PID
                    pinfodisp->cpuloadcntarray[0] = utime + stime;
                    pinfodisp->cpuloadcntarray_prev[0] = utime + stime;
                }
            }
        }
        fclose(fp);
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = (double)ts.tv_sec + 1.0e-9 * ts.tv_nsec;

    if (pinfodisp->sampletimearray[0] > 0) {
        double dt = now - pinfodisp->sampletimearray[0];
        if (dt > 0.05) { // Ensure at least 50ms between samples
            long long dcnt = pinfodisp->cpuloadcntarray[0] - pinfodisp->cpuloadcntarray_prev[0];
            long ticks_per_sec = sysconf(_SC_CLK_TCK);
            
            float instantaneous_cpu = 100.0 * (double)dcnt / (dt * ticks_per_sec);
            pinfodisp->subprocCPUloadarray[0] = instantaneous_cpu;
            
            // 1s averaged CPU usage using Exponential Moving Average
            // alpha = 1 - exp(-dt / tau), where tau = 1.0s
            float alpha = 1.0 - exp(-dt / 1.0);
            pinfodisp->subprocCPUloadarray_timeaveraged[0] = (1.0 - alpha) * pinfodisp->subprocCPUloadarray_timeaveraged[0] + alpha * instantaneous_cpu;

            pinfodisp->sampletimearray_prev[0] = pinfodisp->sampletimearray[0];
            pinfodisp->sampletimearray[0] = now;
        }
    } else {
        // First sample
        pinfodisp->sampletimearray[0] = now;
        pinfodisp->sampletimearray_prev[0] = now;
        pinfodisp->subprocCPUloadarray[0] = 0.0;
        pinfodisp->subprocCPUloadarray_timeaveraged[0] = 0.0;
    }

    return 0;
}
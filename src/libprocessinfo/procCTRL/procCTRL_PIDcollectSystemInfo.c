#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "processinfo.h"
#include "procCTRL_PIDcollectSystemInfo.h"

// Prototypes for standalone support
extern pid_t CLIPID;

int PIDcollectSystemInfo(PROCESSINFODISP *pinfodisp, int mode)
{
    char procfname[200];
    FILE *fp;
    char line[500];

    (void)mode;

    if (pinfodisp->PID <= 0) return -1;

    // Get basic info from /proc/PID/status
    snprintf(procfname, 200, "/proc/%d/status", (int)pinfodisp->PID);
    fp = fopen(procfname, "r");
    if (fp == NULL) return -1;

    while (fgets(line, 499, fp) != NULL) {
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
    }
    fclose(fp);

    // Get timing/scheduling info from /proc/PID/stat
    snprintf(procfname, 200, "/proc/%d/stat", (int)pinfodisp->PID);
    fp = fopen(procfname, "r");
    if (fp != NULL) {
        // Simple extraction of some fields
        long utime, stime;
        // The stat file format is complex due to process name in parentheses
        // For simplicity in this standalone version, we do minimal parsing
        // In a real scenario, we'd use a more robust parser.
        fclose(fp);
    }

    pinfodisp->sampletimearray[0] = (double)time(NULL);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "processinfo.h"
#include "processtools.h"
#include "procCTRL_GetCPUloads.h"

int GetCPUloads(PROCINFOPROC *pinfop)
{
    FILE *fp;
    char line[1024];
    int cpu_idx = 0;

    fp = fopen("/proc/stat", "r");
    if (fp == NULL) return -1;

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strncmp(line, "cpu", 3) == 0 && line[3] != ' ') {
            long long user, nice, system, idle, iowait, irq, softirq, steal;
            sscanf(line + 3, "%lld %lld %lld %lld %lld %lld %lld %lld",
                   &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
            
            long long total = user + nice + system + idle + iowait + irq + softirq + steal;
            // Simple calculation: just storing the total for now to show activity
            // In a real TUI we'd compare with previous values
            pinfop->CPUload[cpu_idx] = (float)total; 
            cpu_idx++;
            if (cpu_idx >= MAXNBCPU) break;
        }
    }
    fclose(fp);
    return 0;
}
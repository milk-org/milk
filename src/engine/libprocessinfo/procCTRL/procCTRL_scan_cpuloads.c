/**
 * @file procCTRL_scan_cpuloads.c
 * @brief CPU load collection for procCTRL scanner
 */

#include <stdio.h>
#include <string.h>

#include "processinfo.h"
#include "procCTRL_scan_internal.h"

// Inline simplified CPU load collector
void Scan_GetCPUloads(PROCSCAN_SHM *scan_shm) {
    static long long prev_user[MAXNBCPU], prev_nice[MAXNBCPU], prev_system[MAXNBCPU];
    static long long prev_idle[MAXNBCPU], prev_iowait[MAXNBCPU], prev_irq[MAXNBCPU], prev_softirq[MAXNBCPU], prev_steal[MAXNBCPU];
    static int initialized = 0;

    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) return;

    char line[1024];
    int cpu_idx = 0;
    while (fgets(line, sizeof(line), fp) && cpu_idx < MAXNBCPU) {
        if (strncmp(line, "cpu", 3) == 0 && line[3] != ' ') { // cpu0, cpu1...
            long long user, nice, system, idle, iowait, irq, softirq, steal;
            if (sscanf(line + 3, "%lld %lld %lld %lld %lld %lld %lld %lld",
                   &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) == 8) {

                if (initialized) {
                    long long total_prev = prev_user[cpu_idx] + prev_nice[cpu_idx] + prev_system[cpu_idx] + prev_idle[cpu_idx] +
                                           prev_iowait[cpu_idx] + prev_irq[cpu_idx] + prev_softirq[cpu_idx] + prev_steal[cpu_idx];
                    long long total_cur = user + nice + system + idle + iowait + irq + softirq + steal;
                    long long total_diff = total_cur - total_prev;
                    long long idle_diff = idle - prev_idle[cpu_idx];
                    
                    if (total_diff > 0) {
                        scan_shm->CPUload[cpu_idx] = (float)(total_diff - idle_diff) / total_diff;
                    }
                }

                prev_user[cpu_idx] = user; prev_nice[cpu_idx] = nice; prev_system[cpu_idx] = system;
                prev_idle[cpu_idx] = idle; prev_iowait[cpu_idx] = iowait; prev_irq[cpu_idx] = irq;
                prev_softirq[cpu_idx] = softirq; prev_steal[cpu_idx] = steal;
                
                cpu_idx++;
            }
        }
    }
    scan_shm->NBcpus = cpu_idx;
    initialized = 1;
    fclose(fp);
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "perfbench.h"

#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

/**
 * Suppress -Wunused-result on calls where we
 * intentionally discard the return value.
 */
#ifndef IGNORE_RESULT
#    define IGNORE_RESULT(x) \
        do                   \
        {                    \
            if (x)           \
            {                \
            }                \
        } while (0)
#endif

/* ================================================================
 * perf_event_open syscall wrapper
 * ============================================================= */

static long _perf_event_open(struct perf_event_attr *attr,
                             pid_t                   pid,
                             int                     cpu,
                             int                     group_fd,
                             unsigned long           flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/* ================================================================
 * Open perf counters for a set of events
 * ============================================================= */

/**
 * @brief Open all perf_event fds for a child PID.
 *
 * Opens N_PERF_EVS file descriptors tracking the
 * given pid.  Descriptors are stored in fds[].
 * Returns number of successfully opened fds.
 * Events that fail (no kernel support or permissions)
 * are stored as -1 and skipped gracefully.
 *
 * @param pid    Process to monitor
 * @param fds    Output array of size N_PERF_EVS
 * @return       Number of fds successfully opened
 */
int perf_open_all(pid_t pid, int *fds)
{
    int nok = 0;
    for (int i = 0; i < N_PERF_EVS; i++)
    {
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.type           = PERF_EVS[i].type;
        attr.size           = sizeof(attr);
        attr.config         = PERF_EVS[i].config;
        attr.disabled       = 1;
        attr.exclude_kernel = 0;
        attr.exclude_hv     = 1;
        attr.inherit        = 1;

        fds[i] = (int) _perf_event_open(&attr, pid, -1, -1, 0);
        if (fds[i] >= 0)
        {
            ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
            ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
            nok++;
        }
    }
    return nok;
}

/**
 * @brief Disable, read, and close all perf fds.
 *
 * @param fds    Array of N_PERF_EVS fds
 * @param phase  Output hw_phase_t to fill
 */
void perf_read_close(int *fds, hw_phase_t *phase)
{
    phase->valid = 0;
    for (int i = 0; i < N_PERF_EVS; i++)
    {
        phase->v[i] = 0;
        if (fds[i] < 0)
        {
            continue;
        }
        ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
        IGNORE_RESULT(read(fds[i], &phase->v[i], sizeof(long long)));
        close(fds[i]);
        fds[i]       = -1;
        phase->valid = 1;
    }
}

/* All events we open.  Must stay in sync with idx_* enums. */
const perf_ev_t PERF_EVS[] = {
    { "cycles", "cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES },
    { "bus-cycles", "bus_cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES },
    { "instructions", "instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS },
    /* L1d */
    { "L1-dcache-loads", "L1_dcache_loads", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16) },
    { "L1-dcache-load-misses", "L1_dcache_misses", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    { "L1-dcache-stores", "L1_dcache_stores", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16) },
    /* L1i */
    { "L1-icache-loads", "L1_icache_loads", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16) },
    { "L1-icache-load-misses", "L1_icache_misses", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    /* iTLB */
    { "iTLB-load-misses", "iTLB_misses", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    /* LLC */
    { "LLC-loads", "LLC_loads", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16) },
    { "LLC-load-misses", "LLC_misses", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    { "LLC-stores", "LLC_stores", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16) },
    { "LLC-store-misses", "LLC_store_misses", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    /* dTLB */
    { "dTLB-loads", "dTLB_loads", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16) },
    { "dTLB-load-misses", "dTLB_misses", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    { "dTLB-store-misses", "dTLB_store_misses", PERF_TYPE_HW_CACHE,
      (PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    /* stalls */
    { "stalled-cycles-frontend", "stalled_cycles_frontend", PERF_TYPE_HARDWARE,
      PERF_COUNT_HW_STALLED_CYCLES_FRONTEND },
    { "stalled-cycles-backend", "stalled_cycles_backend", PERF_TYPE_HARDWARE,
      PERF_COUNT_HW_STALLED_CYCLES_BACKEND },
    /* branch */
    { "branches", "branches", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS },
    { "branch-misses", "branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES },
    /* software */
    { "page-faults", "page_faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS },
    { "minor-faults", "minor_faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN },
    { "major-faults", "major_faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ },
    { "cpu-migrations", "cpu_migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS },
    { "context-switches", "context_switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES },
    { "task-clock", "task_clock_ns", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK },
};

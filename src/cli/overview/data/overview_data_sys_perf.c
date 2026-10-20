// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

#define OV_PERF_NEV 6 /* instructions, cache, branch, L1D read, LLC read, DTLB read misses */

/* One counter group on the inspected process's main thread: the leader is the first event that opened */
static int s_perf_pid = -1;
static int s_perf_fd[OV_PERF_NEV];
static int s_perf_slot[OV_PERF_NEV]; /* event -> position in the group read, -1 = not counted */
static int s_perf_n;                 /* events in the group */
static int s_perf_leader = -1;
static uint64_t s_perf_prev[OV_PERF_NEV];
static int64_t  s_perf_prev_loopcnt;

/**
 * _perf_event_open - Direct syscall wrapper for perf_event_open
 * @attr:     Perf event attribute structure
 * @pid:      Process ID to monitor (-1 for any process)
 * @cpu:      CPU core to monitor (-1 for any CPU)
 * @group_fd: Group leader file descriptor or -1
 * @flags:    Syscall control flags
 *
 * Return: File descriptor on success, -1 on failure with errno set.
 */
static long _perf_event_open(struct perf_event_attr *attr,
                             pid_t                   pid,
                             int                     cpu,
                             int                     group_fd,
                             unsigned long           flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/** open_perf_fd - open one user-space counter of @pid in the group led by @leader (-1: as the leader). */
static int open_perf_fd(pid_t pid, uint32_t type, uint64_t config, int leader)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type           = type;
    attr.size           = sizeof(attr);
    attr.config         = config;
    attr.disabled       = (leader < 0); /* the group starts when its leader is enabled */
    attr.exclude_kernel = 1;            /* also allowed at perf_event_paranoid = 2 */
    attr.exclude_hv     = 1;
    attr.read_format =
        PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return (int) _perf_event_open(&attr, pid, -1, leader, PERF_FLAG_FD_CLOEXEC);
}

/** pid_perf_close - stop counting: close the group (RESOURCES hidden, or exit). */
void pid_perf_close(void)
{
    for (int e = 0; e < OV_PERF_NEV; e++)
    {
        if (s_perf_fd[e] > 0)
        {
            close(s_perf_fd[e]);
        }
        s_perf_fd[e]   = -1;
        s_perf_slot[e] = -1;
    }
    s_perf_pid    = -1;
    s_perf_leader = -1;
    s_perf_n      = 0;
}

/** perf_open_group - open the counters of @pid as one group, skipping events the CPU does not have. */
static void perf_open_group(pid_t pid)
{
    static const struct
    {
        uint32_t type;
        uint64_t config;
    } ev[OV_PERF_NEV] = {
        { PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS },
        { PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES },
        { PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES },
        { PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                  (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
        { PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                  (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
        { PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                  (PERF_COUNT_HW_CACHE_RESULT_MISS << 16) },
    };
    pid_perf_close();
    for (int e = 0; e < OV_PERF_NEV; e++)
    {
        int fd = open_perf_fd(pid, ev[e].type, ev[e].config, s_perf_leader);
        if (fd < 0)
        {
            continue;
        }
        s_perf_fd[e]   = fd;
        s_perf_slot[e] = s_perf_n++;
        s_perf_leader  = (s_perf_leader < 0) ? fd : s_perf_leader;
    }
    if (s_perf_leader >= 0)
    {
        ioctl(s_perf_leader, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(s_perf_leader, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    s_perf_pid = pid;
    memset(s_perf_prev, 0, sizeof(s_perf_prev));
}

/**
 * pid_read_perf_counters - hardware counters of @pid's main thread, and their increase per loop iteration
 * since the last read in which @loopcnt advanced. One read() for the whole group; counts are scaled when
 * the PMU multiplexed them. Requires CAP_PERFMON or root, or perf_event_paranoid <= 2.
 */
int pid_read_perf_counters(pid_t pid, int64_t loopcnt, ov_perf_counters_t *out)
{
    if (pid <= 0 || !out)
    {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (s_perf_pid != pid)
    {
        perf_open_group(pid);
        s_perf_prev_loopcnt = loopcnt;
    }
    uint64_t buf[3 + OV_PERF_NEV];
    if (s_perf_leader < 0 ||
        read(s_perf_leader, buf, sizeof(buf)) < (ssize_t) (3 * sizeof(uint64_t)))
    {
        return -1;
    }
    double  scale = (buf[2] > 0) ? (double) buf[1] / (double) buf[2] : 0.0; /* enabled / running */
    int64_t d_loops             = loopcnt - s_perf_prev_loopcnt;
    uint64_t *tot[OV_PERF_NEV]  = { &out->instructions, &out->cache_misses, &out->branch_misses,
                                    &out->l1d_misses,   &out->llc_misses,   &out->dtlb_misses };
    double   *rate[OV_PERF_NEV] = { &out->inst_per_loop,        &out->cache_miss_per_loop,
                                    &out->branch_miss_per_loop, &out->l1d_miss_per_loop,
                                    &out->llc_miss_per_loop,    &out->dtlb_miss_per_loop };
    for (int e = 0; e < OV_PERF_NEV; e++)
    {
        if (s_perf_slot[e] < 0 || (uint64_t) s_perf_slot[e] >= buf[0])
        {
            continue;
        }
        uint64_t v = (uint64_t) ((double) buf[3 + s_perf_slot[e]] * scale);
        *tot[e]    = v;
        if (d_loops > 0 && v >= s_perf_prev[e])
        {
            *rate[e] = (double) (v - s_perf_prev[e]) / (double) d_loops;
        }
        if (d_loops > 0)
        {
            s_perf_prev[e] = v; /* the counts and the loop counter move on together */
        }
    }
    if (d_loops > 0)
    {
        s_perf_prev_loopcnt = loopcnt;
    }
    return 0;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include "overview_ansi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <signal.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

static int     s_perf_pid          = -1;
static int64_t s_perf_prev_loopcnt = 0;

static int s_perf_fd_inst   = -1;
static int s_perf_fd_cache  = -1;
static int s_perf_fd_branch = -1;
static int s_perf_fd_l1d    = -1;
static int s_perf_fd_llc    = -1;
static int s_perf_fd_dtlb   = -1;

static uint64_t s_perf_prev_inst   = 0;
static uint64_t s_perf_prev_cache  = 0;
static uint64_t s_perf_prev_branch = 0;
static uint64_t s_perf_prev_l1d    = 0;
static uint64_t s_perf_prev_llc    = 0;
static uint64_t s_perf_prev_dtlb   = 0;

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
static long _perf_event_open(
    struct perf_event_attr *attr,
    pid_t                   pid,
    int                     cpu,
    int                     group_fd,
    unsigned long           flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/**
 * @brief Open a perf_event file descriptor.
 *
 * Configures hardware performance counters for
 * the specified event type.
 */
static int open_perf_fd(pid_t pid, uint32_t type, uint64_t config)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type           = type;
    attr.size           = sizeof(attr);
    attr.config         = config;
    attr.disabled       = 1;
    attr.exclude_kernel = 0;
    attr.exclude_hv     = 1;
    attr.inherit        = 1;
    int fd              = (int) _perf_event_open(&attr, pid, -1, -1, 0);
    if (fd >= 0)
    {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    return fd;
}

/**
 * pid_read_perf_counters - get hardware metrics via perf_event_open
 * Requires CAP_PERFMON or root, or perf_event_paranoid <= 2
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
        if (s_perf_fd_inst >= 0)
        {
            close(s_perf_fd_inst);
        }
        if (s_perf_fd_cache >= 0)
        {
            close(s_perf_fd_cache);
        }
        if (s_perf_fd_branch >= 0)
        {
            close(s_perf_fd_branch);
        }
        if (s_perf_fd_l1d >= 0)
        {
            close(s_perf_fd_l1d);
        }
        if (s_perf_fd_llc >= 0)
        {
            close(s_perf_fd_llc);
        }
        if (s_perf_fd_dtlb >= 0)
        {
            close(s_perf_fd_dtlb);
        }

        s_perf_fd_inst   = open_perf_fd(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        s_perf_fd_cache  = open_perf_fd(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
        s_perf_fd_branch = open_perf_fd(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

        s_perf_fd_l1d =
            open_perf_fd(pid, PERF_TYPE_HW_CACHE,
                         (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                             (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        s_perf_fd_llc = open_perf_fd(pid, PERF_TYPE_HW_CACHE,
                                     (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        s_perf_fd_dtlb =
            open_perf_fd(pid, PERF_TYPE_HW_CACHE,
                         (PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                             (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));

        s_perf_pid          = pid;
        s_perf_prev_loopcnt = loopcnt;
        s_perf_prev_inst    = 0;
        s_perf_prev_cache   = 0;
        s_perf_prev_branch  = 0;
        s_perf_prev_l1d     = 0;
        s_perf_prev_llc     = 0;
        s_perf_prev_dtlb    = 0;
    }

    int ok       = 0;
    int expected = 0;

    int64_t d_loops = loopcnt - s_perf_prev_loopcnt;
    if (d_loops <= 0)
    {
        d_loops = 0;
    }

#define READ_FD(fd, outval, prevval, rateval)                         \
    do                                                                \
    {                                                                 \
        if ((fd) >= 0)                                                \
        {                                                             \
            expected++;                                               \
            uint64_t val = 0;                                         \
            if (read((fd), &val, sizeof(val)) == sizeof(val))         \
            {                                                         \
                (outval) = val;                                       \
                ok++;                                                 \
                if (d_loops > 0 && val >= (prevval))                  \
                {                                                     \
                    (rateval) = (double) (val - (prevval)) / d_loops; \
                }                                                     \
                (prevval) = val;                                      \
            }                                                         \
        }                                                             \
    } while (0)

    READ_FD(s_perf_fd_inst, out->instructions, s_perf_prev_inst, out->inst_per_loop);
    READ_FD(s_perf_fd_cache, out->cache_misses, s_perf_prev_cache, out->cache_miss_per_loop);
    READ_FD(s_perf_fd_branch, out->branch_misses, s_perf_prev_branch, out->branch_miss_per_loop);
    READ_FD(s_perf_fd_l1d, out->l1d_misses, s_perf_prev_l1d, out->l1d_miss_per_loop);
    READ_FD(s_perf_fd_llc, out->llc_misses, s_perf_prev_llc, out->llc_miss_per_loop);
    READ_FD(s_perf_fd_dtlb, out->dtlb_misses, s_perf_prev_dtlb, out->dtlb_miss_per_loop);
#undef READ_FD

    if (d_loops > 0)
    {
        s_perf_prev_loopcnt = loopcnt;
    }

    return (expected > 0 && ok == expected) ? 0 : -1;
}

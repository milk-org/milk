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

/**
 * ov_datatype_name - short name for data type code.
 */
const char *ov_datatype_name(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "UI8";
    case _DATATYPE_INT8:
        return "SI8";
    case _DATATYPE_UINT16:
        return "U16";
    case _DATATYPE_INT16:
        return "S16";
    case _DATATYPE_UINT32:
        return "U32";
    case _DATATYPE_INT32:
        return "S32";
    case _DATATYPE_UINT64:
        return "U64";
    case _DATATYPE_INT64:
        return "S64";
    case _DATATYPE_FLOAT:
        return "F32";
    case _DATATYPE_DOUBLE:
        return "F64";
    case _DATATYPE_COMPLEX_FLOAT:
        return "CF";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "CD";
    default:
        return "???";
    }
}
/* suppress unused-function warning when header is
 * included but ov_datatype_name is only used by
 * the render code */
__attribute__((unused)) static const char *ov_datatype_name_ref =
    (const char *) (uintptr_t) ov_datatype_name;

#include <sys/resource.h>

/**
 * ov_sys_get_cpu_usage - compute self process smoothed CPU utilization.
 *
 * Return: Estimated CPU percentage (0.0 to 100.0 * num_cores).
 */
double ov_sys_get_cpu_usage(void)
{
    static struct rusage   last_usage;
    static struct timespec last_time;
    static int             initialized  = 0;
    static double          smoothed_cpu = 0.0;

    struct rusage   current_usage;
    struct timespec current_time;

    getrusage(RUSAGE_SELF, &current_usage);
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    if (!initialized)
    {
        last_usage  = current_usage;
        last_time   = current_time;
        initialized = 1;
        return 0.0;
    }

    double dt = (current_time.tv_sec - last_time.tv_sec) +
                (current_time.tv_nsec - last_time.tv_nsec) / 1e9;

    if (dt >= 0.5) /* update every 0.5s */
    {
        double d_utime = (current_usage.ru_utime.tv_sec - last_usage.ru_utime.tv_sec) +
                         (current_usage.ru_utime.tv_usec - last_usage.ru_utime.tv_usec) / 1e6;
        double d_stime = (current_usage.ru_stime.tv_sec - last_usage.ru_stime.tv_sec) +
                         (current_usage.ru_stime.tv_usec - last_usage.ru_stime.tv_usec) / 1e6;

        double inst_cpu = 100.0 * (d_utime + d_stime) / dt;
        smoothed_cpu    = inst_cpu;
        last_usage      = current_usage;
        last_time       = current_time;
    }
    return smoothed_cpu;
}

/**
 * ov_sys_get_bandwidth_usage - compute terminal delta rendering bandwidth.
 *
 * Return: Smoothed bandwidth in kB/s.
 */
double ov_sys_get_bandwidth_usage(void)
{
    static struct timespec last_time;
    static uint64_t        last_bytes  = 0;
    static int             initialized = 0;
    static double          smoothed_bw = 0.0;

    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    if (!initialized)
    {
        last_time   = current_time;
        last_bytes  = ov__total_bytes_rendered;
        initialized = 1;
        return 0.0;
    }

    double dt = (current_time.tv_sec - last_time.tv_sec) +
                (current_time.tv_nsec - last_time.tv_nsec) / 1e9;

    if (dt >= 0.5) /* update every 0.5s */
    {
        uint64_t d_bytes = ov__total_bytes_rendered - last_bytes;
        double   inst_bw = (double) d_bytes / 1024.0 / dt;
        smoothed_bw      = inst_bw;
        last_bytes       = ov__total_bytes_rendered;
        last_time        = current_time;
    }
    return smoothed_bw;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_timing.h
 * @brief Loop timing statistics computed by readers from a process's timing ring
 */

#ifndef _PROCESSINFO_TIMING_H
#define _PROCESSINFO_TIMING_H

#include <stdint.h>

#include "processinfo.h"

#ifdef __cplusplus
extern "C"
{
#endif

int processinfo_timing_median(const PROCESSINFO *pinfo, int64_t *iter_ns, int64_t *exec_ns);

/** Loop timing over the ring: medians and maxima of the iteration interval and of the execution time, and
 * the jitter (largest |interval - median|), all in ns, -1 without samples. Any output may be NULL. */
int processinfo_timing_stats(const PROCESSINFO *pinfo,
                             int64_t           *iter_med_ns,
                             int64_t           *iter_max_ns,
                             int64_t           *exec_med_ns,
                             int64_t           *exec_max_ns,
                             int64_t           *jitter_ns);

#ifdef __cplusplus
}
#endif

#endif /* _PROCESSINFO_TIMING_H */

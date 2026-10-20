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

#ifdef __cplusplus
}
#endif

#endif /* _PROCESSINFO_TIMING_H */

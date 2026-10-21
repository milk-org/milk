// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stream_monitorlimits.h
 * @brief Stream monitorlimits module
 */

/** @file stream_delay,h
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_streammlim;

errno_t stream_monitorlimits_FPCONF();

errno_t stream_monitorlimits_RUN();

errno_t stream_monitorlimits(const char *instreamname);

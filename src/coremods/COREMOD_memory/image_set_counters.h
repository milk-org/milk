// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_set_counters.h
 * @brief Set image flags / counters
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_imsetcnt0;
extern const MILK_RUNNABLE runnable_COREMOD_memory_imsetstatus;
extern const MILK_RUNNABLE runnable_COREMOD_memory_imsetcnt1;

errno_t COREMOD_MEMORY_image_set_status(const char *IDname, int status);

errno_t COREMOD_MEMORY_image_set_cnt0(const char *IDname, int cnt0);

errno_t COREMOD_MEMORY_image_set_cnt1(const char *IDname, int cnt1);

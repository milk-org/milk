// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file read_shmimall.h
 * @brief Read all shared memory streams
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_readshmimall;

errno_t read_sharedmem_image_all(const char *strfilter);

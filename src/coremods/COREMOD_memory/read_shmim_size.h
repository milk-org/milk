// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file read_shmim_size.h
 * @brief Read shmim size module
 */

#ifndef COREMOD_MEMORY_READ_SHMIM_SIZE_H
#define COREMOD_MEMORY_READ_SHMIM_SIZE_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_readshmimsize;

imageID read_sharedmem_image_size(const char *name, const char *fname);

#endif

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file read_shmim.h
 * @brief Read shmim module
 */

/**
 * @file    read_shmim.h
 */

#ifndef COREMOD_MEMORY_READ_SHMIM_H
#define COREMOD_MEMORY_READ_SHMIM_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_readshmim;

imageID read_sharedmem_image(const char *sname, IMAGE *imagearray, long NB_images);

#endif

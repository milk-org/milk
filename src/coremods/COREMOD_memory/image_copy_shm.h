// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_copy_shm.h
 * @brief Image copy shm module
 */

#ifndef COREMOD_MEMORY_IMAGE_COPY_SHM_H
#define COREMOD_MEMORY_IMAGE_COPY_SHM_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_image_copy_shm;

errno_t image_copy_shm(const char *inname, const char *outname);
errno_t image_copy_shm_IMGID(IMGID *imgin, IMGID *imgout);

#endif

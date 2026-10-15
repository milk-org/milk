// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_make3D.h
 * @brief Image make3d module
 */

#ifndef COREMOD_MEMORY_MK3DIMAGE_H
#define COREMOD_MEMORY_MK3DIMAGE_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_image_make3D;

imageID make_3Dimage(const char *name, uint32_t xsize, uint32_t ysize, uint32_t zsize);
imageID make_3Dimage_IMGID(IMGID *img);

#endif

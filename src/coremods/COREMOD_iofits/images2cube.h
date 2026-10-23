// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file images2cube.h
 * @brief Combine individual images into cube
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_iofits_images2cube;

errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name);

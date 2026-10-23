// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_mk_complex_from_amph.h
 * @brief Image mk complex from amph module
 */

#ifndef COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_AMPH_H
#define COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_AMPH_H

errno_t mk_complex_from_amph_IMGID(IMGID *imginamp, IMGID *imginpha, IMGID *imgoutC);

errno_t mk_complex_from_amph(const char *am_name,
                             const char *ph_name,
                             const char *out_name,
                             int         sharedmem);

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_image_mk_complex_from_amph;

#endif

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_mk_complex_from_reim.h
 * @brief Image mk complex from reim module
 */

#ifndef COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_REIM_H
#define COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_REIM_H

errno_t mk_complex_from_reim(const char *re_name,
                             const char *im_name,
                             const char *out_name,
                             int         sharedmem);

errno_t mk_complex_from_reim_IMGID(IMGID *imgre, IMGID *imgim, IMGID *imgout);

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_image_mk_complex_from_reim;

#endif

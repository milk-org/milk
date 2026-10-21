// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_filter.c
 * @brief   Image filtering
 *
 */


// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "imgfilt"

// Module short description
#define MODULE_DESCRIPTION "Image filtering"

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_filter.h"

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_image_filter_gaussfilter,
    &runnable_image_filter_fconvolve,
    &runnable_image_filter_im2Dfilter_1pixbblurr,
};

MILK_MODULE_RUNNABLES(image_filter, NULL, runnables);

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    COREMOD_arith.c
 * @brief   Arithmeric operations on images
 *
 * Addition, multiplication and much more
 *
 *
 */

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "arith"

// Module short description
#define MODULE_DESCRIPTION "Image arithmetic operations"

#include "milk_runnable.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_arith/COREMOD_arith.h"

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_COREMOD_arith_extract3Dim,
    &runnable_COREMOD_arith_extractim,
    &runnable_COREMOD_arith_image_arith__im_f_f__im,
    &runnable_COREMOD_arith_image_merge3D,
    &runnable_COREMOD_arith_image_norm,
    &runnable_COREMOD_arith_image_slicenormalize,
    &runnable_COREMOD_arith_image_cropmask,
    &runnable_COREMOD_arith_image_crop2D,
    &runnable_COREMOD_arith_image_multicrop2D,
    &runnable_COREMOD_arith_image_set_1Dpixrange,
    &runnable_COREMOD_arith_image_set_2Dpix,
    &runnable_COREMOD_arith_image_set_3Daxes,
    &runnable_COREMOD_arith_image_set_col,
    &runnable_COREMOD_arith_image_set_row,
    &runnable_COREMOD_arith_image_setzero,
    &runnable_COREMOD_arith_image_pixremap,
    &runnable_COREMOD_arith_image_pixunmap,
    &runnable_COREMOD_arith_image_unfold,
    &runnable_COREMOD_arith_image_vecmult,
};

MILK_MODULE_RUNNABLES(COREMOD_arith, NULL, runnables);

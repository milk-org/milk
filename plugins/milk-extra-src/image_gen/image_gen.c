// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_gen.c
 * @brief   Generate frequently used image(s)
 *
 * Creates images for misc applications
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
#define MODULE_SHORTNAME_DEFAULT "imgen"

// Module short description
#define MODULE_DESCRIPTION "Creating images (shapes, useful functions and patterns)"

#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CFITSIO
#    include <fitsio.h> /* required by every program that uses CFITSIO  */
#endif

#include "milk_runnable.h"

#include "COREMOD_arith/COREMOD_arith.h"
#ifdef USE_CFITSIO
#    include "COREMOD_iofits/COREMOD_iofits.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

#include "statistic/statistic.h"

#include "image_gen/image_gen.h"

#define OMP_NELEMENT_LIMIT 1000000

#define SWAP(x, y) \
    tmp = (x);     \
    x   = (y);     \
    y   = tmp;

#define PI 3.14159265358979323846264338328

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_image_gen_imgen_segs2wfmodes,
    &runnable_image_gen_imgen_mkrect,
    &runnable_image_gen_imgen_mkgridpix,
    &runnable_image_gen_imgen_mkline,
    &runnable_image_gen_imgen_mkdist,
    &runnable_image_gen_imgen_mkrndim,
    &runnable_image_gen_imgen_mkhexsegpup,
    &runnable_image_gen_imgen_mkrndgim,
    &runnable_image_gen_imgen_mkslopexy,
    &runnable_image_gen_imgen_mklincoord,
    &runnable_image_gen_imgen_im2coord,
    &runnable_image_gen_imgen_mkfiberclpoverlap,
    &runnable_image_gen_imgen_mkgauss,
    &runnable_image_gen_mkdisk,
    &runnable_image_gen_mkpolygon,
    &runnable_image_gen_mkspdisk,
    &runnable_image_gen_voronoi,
    &runnable_image_gen_mkrandomim,
};

MILK_MODULE_RUNNABLES(image_gen, NULL, runnables);

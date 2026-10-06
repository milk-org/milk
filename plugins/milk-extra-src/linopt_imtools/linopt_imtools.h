// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    linopt_imtools.h
 * @brief   Function prototypes for linear algebra tools
 *
 * CPU-based lineal algebra tools: decomposition, SVD etc...
 *
 */

#ifndef _LINOPTIMTOOLS_H
#define _LINOPTIMTOOLS_H

#include "compute_SVDdecomp.h"
#include "compute_SVDpseudoInverse.h"
#include "image_construct.h"
#include "image_fitModes.h"
#include "image_to_vec.h"
#include "lin1Dfit.h"
#include "linRM_from_inout.h"
#include "makeCPAmodes.h"
#include "makeCosRadModes.h"
#include "mask_to_pixtable.h"
#include "vec_to_2Dimage.h"

void __attribute__((constructor)) libinit_linopt_imtools();

imageID linopt_imtools_make1Dpolynomials(const char *IDout_name,
                                         long        NBpts,
                                         long        MaxOrder,
                                         float       r0pix);

#endif

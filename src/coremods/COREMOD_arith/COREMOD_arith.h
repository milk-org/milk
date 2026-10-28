// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file COREMOD_arith.h
 * @brief void __attribute__ ((constructor)) libinit_COREMOD_arith();
 */

#ifndef _ARITH_H
#define _ARITH_H

//void __attribute__ ((constructor)) libinit_COREMOD_arith();

//int init_COREMOD_arith();

#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_coord.h"
#include "COREMOD_arith/image_elem_block.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_arith/image_arith__Cim_Cim__Cim.h"
#include "COREMOD_arith/image_arith__im__im.h"
#include "COREMOD_arith/image_arith__im_f__im.h"
#include "COREMOD_arith/image_arith__im_f_f__im.h"
#include "COREMOD_arith/image_arith__im_im__im.h"
#include "COREMOD_arith/image_crop.h"
#include "COREMOD_arith/image_multicrop2D.h"
#include "COREMOD_arith/image_cropmask.h"
#include "COREMOD_arith/image_merge3D.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_arith/image_total.h"
#include "COREMOD_arith/imfunctions.h"

#endif

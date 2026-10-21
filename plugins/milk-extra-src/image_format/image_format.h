// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_format.h
 * @brief Image format module
 */

#ifndef _IMAGEFORMATMODULE_H
#define _IMAGEFORMATMODULE_H

#include "milk_runnable.h"

#include "image_format/CR2toFITS.h"
#include "image_format/CR2tomov.h"
#include "image_format/FITS_to_floatbin_lock.h"
#include "image_format/FITS_to_ushortintbin_lock.h"
#include "image_format/FITStorgbFITSsimple.h"
#include "image_format/imtoASCII.h"
#include "image_format/loadCR2toFITSRGB.h"
#include "image_format/readPGM.h"
#include "image_format/read_binary32f.h"
#include "image_format/writeBMP.h"

extern const MILK_RUNNABLE runnable_image_format_extract_RGGBchan;
extern const MILK_RUNNABLE runnable_image_format_combineHDR;
extern const MILK_RUNNABLE runnable_image_format_extract_utr;
extern const MILK_RUNNABLE runnable_image_format_stream_temporal_stats;
extern const MILK_RUNNABLE runnable_image_format_imtoASCII;
extern const MILK_RUNNABLE runnable_image_format_writeBMP;
extern const MILK_RUNNABLE runnable_image_format_CR2toFITS;
extern const MILK_RUNNABLE runnable_image_format_loadCR2toFITSRGB;
extern const MILK_RUNNABLE runnable_image_format_FITS_to_floatbin_lock;
extern const MILK_RUNNABLE runnable_image_format_FITS_to_ushortintbin_lock;
extern const MILK_RUNNABLE runnable_image_format_read_binary32f;

#endif // _IMAGEFORMATMODULE_H

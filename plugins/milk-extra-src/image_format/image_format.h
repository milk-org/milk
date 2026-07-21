// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_format.h
 * @brief Image format module
 */

#ifndef _IMAGEFORMATMODULE_H
#define _IMAGEFORMATMODULE_H

#include "image_format/CR2toFITS.h"
#include "image_format/CR2tomov.h"
#include "image_format/FITS_to_floatbin_lock.h"
#include "image_format/FITS_to_ushortintbin_lock.h"
#include "image_format/FITStorgbFITSsimple.h"
#include "image_format/extract_RGGBchan.h"
#include "image_format/extract_utr.h"
#include "image_format/imtoASCII.h"
#include "image_format/loadCR2toFITSRGB.h"
#include "image_format/readPGM.h"
#include "image_format/read_binary32f.h"
#include "image_format/stream_temporal_stats.h"
#include "image_format/writeBMP.h"

#endif // _IMAGEFORMATMODULE_H

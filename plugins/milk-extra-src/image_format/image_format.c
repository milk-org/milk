// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_format.c
 * @brief   Convert between image formats
 *
 * read and write images other than FITS
 *
 */

#define MODULE_SHORTNAME_DEFAULT "imgformat"
#define MODULE_DESCRIPTION "Conversion between image format, I/O"

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_format.h"


/*typedef struct
{
    int rows;
    int cols;
    unsigned char *data;
} sImage;
*/
/* This pragma is necessary so that the data in the structures is aligned to 2-byte
   boundaries.  Some different compilers have a different syntax for this line.  For
   example, if you're using cc on Solaris, the line should be #pragma pack(2).
*/
//#pragma pack(2)

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_image_format_extract_RGGBchan,
    &runnable_image_format_combineHDR,
    &runnable_image_format_extract_utr,
    &runnable_image_format_stream_temporal_stats,
    &runnable_image_format_imtoASCII,
    &runnable_image_format_writeBMP,
    &runnable_image_format_CR2toFITS,
    &runnable_image_format_loadCR2toFITSRGB,
    &runnable_image_format_FITS_to_floatbin_lock,
    &runnable_image_format_FITS_to_ushortintbin_lock,
    &runnable_image_format_read_binary32f,
};

MILK_MODULE_RUNNABLES(image_format, NULL, runnables);

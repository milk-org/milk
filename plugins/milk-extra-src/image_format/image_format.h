#ifndef _IMAGEFORMATMODULE_H
#define _IMAGEFORMATMODULE_H

void __attribute__((constructor)) libinit_image_format();

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

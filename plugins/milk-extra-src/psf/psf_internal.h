/**
 * @file psf_internal.h
 * @brief Internal header for psf module
 */

#ifndef _PSF_INTERNAL_H
#define _PSF_INTERNAL_H

#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "image_basic/image_basic.h"
#include "image_filter/image_filter.h"
#include "image_gen/image_gen.h"
#include "fft/fft.h"
#include "psf/psf.h"

extern double FWHM_MEASURED;

#endif

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_gen_internal.h
 * @brief   Internal declarations for the image_gen module
 */

#ifndef IMAGE_GEN_INTERNAL_H
#define IMAGE_GEN_INTERNAL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CFITSIO
#    include <fitsio.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif

#include "COREMOD_arith/COREMOD_arith.h"
#ifdef USE_CFITSIO
#    include "COREMOD_iofits/COREMOD_iofits.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

#include "statistic/statistic.h"

#include "image_gen/image_gen.h"

#endif // IMAGE_GEN_INTERNAL_H

/**
 * @file    dofft_internal.h
 * @brief   Internal declarations shared between dofft split files
 *
 * Not part of the public API. Shared by dofft.c,
 * dofft_1d.c, and dofft_2d.c.
 */

#ifndef DOFFT_INTERNAL_H
#define DOFFT_INTERNAL_H

#include <fftw3.h>

#include "dofft.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "wisdom.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO/ImageStreamIO.h"

#define FFTWOPTMODE FFTW_ESTIMATE

/**
 * array_index() - Map power-of-2 size to index
 * @size: Array dimension (must be power of 2)
 *
 * Returns log2(size) for powers of 2 from 1 to
 * 16384, or 100 for non-power-of-2 sizes.
 */
int array_index(long size);

/* Internal FFT implementations (direction arg) */

imageID FFT_do1dfft(
    const char *__restrict in_name,
    const char *__restrict out_name,
    int dir);

imageID FFT_do2dfft(
    const char *in_name,
    const char *out_name,
    int dir);

imageID FFT_do2drfft(
    const char *__restrict in_name,
    const char *__restrict out_name,
    int dir);

#endif /* DOFFT_INTERNAL_H */

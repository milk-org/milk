// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file fft.h
 * @brief Fft module
 */

#ifndef _FFT_H
#define _FFT_H

#include "fft/DFT.h"
#include "fft/dofft.h"
#include "fft/fft_autocorrelation.h"
#include "fft/fft_structure_function.h"
#include "fft/fftcorrelation.h"
#include "fft/ffttranslate.h"
#include "fft/fftzoom.h"
#include "fft/init_fftwplan.h"
#include "fft/permut.h"
#include "fft/pup2foc.h"
#include "fft/testfftspeed.h"
#include "fft/wisdom.h"

int fft_setoffsets(long o1, long o2);

#endif

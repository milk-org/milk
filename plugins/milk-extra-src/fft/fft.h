// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file fft.h
 * @brief Fft module
 */

#ifndef _FFT_H
#define _FFT_H

#include "milk_runnable.h"

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

extern const MILK_RUNNABLE runnable_fft_init_fftwplan;
extern const MILK_RUNNABLE runnable_fft_permut;
extern const MILK_RUNNABLE runnable_fft_dofft;
extern const MILK_RUNNABLE runnable_fft_dofft_1dfft;
extern const MILK_RUNNABLE runnable_fft_dofft_1drfft;
extern const MILK_RUNNABLE runnable_fft_testfftspeed;
extern const MILK_RUNNABLE runnable_fft_ffttranslate;
extern const MILK_RUNNABLE runnable_fft_fftcorrelation;
extern const MILK_RUNNABLE runnable_fft_pup2foc;

int fft_setoffsets(long o1, long o2);

#endif

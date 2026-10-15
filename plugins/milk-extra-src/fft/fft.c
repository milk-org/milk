// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fft.c
 * @brief   Fourier Transform
 *
 * Wrapper to fftw and FFT tools
 *
 */

#define MODULE_SHORTNAME_DEFAULT "fft"
#define MODULE_DESCRIPTION "FFTW wrapper and FFT-related functions"

#include <fftw3.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fft.h"
#include "wisdom.h"

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_fft_init_fftwplan, &runnable_fft_permut,         &runnable_fft_dofft,
    &runnable_fft_dofft_1dfft,   &runnable_fft_dofft_1drfft,   &runnable_fft_testfftspeed,
    &runnable_fft_ffttranslate,  &runnable_fft_fftcorrelation, &runnable_fft_pup2foc,
};

MILK_MODULE_RUNNABLES(fft, NULL, runnables);

static void __attribute__((constructor)) fftw_init()
{
#ifdef FFTWMT
    printf("Multi-threaded fft enabled, max threads = %d\n", omp_get_max_threads());
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
    // load fftw wisdom
    import_wisdom();

    __milk_module_info.constructor_called = 1;
}


static void __attribute__((destructor)) close_fftwlib()
{
    if (__milk_module_info.constructor_called == 1)
    {
        fftw_forget_wisdom();
        fftwf_forget_wisdom();

#ifdef FFTWMT
        fftw_cleanup_threads();
        fftwf_cleanup_threads();
#else
        fftw_cleanup();
        fftwf_cleanup();
#endif
    }
}
int fft_setNthreads(__attribute__((unused)) int nt)
{
//   printf("set number of thread to %d (FFTWMT)\n",nt);
#ifdef FFTWMT
    fftwf_cleanup_threads();
    fftwf_cleanup();

    //  printf("Multi-threaded fft enabled, max threads = %d\n",nt);
    fftwf_init_threads();
    fftwf_plan_with_nthreads(nt);
#endif

    import_wisdom();

    return RETURN_SUCCESS;
}

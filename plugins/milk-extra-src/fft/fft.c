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

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "CLIcore.h"
#endif

#include "wisdom.h"

#ifndef MILK_NO_CLI
#    include "dofft.h"
#    include "fftcorrelation.h"
#    include "ffttranslate.h"
#    include "init_fftwplan.h"
#    include "permut.h"
#    include "testfftspeed.h"

#    include "pup2foc.h"

static errno_t init_module_CLI()
{
    CLIADDCMD_milkfft__init_fftwplan();
    CLIADDCMD_milkfft__permut();
    CLIADDCMD_milkfft__dofft();
    CLIADDCMD_milkfft__testfftspeed();
    CLIADDCMD_milkfft__ffttranslate();
    CLIADDCMD_milkfft__fftcorrelation();

    CLIADDCMD_milk_fft__pup2foc();
    return RETURN_SUCCESS;
}

MILK_MODULE(fft, init_module_CLI, NULL);

static void __attribute__((constructor)) fftw_init()
{
#    ifdef FFTWMT
    printf("Multi-threaded fft enabled, max threads = %d\n", omp_get_max_threads());
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
#    endif
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

#    ifdef FFTWMT
        fftw_cleanup_threads();
        fftwf_cleanup_threads();
#    else
        fftw_cleanup();
        fftwf_cleanup();
#    endif
    }
}
#endif /* MILK_NO_CLI */

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

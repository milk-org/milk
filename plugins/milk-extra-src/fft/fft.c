/**
 * @file    fft.c
 * @brief   Fourier Transform
 *
 * Wrapper to fftw and FFT tools
 *
 */

#define MODULE_SHORTNAME_DEFAULT "fft"
#define MODULE_DESCRIPTION       "FFTW wrapper and FFT-related functions"

#include <fftw3.h>

#include "CommandLineInterface/CLIcore.h"

#include "dofft.h"
#include "fftcorrelation.h"
#include "ffttranslate.h"
#include "init_fftwplan.h"
#include "permut.h"
#include "testfftspeed.h"
#include "wisdom.h"

// auto-generate libinit_<modulename>
// initialize INITSTATUS_<modulename>
INIT_MODULE_LIB(fft)


static errno_t init_module_CLI()
{

#ifdef FFTWMT
    printf("Multi-threaded fft enabled, max threads = %d\n",
           omp_get_max_threads());
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
#endif

    // load fftw wisdom
    import_wisdom();

    //fftwf_set_timelimit(1000.0);
    //fftw_set_timelimit(1000.0);

    init_fftwplan_addCLIcmd();
    permut_addCLIcmd();
    dofft_addCLIcmd();
    testfftspeed_addCLIcmd();
    ffttranslate_addCLIcmd();
    fftcorrelation_addCLIcmd();

    return RETURN_SUCCESS;
}

static void __attribute__((destructor)) close_fftwlib()
{
    if(INITSTATUS_fft == 1)
    {
        fftw_forget_wisdom();
        fftwf_forget_wisdom();

#ifdef FFTWMT
        fftw_cleanup_threads();
        fftwf_cleanup_threads();
#endif

#ifndef FFTWMT
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

    return (0);
}

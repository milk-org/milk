/** @file testfftspeed.c
 */

#include <fftw3.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"

// Forward declaration
int test_fftspeed(int nmax);

/* =========================================
 *  V2 PARAMS
 * ======================================= */

static long long p_nmax = 10;

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "testfftspeed",
    .cmdkey      = "testfftspeed",
    .description = "test FFTW speed",
    .description_long =
        "Benchmark FFTW performance for a given image size and transform type. Measures throughput in megapixels/s and reports optimal plan wisdom."
};

#define FPS_PARAMS(X) \
    X(".nmax", &p_nmax, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "max power of 2")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};
static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static MILK_HOT errno_t compute_function()
{
    test_fftspeed((int) p_nmax);
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_milkfft__testfftspeed()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/** @brief Test FFT speed (fftw)
 *
 */

int test_fftspeed(int nmax)
{
    int  n;
    long size;
    int  nbiter, iter;

    struct timespec tS0;
    struct timespec tS1;
    struct timespec tS2;
    double          ti0, ti1, ti2;
    double          dt1;
    //struct timeval tv;
    //int nb_threads=1;
    //int nb_threads_max = 8;

    /*  printf("%ld ticks per second\n",CLOCKS_PER_SEC);*/
    nbiter = 10000;
    size   = 2;

    printf("Testing complex FFT, nxn pix\n");

    printf("size(pix)");
#ifdef FFTWMT
    for(nb_threads = 1; nb_threads < nb_threads_max; nb_threads++)
    {
        printf("%13d", nb_threads);
    }
#endif
    printf("\n");

    size = 2;
    for(n = 0; n < nmax; n++)
    {
        printf("%9ld", size);
#ifdef FFTWMT
        for(nb_threads = 1; nb_threads < nb_threads_max; nb_threads++)
        {
            fft_setNthreads(nb_threads);
#endif

#if _POSIX_TIMERS > 0
            clock_gettime(CLOCK_MILK, &tS0);
#else
            gettimeofday(&tv, NULL);
            tS0.tv_sec  = tv.tv_sec;
            tS0.tv_nsec = tv.tv_usec * 1000;
#endif

            //	  clock_gettime(CLOCK_MILK, &tS0);
            for(iter = 0; iter < nbiter; iter++)
            {
                create_2DCimage_ID("tmp", size, size, NULL);
                do2dfft("tmp", "tmpf");
                delete_image_ID("tmp", DELETE_IMAGE_ERRMODE_WARNING);
                delete_image_ID("tmpf", DELETE_IMAGE_ERRMODE_WARNING);
            }

#if _POSIX_TIMERS > 0
            clock_gettime(CLOCK_MILK, &tS1);
#else
            gettimeofday(&tv, NULL);
            tS1.tv_sec  = tv.tv_sec;
            tS1.tv_nsec = tv.tv_usec * 1000;
#endif
            //	  clock_gettime(CLOCK_MILK, &tS1);

            for(iter = 0; iter < nbiter; iter++)
            {
                create_2DCimage_ID("tmp", size, size, NULL);
                delete_image_ID("tmp", DELETE_IMAGE_ERRMODE_WARNING);
            }

#if _POSIX_TIMERS > 0
            clock_gettime(CLOCK_MILK, &tS2);
#else
            gettimeofday(&tv, NULL);
            tS2.tv_sec  = tv.tv_sec;
            tS2.tv_nsec = tv.tv_usec * 1000;
#endif
            //clock_gettime(CLOCK_MILK, &tS2);

            ti0 = 1.0 * tS0.tv_sec + 0.000000001 * tS0.tv_nsec;
            ti1 = 1.0 * tS1.tv_sec + 0.000000001 * tS1.tv_nsec;
            ti2 = 1.0 * tS2.tv_sec + 0.000000001 * tS2.tv_nsec;
            dt1 = 1.0 * (ti1 - ti0) - 1.0 * (ti2 - ti1);

            dt1 /= nbiter;

            printf("%10.3f ms", dt1 * 1000.0);
            //printf("Complex FFT %ldx%ld [%d threads] : %f ms  [%ld]\n",size,size,nb_threads,dt1*1000.0,nbiter);
            fflush(stdout);
#ifdef FFTWMT
        }
#endif
        printf("\n");
        nbiter = 0.1 / dt1;
        if(nbiter < 2)
        {
            nbiter = 2;
        }
        size = size * 2;
    }

    return (0);
}

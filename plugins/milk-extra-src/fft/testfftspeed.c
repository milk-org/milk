/** @file testfftspeed.c
 */

#include <fftw3.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"

// ==========================================
// Forward declaration(s)
// ==========================================

int test_fftspeed(int nmax);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t test_fftspeed_cli()
{
    if(CLI_checkarg(1, CLIARG_INT64) == 0)
    {
        test_fftspeed((int) data.cmdargtoken[1].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t testfftspeed_addCLIcmd()
{
    RegisterCLIcommand("testfftspeed",
                       __FILE__,
                       test_fftspeed_cli,
                       "test FFTW speed",
                       "no argument",
                       "testfftspeed",
                       "int test_fftwspeed(int nmax)");

    return RETURN_SUCCESS;
}

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
            clock_gettime(CLOCK_REALTIME, &tS0);
#else
            gettimeofday(&tv, NULL);
            tS0.tv_sec  = tv.tv_sec;
            tS0.tv_nsec = tv.tv_usec * 1000;
#endif

            //	  clock_gettime(CLOCK_REALTIME, &tS0);
            for(iter = 0; iter < nbiter; iter++)
            {
                create_2DCimage_ID("tmp", size, size, NULL);
                do2dfft("tmp", "tmpf");
                delete_image_ID("tmp", DELETE_IMAGE_ERRMODE_WARNING);
                delete_image_ID("tmpf", DELETE_IMAGE_ERRMODE_WARNING);
            }

#if _POSIX_TIMERS > 0
            clock_gettime(CLOCK_REALTIME, &tS1);
#else
            gettimeofday(&tv, NULL);
            tS1.tv_sec  = tv.tv_sec;
            tS1.tv_nsec = tv.tv_usec * 1000;
#endif
            //	  clock_gettime(CLOCK_REALTIME, &tS1);

            for(iter = 0; iter < nbiter; iter++)
            {
                create_2DCimage_ID("tmp", size, size, NULL);
                delete_image_ID("tmp", DELETE_IMAGE_ERRMODE_WARNING);
            }

#if _POSIX_TIMERS > 0
            clock_gettime(CLOCK_REALTIME, &tS2);
#else
            gettimeofday(&tv, NULL);
            tS2.tv_sec  = tv.tv_sec;
            tS2.tv_nsec = tv.tv_usec * 1000;
#endif
            //clock_gettime(CLOCK_REALTIME, &tS2);

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

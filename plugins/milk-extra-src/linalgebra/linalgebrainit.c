// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file linalgebrainit.c
 */

// MILK_CMAKE_MANDATE_CUDA
// MILK_CMAKE_REQUEST_MAGMA

#ifdef HAVE_MAGMA
#    include "magma_lapack.h"
#    include "magma_v2.h"
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
#    include "computeunit.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#ifdef HAVE_CUDA

extern int cuda_deviceCount;

// ==========================================
// Forward declaration(s)
// ==========================================

int LINALGEBRA_init();

// ==========================================
// Gen 4 V2 CLI command: linalgebrainit
// ==========================================

MILK_RUNNABLE_RT_DEFINE(li);

static errno_t li_compute(__attribute__((unused)) const MILK_RUNNABLE *self)
{
    LINALGEBRA_init();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_linalgebra_linalgebrainit = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "linalgebrainit",
                 .cmdkey           = "linalgebrainit",
                 .description      = "init linalgebra",
                 .description_long = "Initialize linear algebra subsystem. Sets up BLAS/LAPACK "
                                     "configuration and GPU context if available." },
    .srcfile = __FILE__,
    .compute = li_compute,
    .rt      = &li_rt,
};


/**
 * @brief Initialize CUDA and MAGMA
 *
 * Finds CUDA devices
 * Initializes CUDA and MAGMA libraries
 *
 * @return number of CUDA devices found
 *
 */
int LINALGEBRA_init()
{
    int                   device;
    struct cudaDeviceProp deviceProp;
    int                   devicecntMax = 100;

    cudaGetDeviceCount(&cuda_deviceCount);
    if (cuda_deviceCount > devicecntMax)
    {
        cuda_deviceCount = 0;
    }
    if (cuda_deviceCount < 0)
    {
        cuda_deviceCount = 0;
    }

    printf("%s: %d devices found\n", __func__, cuda_deviceCount);
    printf("\n");
    for (device = 0; device < cuda_deviceCount; ++device)
    {
        cudaGetDeviceProperties(&deviceProp, device);

        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, device);

        printf("Device %d [ %20s ]  has compute capability %d.%d.\n", device, deviceProp.name,
               deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %.0f MBytes "
               "(%llu bytes)\n",
               (float) deviceProp.totalGlobalMem / 1048576.0f,
               (unsigned long long) deviceProp.totalGlobalMem);
        printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
               "GHz)\n",
               clockRate * 1e-3f, clockRate * 1e-6f);
        printf("\n");
#    ifdef HAVE_MAGMA
        printf("Using MAGMA library\n");
        magma_print_environment();
#    endif

        printf("\n");
    }

    return ((int) cuda_deviceCount);
}


void *GPU_scanDevices(void *deviceCount_void_ptr)
{
    int                  *devcnt_ptr = (int *) deviceCount_void_ptr;
    int                   device;
    struct cudaDeviceProp deviceProp;
    int                   devicecntMax = 100;


    printf("Scanning for GPU devices ...\n");
    fflush(stdout);

    cudaGetDeviceCount(&cuda_deviceCount);
    if (cuda_deviceCount > devicecntMax)
    {
        cuda_deviceCount = 0;
    }
    if (cuda_deviceCount < 0)
    {
        cuda_deviceCount = 0;
    }

    printf("%s: %d devices found\n", __func__, cuda_deviceCount);
    fflush(stdout);

    printf("\n");
    for (device = 0; device < cuda_deviceCount; ++device)
    {
        cudaGetDeviceProperties(&deviceProp, device);

        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, device);


        printf("Device %d [ %20s ]  has compute capability %d.%d.\n", device, deviceProp.name,
               deviceProp.major, deviceProp.minor);

        printf("  Total amount of global memory:                 %.0f MBytes "
               "(%llu bytes)\n",
               (float) deviceProp.totalGlobalMem / 1048576.0f,
               (unsigned long long) deviceProp.totalGlobalMem);

        printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);

        printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
               "GHz)\n",
               clockRate * 1e-3f, clockRate * 1e-6f);

        printf("\n");
    }

    printf("Done scanning for GPU devices\n");
    fflush(stdout);

    *devcnt_ptr = cuda_deviceCount;

    return NULL;
}

#endif

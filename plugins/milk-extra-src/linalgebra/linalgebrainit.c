/** @file linalgebrainit.c
 */

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
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
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

static FPS_APP_INFO FPS_app_info_li = {
    .fps_name         = "linalgebrainit",
    .cmdkey           = "linalgebrainit",
    .description      = "init linalgebra",
    .description_long = "Initialize linear algebra subsystem. Sets up BLAS/LAPACK configuration "
                        "and GPU context if available."
};
#    define FPS_PARAMS_LI(X)
#    include "fps.h"
static FPS_CLI_BINDING                   li_b[]     = { { 0 } };
static const int                         li_nb      = 0;
static CLICMDARGDEF                      farg[]     = { { 0 } };
static CLICMDDATA                        CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS                       li_cms     = { 0 };
static __attribute__((constructor)) void init_li(void)
{
    strncpy(CLIcmddata.key, FPS_app_info_li.cmdkey, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, FPS_app_info_li.description,
            sizeof(CLIcmddata.description) - 1);
    CLIcmddata.nbarg         = 0;
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags         = CLICMDFLAG_FPS;
    if (!CLIcmddata.cmdsettings)
    {
        CLIcmddata.cmdsettings = &li_cms;
    }
}
static errno_t li_compute(void)
{
    LINALGEBRA_init();
    return RETURN_SUCCESS;
}
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_li, farg, &CLIcmddata, li_b, li_nb,
                                        li_compute);
}

errno_t linalgebrainit_addCLIcmd()
{
    safe_fps_fill_farg_examples(farg, li_b, li_nb);
    INSERT_STD_CLIREGISTERFUNC;

    return RETURN_SUCCESS;
}

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

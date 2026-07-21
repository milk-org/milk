// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cublas_Coeff2Map_Loop.c
 * @brief Cublas coeff2map loop module
 */

// MILK_CMAKE_MANDATE_CUDA

#include <semaphore.h>

#include <cublas_v2.h>

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

extern int cuda_deviceCount;

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t LINALGEBRA_Coeff2Map_Loop(const char *IDmodes_name,
                                  const char *IDcoeff_name,
                                  int         GPUindex,
                                  const char *IDoutmap_name,
                                  int         offsetmode,
                                  const char *IDoffset_name);

// ==========================================
// Gen 4 V2 CLI commands
// ==========================================

/* ===== Command: cudacoeff2map ===== */
static char         cm_m[FUNCTION_PARAMETER_STRMAXLEN] = "modes";
static char         cm_c[FUNCTION_PARAMETER_STRMAXLEN] = "coeff";
static int64_t      cm_gpu                             = 4;
static char         cm_o[FUNCTION_PARAMETER_STRMAXLEN] = "outmap";
static FPS_APP_INFO FPS_app_info_cm                    = {
    .fps_name         = "cudacoeff2map",
    .cmdkey           = "cudacoeff2map",
    .description      = "CUDA multiply vector by modes",
    .description_long = "GPU-accelerated coefficient-to-map conversion using cuBLAS. Continuously "
                        "transforms modal coefficients into spatial maps."
};
#define FPS_PARAMS_CM(X)                                                        \
    X(".modes", cm_m, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "modes")      \
    X(".coeff", cm_c, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "coeff")      \
    X(".gpuindex", &cm_gpu, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "GPU index") \
    X(".outmap", cm_o, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output map")
#include "fps.h"
static FPS_CLI_BINDING                   cm_b[]    = { FPS_PARAMS_CM(FPS_X_BINDING) };
static const int                         cm_nb     = sizeof(cm_b) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF                      cm_farg[] = { FPS_PARAMS_CM(FPS_X_FARG) };
static CLICMDDATA                        cm_d   = { "", "", CLICMD_FIELDS_DEFAULTS_W_ARG(cm_farg) };
static CMDSETTINGS                       cm_cms = { 0 };
static __attribute__((constructor)) void init_cm(void)
{
    strncpy(cm_d.key, FPS_app_info_cm.cmdkey, sizeof(cm_d.key) - 1);
    strncpy(cm_d.description, FPS_app_info_cm.description, sizeof(cm_d.description) - 1);
    cm_d.nbarg         = sizeof(cm_farg) / sizeof(CLICMDARGDEF);
    cm_d.funcfpscliarg = cm_farg;
    cm_d.flags         = CLICMDFLAG_FPS;
    if (!cm_d.cmdsettings)
    {
        cm_d.cmdsettings = &cm_cms;
    }
}
static errno_t cm_compute(void)
{
    LINALGEBRA_Coeff2Map_Loop(cm_m, cm_c, (int) cm_gpu, cm_o, 0, " ");
    return RETURN_SUCCESS;
}
static errno_t cm_CLIfunc(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_cm, cm_farg, &cm_d, cm_b, cm_nb, cm_compute);
}

/* ===== Command: cudacoeffo2map ===== */
static char         co_m[FUNCTION_PARAMETER_STRMAXLEN]   = "modes";
static char         co_c[FUNCTION_PARAMETER_STRMAXLEN]   = "coeff";
static int64_t      co_gpu                               = 4;
static char         co_o[FUNCTION_PARAMETER_STRMAXLEN]   = "outmap";
static char         co_off[FUNCTION_PARAMETER_STRMAXLEN] = "offsetim";
static FPS_APP_INFO FPS_app_info_co                      = {
    .fps_name         = "cudacoeffo2map",
    .cmdkey           = "cudacoeffo2map",
    .description      = "CUDA coeff2map with offset",
    .description_long = "GPU-accelerated coefficient-to-map conversion using cuBLAS. Continuously "
                        "transforms modal coefficients into spatial maps."
};
#define FPS_PARAMS_CO(X)                                                        \
    X(".modes", co_m, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "modes")      \
    X(".coeff", co_c, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "coeff")      \
    X(".gpuindex", &co_gpu, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "GPU index") \
    X(".outmap", co_o, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output")    \
    X(".offset", co_off, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "offset")
static FPS_CLI_BINDING                   co_b[]    = { FPS_PARAMS_CO(FPS_X_BINDING) };
static const int                         co_nb     = sizeof(co_b) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF                      co_farg[] = { FPS_PARAMS_CO(FPS_X_FARG) };
static CLICMDDATA                        co_d   = { "", "", CLICMD_FIELDS_DEFAULTS_W_ARG(co_farg) };
static CMDSETTINGS                       co_cms = { 0 };
static __attribute__((constructor)) void init_co(void)
{
    strncpy(co_d.key, FPS_app_info_co.cmdkey, sizeof(co_d.key) - 1);
    strncpy(co_d.description, FPS_app_info_co.description, sizeof(co_d.description) - 1);
    co_d.nbarg         = sizeof(co_farg) / sizeof(CLICMDARGDEF);
    co_d.funcfpscliarg = co_farg;
    co_d.flags         = CLICMDFLAG_FPS;
    if (!co_d.cmdsettings)
    {
        co_d.cmdsettings = &co_cms;
    }
}
static errno_t co_compute(void)
{
    LINALGEBRA_Coeff2Map_Loop(co_m, co_c, (int) co_gpu, co_o, 1, co_off);
    return RETURN_SUCCESS;
}
static errno_t co_CLIfunc(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_co, co_farg, &co_d, co_b, co_nb, co_compute);
}

errno_t Coeff2Map_Loop_addCLIcmd()
{
    {
        safe_fps_fill_farg_examples(cm_farg, cm_b, cm_nb);
        int cmdi         = RegisterCLIcmd(cm_d, cm_CLIfunc);
        cm_d.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(co_farg, co_b, co_nb);
        int cmdi         = RegisterCLIcmd(co_d, co_CLIfunc);
        co_d.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}

//
// single GPU
// semaphore input = 3
//
errno_t LINALGEBRA_Coeff2Map_Loop(const char *IDmodes_name,
                                  const char *IDcoeff_name,
                                  int         GPUindex,
                                  const char *IDoutmap_name,
                                  int         offsetmode,
                                  const char *IDoffset_name)
{
    long    NBmodes;
    imageID IDmodes;
    imageID IDcoeff;
    imageID IDoutmap;

    cublasHandle_t        cublasH       = NULL;
    cublasStatus_t        cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t           cudaStat      = cudaSuccess;
    struct cudaDeviceProp deviceProp;

    float *d_modes  = NULL; // linear memory of GPU
    float *d_coeff  = NULL;
    float *d_outmap = NULL;

    float           alpha = 1.0;
    float           beta  = 0.0;
    int             loopOK;
    struct timespec ts;
    long            iter;
    uint64_t        cnt;
    long            scnt;
    int             semval;
    int             semr;

    int devicecntMax = 100;

    imageID IDoffset;

    printf("entering LINALGEBRA_Coeff2Map_Loop\n");
    printf("offsetmode = %d\n", offsetmode);
    fflush(stdout);

    if (offsetmode == 1)
    {
        beta     = 1.0;
        IDoffset = image_ID(IDoffset_name, dcimg, dcnimg);

        if (IDoffset == -1)
        {
            printf("ERROR: image \"%s\" does not exist\n", IDoffset_name);
            exit(0);
        }
    }

    IDoutmap = image_ID(IDoutmap_name, dcimg, dcnimg);
    if (IDoutmap == -1)
    {
        printf("ERROR: missing output stream\n");
        exit(0);
    }

    cudaGetDeviceCount(&cuda_deviceCount);
    printf("%s : %d devices found\n", __func__, cuda_deviceCount);
    fflush(stdout);
    if (cuda_deviceCount > devicecntMax)
    {
        cuda_deviceCount = 0;
    }
    if (cuda_deviceCount < 0)
    {
        cuda_deviceCount = 0;
    }


    printf("\n");
    for (int k = 0; k < cuda_deviceCount; ++k)
    {
        cudaGetDeviceProperties(&deviceProp, k);

        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, k);

        printf("Device %d [ %20s ]  has compute capability %d.%d.\n", k, deviceProp.name,
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

    if (GPUindex < cuda_deviceCount)
    {
        cudaSetDevice(GPUindex);
    }
    else
    {
        printf("Invalid Device : %d / %d\n", GPUindex, cuda_deviceCount);
        exit(0);
    }

    printf("Create cublas handle ...");
    fflush(stdout);
    cublas_status = cublasCreate(&cublasH);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    printf(" done\n");
    fflush(stdout);

    // load modes to GPU

    IDcoeff = image_ID(IDcoeff_name, dcimg, dcnimg);
    NBmodes = 1;
    for (uint8_t k = 0; k < dcimg[IDcoeff].md[0].naxis; k++)
    {
        NBmodes *= dcimg[IDcoeff].md[0].size[k];
    }

    IDmodes = image_ID(IDmodes_name, dcimg, dcnimg);
    uint64_t mdim;
    if (dcimg[IDmodes].md[0].naxis == 3)
    {
        mdim = dcimg[IDmodes].md[0].size[0] * dcimg[IDmodes].md[0].size[1];
    }
    else
    {
        mdim = dcimg[IDmodes].md[0].size[0];
    }

    printf("Allocating d_modes. Size = %lu x %ld, total = %ld\n", mdim, NBmodes,
           sizeof(float) * mdim * NBmodes);
    fflush(stdout);
    cudaStat = cudaMalloc((void **) &d_modes, sizeof(float) * mdim * NBmodes);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_DMmodes returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("cudaMemcpy ID %ld  -> d_modes\n", IDmodes);
    fflush(stdout);
    list_image_ID();
    cudaStat = cudaMemcpy(d_modes, dcimg[IDmodes].array.F, sizeof(float) * mdim * NBmodes,
                          cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }

    // create d_outmap
    printf("Allocating d_outmap. Size = %ld,  total = %ld\n", mdim, sizeof(float) * mdim);
    fflush(stdout);
    cudaStat = cudaMalloc((void **) &d_outmap, sizeof(float) * mdim);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_outmap returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }

    // create d_coeff
    printf("Allocating d_coeff. Size = %ld,  total = %ld\n", NBmodes, sizeof(float) * NBmodes);
    fflush(stdout);
    cudaStat = cudaMalloc((void **) &d_coeff, sizeof(float) * NBmodes);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_coeff returned error code %d, line(%d)\n", cudaStat, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (sigaction(SIGINT, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGTERM, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGBUS, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGSEGV, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGABRT, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGHUP, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGPIPE, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGSEGV, &dcsigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    loopOK = 1;
    iter   = 0;

    printf("ENTERING LOOP, %ld modes (offsetmode = %d)\n", NBmodes, offsetmode);
    fflush(stdout);

    while (loopOK == 1)
    {
        if (dcimg[IDcoeff].md[0].sem == 0)
        {
            while (dcimg[IDcoeff].md[0].cnt0 == cnt) // test if new frame exists
            {
                struct timespec treq, trem;
                treq.tv_sec  = 0;
                treq.tv_nsec = 5000;
                nanosleep(&treq, &trem);
            }
            cnt  = dcimg[IDcoeff].md[0].cnt0;
            semr = 0;
        }
        else
        {
            if (clock_gettime(CLOCK_MILK, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 1;
            semr = ImageStreamIO_semtimedwait(dcimg + IDcoeff, 3, &ts);

            if (iter == 0)
            {
                //  printf("driving semaphore to zero ... ");
                // fflush(stdout);
                semval = ImageStreamIO_semvalue(dcimg + IDcoeff, 2);
                for (scnt = 0; scnt < semval; scnt++)
                {
                    printf("WARNING %s %d  : sem_trywait on semptr2\n", __FILE__, __LINE__);
                    fflush(stdout);
                    ImageStreamIO_semtrywait(dcimg + IDcoeff, 2);
                }
                // printf("done\n");
                // fflush(stdout);
            }
        }

        if (semr == 0)
        {
            //  printf("Compute\n");
            //  fflush(stdout);

            // send vector back to GPU
            cudaStat = cudaMemcpy(d_coeff, dcimg[IDcoeff].array.F, sizeof(float) * NBmodes,
                                  cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess)
            {
                printf("cudaMemcpy returned error code %d, line(%d)\n", cudaStat, __LINE__);
                exit(EXIT_FAILURE);
            }

            if (offsetmode == 1)
            {
                cudaStat = cudaMemcpy(d_outmap, dcimg[IDoffset].array.F, sizeof(float) * mdim,
                                      cudaMemcpyHostToDevice);
                if (cudaStat != cudaSuccess)
                {
                    printf("cudaMemcpy returned error code %d, line(%d)\n", cudaStat, __LINE__);
                    exit(EXIT_FAILURE);
                }
            }

            // compute
            cublas_status = cublasSgemv(cublasH, CUBLAS_OP_N, mdim, NBmodes, &alpha, d_modes, mdim,
                                        d_coeff, 1, &beta, d_outmap, 1);
            if (cublas_status != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasSgemv returned error code %d, line(%d)\n", cublas_status, __LINE__);
                fflush(stdout);
                if (cublas_status == CUBLAS_STATUS_NOT_INITIALIZED)
                {
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                }
                if (cublas_status == CUBLAS_STATUS_INVALID_VALUE)
                {
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                }
                if (cublas_status == CUBLAS_STATUS_ARCH_MISMATCH)
                {
                    printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                }
                if (cublas_status == CUBLAS_STATUS_EXECUTION_FAILED)
                {
                    printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                }

                printf("GPU index                           = %d\n", GPUindex);

                printf("CUBLAS_OP_N                         = %d\n", CUBLAS_OP_N);
                printf("alpha                               = %f\n", alpha);
                printf("alpha                               = %f\n", beta);
                printf("m                                   = %d\n", (int) mdim);
                printf("NBmodes                             = %d\n", (int) NBmodes);
                fflush(stdout);
                exit(EXIT_FAILURE);
            }

            // copy result
            dcimg[IDoutmap].md[0].write = 1;
            cudaStat = cudaMemcpy(dcimg[IDoutmap].array.F, d_outmap, sizeof(float) * mdim,
                                  cudaMemcpyDeviceToHost);
            semval   = ImageStreamIO_semvalue(dcimg + IDoutmap, 0);
            if (semval < SEMAPHORE_MAXVAL)
            {
                ImageStreamIO_sempost(dcimg + IDoutmap, 0);
            }
            semval = ImageStreamIO_semvalue(dcimg + IDoutmap, 1);
            if (semval < SEMAPHORE_MAXVAL)
            {
                ImageStreamIO_sempost(dcimg + IDoutmap, 1);
            }
            dcimg[IDoutmap].md[0].cnt0++;
            dcimg[IDoutmap].md[0].write = 0;
        }

        if ((dcsigINT == 1) || (dcsigTERM == 1) || (dcsigABRT == 1) || (dcsigBUS == 1) ||
            (dcsigSEGV == 1) || (dcsigHUP == 1) || (dcsigPIPE == 1))
        {
            loopOK = 0;
        }

        iter++;
    }

    cudaFree(d_modes);
    cudaFree(d_outmap);
    cudaFree(d_coeff);

    if (cublasH)
    {
        cublasDestroy(cublasH);
    }

    return RETURN_SUCCESS;
}

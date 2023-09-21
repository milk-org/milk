/** @file Coeff2Map_Loop.c
 */

#ifdef HAVE_CUDA

// include sem_timedwait
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <semaphore.h>

#include <cublas_v2.h>

#include "CommandLineInterface/CLIcore.h"

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
// Command line interface wrapper function(s)
// ==========================================

static errno_t LINALGEBRA_Coeff2Map_Loop_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 4) ==
            0)
    {
        LINALGEBRA_Coeff2Map_Loop(data.cmdargtoken[1].val.string,
                                  data.cmdargtoken[2].val.string,
                                  data.cmdargtoken[3].val.numl,
                                  data.cmdargtoken[4].val.string,
                                  0,
                                  " ");

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t LINALGEBRA_Coeff2Map_offset_Loop_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 4) + CLI_checkarg(5, 4) ==
            0)
    {
        LINALGEBRA_Coeff2Map_Loop(data.cmdargtoken[1].val.string,
                                  data.cmdargtoken[2].val.string,
                                  data.cmdargtoken[3].val.numl,
                                  data.cmdargtoken[4].val.string,
                                  1,
                                  data.cmdargtoken[5].val.string);

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

errno_t Coeff2Map_Loop_addCLIcmd()
{

    RegisterCLIcommand(
        "cudacoeff2map",
        __FILE__,
        LINALGEBRA_Coeff2Map_Loop_cli,
        "CUDA multiply vector by modes",
        "<modes> <coeffs vector> <GPU index [long]> <output map>",
        "cudacoeff2map modes coeff 4 outmap",
        "int LINALGEBRA_Coeff2Map_Loop(const char *IDmodes_name, const char "
        "*IDcoeff_name, int GPUindex, "
        "const char *IDoutmap_name, int offsetmode, const char "
        "*IDoffset_name)");

    RegisterCLIcommand(
        "cudacoeffo2map",
        __FILE__,
        LINALGEBRA_Coeff2Map_offset_Loop_cli,
        "CUDA multiply vector by modes and add offset",
        "<modes> <coeffs vector> <GPU index [long]> <output "
        "map> <offset image>",
        "cudacoeffo2map modes coeff 4 outmap offsetim",
        "int LINALGEBRA_Coeff2Map_Loop(const char *IDmodes_name, "
        "const char *IDcoeff_name, int GPUindex, "
        "const char *IDoutmap_name, int offsetmode, const char "
        "*IDoffset_name)");

    return RETURN_SUCCESS;
}

//
// single GPU
// semaphore input = 3
//
errno_t LINALGEBRA_Coeff2Map_Loop(
    const char *IDmodes_name,
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

    if(offsetmode == 1)
    {
        beta     = 1.0;
        IDoffset = image_ID(IDoffset_name);

        if(IDoffset == -1)
        {
            printf("ERROR: image \"%s\" does not exist\n", IDoffset_name);
            exit(0);
        }
    }

    IDoutmap = image_ID(IDoutmap_name);
    if(IDoutmap == -1)
    {
        printf("ERROR: missing output stream\n");
        exit(0);
    }

    cudaGetDeviceCount(&cuda_deviceCount);
    printf("%s : %d devices found\n", __func__, cuda_deviceCount);
    fflush(stdout);
    if(cuda_deviceCount > devicecntMax)
    {
        cuda_deviceCount = 0;
    }
    if(cuda_deviceCount < 0)
    {
        cuda_deviceCount = 0;
    }



    printf("\n");
    for(int k = 0; k < cuda_deviceCount; ++k)
    {
        cudaGetDeviceProperties(&deviceProp, k);
        printf("Device %d [ %20s ]  has compute capability %d.%d.\n",
               k,
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
        printf(
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            (float) deviceProp.totalGlobalMem / 1048576.0f,
            (unsigned long long) deviceProp.totalGlobalMem);
        printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
        printf(
            "  GPU Clock rate:                                %.0f MHz (%0.2f "
            "GHz)\n",
            deviceProp.clockRate * 1e-3f,
            deviceProp.clockRate * 1e-6f);
        printf("\n");
    }

    if(GPUindex < cuda_deviceCount)
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
    if(cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    printf(" done\n");
    fflush(stdout);

    // load modes to GPU

    IDcoeff = image_ID(IDcoeff_name);
    NBmodes = 1;
    for(uint8_t k = 0; k < data.image[IDcoeff].md[0].naxis; k++)
    {
        NBmodes *= data.image[IDcoeff].md[0].size[k];
    }

    IDmodes = image_ID(IDmodes_name);
    uint64_t mdim;
    if(data.image[IDmodes].md[0].naxis == 3)
    {
        mdim = data.image[IDmodes].md[0].size[0] *
               data.image[IDmodes].md[0].size[1];
    }
    else
    {
        mdim = data.image[IDmodes].md[0].size[0];
    }

    printf("Allocating d_modes. Size = %lu x %ld, total = %ld\n",
           mdim,
           NBmodes,
           sizeof(float) * mdim * NBmodes);
    fflush(stdout);
    cudaStat = cudaMalloc((void **) &d_modes, sizeof(float) * mdim * NBmodes);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_DMmodes returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("cudaMemcpy ID %ld  -> d_modes\n", IDmodes);
    fflush(stdout);
    list_image_ID();
    cudaStat = cudaMemcpy(d_modes,
                          data.image[IDmodes].array.F,
                          sizeof(float) * mdim * NBmodes,
                          cudaMemcpyHostToDevice);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    // create d_outmap
    printf("Allocating d_outmap. Size = %ld,  total = %ld\n",
           mdim,
           sizeof(float) * mdim);
    fflush(stdout);
    cudaStat = cudaMalloc((void **) &d_outmap, sizeof(float) * mdim);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_outmap returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    // create d_coeff
    printf("Allocating d_coeff. Size = %ld,  total = %ld\n",
           NBmodes,
           sizeof(float) * NBmodes);
    fflush(stdout);
    cudaStat = cudaMalloc((void **) &d_coeff, sizeof(float) * NBmodes);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_coeff returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    if(sigaction(SIGINT, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGTERM, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGBUS, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGSEGV, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGABRT, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGHUP, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGPIPE, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    if(sigaction(SIGSEGV, &data.sigact, NULL) == -1)
    {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    loopOK = 1;
    iter   = 0;

    printf("ENTERING LOOP, %ld modes (offsetmode = %d)\n", NBmodes, offsetmode);
    fflush(stdout);

    while(loopOK == 1)
    {

        if(data.image[IDcoeff].md[0].sem == 0)
        {
            while(data.image[IDcoeff].md[0].cnt0 ==
                    cnt) // test if new frame exists
            {
                struct timespec treq, trem;
                treq.tv_sec  = 0;
                treq.tv_nsec = 5000;
                nanosleep(&treq, &trem);
            }
            cnt  = data.image[IDcoeff].md[0].cnt0;
            semr = 0;
        }
        else
        {

            if(clock_gettime(CLOCK_MILK, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 1;
            semr = ImageStreamIO_semtimedwait(data.image+IDcoeff, 3, &ts);

            if(iter == 0)
            {
                //  printf("driving semaphore to zero ... ");
                // fflush(stdout);
                semval = ImageStreamIO_semvalue(data.image+IDcoeff, 2);
                for(scnt = 0; scnt < semval; scnt++)
                {
                    printf("WARNING %s %d  : sem_trywait on semptr2\n",
                           __FILE__,
                           __LINE__);
                    fflush(stdout);
                    ImageStreamIO_semtrywait(data.image+IDcoeff, 2);
                }
                // printf("done\n");
                // fflush(stdout);
            }
        }

        if(semr == 0)
        {
            //  printf("Compute\n");
            //  fflush(stdout);

            // send vector back to GPU
            cudaStat = cudaMemcpy(d_coeff,
                                  data.image[IDcoeff].array.F,
                                  sizeof(float) * NBmodes,
                                  cudaMemcpyHostToDevice);
            if(cudaStat != cudaSuccess)
            {
                printf("cudaMemcpy returned error code %d, line(%d)\n",
                       cudaStat,
                       __LINE__);
                exit(EXIT_FAILURE);
            }

            if(offsetmode == 1)
            {
                cudaStat = cudaMemcpy(d_outmap,
                                      data.image[IDoffset].array.F,
                                      sizeof(float) * mdim,
                                      cudaMemcpyHostToDevice);
                if(cudaStat != cudaSuccess)
                {
                    printf("cudaMemcpy returned error code %d, line(%d)\n",
                           cudaStat,
                           __LINE__);
                    exit(EXIT_FAILURE);
                }
            }

            // compute
            cublas_status = cublasSgemv(cublasH,
                                        CUBLAS_OP_N,
                                        mdim,
                                        NBmodes,
                                        &alpha,
                                        d_modes,
                                        mdim,
                                        d_coeff,
                                        1,
                                        &beta,
                                        d_outmap,
                                        1);
            if(cublas_status != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasSgemv returned error code %d, line(%d)\n",
                       cublas_status,
                       __LINE__);
                fflush(stdout);
                if(cublas_status == CUBLAS_STATUS_NOT_INITIALIZED)
                {
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                }
                if(cublas_status == CUBLAS_STATUS_INVALID_VALUE)
                {
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                }
                if(cublas_status == CUBLAS_STATUS_ARCH_MISMATCH)
                {
                    printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                }
                if(cublas_status == CUBLAS_STATUS_EXECUTION_FAILED)
                {
                    printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                }

                printf("GPU index                           = %d\n", GPUindex);

                printf("CUBLAS_OP_N                         = %d\n",
                       CUBLAS_OP_N);
                printf("alpha                               = %f\n", alpha);
                printf("alpha                               = %f\n", beta);
                printf("m                                   = %d\n",
                       (int) mdim);
                printf("NBmodes                             = %d\n",
                       (int) NBmodes);
                fflush(stdout);
                exit(EXIT_FAILURE);
            }

            // copy result
            data.image[IDoutmap].md[0].write = 1;
            cudaStat = cudaMemcpy(data.image[IDoutmap].array.F,
                                  d_outmap,
                                  sizeof(float) * mdim,
                                  cudaMemcpyDeviceToHost);
            semval = ImageStreamIO_semvalue(data.image+IDoutmap, 0);
            if(semval < SEMAPHORE_MAXVAL)
            {
                ImageStreamIO_sempost(data.image+IDoutmap, 0);
            }
            semval = ImageStreamIO_semvalue(data.image+IDoutmap, 1);
            if(semval < SEMAPHORE_MAXVAL)
            {
                ImageStreamIO_sempost(data.image+IDoutmap, 1);
            }
            data.image[IDoutmap].md[0].cnt0++;
            data.image[IDoutmap].md[0].write = 0;
        }

        if((data.signal_INT == 1) || (data.signal_TERM == 1) ||
                (data.signal_ABRT == 1) || (data.signal_BUS == 1) ||
                (data.signal_SEGV == 1) || (data.signal_HUP == 1) ||
                (data.signal_PIPE == 1))
        {
            loopOK = 0;
        }

        iter++;
    }

    cudaFree(d_modes);
    cudaFree(d_outmap);
    cudaFree(d_coeff);

    if(cublasH)
    {
        cublasDestroy(cublasH);
    }

    return RETURN_SUCCESS;
}

#endif

/**
 * @file SGEMM.c
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "CommandLineInterface/timeutils.h"

#include "SGEMM.h"




#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <device_types.h>
#include <pthread.h>
#endif


// CPU mode: Use MKL if available
// Otherwise use openBLAS
//
#ifdef HAVE_MKL
#include "mkl.h"
#include "mkl_lapacke.h"
#define BLASLIB "IntelMKL"
#else
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#define BLASLIB "OpenBLAS"
#endif
#endif



static char *inmatA;
static long  fpi_inmatA;

static char *inmatB;
static long  fpi_inmatB;

static uint64_t *transpA;
static long      fpi_transpA;

static uint64_t *transpB;
static long      fpi_transpB;

static char *outM;
static long  fpi_outM;

static int32_t *GPUdevice;
static long     fpi_GPUdevice;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".matA",
        "input matrix A",
        "matA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmatA,
        &fpi_inmatA
    },
    {
        CLIARG_IMG,
        ".matB",
        "output matrix B",
        "matA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmatB,
        &fpi_inmatB
    },
    {
        CLIARG_ONOFF,
        ".transpA",
        "transpose A",
        "OFF",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &transpA,
        &fpi_transpA
    },
    {
        CLIARG_ONOFF,
        ".transpB",
        "transpose B",
        "OFF",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &transpB,
        &fpi_transpB
    },

    {
        CLIARG_STR,
        ".outM",
        "output matrix",
        "out",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outM,
        &fpi_outM
    },
    {
        // using GPU (99 : no GPU, otherwise GPU device)
        CLIARG_INT32,
        ".GPUdevice",
        "GPU device, 99 for CPU",
        "-1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUdevice,
        &fpi_GPUdevice
    }
};


static CLICMDDATA CLIcmddata =
{
    "sgemm", "matrix-matrix multiply", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("CPU or GPU matrix-matrix multiplication\n");

    return RETURN_SUCCESS;
}


errno_t computeSGEMM(
    IMGID imginA,
    IMGID imginB,
    IMGID *outimg,
    int TranspA,
    int TranspB,
    int GPUdev
)
{
    DEBUG_TRACE_FSTART();

    int SGEMMcomputed = 0;

    // Get input matrices A and B sizes (MxN)
    // if 3D cubes, group first 2 dimensions into M

    int inA_Mdim;
    int inA_Mdim0;
    int inA_Mdim1;
    int inA_Ndim;
    int inA_Ndim0;
    int inA_Ndim1;

    if(imginA.md->naxis == 3)
    {
        //printf("inA_Mdim   : %d x %d\n", imginA.md->size[0], imginA.md->size[1]);
        inA_Mdim = imginA.md->size[0] * imginA.md->size[1];
        inA_Mdim0 = imginA.md->size[0];
        inA_Mdim1 = imginA.md->size[1];

        //printf("inA_Ndim    : %d\n", imginA.md->size[2]);
        inA_Ndim = imginA.md->size[2];
        inA_Ndim0 = imginA.md->size[2];
        inA_Ndim1 = 1;
    }
    else
    {
        //printf("inA_Mdim   : %d\n", imginA.md->size[0]);
        inA_Mdim = imginA.md->size[0];
        inA_Mdim0 = imginA.md->size[1];
        inA_Mdim1 = 1;

        //printf("inNdim    : %d\n", imginA.md->size[1]);
        inA_Ndim = imginA.md->size[1];
        inA_Ndim0 = imginA.md->size[1];
        inA_Ndim1 = 1;
    }


    int inB_Mdim;
    int inB_Mdim0;
    int inB_Mdim1;
    int inB_Ndim;
    int inB_Ndim0;
    int inB_Ndim1;

    if(imginB.md->naxis == 3)
    {
        //printf("inB_Mdim   : %d x %d\n", imginB.md->size[0], imginB.md->size[1]);
        inB_Mdim = imginB.md->size[0] * imginB.md->size[1];
        inB_Mdim0 = imginB.md->size[0];
        inB_Mdim1 = imginB.md->size[1];

        //printf("inB_Ndim    : %d\n", imginB.md->size[2]);
        inB_Ndim = imginB.md->size[2];
        inB_Ndim0 = imginB.md->size[2];
        inB_Ndim1 = 1;
    }
    else
    {
        //printf("inB_Mdim   : %d\n", imginB.md->size[0]);
        inB_Mdim = imginB.md->size[0];
        inB_Mdim0 = imginB.md->size[1];
        inB_Mdim1 = 1;

        //printf("inB_Ndim    : %d\n", imginB.md->size[1]);
        inB_Ndim = imginB.md->size[1];
        inB_Ndim0 = imginB.md->size[1];
        inB_Ndim1 = 1;
    }


    // input to SGEMM function
    int Mdim, Ndim, Kdim;
    int Mdim0, Ndim0, Kdim0;
    int Mdim1, Ndim1, Kdim1;


    // if no transpose
    Mdim = inA_Mdim;
    Mdim0 = inA_Mdim0;
    Mdim1 = inA_Mdim1;

    Ndim  = inB_Ndim;
    Ndim0 = inB_Ndim0;
    Ndim1 = inB_Ndim1;

    Kdim = inA_Ndim;

    if ( TranspA == 1 )
    {
        Mdim = inA_Ndim;
        Mdim0 = inA_Ndim0;
        Mdim1 = inA_Ndim1;

        Kdim = inA_Mdim;

    }
    if ( TranspB == 1 )
    {
        Ndim  = inB_Mdim;
        Ndim0 = inB_Mdim0;
        Ndim1 = inB_Mdim1;
    }

    //printf("T %d %d  -> SGEMM  M=%d, N=%d, K=%d\n", TranspA, TranspB, Mdim, Ndim, Kdim);


    // Create output
    //
    int outMdim = Mdim;
    int outNdim = Ndim;
    if( Mdim1 == 1)
    {
        // 2D output
        outimg->naxis = 2;
        outimg->size[0] = outMdim;
        outimg->size[1] = outNdim;
        outimg->size[2] = 1;

    }
    else
    {
        // 3D output
        outimg->naxis = 3;
        outimg->size[0] = Mdim0;
        outimg->size[1] = Mdim1;
        outimg->size[2] = outNdim;
    }
    outimg->datatype = _DATATYPE_FLOAT;
    createimagefromIMGID(outimg);






    if( (GPUdev >= 0) && (GPUdev <= 99))
    {
#ifdef HAVE_CUDA
        //printf("Running SGEMM on GPU device %d\n", GPUdev);
        //fflush(stdout);

        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;


        float *d_inmatA;

        {
            cudaError_t cudaStat = cudaMalloc((void **)&d_inmatA, imginA.md->nelement * sizeof(float));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                return EXIT_FAILURE;
            }
        }


        {
            cudaError_t stat = cudaMemcpy(d_inmatA, imginA.im->array.F, imginA.md->nelement * sizeof(float),
                                          cudaMemcpyHostToDevice);
            if (stat != cudaSuccess) {
                printf ("cudaMemcpy failed\n");
                return EXIT_FAILURE;
            }
        }




        float *d_inmatB;

        {
            cudaError_t cudaStat = cudaMalloc((void **)&d_inmatB, imginB.md->nelement * sizeof(float));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                return EXIT_FAILURE;
            }
        }


        {
            cudaError_t stat = cudaMemcpy(d_inmatB, imginB.im->array.F, imginB.md->nelement * sizeof(float),
                                          cudaMemcpyHostToDevice);
            if (stat != cudaSuccess) {
                printf ("cudaMemcpy failed\n");
                return EXIT_FAILURE;
            }
        }





        float *d_outmat;
        {
            cudaError_t cudaStat = cudaMalloc((void **)&d_outmat, outimg->md->nelement * sizeof(float));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                return EXIT_FAILURE;
            }
        }


        // Create a handle for CUBLAS
        cublasHandle_t handle;
        {
            cublasStatus_t stat = cublasCreate(&handle);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("cublasCreate failed\n");
                return EXIT_FAILURE;
            }
        }


        // Do the actual multiplication

        cublasOperation_t OPA = CUBLAS_OP_N;
        cublasOperation_t OPB = CUBLAS_OP_N;
        if ( TranspA == 1 )
        {
            OPA = CUBLAS_OP_T;
        }
        if ( TranspB == 1 )
        {
            OPB = CUBLAS_OP_T;
        }


        {
            cublasStatus_t stat = cublasSgemm(handle, OPA, OPB,
                                              Mdim, Ndim, Kdim, alpha,
                                              d_inmatA, inA_Mdim,
                                              d_inmatB, inB_Mdim,
                                              beta, d_outmat, outMdim);

            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("cublasSgemm failed\n");
                return EXIT_FAILURE;
            }
        }

        cublasDestroy(handle);


        {
            cudaError_t stat = cudaMemcpy(outimg->im->array.F, d_outmat,
                                          outimg->md->nelement * sizeof(float), cudaMemcpyDeviceToHost);
            if (stat != cudaSuccess) {
                printf ("cudaMemcpy failed\n");
                return EXIT_FAILURE;
            }
        }

        cudaFree(d_inmatA);
        cudaFree(d_inmatB);
        cudaFree(d_outmat);

        SGEMMcomputed = 1;

#endif
    }


    if ( SGEMMcomputed == 0)
    {
//        printf("Running SGEMM on CPU\n");
//        fflush(stdout);

        CBLAS_TRANSPOSE OPA = CblasNoTrans;
        if(TranspA == 1 )
        {
            OPA = CblasTrans;
        }

        CBLAS_TRANSPOSE OPB = CblasNoTrans;
        if(TranspB == 1 )
        {
            OPB = CblasTrans;
        }

        cblas_sgemm(CblasColMajor, OPA, OPB,
                    Mdim, Ndim, Kdim, 1.0,
                    imginA.im->array.F, inA_Mdim,
                    imginB.im->array.F, inB_Mdim,
                    0.0, outimg->im->array.F, outMdim);
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginA = mkIMGID_from_name(inmatA);
    resolveIMGID(&imginA, ERRMODE_ABORT);

    IMGID imginB = mkIMGID_from_name(inmatB);
    resolveIMGID(&imginB, ERRMODE_ABORT);



    IMGID imgM  = mkIMGID_from_name(outM);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {


        computeSGEMM(imginA, imginB, &imgM, *transpA, *transpB, *GPUdevice);


    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_linalgebra__SGEMM()
{

    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}


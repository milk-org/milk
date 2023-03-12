/**
 * @file SingularValueDecomp.c
 *
 */


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "CommandLineInterface/timeutils.h"



// Use MKL if available
// Otherwise use openBLAS
//
#ifdef HAVE_MKL
#include "mkl.h"
#define BLASLIB "IntelMKL"
#else
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#define BLASLIB "OpenBLAS"
#endif
#endif




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




static char *inM;
static long  fpi_inM;

static char *outU;
static long  fpi_outU;

static char *outV;
static long  fpi_outV;

static float *svdlim;
static long   fpi_svdlim;

static int32_t *GPUdevice;
static long     fpi_GPUdevice;





static CLICMDARGDEF farg[] =
{
    {
        // input
        CLIARG_IMG,
        ".inM",
        "input matrix",
        "inM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inM,
        &fpi_inM
    },
    {
        // output U
        CLIARG_STR,
        ".outU",
        "output U",
        "outU",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outU,
        &fpi_outU
    },
    {
        // output V
        CLIARG_STR,
        ".outV",
        "output V",
        "outV",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outV,
        &fpi_outV
    },
    {
        // Singular Value Decomposition limit
        CLIARG_FLOAT32,
        ".svdlim",
        "SVD limit",
        "0.01",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &svdlim,
        &fpi_svdlim
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



// Optional custom configuration setup.
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        data.fpsptr->parray[fpi_inM].fpflag |=
            FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
    }

    return RETURN_SUCCESS;
}




// Optional custom configuration checks.
// Runs at every configuration check loop iteration
//
static errno_t customCONFcheck()
{

    if(data.fpsptr != NULL)
    {
    }

    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata =
{
    "compSVD", "compute SVD", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




/**
 * @brief Compute SVD of indimM x indimN matrix
 *
 */
errno_t compute_SVD(
    IMGID imgin,
    float SVDeps,
    int GPUdev
)
{


    int inNdim;
    int inMdim;

    if(imgin.md->naxis == 3)
    {
        printf("inMdim   : %d x %d\n", imgin.md->size[0], imgin.md->size[1]);
        inMdim = imgin.md->size[0] * imgin.md->size[1];

        printf("inNdim    : %d\n", imgin.md->size[2]);
        inNdim = imgin.md->size[2];
    }
    else
    {
        printf("inMdim   : %d x %d\n", imgin.md->size[0], imgin.md->size[1]);
        inMdim = imgin.md->size[0] * imgin.md->size[1];

        printf("inNdim    : %d\n", imgin.md->size[2]);
        inNdim = imgin.md->size[2];
    }


    // Orient matrix such that it is tall (M>N)

    enum matrix_shape {inMshape_tall, inMshape_wide} mshape;
    uint32_t Mdim = 0;
    uint32_t Ndim = 0;

    if( inNdim < inMdim )
    {
        // input matrix is tall
        // this is the default
        // notations follow this case
        //
        printf("CASE NBMODE < NBACT \n");
        mshape = inMshape_tall;
        Mdim = inMdim;
        Ndim = inNdim;

    }
    else
    {
        printf("CASE NBMODE > NBACT \n");
        mshape = inMshape_wide;
        Mdim = inNdim;
        Ndim = inMdim;
    }


    // create eigenvectors array
    IMGID imgmV = makeIMGID_2D("mV", Ndim, Ndim);
    createimagefromIMGID(&imgmV);

    // create eigenvalues array
    IMGID imgeval = makeIMGID_2D("eigenval", Ndim, 1);
    createimagefromIMGID(&imgeval);





    {
        // create ATA
        // note that this is AAT if inNdim > inMdim (inMshape_wide)
        //
        IMGID imgATA = makeIMGID_2D("ATA", Ndim, Ndim);
        createimagefromIMGID(&imgATA);

        {
            int SGEMMcomputed = 0;
            if( (GPUdev >= 0) && (GPUdev <= 99))
            {
#ifdef HAVE_CUDA
                printf("Running SGEMM 1 on GPU device %d\n", *GPUdevice);
                fflush(stdout);

                const float alf = 1;
                const float bet = 0;
                const float *alpha = &alf;
                const float *beta = &bet;

                float *d_inM;
                cudaMalloc((void **)&d_inM, imgin.md->nelement * sizeof(float));
                cudaMemcpy(d_inM, imgin.im->array.F, imgin.md->nelement * sizeof(float), cudaMemcpyHostToDevice);

                float *d_ATA;
                cudaMalloc((void **)&d_ATA, imgATA.md->nelement * sizeof(float));

                // Create a handle for CUBLAS
                cublasHandle_t handle;
                cublasCreate(&handle);

                // Do the actual multiplication
                cublasOperation_t OP0 = CUBLAS_OP_T;
                cublasOperation_t OP1 = CUBLAS_OP_N;
                if ( mshape == inMshape_wide )
                {
                    OP0 = CUBLAS_OP_N;
                    OP1 = CUBLAS_OP_T;
                }

                cublasSgemm(handle, OP0, OP1,
                            Ndim, Ndim, Mdim, alpha, d_inM, inMdim, d_inM, inMdim, beta, d_ATA, Ndim);

                cublasDestroy(handle);

                cudaMemcpy(imgATA.im->array.F, d_ATA, imgATA.md->nelement * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_inM);
                cudaFree(d_ATA);

                SGEMMcomputed = 1;
#endif
            }
            if ( SGEMMcomputed == 0)
            {
                printf("Running SGEMM 1 on CPU\n");
                fflush(stdout);

                CBLAS_TRANSPOSE OP0 = CblasTrans;
                CBLAS_TRANSPOSE OP1 = CblasNoTrans;
                if ( mshape == inMshape_wide )
                {
                    OP0 = CblasNoTrans;
                    OP1 = CblasTrans;
                }

                cblas_sgemm(CblasColMajor, OP0, OP1,
                            Ndim, Ndim, Mdim, 1.0, imgin.im->array.F, inMdim, imgin.im->array.F, inMdim, 0.0, imgATA.im->array.F, Ndim);
            }
        }


        float *d = (float*) malloc(sizeof(float)*Ndim);
        float *e = (float*) malloc(sizeof(float)*Ndim);
        float *t = (float*) malloc(sizeof(float)*Ndim);


#ifdef HAVE_MKL
        mkl_set_interface_layer(MKL_INTERFACE_ILP64);
#endif

        LAPACKE_ssytrd(LAPACK_COL_MAJOR, 'U', Ndim, (float*) imgATA.im->array.F, Ndim, d, e, t);


        // Assemble Q matrix
        LAPACKE_sorgtr(LAPACK_COL_MAJOR, 'U', Ndim, imgATA.im->array.F, Ndim, t );




        // compute all eigenvalues and eivenvectors -> imgmV
        //
        memcpy(imgmV.im->array.F, imgATA.im->array.F, sizeof(float)*Ndim*Ndim);
        LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'V', Ndim, d, e, imgmV.im->array.F, Ndim);
        memcpy(imgeval.im->array.F, d, sizeof(float)*Ndim);


        free(d);
        free(e);
        free(t);

        delete_image(imgATA, DELETE_IMAGE_ERRMODE_EXIT);

        // This is matV if inMshape_tall, matU if inMshape_wide
        //save_fits("mV", "mV.fits");
    }






    // create mU (if inMshape_tall)
    // create mV (if inMshape_wide)
    // (only non-zero part allocated)
    //
    IMGID imgmU = makeIMGID_2D("mU", Mdim, Ndim);
    createimagefromIMGID(&imgmU);


    // Compute mU (only non-zero part allocated)
    // Multiply RMmodesDM by Vmat
    //

    {
        int SGEMMcomputed = 0;
        if( (*GPUdevice >= 0) && (*GPUdevice <= 99))
        {
#ifdef HAVE_CUDA
            printf("Running SGEMM 2 on GPU device %d\n", *GPUdevice);
            fflush(stdout);

            const float alf = 1;
            const float bet = 0;
            const float *alpha = &alf;
            const float *beta = &bet;

            float *d_inM;
            cudaMalloc((void **)&d_inM, imgin.md->nelement * sizeof(float));
            cudaMemcpy(d_inM, imgin.im->array.F, imgin.md->nelement * sizeof(float), cudaMemcpyHostToDevice);

            float *d_mV;
            cudaMalloc((void **)&d_mV, imgmV.md->nelement * sizeof(float));
            cudaMemcpy(d_mV, imgmV.im->array.F, imgmV.md->nelement * sizeof(float), cudaMemcpyHostToDevice);

            float *d_mU;
            cudaMalloc((void **)&d_mU, imgmU.md->nelement * sizeof(float));

            cublasHandle_t handle;
            cublasCreate(&handle);

            cublasOperation_t OP0 = CUBLAS_OP_N;
            if ( mshape == inMshape_wide )
            {
                OP0 = CUBLAS_OP_T;
            }
            cublasSgemm(handle, OP0, CUBLAS_OP_N,
                        Mdim, Ndim, Ndim, alpha, d_inM, inMdim, d_mV, Ndim, beta, d_mU, Mdim);

            cublasDestroy(handle);

            cudaMemcpy(imgmU.im->array.F, d_mU, imgmU.md->nelement * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_inM);
            cudaFree(d_mV);
            cudaFree(d_mU);

            SGEMMcomputed = 1;
#endif
        }

        if ( SGEMMcomputed == 0 )
        {
            printf("Running SGEMM 2 on CPU\n");
            fflush(stdout);

            CBLAS_TRANSPOSE OP0 = CblasNoTrans;
            if ( mshape == inMshape_wide )
            {
                OP0 = CblasTrans;
            }

            cblas_sgemm (CblasColMajor, OP0, CblasNoTrans,
                         Mdim, Ndim, Ndim, 1.0, imgin.im->array.F, inMdim, imgmV.im->array.F, Ndim, 0.0, imgmU.im->array.F, Mdim);

        }
    }








    return RETURN_SUCCESS;
}








static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM = mkIMGID_from_name(inM);
    resolveIMGID(&imginM, ERRMODE_ABORT);



    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {


        compute_SVD(imginM, *svdlim, *GPUdevice);


    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_cudacomp__compSVD()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

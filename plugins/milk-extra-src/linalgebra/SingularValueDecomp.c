/**
 * @file SingularValueDecomp.c
 *
 */


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "CommandLineInterface/timeutils.h"

#include "SingularValueDecomp.h"


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

static char *outev;
static long  fpi_outev;

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
        // output ev
        CLIARG_STR,
        ".outev",
        "output eigenval",
        "outev",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outev,
        &fpi_outev
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
 * compSVDmode
 * COMPSVD_SKIP_BIGMAT  skip big (U of V) matrix computation
 */
errno_t compute_SVD(
    IMGID imgin,
    IMGID imgU,
    IMGID imgeigenval,
    IMGID imgV,
    int GPUdev,
    uint64_t compSVDmode
)
{


    int inNdim, inNdim0, inNdim1;
    int inMdim, inMdim0, inMdim1;

    if(imgin.md->naxis == 3)
    {
        printf("inMdim   : %d x %d\n", imgin.md->size[0], imgin.md->size[1]);
        inMdim = imgin.md->size[0] * imgin.md->size[1];
        inMdim0 = imgin.md->size[0];
        inMdim1 = imgin.md->size[1];

        printf("inNdim    : %d\n", imgin.md->size[2]);
        inNdim = imgin.md->size[2];
        inNdim0 = imgin.md->size[2];
        inNdim1 = 1;
    }
    else
    {
        printf("inMdim   : %d\n", imgin.md->size[0]);
        inMdim = imgin.md->size[0];
        inMdim0 = imgin.md->size[1];
        inMdim1 = 1;

        printf("inNdim    : %d\n", imgin.md->size[1]);
        inNdim = imgin.md->size[1];
        inNdim0 = imgin.md->size[1];
        inNdim1 = 1;
    }


    // Orient matrix such that it is tall (M>N)

    enum matrix_shape {inMshape_tall, inMshape_wide} mshape;

    uint32_t Mdim = 0;
    uint32_t Mdim0 = 0;
    uint32_t Mdim1 = 0;

    uint32_t Ndim = 0;
    uint32_t Ndim0 = 0;
    uint32_t Ndim1 = 0;

    if( inNdim < inMdim )
    {
        // input matrix is tall
        // this is the default
        // notations follow this case
        //
        printf("CASE inNdim < inMdim (tall)\n");
        mshape = inMshape_tall;

        Mdim = inMdim;
        Mdim0 = inMdim0;
        Mdim1 = inMdim1;

        Ndim = inNdim;
        Ndim0 = inNdim0;
        Ndim1 = inNdim1;
    }
    else
    {
        printf("CASE inNdim > inMdim (wide)\n");
        mshape = inMshape_wide;

        Mdim = inNdim;
        Mdim0 = inNdim0;
        Mdim1 = inNdim1;

        Ndim = inMdim;
        Ndim0 = inMdim0;
        Ndim1 = inMdim1;
    }


    printf("inNdim               = %d  (%d x %d)\n", inNdim, inNdim0, inNdim1);
    printf("inMdim               = %d  (%d x %d)\n", inMdim, inMdim0, inMdim1);

    printf("  Ndim               = %d  (%d x %d)\n",   Ndim, Ndim0, Ndim1);
    printf("  Mdim               = %d  (%d x %d)\n",   Mdim, Mdim0, Mdim1);


    // from here on, Mdim > Ndim

    // create eigenvalues array
    if( imgeigenval.ID == -1)
    {
        imgeigenval.naxis   = 2;
        imgeigenval.size[0] = Ndim;
        imgeigenval.size[1] = 1;
        createimagefromIMGID(&imgeigenval);
    }


    // singular vectors array, small dimension
    IMGID imgmNsvec;


    {
        // create ATA
        // note that this is AAT if inNdim > inMdim (inMshape_wide)
        //
        IMGID imgATA = makeIMGID_2D("ATA", Ndim, Ndim);
        createimagefromIMGID(&imgATA);

        list_image_ID();

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

                cudaError_t cudaStat;
                cublasStatus_t stat;

                float *d_inmat;

                cudaStat = cudaMalloc((void **)&d_inmat, imgin.md->nelement * sizeof(float));
                if (cudaStat != cudaSuccess) {
                    printf ("device memory allocation failed");
                    return EXIT_FAILURE;
                }


                stat = cudaMemcpy(d_inmat, imgin.im->array.F, imgin.md->nelement * sizeof(float), cudaMemcpyHostToDevice);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cudaMemcpy failed\n");
                    return EXIT_FAILURE;
                }


                float *d_ATA;
                cudaStat = cudaMalloc((void **)&d_ATA, imgATA.md->nelement * sizeof(float));
                if (cudaStat != cudaSuccess) {
                    printf ("device memory allocation failed");
                    return EXIT_FAILURE;
                }


                // Create a handle for CUBLAS
                cublasHandle_t handle;
                stat = cublasCreate(&handle);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cublasCreate failed\n");
                    return EXIT_FAILURE;
                }


                // Do the actual multiplication
                cublasOperation_t OP0 = CUBLAS_OP_T;
                cublasOperation_t OP1 = CUBLAS_OP_N;
                if ( mshape == inMshape_wide )
                {
                    OP0 = CUBLAS_OP_N;
                    OP1 = CUBLAS_OP_T;
                }



                printf("imgin.md->nelement   = %ld\n", imgin.md->nelement);
                fflush(stdout);

                printf("imgATA.md->nelement  = %ld\n", imgATA.md->nelement);
                fflush(stdout);


                stat = cublasSgemm(handle, OP0, OP1,
                                   Ndim, Ndim, Mdim, alpha,
                                   d_inmat, inMdim,
                                   d_inmat, inMdim,
                                   beta, d_ATA, Ndim);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cublasSgemm failed\n");
                    return EXIT_FAILURE;
                }

                cublasDestroy(handle);

                stat = cudaMemcpy(imgATA.im->array.F, d_ATA, imgATA.md->nelement * sizeof(float), cudaMemcpyDeviceToHost);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cudaMemcpy failed\n");
                    return EXIT_FAILURE;
                }

                cudaFree(d_inmat);
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
                            Ndim, Ndim, Mdim, 1.0,
                            imgin.im->array.F, inMdim,
                            imgin.im->array.F, inMdim,
                            0.0, imgATA.im->array.F, Ndim);
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


        if(mshape == inMshape_tall)
        {
            imgmNsvec = makeIMGID_2D("svdV", Ndim, Ndim);

            /*
                        if( imgV.ID == -1)
                        {
                            imgeigenval.naxis   = 2;
                            imgeigenval.size[0] = Ndim;
                            imgeigenval.size[1] = 1;
                            createimagefromIMGID(&imgeigenval);
                        }*/
        }
        else
        {
            imgmNsvec = makeIMGID_3D("svdU", Ndim0, Ndim1, Ndim);
        }

        createimagefromIMGID(&imgmNsvec);


        memcpy(imgmNsvec.im->array.F, imgATA.im->array.F, sizeof(float)*Ndim*Ndim);
        LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'V', Ndim, d, e, imgmNsvec.im->array.F, Ndim);
        memcpy(imgeigenval.im->array.F, d, sizeof(float)*Ndim);


        free(d);
        free(e);
        free(t);

        delete_image(imgATA, DELETE_IMAGE_ERRMODE_EXIT);

        // This is matV if inMshape_tall, matU if inMshape_wide
        //save_fits("mV", "mV.fits");
    }




    if(compSVDmode & COMPSVD_SKIP_BIGMAT)
    {

        // create mU (if inMshape_tall)
        // create mV (if inMshape_wide)
        // (only non-zero part allocated)
        //

        IMGID imgmMsvec;
        if(mshape == inMshape_tall)
        {
            imgmMsvec = makeIMGID_3D("svdU", Mdim0, Mdim1, Ndim);
        }
        else
        {
            imgmMsvec = makeIMGID_2D("svdV", Mdim, Ndim);
        }

        createimagefromIMGID(&imgmMsvec);


        list_image_ID();



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

                cudaError_t cudaStat;
                cublasStatus_t stat;

                float *d_inmat;
                cudaStat = cudaMalloc((void **)&d_inmat, imgin.md->nelement * sizeof(float));
                if (cudaStat != cudaSuccess) {
                    printf ("device memory allocation failed");
                    return EXIT_FAILURE;
                }

                stat = cudaMemcpy(d_inmat, imgin.im->array.F, imgin.md->nelement * sizeof(float), cudaMemcpyHostToDevice);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cudaMemcpy failed\n");
                    return EXIT_FAILURE;
                }

                float *d_mNsvec;
                cudaStat = cudaMalloc((void **)&d_mNsvec, imgmNsvec.md->nelement * sizeof(float));
                if (cudaStat != cudaSuccess) {
                    printf ("device memory allocation failed");
                    return EXIT_FAILURE;
                }


                stat = cudaMemcpy(d_mNsvec, imgmNsvec.im->array.F, imgmNsvec.md->nelement * sizeof(float), cudaMemcpyHostToDevice);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cudaMemcpy failed\n");
                    return EXIT_FAILURE;
                }

                float *d_mMsvec;
                cudaStat = cudaMalloc((void **)&d_mMsvec, imgmMsvec.md->nelement * sizeof(float));
                if (cudaStat != cudaSuccess) {
                    printf ("device memory allocation failed");
                    return EXIT_FAILURE;
                }

                cublasHandle_t handle;
                stat = cublasCreate(&handle);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cublasCreate failed\n");
                    return EXIT_FAILURE;
                }

                cublasOperation_t OP0 = CUBLAS_OP_N;
                if ( mshape == inMshape_wide )
                {
                    OP0 = CUBLAS_OP_T;
                }


                stat = cublasSgemm(handle, OP0, CUBLAS_OP_N,
                                   Mdim, Ndim, Ndim, alpha,
                                   d_inmat, inMdim,
                                   d_mNsvec, Ndim,
                                   beta, d_mMsvec, Mdim);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cublasSgemm failed\n");
                    return EXIT_FAILURE;
                }


                stat = cublasDestroy(handle);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cublasCreate failed\n");
                    return EXIT_FAILURE;
                }

                stat = cudaMemcpy(imgmMsvec.im->array.F, d_mMsvec, imgmMsvec.md->nelement * sizeof(float), cudaMemcpyDeviceToHost);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("cudaMemcpy failed\n");
                    return EXIT_FAILURE;
                }


                cudaFree(d_inmat);
                cudaFree(d_mNsvec);
                cudaFree(d_mMsvec);

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
                             Mdim, Ndim, Ndim, 1.0,
                             imgin.im->array.F, inMdim,
                             imgmNsvec.im->array.F, Ndim,
                             0.0, imgmMsvec.im->array.F, Mdim);
            }
        }





        // normalize cols of imgmMsvec
        for(uint32_t jj=0; jj<Ndim; jj++)
        {
            for(uint32_t ii=0; ii< Mdim; ii++)
            {
                imgmMsvec.im->array.F[jj*Mdim+ii] /= sqrt(imgeigenval.im->array.F[jj]);
            }
        }
    }



    return RETURN_SUCCESS;
}








static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM = mkIMGID_from_name(inM);
    resolveIMGID(&imginM, ERRMODE_ABORT);



    IMGID imgU  = mkIMGID_from_name(outU);
    IMGID imgev = mkIMGID_from_name(outev);
    IMGID imgV  = mkIMGID_from_name(outV);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {


        compute_SVD(imginM, imgU, imgev, imgV, *GPUdevice, 0);


    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_linalgebra__compSVD()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

/**
 * @file SingularValueDecomp.c
 *
 */

#include <math.h>

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

static uint32_t *maxNBmode;
static long   fpi_maxNBmode;


static int32_t *GPUdevice;
static long     fpi_GPUdevice;

static uint64_t *compmode;
static long     fpi_compmode;



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
        CLIARG_UINT32,
        ".maxNBmode",
        "Maximum number of modes",
        "10000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &maxNBmode,
        &fpi_maxNBmode
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
    },
    {
        // optional computations
        CLIARG_UINT64,
        ".compmode",
        "flag: optional computations and checks",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &compmode,
        &fpi_compmode
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
    printf("CPU or GPU. Set .GPIdevice to -1 for CPU\n");

    printf("Optional computations and checks specified by bitmask flag .compmode :\n");
    printf("bit dec  description\n");
    printf(" 0    1  Skip big matrix (U or V) computation\n");
    printf(" 1    2  Compute pseudo-inverse, using svdlim for regularization\n");
    printf("         Inverse stored as image psinv\n");
    printf("         Only supported for tall input matrix\n");
    printf(" 2    4  Check pseudo-inverse: compute input x psinv product\n");
    printf("         result stored as image psinvcheck\n");
    printf("         Only supported for tall input matrix\n");
    printf(" 3    8  Reconstruct original image\n");
    printf("\n");
    printf("Example: compmode=6 will compute psinv and check it\n");
    printf("\n");
    printf("To run PCA on a sequence of images, input should be 3D cube of images\n");
    printf("datacube U contains principal components\n");
    printf("vextor outev are eigenvalues (magnitude) of each component\n");
    printf("datacube V is decoding matrix\n");

    return RETURN_SUCCESS;
}




static errno_t computeSGEMM(
    IMGID imginA,
    IMGID imginB,
    IMGID *outimg,
    int TranspA,
    int TranspB,
    int GPUdev
)
{

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



    return RETURN_SUCCESS;
}









/**
 * @brief Compute SVD of indimM x indimN matrix
 *
 * Decompose matrix imgin as:
 * imgU imgeigenval imgV^T
 *
 * Using column-major indexing
 *
 *
 *
 * compSVDmode flags:
 * COMPSVD_SKIP_BIGMAT  skip big (U of V) matrix computation
 */
errno_t compute_SVD(
    IMGID    imgin,
    IMGID    imgU,
    IMGID    imgeigenval,
    IMGID    imgV,
    float    SVlimit,
    uint32_t SVDmaxNBmode,
    int      GPUdev,
    uint64_t compSVDmode
)
{
    // input dimensions
    // input matrix is inMdim x inNdim, column-major
    //
    int inNdim, inNdim0, inNdim1;
    int inMdim, inMdim0, inMdim1;

    if(imgin.md->naxis == 3)
    {
        //printf("inMdim   : %d x %d\n", imgin.md->size[0], imgin.md->size[1]);
        inMdim = imgin.md->size[0] * imgin.md->size[1];
        inMdim0 = imgin.md->size[0];
        inMdim1 = imgin.md->size[1];

        //printf("inNdim    : %d\n", imgin.md->size[2]);
        inNdim = imgin.md->size[2];
        inNdim0 = imgin.md->size[2];
        inNdim1 = 1;
    }
    else
    {
        //printf("inMdim   : %d\n", imgin.md->size[0]);
        inMdim = imgin.md->size[0];
        inMdim0 = imgin.md->size[1];
        inMdim1 = 1;

        //printf("inNdim    : %d\n", imgin.md->size[1]);
        inNdim = imgin.md->size[1];
        inNdim0 = imgin.md->size[1];
        inNdim1 = 1;
    }


    // Orient matrix such that it is tall (M > N)
    //



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
        //printf("CASE inNdim < inMdim (tall)\n");
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
        //printf("CASE inNdim > inMdim (wide)\n");
        mshape = inMshape_wide;

        Mdim = inNdim;
        Mdim0 = inNdim0;
        Mdim1 = inNdim1;

        Ndim = inMdim;
        Ndim0 = inMdim0;
        Ndim1 = inMdim1;
    }


    //printf("inNdim               = %d  (%d x %d)\n", inNdim, inNdim0, inNdim1);
    //printf("inMdim               = %d  (%d x %d)\n", inMdim, inMdim0, inMdim1);

    //printf("  Ndim               = %d  (%d x %d)\n",   Ndim, Ndim0, Ndim1);
    //printf("  Mdim               = %d  (%d x %d)\n",   Mdim, Mdim0, Mdim1);


    // from here on, Mdim > Ndim



    // create eigenvalues array if needed
    if( imgeigenval.ID == -1)
    {
        imgeigenval.naxis   = 2;
        imgeigenval.size[0] = Ndim;
        imgeigenval.size[1] = 1;
        createimagefromIMGID(&imgeigenval);
    }











    IMGID imgATA;
    {
        // create ATA
        // note that this is AAT if inNdim > inMdim (inMshape_wide)
        //
        int TranspA = 1;
        int TranspB = 0;
        if ( mshape == inMshape_wide )
        {
            TranspA = 0;
            TranspB = 1;
        }
        strcpy(imgATA.name, "ATA");
        computeSGEMM(imgin, imgin, &imgATA, TranspA, TranspB, GPUdev);
    }




    // singular vectors array, small dimension
    // matrix (U or V) is square
    //
    IMGID *imgmNsvec;
    float evalmax;
    long NBmode = 0;
    {
        // eigendecomposition
        //
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
        //memcpy(imgmNsvec->im->array.F, imgATA.im->array.F, sizeof(float)*Ndim*Ndim);
        LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'V', Ndim, d, e, imgATA.im->array.F, Ndim);


        // How many modes to keep ?
        evalmax = d[Ndim-1];
        {
            long modecnt = 0;
            for(int k=0; k<Ndim; k++)
            {
                if( d[k] > SVlimit*SVlimit*evalmax )
                {
                    modecnt++;
                }
            }
            NBmode = modecnt;
            if ( modecnt > SVDmaxNBmode )
            {
                NBmode = SVDmaxNBmode;
            }
        }
        printf("KEEPING %ld MODES\n", NBmode);



        if(mshape == inMshape_tall)
        {
            imgmNsvec = &imgV;

            if( imgV.ID == -1)
            {
                imgV.naxis = 2;
                imgV.size[0] = inNdim;
                imgV.size[1] = NBmode; //inNdim;
                createimagefromIMGID(&imgV);
            }
        }
        else
        {
            imgmNsvec = &imgU;

            if( imgU.ID == -1)
            {
                imgU.naxis = imgin.md->naxis;
                if(imgin.md->naxis == 3)
                {
                    imgU.size[0] = inMdim0;
                    imgU.size[1] = inMdim1;
                    imgU.size[2] = NBmode; //inMdim;
                }
                else
                {
                    imgU.size[0] = inMdim;
                    imgU.size[1] = NBmode; //inMdim;
                }
                createimagefromIMGID(&imgU);
            }
        }


        // re-order from largest to smallest
        for(int k=0; k<NBmode; k++)
        {
            char * ptr0 = (char*) &imgATA.im->array.F[(Ndim-k-1)*Ndim];
            char * ptr1 = (char*) &imgmNsvec->im->array.F[k*Ndim];

            memcpy(ptr1, ptr0, sizeof(float)*Ndim);

            imgeigenval.im->array.F[k] = d[Ndim-k-1];
        }


        free(d);
        free(e);
        free(t);

        // imgmNsvec is matV if inMshape_tall, matU if inMshape_wide
    }
    delete_image(&imgATA, DELETE_IMAGE_ERRMODE_EXIT);










    if( !(compSVDmode & COMPSVD_SKIP_BIGMAT) )
    {
        // create mU (if inMshape_tall)
        // create mV (if inMshape_wide)
        // (only non-zero part allocated)
        //


        // Compute mU (only non-zero part allocated)
        //
        IMGID *imgmMsvec;
        {
            int TranspA = 0;
            int TranspB = 0;
            if ( mshape == inMshape_wide )
            {
                TranspA = 1;
            }

            if(mshape == inMshape_tall)
            {
                computeSGEMM(imgin, *imgmNsvec, &imgU, TranspA, TranspB, GPUdev);
                imgmMsvec = &imgU;
            }
            else
            {
                computeSGEMM(imgin, *imgmNsvec, &imgV, TranspA, TranspB, GPUdev);
                imgmMsvec = &imgV;
            }
        }




        // find largest eigenvalue
        //
        /*
        float evalmax = fabs(imgeigenval.im->array.F[0]);
        for(uint32_t jj=1; jj<Ndim; jj++)
        {
            float eval = fabs(imgeigenval.im->array.F[jj]);
            if(eval > evalmax)
            {
                evalmax = eval;
            }
        }*/


        // normalize cols of imgmMsvec
        // Report number of modes kept
        //
        long SVkeptcnt = 0;
        for(uint32_t jj=0; jj<NBmode; jj++)
        {

            float normfact = 0.0;
            float eval = fabs(imgeigenval.im->array.F[jj]);
            float evaln = eval / evalmax;
            if( evaln > SVlimit*SVlimit )
            {
                normfact = 1.0 / sqrt(eval);
                SVkeptcnt ++;
            }

            for(uint32_t ii=0; ii< Mdim; ii++)
            {
                imgmMsvec->im->array.F[jj*Mdim+ii] *= normfact;
            }
        }
        printf("LIMIT = %g  - Keeping %ld / %u modes\n", SVlimit, SVkeptcnt, Ndim);





        // Compute pseudo-inverse
        //
        if( (compSVDmode & COMPSVD_COMP_PSINV) )
        {
            // assumes tall matrix
            //

            IMGID imgmNsvec1 = mkIMGID_from_name("matNtemp");
            if( imgmNsvec1.ID == -1)
            {
                imgmNsvec1.naxis = 2;

                imgmNsvec1.size[0] = Ndim;
                imgmNsvec1.size[1] = NBmode;

                createimagefromIMGID(&imgmNsvec1);
            }


            // multiply by inverse of singular values
            //
            for(uint32_t jj=0; jj<NBmode; jj++)
            {
                float normfact = 0.0;
                float eval = fabs(imgeigenval.im->array.F[jj]);
                float evaln = eval / evalmax;
                if( evaln > SVlimit*SVlimit )
                {
                    normfact = 1.0 / sqrt(eval);

                }

                for(uint32_t ii=0; ii < Ndim; ii++)
                {
                    imgmNsvec1.im->array.F[jj*Ndim+ii] =
                        imgmNsvec->im->array.F[jj*Ndim+ii] * normfact;
                    // / sqrt(imgeigenval.im->array.F[jj]);
                }
            }



            IMGID imgpsinv;
            {
                int TranspA = 0;
                int TranspB = 1;
                strcpy(imgpsinv.name, "psinv");
                computeSGEMM(imgmNsvec1, *imgmMsvec, &imgpsinv, TranspA, TranspB, GPUdev);

                delete_image(&imgmNsvec1, DELETE_IMAGE_ERRMODE_EXIT);
            }




            // Check inverse
            //
            if( (compSVDmode & COMPSVD_COMP_CHECKPSINV) )
            {

                IMGID imgpsinvcheck = mkIMGID_from_name("psinvcheck");
                if(mshape == inMshape_tall)
                {
                    // inNdim < inMdim
                    computeSGEMM(imgpsinv, imgin, &imgpsinvcheck, 0, 0, GPUdev);
                }
            }
        }

    }



    // Compute un-normalized modes U
    // Singular Values included in modes U
    //
    if ( imgU.ID != -1 )
    {
        // un-normalized modes
        IMGID imgunmodes = mkIMGID_from_name("SVDunmodes");
        imgunmodes.naxis = imgU.md->naxis;
        imgunmodes.datatype = imgU.md->datatype;
        imgunmodes.size[0] = imgU.md->size[0];
        imgunmodes.size[1] = imgU.md->size[1];
        imgunmodes.size[2] = imgU.md->size[2];
        createimagefromIMGID(&imgunmodes);

        int lastaxis = imgunmodes.naxis-1;
        long framesize = imgunmodes.size[0];
        if(lastaxis==2)
        {
            framesize *= imgunmodes.size[1];
        }

        for(int kk=0; kk<imgunmodes.size[lastaxis]; kk++)
        {
            float mfact = sqrt(imgeigenval.im->array.F[kk]);
            for(long ii=0; ii<framesize; ii++)
            {
                imgunmodes.im->array.F[kk*framesize+ii] = imgU.im->array.F[kk*framesize+ii] * mfact;
            }
        }

        IMGID iminrec = mkIMGID_from_name("SVDinrec");
        computeSGEMM(imgunmodes, imgV, &iminrec, 0, 1, GPUdev);
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


        compute_SVD(imginM, imgU, imgev, imgV, *svdlim, *maxNBmode, *GPUdevice, *compmode);


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

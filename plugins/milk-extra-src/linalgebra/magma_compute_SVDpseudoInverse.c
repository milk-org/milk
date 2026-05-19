/** @file magma_compute_SVDpseudoInverse.c
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <device_types.h>
#include <pthread.h>

#ifdef HAVE_MAGMA

#include "magma_lapack.h"
#include "magma_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "timeutils.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "linalgebra_types.h"

extern int INIT_MAGMA;

// queue for default magma device
extern magma_queue_t magmaqueue;

static long MAGMAloop_iter = 0;


static magma_int_t  magma_aux_iwork[1];
static magma_int_t  magma_lwork, magma_liwork;
static magma_int_t *magma_iwork;


/**
 * @brief Internal context for SVD pseudo-inverse
 *
 * Bundles matrix dimensions, device/host buffer
 * pointers, precision flag, and timing data so
 * that helper functions share state without long
 * parameter lists.
 */
typedef struct
{
    magma_int_t M;     /**< rows (sensors)    */
    magma_int_t N;     /**< cols (actuators)  */
    magma_int_t info;  /**< MAGMA return code */

    int MAGMAfloat;    /**< 1=float, 0=double */
    int testmode;
    int LOOPmode;
    int verbose;
    int magmaXmode;
    int dAinvMODE;
    magma_int_t mout;

    imageID  ID_Rmatrix;
    imageID  ID_Cmatrix;
    uint8_t  datatype;
    double   SVDeps;
    long     MaxNBmodes;
    long     MaxNBmodes1;

    magma_device_t *devicearray;

    /* Double-precision host/device buffers */
    double *h_A,    *d_A;
    double *h_AtA,  *d_AtA;
    double *h_VT1,  *d_VT1;
    double *d_M2;
    double *d_Ainv, *h_Ainv;
    double *w1;
    double *h_R,    *h_work;

    /* Single-precision host/device buffers */
    float *fh_A,    *fd_A;
    float *fh_AtA,  *fd_AtA;
    float *fh_VT1,  *fd_VT1;
    float *fd_M2;
    float *fd_Ainv, *fh_Ainv;
    float *fw1;
    float *fh_R,    *fh_work;

    /* Timing checkpoints */
    struct timespec t[14];

} svdpinv_ctx;


// ==========================================
// Forward declaration(s)
// ==========================================

errno_t LINALGEBRA_magma_compute_SVDpseudoInverse(
    const char *ID_Rmatrix_name,
    const char *ID_Cmatrix_name,
    double SVDeps, long MaxNBmodes,
    const char *ID_VTmatrix_name,
    int LOOPmode, int testmode,
    int precision, int GPUdevice,
    imageID *outID);

// ==========================================
// Gen 4 V2 CLI command: linalgebrapsinv
// ==========================================

static char pi_r[FUNCTION_PARAMETER_STRMAXLEN]
    = "matA";
static char pi_c[FUNCTION_PARAMETER_STRMAXLEN]
    = "matAinv";
static double pi_eps = 0.01;
static int64_t pi_nm = 100;
static char pi_vt[FUNCTION_PARAMETER_STRMAXLEN]
    = "VTmat";
static FPS_APP_INFO FPS_app_info_pi = {
    .fps_name = "linalgebrapsinv",
    .cmdkey   = "linalgebrapsinv",
    .description =
        "compute pseudo inverse",
    .description_long =
        "Compute the Moore-Penrose pseudo-inverse of a matrix via SVD using the MAGMA GPU library."
};
#define FPS_PARAMS_PI(X) \
    X(".inmat", pi_r, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input mat") \
    X(".outmat", pi_c, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".svdeps", &pi_eps, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "SVD eps") \
    X(".nbmodes", &pi_nm, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "max modes") \
    X(".vtmat", pi_vt, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "VT matrix")
#include "fps.h"
static FPS_CLI_BINDING pi_b[] = {
    FPS_PARAMS_PI(FPS_X_BINDING) };
static const int pi_nb =
    sizeof(pi_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS_PI(FPS_X_FARG) };
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS pi_cms = {0};
static __attribute__((constructor))
void init_pi(void) {
    strncpy(CLIcmddata.key,
        FPS_app_info_pi.cmdkey,
        sizeof(CLIcmddata.key)-1);
    strncpy(CLIcmddata.description,
        FPS_app_info_pi.description,
        sizeof(CLIcmddata.description)-1);
    CLIcmddata.nbarg =
        sizeof(farg)/sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags = CLICMDFLAG_FPS;
    if(!CLIcmddata.cmdsettings)
        CLIcmddata.cmdsettings = &pi_cms;
}
static errno_t pi_compute(void) {
    LINALGEBRA_magma_compute_SVDpseudoInverse(
        pi_r, pi_c, pi_eps,
        (long)pi_nm, pi_vt,
        0, 0, 64, 0, NULL);
    return RETURN_SUCCESS;
}
static errno_t CLIfunction(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_pi, farg,
        &CLIcmddata,
        pi_b, pi_nb, pi_compute);
}

errno_t magma_compute_SVDpseudoInverse_addCLIcmd()
{
    safe_fps_fill_farg_examples(
        farg, pi_b, pi_nb);
    INSERT_STD_CLIREGISTERFUNC;

    return RETURN_SUCCESS;
}


// =========================================================
//  Static helpers for SVD pseudo-inverse computation
// =========================================================


/**
 * @brief Read input matrix dimensions
 *
 * Resolves the input image and extracts M (rows)
 * and N (columns) from its axis sizes.
 */
static void svdpinv_read_dims(
    svdpinv_ctx *ctx,
    const char  *name)
{
    ctx->ID_Rmatrix = image_ID(
        name, dcimg, dcnimg);
    ctx->datatype =
        dcimg[ctx->ID_Rmatrix].md[0].datatype;

    if (dcimg[ctx->ID_Rmatrix].md[0].naxis == 3)
    {
        ctx->M =
            dcimg[ctx->ID_Rmatrix].md[0].size[0]
            * dcimg[ctx->ID_Rmatrix]
                  .md[0]
                  .size[1];
        ctx->N =
            dcimg[ctx->ID_Rmatrix].md[0].size[2];

        if (ctx->verbose == 1)
        {
            printf(
                "3D image -> %ld %ld\n",
                (long) ctx->M,
                (long) ctx->N);
            fflush(stdout);
        }
    }
    else
    {
        ctx->M =
            dcimg[ctx->ID_Rmatrix].md[0].size[0];
        ctx->N =
            dcimg[ctx->ID_Rmatrix].md[0].size[1];

        if (ctx->verbose == 1)
        {
            printf(
                "2D image -> %ld %ld\n",
                (long) ctx->M,
                (long) ctx->N);
            fflush(stdout);
        }
    }

    //TEST
    //for(ii=0;ii<N;ii++)
    //dcimg[ID_Rmatrix].array.F[ii*M+ii] += 1.0f;

    if (ctx->verbose == 1)
    {
        printf(
            "magma :    M = %ld , N = %ld\n",
            (long) ctx->M, (long) ctx->N);
        fflush(stdout);
    }
}


/**
 * @brief Initialize MAGMA and select GPU device
 *
 * Initializes the MAGMA library on first use,
 * selects the requested GPU, and creates the
 * MAGMA queue on the first iteration.
 */
static void svdpinv_init_device(
    svdpinv_ctx *ctx,
    int          GPUdevice)
{
    if (INIT_MAGMA == 0)
    {
        if (ctx->verbose == 1)
        {
            printf("INITIALIZE MAGMA\n");
            fflush(stdout);
        }
        magma_init();
        magma_print_environment();

        INIT_MAGMA = 1;
    }

    printf("Selecting device\n");
    fflush(stdout);

    magma_int_t num_dev;
    ctx->devicearray =
        (magma_device_t *) malloc(
            sizeof(magma_device_t) * 10);
    magma_getdevices(
        ctx->devicearray, 10, &num_dev);
    printf("%d devices detected\n", num_dev);

    printf("Selecting device %d\n", GPUdevice);
    magma_setdevice(
        ctx->devicearray[GPUdevice]);
    fflush(stdout);

    if (MAGMAloop_iter == 0)
    {
        magma_queue_create(
            ctx->devicearray[GPUdevice],
            &magmaqueue);
    }
}


/**
 * @brief Allocate device and host buffers
 *
 * Allocates all MAGMA buffers on the first
 * iteration.  On subsequent iterations in
 * LOOPmode, buffers are reused.
 */
static void svdpinv_alloc(svdpinv_ctx *ctx)
{
    if (MAGMAloop_iter != 0)
    {
        return;
    }

    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    if (ctx->MAGMAfloat == 0)
    {
        printf(
            "MAGMA allocating double"
            " %d x %d = %ld byte\n",
            (int) M,
            (int) N,
            sizeof(double) * M * N);

        if (MAGMA_SUCCESS !=
            magma_dmalloc(&ctx->d_A, M * N))
        {
            fprintf(
                stderr,
                "!!!! magma_malloc failed\n");
            magma_finalize();
            exit(-1);
        }
        TESTING_DMALLOC_CPU(ctx->h_A, M * N);

        TESTING_DMALLOC_CPU(
            ctx->h_AtA, N * N);
        TESTING_DMALLOC_DEV(
            ctx->d_AtA, N * N);

        TESTING_DMALLOC_CPU(
            ctx->h_VT1, N * N);
        TESTING_DMALLOC_DEV(
            ctx->d_VT1, N * N);
        TESTING_DMALLOC_DEV(
            ctx->d_M2, N * N);

        TESTING_DMALLOC_CPU(
            ctx->h_Ainv, N * M);
    }
    else
    {
        TESTING_SMALLOC_CPU(
            ctx->fh_A, M * N);
        printf(
            "Allocating magmaf_d_A"
            " on device ...\n");
        fflush(stdout);
        TESTING_SMALLOC_DEV(
            ctx->fd_A, M * N);
        printf(" ... done\n");
        fflush(stdout);

        TESTING_SMALLOC_CPU(
            ctx->fh_AtA, N * N);
        TESTING_SMALLOC_DEV(
            ctx->fd_AtA, N * N);

        TESTING_SMALLOC_CPU(
            ctx->fh_VT1, N * N);
        TESTING_SMALLOC_DEV(
            ctx->fd_VT1, N * N);
        TESTING_SMALLOC_DEV(
            ctx->fd_M2, N * N);

        TESTING_SMALLOC_CPU(
            ctx->fh_Ainv, N * M);
    }
}


/**
 * @brief STEP 1: Load input matrix to device
 *
 * Copies the input matrix from the image array
 * to the MAGMA host/device buffers.  In test
 * mode, also saves the input to a FITS file.
 */
static void svdpinv_load_input(
    svdpinv_ctx *ctx)
{
    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    if (ctx->datatype == _DATATYPE_FLOAT)
    {
        if (ctx->MAGMAfloat == 1)
        {
            if (ctx->testmode == 1)
            {
                memcpy(
                    ctx->fh_A,
                    dcimg[ctx->ID_Rmatrix]
                        .array.F,
                    sizeof(float) * M * N);
                magma_ssetmatrix(
                    M, N,
                    ctx->fh_A, M,
                    ctx->fd_A, M,
                    magmaqueue);
            }
            else
            {
                magma_ssetmatrix(
                    M, N,
                    dcimg[ctx->ID_Rmatrix]
                        .array.F,
                    M,
                    ctx->fd_A, M,
                    magmaqueue);
            }
        }
        else
        {
            for (long ii = 0; ii < M * N; ii++)
            {
                ctx->h_A[ii] =
                    dcimg[ctx->ID_Rmatrix]
                        .array.F[ii];
            }
            magma_dsetmatrix(
                M, N,
                ctx->h_A, M,
                ctx->d_A, M,
                magmaqueue);
        }
    }
    else
    {
        if (ctx->MAGMAfloat == 1)
        {
            for (long ii = 0; ii < M * N; ii++)
            {
                ctx->fh_A[ii] =
                    dcimg[ctx->ID_Rmatrix]
                        .array.D[ii];
            }
            magma_ssetmatrix(
                M, N,
                ctx->fh_A, M,
                ctx->fd_A, M,
                magmaqueue);
        }
        else
        {
            if (ctx->testmode == 1)
            {
                memcpy(
                    ctx->h_A,
                    dcimg[ctx->ID_Rmatrix]
                        .array.D,
                    sizeof(double) * M * N);
                magma_dsetmatrix(
                    M, N,
                    ctx->h_A, M,
                    ctx->d_A, M,
                    magmaqueue);
            }
            else
            {
                magma_dsetmatrix(
                    M, N,
                    dcimg[ctx->ID_Rmatrix]
                        .array.D,
                    M,
                    ctx->d_A, M,
                    magmaqueue);
            }
        }
    }

    if (ctx->testmode == 1)
    {
        imageID ID_A;

        FUNC_CHECK_RETURN(
            create_2Dimage_ID(
                "mA", M, N, &ID_A));

        if (ctx->MAGMAfloat == 1)
        {
            for (long ii = 0; ii < M * N; ii++)
            {
                dcimg[ID_A].array.F[ii] =
                    ctx->fh_A[ii];
            }
        }
        else
        {
            for (long ii = 0; ii < M * N; ii++)
            {
                dcimg[ID_A].array.F[ii] =
                    ctx->h_A[ii];
            }
        }

        FUNC_CHECK_RETURN(
            save_fits(
                "mA", "test_mA.QDWH.fits"));

        FUNC_CHECK_RETURN(
            delete_image_ID(
                "mA",
                DELETE_IMAGE_ERRMODE_WARNING));
    }
}


/**
 * @brief STEP 3: Compute A^T A on GPU
 *
 * Computes the symmetric product trans(A) x A
 * using a single MAGMA syrk/gemm call.  In test
 * mode, copies the result to host and saves it.
 */
static void svdpinv_compute_AtA(
    svdpinv_ctx *ctx)
{
    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    if (ctx->MAGMAfloat == 1)
    {
        magma_ssyrk(
            MagmaLower,
            MagmaTrans,
            N, M,
            1.0, ctx->fd_A, M,
            0.0, ctx->fd_AtA, N,
            magmaqueue);
        magmablas_ssymmetrize(
            MagmaLower, N,
            ctx->fd_AtA, N,
            magmaqueue);

        // Slower alternative
        //magma_sgemm(MagmaTrans,
        //    MagmaNoTrans, N, N, M,
        //    1.0, ctx->fd_A, M,
        //    ctx->fd_A, M,
        //    0.0, ctx->fd_AtA, N,
        //    magmaqueue);
    }
    else
    {
        magma_dgemm(
            MagmaTrans,
            MagmaNoTrans,
            N, N, M,
            1.0, ctx->d_A, M,
            ctx->d_A, M,
            0.0, ctx->d_AtA, N,
            magmaqueue);
    }

    if (ctx->testmode == 1)
    {
        if (ctx->MAGMAfloat == 1)
        {
            magma_sgetmatrix(
                N, N,
                ctx->fd_AtA, N,
                ctx->fh_AtA, N,
                magmaqueue);
        }
        else
        {
            magma_dgetmatrix(
                N, N,
                ctx->d_AtA, N,
                ctx->h_AtA, N,
                magmaqueue);
        }

        imageID ID_AtA;
        FUNC_CHECK_RETURN(
            create_2Dimage_ID(
                "mAtA", N, N, &ID_AtA));

        if (ctx->MAGMAfloat == 1)
        {
            for (long ii = 0; ii < N * N; ii++)
            {
                dcimg[ID_AtA].array.F[ii] =
                    ctx->fh_AtA[ii];
            }
        }
        else
        {
            for (long ii = 0; ii < N * N; ii++)
            {
                dcimg[ID_AtA].array.F[ii] =
                    ctx->h_AtA[ii];
            }
        }
        FUNC_CHECK_RETURN(
            save_fits("mAtA", "test_mAtA.fits"));
        FUNC_CHECK_RETURN(
            delete_image_ID(
                "mAtA",
                DELETE_IMAGE_ERRMODE_IGNORE));
    }
}


/**
 * @brief STEP 4: Eigenvalue decomposition
 *
 * Queries workspace size on first iteration,
 * allocates workspace buffers, then runs the
 * symmetric eigendecomposition on the GPU.
 * Frees workspace when LOOPmode is 0.
 *
 * Sets ctx->t[4] internally (between workspace
 * setup and the actual decomposition call).
 */
static void svdpinv_eigendecomp(
    svdpinv_ctx *ctx)
{
    magma_int_t N = ctx->N;

    /* --- workspace query (first iter) --- */
    if (MAGMAloop_iter == 0)
    {
        if (ctx->MAGMAfloat == 1)
        {
            float auxf_work[1];

            if (ctx->magmaXmode == 1)
            {
                magma_ssyevdx_gpu(
                    MagmaVec,
                    MagmaRangeI,
                    MagmaLower,
                    N, NULL, N,
                    0.0, 1.0,
                    N - ctx->MaxNBmodes, N,
                    NULL, NULL,
                    NULL, N,
                    auxf_work, -1,
                    magma_aux_iwork, -1,
                    &ctx->info);
            }
            else
            {
                magma_ssyevd_gpu(
                    MagmaVec,
                    MagmaLower,
                    N, NULL, N,
                    NULL, NULL, N,
                    auxf_work, -1,
                    magma_aux_iwork, -1,
                    &ctx->info);
            }
            // -> change to 2-stage magma SVD
            // evd -> evr
            // PALSMA

            // alt -> LQ reduction -> SVD
            // magma_dgsvd (more stable)

            magma_lwork =
                (magma_int_t)
                MAGMA_S_REAL(auxf_work[0]);
        }
        else
        {
            double aux_work[1];

            magma_dsyevd_gpu(
                MagmaVec,
                MagmaLower,
                N, NULL, N,
                NULL, NULL, N,
                aux_work, -1,
                magma_aux_iwork, -1,
                &ctx->info);
            magma_lwork =
                (magma_int_t)
                MAGMA_S_REAL(aux_work[0]);
        }

        magma_liwork = magma_aux_iwork[0];
    }

    /* --- allocate workspace (first iter) --- */
    if (MAGMAloop_iter == 0)
    {
        if (ctx->MAGMAfloat == 1)
        {
            TESTING_MALLOC_CPU(
                magma_iwork,
                magma_int_t,
                magma_liwork);
            TESTING_MALLOC_PIN(
                ctx->fh_work,
                float,
                magma_lwork);
            TESTING_MALLOC_CPU(
                ctx->fw1, float, N);
            TESTING_MALLOC_PIN(
                ctx->fh_R, float, N * N);
        }
        else
        {
            TESTING_MALLOC_CPU(
                magma_iwork,
                magma_int_t,
                magma_liwork);
            TESTING_MALLOC_PIN(
                ctx->h_work,
                double,
                magma_lwork);
            TESTING_MALLOC_CPU(
                ctx->w1, double, N);
            TESTING_MALLOC_PIN(
                ctx->h_R, double, N * N);
        }
    }

    /* timing: after workspace setup */
    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx->t[4]);

    /* --- eigendecomposition call --- */
    if (ctx->MAGMAfloat == 1)
    {
        if (ctx->magmaXmode == 1)
        {
            magma_ssyevdx_gpu(
                MagmaVec,
                MagmaRangeI,
                MagmaLower,
                N,
                ctx->fd_AtA, N,
                0.0, 1.0,
                N - ctx->MaxNBmodes, N,
                &ctx->mout,
                ctx->fw1,
                ctx->fh_R, N,
                ctx->fh_work, magma_lwork,
                magma_iwork, magma_liwork,
                &ctx->info);
        }
        else
        {
            magma_ssyevd_gpu(
                MagmaVec,
                MagmaLower,
                N,
                ctx->fd_AtA, N,
                ctx->fw1,
                ctx->fh_R, N,
                ctx->fh_work, magma_lwork,
                magma_iwork, magma_liwork,
                &ctx->info);
        }
    }
    else
    {
        // CODE CAN HANG HERE - THIS HAPPENS
        // ONCE OUT OF multiple 1000s EXECUTIONS
        // WHEN RUNNING IN A LOOP..
        // SEEMS TO BE A MAGMA ISSUE

        magma_dsyevd_gpu(
            MagmaVec,
            MagmaLower,
            N,
            ctx->d_AtA, N,
            ctx->w1,
            ctx->h_R, N,
            ctx->h_work, magma_lwork,
            magma_iwork, magma_liwork,
            &ctx->info);

        if (ctx->verbose == 1)
        {
            printf(" DONE\n");
            fflush(stdout);
        }
    }

    /* --- free workspace if one-shot --- */
    if (ctx->LOOPmode == 0)
    {
        TESTING_FREE_CPU(magma_iwork);

        if (ctx->MAGMAfloat == 1)
        {
            TESTING_FREE_PIN(ctx->fh_R);
            TESTING_FREE_PIN(ctx->fh_work);
        }
        else
        {
            TESTING_FREE_PIN(ctx->h_R);
            TESTING_FREE_PIN(ctx->h_work);
        }
    }
}


/**
 * @brief STEP 5: Select eigenvalue modes
 *
 * Applies the SVDeps^2 threshold to keep only
 * eigenvalues above the limit.  Stores the
 * resulting mode count in ctx->MaxNBmodes1.
 * Optionally dumps eigenvalues and mode count
 * to files in test mode.
 */
static void svdpinv_select_modes(
    svdpinv_ctx *ctx)
{
    magma_int_t N = ctx->N;

    /* Dump eigenvalues in test mode */
    if (ctx->testmode == 1)
    {
        char fname[STRINGMAXLEN_FILENAME];
        WRITE_FILENAME(fname, "eigenv.dat");
        FILE *fp;

        if ((fp = fopen(fname, "w")) == NULL)
        {
            printf(
                "ERROR: cannot create"
                " file \"%s\"\n",
                fname);
            abort();
        }
        if (ctx->MAGMAfloat == 1)
        {
            for (long k = 0; k < N; k++)
            {
                fprintf(fp,
                    "%5ld %20.8g"
                    "  %20.8f  %g\n",
                    k,
                    ctx->fw1[N - k - 1],
                    ctx->fw1[N - k - 1]
                        / ctx->fw1[N - 1],
                    ctx->SVDeps
                        * ctx->SVDeps);
            }
        }
        else
        {
            for (long k = 0; k < N; k++)
            {
                fprintf(fp,
                    "%5ld %20.8g"
                    "  %20.8f  %g\n",
                    k,
                    ctx->w1[N - k - 1],
                    ctx->w1[N - k - 1]
                        / ctx->w1[N - 1],
                    ctx->SVDeps
                        * ctx->SVDeps);
            }
        }
        fclose(fp);
    }

    /// w1 values are EIGENVALUES of AT A
    /// Note: w1 values are the SQUARE of the
    /// singular values of A

    /* eigenvalue threshold */
    DEBUG_TRACEPOINT("Set eigenvalue limit");
    double egvlim;
    if (ctx->MAGMAfloat == 1)
    {
        egvlim = ctx->SVDeps * ctx->SVDeps
                 * ctx->fw1[N - 1];
    }
    else
    {
        egvlim = ctx->SVDeps * ctx->SVDeps
                 * ctx->w1[N - 1];
    }

    long MaxNBmodes1 = ctx->MaxNBmodes;
    if (MaxNBmodes1 > N)
    {
        MaxNBmodes1 = N;
    }
    if (MaxNBmodes1 > ctx->M)
    {
        MaxNBmodes1 = ctx->M;
    }

    long mode = 0;
    if (ctx->MAGMAfloat == 1)
    {
        while ((mode < MaxNBmodes1)
               && (ctx->fw1[N - mode - 1]
                   > egvlim))
        {
            mode++;
        }
    }
    else
    {
        while ((mode < MaxNBmodes1)
               && (ctx->w1[N - mode - 1]
                   > egvlim))
        {
            mode++;
        }
    }

    if (ctx->verbose == 1)
    {
        printf(
            "Keeping %ld modes"
            "  (SVDeps = %g -> %g,"
            " MaxNBmodes = %ld -> %ld)\n",
            mode,
            ctx->SVDeps,
            egvlim,
            ctx->MaxNBmodes,
            MaxNBmodes1);
        fflush(stdout);
    }

    if (ctx->testmode == 1)
    {
        FILE *fp =
            fopen("test_SVDmodes.log", "w");
        fprintf(fp,
                "%6ld %6ld\n",
                mode, MaxNBmodes1);
        fclose(fp);
    }

    ctx->MaxNBmodes1 = mode;
    printf(
        "Keeping %ld modes  (SVDeps = %g)\n",
        ctx->MaxNBmodes1, ctx->SVDeps);
}


/**
 * @brief STEP 6+7: Build VT and weighted VT1
 *
 * Copies eigenvectors from device to host, writes
 * them to the VT output image (STEP 6), then
 * builds the weighted VT1 matrix dividing each
 * eigenvector by its eigenvalue (STEP 7).
 */
static void svdpinv_build_VT(
    svdpinv_ctx    *ctx,
    const char     *ID_VTmatrix_name)
{
    magma_int_t N = ctx->N;

    /* --- STEP 6: copy eigenvectors --- */
    DEBUG_TRACEPOINT("Write eigenvectors");

    if (ctx->MAGMAfloat == 1)
    {
        magma_sgetmatrix(
            N, N,
            ctx->fd_AtA, N,
            ctx->fh_AtA, N,
            magmaqueue);
    }
    else
    {
        magma_dgetmatrix(
            N, N,
            ctx->d_AtA, N,
            ctx->h_AtA, N,
            magmaqueue);
    }

    /* write eigenvectors to VT image */
    {
        imageID ID_VT;
        FUNC_CHECK_RETURN(
            create_2Dimage_ID(
                ID_VTmatrix_name,
                N, N, &ID_VT));

        if (ctx->MAGMAfloat == 1)
        {
            for (long ii = 0; ii < N; ii++)
                for (long jj = 0; jj < N; jj++)
                {
                    dcimg[ID_VT]
                        .array.F[jj * N + ii] =
                        ctx->fh_AtA[
                            (N - ii - 1) * N
                            + jj];
                }
        }
        else
        {
            for (long ii = 0; ii < N; ii++)
                for (long jj = 0; jj < N; jj++)
                {
                    dcimg[ID_VT]
                        .array.F[jj * N + ii] =
                        ctx->h_AtA[
                            (N - ii - 1) * N
                            + jj];
                }
        }
    }

    if (ctx->testmode == 1)
    {
        FUNC_CHECK_RETURN(
            save_fits(
                ID_VTmatrix_name,
                "test_mVT.fits"));
    }

    /* --- STEP 7: weighted VT1 --- */
    DEBUG_TRACEPOINT(
        "Write eigenvectors/eigenvalue"
        " to magma_h_VT1 if eigenvalue"
        " > limit");

    if (ctx->MAGMAfloat == 1)
    {
        for (long ii = 0; ii < N; ii++)
            for (long jj = 0; jj < N; jj++)
            {
                if (N - jj - 1
                    < ctx->MaxNBmodes1)
                {
                    ctx->fh_VT1[ii * N + jj] =
                        ctx->fh_AtA[jj * N + ii]
                        / ctx->fw1[jj];
                }
                else
                {
                    ctx->fh_VT1[ii * N + jj] =
                        0.0;
                }
            }
        magma_ssetmatrix(
            N, N,
            ctx->fh_VT1, N,
            ctx->fd_VT1, N,
            magmaqueue);
    }
    else
    {
        for (long ii = 0; ii < N; ii++)
            for (long jj = 0; jj < N; jj++)
            {
                if (N - jj - 1
                    < ctx->MaxNBmodes1)
                {
                    ctx->h_VT1[ii * N + jj] =
                        ctx->h_AtA[jj * N + ii]
                        / ctx->w1[jj];
                }
                else
                {
                    ctx->h_VT1[ii * N + jj] =
                        0.0;
                }
            }
        magma_dsetmatrix(
            N, N,
            ctx->h_VT1, N,
            ctx->d_VT1, N,
            magmaqueue);
    }

    if (ctx->LOOPmode == 0)
    {
        if (ctx->MAGMAfloat == 1)
        {
            TESTING_FREE_CPU(ctx->fh_VT1);
            TESTING_FREE_CPU(ctx->fw1);
        }
        else
        {
            TESTING_FREE_CPU(ctx->h_VT1);
            TESTING_FREE_CPU(ctx->w1);
        }
    }
}


/**
 * @brief STEP 8: Compute M2 = VT1^T * AtA^T
 *
 * M2 is (AT A)^-1.  In test mode, copies M2 to
 * host and saves to FITS.
 */
static void svdpinv_compute_M2(
    svdpinv_ctx *ctx)
{
    magma_int_t N = ctx->N;

    DEBUG_TRACEPOINT(
        "Compute M2 = VT1 VT = (AT A)^-1");

    if (ctx->MAGMAfloat == 1)
    {
        magma_sgemm(
            MagmaTrans, MagmaTrans,
            N, N, N,
            1.0, ctx->fd_VT1, N,
            ctx->fd_AtA, N,
            0.0, ctx->fd_M2, N,
            magmaqueue);
    }
    else
    {
        magma_dgemm(
            MagmaTrans, MagmaTrans,
            N, N, N,
            1.0, ctx->d_VT1, N,
            ctx->d_AtA, N,
            0.0, ctx->d_M2, N,
            magmaqueue);

        if (ctx->verbose == 1)
        {
            printf("-> DONE\n");
            fflush(stdout);
        }
    }

    if (ctx->testmode == 1)
    {
        imageID ID_M2;

        FUNC_CHECK_RETURN(
            create_2Dimage_ID(
                "mM2", N, N, &ID_M2));

        DEBUG_TRACEPOINT("Computing mM2");

        if (ctx->MAGMAfloat == 1)
        {
            float *fh_M2;
            TESTING_SMALLOC_CPU(
                fh_M2, N * N);
            magma_sgetmatrix(
                N, N,
                ctx->fd_M2, N,
                fh_M2, N,
                magmaqueue);
            for (long ii = 0; ii < N; ii++)
                for (long jj = 0; jj < N; jj++)
                {
                    dcimg[ID_M2]
                        .array
                        .F[jj * N + ii] =
                        fh_M2[jj * N + ii];
                }
            TESTING_FREE_CPU(fh_M2);
        }
        else
        {
            double *h_M2;
            TESTING_DMALLOC_CPU(
                h_M2, N * N);
            magma_dgetmatrix(
                N, N,
                ctx->d_M2, N,
                h_M2, N,
                magmaqueue);
            for (long ii = 0; ii < N; ii++)
                for (long jj = 0; jj < N; jj++)
                {
                    dcimg[ID_M2]
                        .array
                        .F[jj * N + ii] =
                        h_M2[jj * N + ii];
                }
            TESTING_FREE_CPU(h_M2);
        }
        DEBUG_TRACEPOINT("Saving mM2");
        FUNC_CHECK_RETURN(
            save_fits("mM2", "test_mM2.fits"));
        FUNC_CHECK_RETURN(
            delete_image_ID(
                "mM2",
                DELETE_IMAGE_ERRMODE_WARNING));
    }

    if (ctx->LOOPmode == 0)
    {
        if (ctx->MAGMAfloat == 1)
        {
            TESTING_FREE_DEV(ctx->fd_VT1);
        }
        else
        {
            TESTING_FREE_DEV(ctx->d_VT1);
        }
    }
}


/**
 * @brief STEP 9: Compute Ainv = M2 * A
 *
 * Performs the final matrix-matrix multiply to
 * produce the pseudo-inverse on the GPU.
 */
static void svdpinv_compute_Ainv(
    svdpinv_ctx *ctx)
{
    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    DEBUG_TRACEPOINT(
        "Compute Ainv = M2 A = (AT A)^-1 A");

    if (MAGMAloop_iter == 0)
    {
        ctx->dAinvMODE = 1;
        if (ctx->MAGMAfloat == 1)
        {
            TESTING_SMALLOC_DEV(
                ctx->fd_Ainv, N * M);
        }
        else
        {
            TESTING_DMALLOC_DEV(
                ctx->d_Ainv, N * M);
        }
    }

    if (ctx->MAGMAfloat == 1)
    {
        magma_sgemm(
            MagmaNoTrans, MagmaNoTrans,
            M, N, N,
            1.0, ctx->fd_A, M,
            ctx->fd_M2, N,
            0.0, ctx->fd_Ainv, M,
            magmaqueue);
    }
    else
    {
        DEBUG_TRACEPOINT(
            "double precision"
            " running magma_dgemm");
        magma_dgemm(
            MagmaNoTrans, MagmaNoTrans,
            M, N, N,
            1.0, ctx->d_A, M,
            ctx->d_M2, N,
            0.0, ctx->d_Ainv, M,
            magmaqueue);
        DEBUG_TRACEPOINT(
            "double precision"
            " magma_dgemm done");
    }

    DEBUG_TRACEPOINT("free");
    if (ctx->LOOPmode == 0)
    {
        if (ctx->MAGMAfloat == 1)
        {
            TESTING_FREE_DEV(ctx->fd_M2);
        }
        else
        {
            TESTING_FREE_DEV(ctx->d_M2);
        }
    }
}


/**
 * @brief Retrieve Ainv from GPU and test-save
 *
 * Copies the pseudo-inverse from device to host.
 * In test mode, also writes it to a FITS file.
 */
static void svdpinv_retrieve_Ainv(
    svdpinv_ctx *ctx)
{
    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    DEBUG_TRACEPOINT("set result");
    if (ctx->MAGMAfloat == 1)
    {
        magma_sgetmatrix(
            M, N,
            ctx->fd_Ainv, M,
            ctx->fh_Ainv, M,
            magmaqueue);
    }
    else
    {
        magma_dgetmatrix(
            M, N,
            ctx->d_Ainv, M,
            ctx->h_Ainv, M,
            magmaqueue);
    }
    DEBUG_TRACEPOINT(
        "end of magma computation");

    if (ctx->testmode == 1)
    {
        imageID ID_Ainv;
        FUNC_CHECK_RETURN(
            create_2Dimage_ID(
                "mAinv", M, N, &ID_Ainv));

        if (ctx->MAGMAfloat == 1)
        {
            for (long ii = 0; ii < M; ii++)
                for (long jj = 0; jj < N; jj++)
                {
                    dcimg[ID_Ainv]
                        .array
                        .F[jj * M + ii] =
                        ctx->fh_Ainv[
                            jj * M + ii];
                }
        }
        else
        {
            for (long ii = 0; ii < M; ii++)
                for (long jj = 0; jj < N; jj++)
                {
                    dcimg[ID_Ainv]
                        .array
                        .F[jj * M + ii] =
                        ctx->h_Ainv[
                            jj * M + ii];
                }
        }

        FUNC_CHECK_RETURN(
            save_fits(
                "mAinv",
                "test_mAinv.fits"));
        FUNC_CHECK_RETURN(
            delete_image_ID(
                "mAinv",
                DELETE_IMAGE_ERRMODE_IGNORE));
    }
}


/**
 * @brief Create the output image
 *
 * On first iteration, creates the output image
 * with matching dimensions.  On subsequent
 * iterations, looks up the existing image.
 */
static void svdpinv_create_output(
    svdpinv_ctx *ctx,
    const char  *ID_Cmatrix_name)
{
    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    if (MAGMAloop_iter == 0)
    {
        uint32_t *arraysizetmp;
        arraysizetmp =
            (uint32_t *) malloc(
                sizeof(uint32_t)
                * dcimg[ctx->ID_Rmatrix]
                      .md[0]
                      .naxis);

        if (dcimg[ctx->ID_Rmatrix]
                .md[0]
                .naxis == 3)
        {
            arraysizetmp[0] =
                dcimg[ctx->ID_Rmatrix]
                    .md[0]
                    .size[0];
            arraysizetmp[1] =
                dcimg[ctx->ID_Rmatrix]
                    .md[0]
                    .size[1];
            arraysizetmp[2] = N;
        }
        else
        {
            arraysizetmp[0] = M;
            arraysizetmp[1] = N;
        }

        {
            IMGID imgcm =
                imgid_make_from_name(
                    ID_Cmatrix_name);
            imgcm.mdt->naxis =
                dcimg[ctx->ID_Rmatrix]
                    .md[0]
                    .naxis;
            for (int a = 0;
                 a < imgcm.mdt->naxis;
                 a++)
            {
                imgcm.mdt->size[a] =
                    arraysizetmp[a];
            }
            imgcm.mdt->datatype =
                ctx->datatype;
            imgcm.im =
                (IMAGE *) calloc(
                    1, sizeof(IMAGE));
            imgid_mkimage(&imgcm);
            ctx->ID_Cmatrix = imgcm.ID;
        }

        free(arraysizetmp);
    }
    else
    {
        ctx->ID_Cmatrix = image_ID(
            ID_Cmatrix_name,
            dcimg, dcnimg);
    }
}


/**
 * @brief Fill output image with Ainv data
 *
 * Copies the pseudo-inverse result from host
 * buffer into the output image pixel array,
 * handling float/double conversion as needed.
 */
static void svdpinv_fill_output(
    svdpinv_ctx *ctx)
{
    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    if (ctx->verbose == 1)
    {
        printf("write result\n");
        fflush(stdout);
    }

    if (ctx->datatype == _DATATYPE_FLOAT)
    {
        if (ctx->MAGMAfloat == 1)
        {
            memcpy(
                dcimg[ctx->ID_Cmatrix]
                    .array.F,
                ctx->fh_Ainv,
                sizeof(float) * M * N);
        }
        else
        {
            for (long ii = 0;
                 ii < M * N;
                 ii++)
            {
                dcimg[ctx->ID_Cmatrix]
                    .array.F[ii] =
                    (float) ctx->h_Ainv[ii];
            }
        }
    }
    else
    {
        if (ctx->MAGMAfloat == 1)
        {
            for (long ii = 0;
                 ii < M * N;
                 ii++)
            {
                dcimg[ctx->ID_Cmatrix]
                    .array.D[ii] =
                    ctx->fh_Ainv[ii];
            }
        }
        else
        {
            memcpy(
                dcimg[ctx->ID_Cmatrix]
                    .array.D,
                ctx->h_Ainv,
                sizeof(double) * M * N);
        }
    }
}


/**
 * @brief Test: verify Ainv * A ≈ Identity
 *
 * Computes the product of Ainv with A on the
 * GPU, copies result to host, and saves to FITS.
 * Only runs when testmode is enabled.
 */
static void svdpinv_test_AinvA(
    svdpinv_ctx *ctx)
{
    if (ctx->testmode != 1)
    {
        return;
    }

    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    if (ctx->MAGMAfloat == 1)
    {
        magma_sgemm(
            MagmaTrans, MagmaNoTrans,
            N, N, M,
            1.0, ctx->fd_A, M,
            ctx->fd_Ainv, M,
            0.0, ctx->fd_AtA, N,
            magmaqueue);
    }
    else
    {
        magma_dgemm(
            MagmaTrans, MagmaNoTrans,
            N, N, M,
            1.0, ctx->d_A, M,
            ctx->d_Ainv, M,
            0.0, ctx->d_AtA, N,
            magmaqueue);
    }

    imageID ID_AinvA;
    FUNC_CHECK_RETURN(
        create_2Dimage_ID(
            "AinvA", N, N, &ID_AinvA));

    if (ctx->MAGMAfloat == 1)
    {
        magma_sgetmatrix(
            N, N,
            ctx->fd_AtA, N,
            ctx->fh_AtA, N,
            magmaqueue);
    }
    else
    {
        magma_dgetmatrix(
            N, N,
            ctx->d_AtA, N,
            ctx->h_AtA, N,
            magmaqueue);
    }

    if (ctx->datatype == _DATATYPE_FLOAT)
    {
        if (ctx->MAGMAfloat == 1)
        {
            memcpy(
                dcimg[ID_AinvA].array.F,
                ctx->fh_AtA,
                sizeof(float) * N * N);
        }
        else
        {
            for (long ii = 0;
                 ii < N * N;
                 ii++)
            {
                dcimg[ID_AinvA]
                    .array.F[ii] =
                    ctx->h_AtA[ii];
            }
        }
    }
    else
    {
        if (ctx->MAGMAfloat == 1)
        {
            for (long ii = 0;
                 ii < N * N;
                 ii++)
            {
                dcimg[ID_AinvA]
                    .array.D[ii] =
                    ctx->fh_AtA[ii];
            }
        }
        else
        {
            memcpy(
                dcimg[ID_AinvA].array.D,
                ctx->h_AtA,
                sizeof(double) * M * N);
        }
    }

    FUNC_CHECK_RETURN(
        save_fits("AinvA", "test_AinvA.fits"));
    FUNC_CHECK_RETURN(
        delete_image_ID(
            "AinvA",
            DELETE_IMAGE_ERRMODE_IGNORE));
}


/**
 * @brief Optional PFfmdat matrix product
 *
 * If a data matrix named "PFfmdat" exists,
 * performs an additional Ainv^T * PFfmdat product
 * for predictive control / sensor fusion.
 */
static void svdpinv_optional_PFgemm(
    svdpinv_ctx *ctx)
{
    magma_int_t M = ctx->M;
    magma_int_t N = ctx->N;

    imageID ID_PFfmdat = image_ID(
        "PFfmdat", dcimg, dcnimg);
    if (ID_PFfmdat == -1)
    {
        return;
    }

    printf(
        "Transp(Ainv)     N x M"
        "   = %d x %d\n", N, M);
    printf(
        "PFfmdat  M x K"
        "           = %d x %d\n",
        dcimg[ID_PFfmdat].md[0].size[0],
        dcimg[ID_PFfmdat].md[0].size[1]);
    long K =
        dcimg[ID_PFfmdat].md[0].size[1];
    printf("K = %ld\n", K);

    float *fd_PFfmdat;
    float *fd_PF;
    float *fh_PF;

    TESTING_SMALLOC_DEV(fd_PFfmdat, M * K);
    TESTING_SMALLOC_DEV(fd_PF, N * K);
    TESTING_SMALLOC_CPU(fh_PF, N * K);

    magma_sgetmatrix(
        N, K, fd_PF, N,
        fh_PF, N, magmaqueue);

    magma_ssetmatrix(
        M, K,
        dcimg[ID_PFfmdat].array.F, M,
        fd_PFfmdat, M,
        magmaqueue);

    magma_sgetmatrix(
        N, K, fd_PF, N,
        fh_PF, N, magmaqueue);

    magma_sgemm(
        MagmaTrans, MagmaNoTrans,
        N, K, M,
        1.0, ctx->fd_Ainv, M,
        fd_PFfmdat, M,
        0.0, fd_PF, N,
        magmaqueue);

    magma_sgetmatrix(
        N, K, fd_PF, N,
        fh_PF, N, magmaqueue);

    imageID ID_PF;
    FUNC_CHECK_RETURN(
        create_2Dimage_ID(
            "psinvPFmat", N, K, &ID_PF));

    memcpy(
        dcimg[ID_PF].array.F,
        fh_PF,
        sizeof(float) * N * K);
    FUNC_CHECK_RETURN(
        save_fits(
            "psinvPFmat",
            "psinvPFmat.fits"));

    TESTING_FREE_DEV(fd_PFfmdat);
    TESTING_FREE_DEV(fd_PF);
    TESTING_FREE_CPU(fh_PF);
}


/**
 * @brief Free all buffers (one-shot mode)
 *
 * Releases device and host memory, destroys the
 * MAGMA queue, and finalizes the library.
 * Only frees when LOOPmode is 0.
 */
static void svdpinv_free(svdpinv_ctx *ctx)
{
    if (ctx->LOOPmode == 0)
    {
        if (ctx->MAGMAfloat == 1)
        {
            TESTING_FREE_CPU(ctx->fh_A);
        }
        else
        {
            TESTING_FREE_CPU(ctx->h_A);
        }
    }

    if (ctx->LOOPmode == 0)
    {
        if (ctx->MAGMAfloat == 1)
        {
            TESTING_FREE_DEV(ctx->fd_A);

            if (ctx->dAinvMODE == 1)
            {
                TESTING_FREE_DEV(
                    ctx->fd_Ainv);
            }

            TESTING_FREE_CPU(ctx->fh_Ainv);
            TESTING_FREE_DEV(ctx->fd_AtA);
            TESTING_FREE_CPU(ctx->fh_AtA);
        }
        else
        {
            TESTING_FREE_DEV(ctx->d_A);

            if (ctx->dAinvMODE == 1)
            {
                TESTING_FREE_DEV(
                    ctx->d_Ainv);
            }

            TESTING_FREE_CPU(ctx->h_Ainv);
            TESTING_FREE_DEV(ctx->d_AtA);
            TESTING_FREE_CPU(ctx->h_AtA);
        }
    }

    if (ctx->LOOPmode == 0)
    {
        free(ctx->devicearray);
        magma_queue_destroy(magmaqueue);
        magma_finalize();
    }
}


/**
 * @brief Print per-step timing breakdown
 *
 * Prints wall-clock time for each computation
 * phase when verbose mode is enabled.
 */
static void svdpinv_print_timing(
    svdpinv_ctx *ctx)
{
    double t01d  = timespec_diff_double(
        ctx->t[0], ctx->t[1]);
    double t12d  = timespec_diff_double(
        ctx->t[1], ctx->t[2]);
    double t23d  = timespec_diff_double(
        ctx->t[2], ctx->t[3]);
    double t34d  = timespec_diff_double(
        ctx->t[3], ctx->t[4]);
    double t45d  = timespec_diff_double(
        ctx->t[4], ctx->t[5]);
    double t56d  = timespec_diff_double(
        ctx->t[5], ctx->t[6]);
    double t67d  = timespec_diff_double(
        ctx->t[6], ctx->t[7]);
    double t78d  = timespec_diff_double(
        ctx->t[7], ctx->t[8]);
    double t89d  = timespec_diff_double(
        ctx->t[8], ctx->t[9]);
    double t910d = timespec_diff_double(
        ctx->t[9], ctx->t[10]);
    double t1011d = timespec_diff_double(
        ctx->t[10], ctx->t[11]);
    double t1112d = timespec_diff_double(
        ctx->t[11], ctx->t[12]);
    double t1213d = timespec_diff_double(
        ctx->t[12], ctx->t[13]);
    double t013d = timespec_diff_double(
        ctx->t[0], ctx->t[13]);

    if (ctx->verbose == 1)
    {
        printf(
            "%6ld  Timing info: \n",
            MAGMAloop_iter);
        printf(
            "  0-1\t[setup]"
            "                           "
            "%12.3f ms\n",
            t01d * 1000.0);
        printf(
            "  1-2\t[copy input to GPU]"
            "               "
            "%12.3f ms\n",
            t12d * 1000.0);
        printf(
            "  2-3\t[compute trans(A) x A]"
            "            "
            "%12.3f ms\n",
            t23d * 1000.0);
        printf(
            "  3-4\t[setup]"
            "                           "
            "%12.3f ms\n",
            t34d * 1000.0);
        printf(
            "  4-5\t[Compute eigenvalues]"
            "             "
            "%12.3f ms\n",
            t45d * 1000.0);
        printf(
            "  5-6\t[Select eigenvalues]"
            "              "
            "%12.3f ms\n",
            t56d * 1000.0);
        printf(
            "  6-7\t[Compute M2]"
            "                      "
            "%12.3f ms\n",
            t67d * 1000.0);
        printf(
            "  7-8\t[Compute Ainv]"
            "                    "
            "%12.3f ms\n",
            t78d * 1000.0);
        printf(
            "  8-9\t[Get Ainv from GPU]"
            "               "
            "%12.3f ms\n",
            t89d * 1000.0);
        printf(
            "  9-10\t[output setup]"
            "                    "
            "%12.3f ms\n",
            t910d * 1000.0);
        printf(
            "  10-11\t[Write output array]"
            "              "
            "%12.3f ms\n",
            t1011d * 1000.0);
        printf(
            "  11-12\t[Test output]"
            "                     "
            "%12.3f ms\n",
            t1112d * 1000.0);
        printf(
            "  12-13\t[Optional gemm]"
            "                   "
            "%12.3f ms\n",
            t1213d * 1000.0);
        printf("\n");
        printf(
            " TOTAL 0-13     %12.3f ms\n",
            t013d * 1000.0);
        fflush(stdout);
    }

    if (ctx->verbose == 1)
    {
        printf("\n");
        fflush(stdout);
    }
}


/**
 *  @brief Computes matrix pseudo-inverse (AT A)^-1 AT, using eigenvector/eigenvalue decomposition of AT A
 *
 *
 * Computes pseuso inverse of a matrix.\n
 * Column-major representation used to match magma and lapack.\n
 * When viewed as an image, matrix leading dimension is size[0] = horizontal axis. When viewed in an image viewer, the first column is on the bottom side with the first element in bottom left corner, so the matrix appears rotated counter-clockwise by 90deg from its conventional representation where first column is on the left side.\n
 * Returns transpose of pseudoinverse.\n
 *
 *
 *
 * ## Matrix representation details
 *
 * Using column-major indexing\n
 * When viewed as a FITS file, the first matrix column (= vector) appears as the bottom line of the FITS image.\n
 * First matrix element is bottom left corner, second element is immediately to the right of it.
 *
 * Noting elements as a[row,column] = a[i,j], elements are accessed as in memory as:
 * 		a[ j * M + i ]
 *
 * FITS file representation (ds9 view) starts from bottom left corner.
 *
 * 		a[000,N-1] -> a[001,N-1] -> ... -> a[M-1,N-1]
 * 		............................................. ^
 * 		a[000,001] -> a[001,001] -> ... -> a[M-1,001] ^
 * 		a[000,000] -> a[001,000] -> ... -> a[M-1,000] ^     : this is the first matrix row
 *
 * Note that a tall input matrix (M>N) will appear short if viewed as an image.
 * To view the FITS file in the conventional matrix view, rotate by 90 deg clockwise.
 *
 *
 *
 * ## Application Notes
 *
 *  Use LOOPmode = 1 for computing the same size SVD, same input and output location
 *
 * ### Use case: Response matrix to compute control matrix
 *
 * When using function to invert AO response matrix with AOloopControl module, input is 2D or 3D image:
 * 		M: number of sensors    (AO control) =  size[0] (2D) = size[0]*size[1] (3D)
 * 		N: number of actuators  (AO control) =  size[1] (2D) =         size[2] (3D)
 *
 * 	We assume M>N
 *
 *
 * ### Use case: Predictive control
 *
 * When using function to compute pseudo-inverse of data matrix (predictive control), input matrix is a 2D image which is the Transpose of the data matrix.
 *		M: number of measurements samples  = size[0] (2D)
 *		N: dimension of each measurement   = size[1] (2D)
 *
 * We assume M>N
 *
 *
 *
 *
 * ## Algorithm details and main computation steps
 *
 * Notations:
 * 	AT is transpose of A
 * 	A+ is pseudo inverse of A
 *
 *  Computes pseudo-inverse : A+ = (AT A)^-1 AT
 *  Inverse of AT A is computed by SVD
 *
 * SVD:   A = U S V^T
 *   U are eigenvectors of A A^T
 *   V are eigenvectors of A^T A, computed at step 4 below
 *
 * Linear algebra reminder: equivalence between (AT A)^-1 AT and V S^-1 UT
 *
 * Definition of pseudoinverse:
 * A+ = (AT A)^-1 AT
 * singular value decomposition of A = U S VT
 * A+ = ( V S UT U S VT )^-1 V S UT
 * Since U is unitary, UT U = Id ->
 * A+ = ( V S^2 VT )^-1 V S UT
 * A+ = VT^-1 S^-2 V^-1 V S UT
 * A+ = V S^-1 UT
 *
 *  Main steps (non-QDWH):
 *
 *  STEP 1 :   Fill input data into magmaf_h_A on host
 *
 *  STEP 2 :   Copy input data to GPU                                 -> magmaf_d_A        (MxN matrix on device)
 *
 *  STEP 3 :   Compute  trans(A) x A   : magmaf_d_A x magmaf_d_A      -> magmaf_d_AtA      (NxN matrix on device)
 *
 *  STEP 4 :   Compute eigenvalues and eigenvectors of A^T A          -> magmaf_d_AtA      (NxN matrix on device)
 *     Calls magma_ssyevd_gpu :
 *     Compute the eigenvalues and optionally eigenvectors of a symmetric real matrix in single precision, GPU interface, big matrix.
 *     This function computes in single precision all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A defined on the device.
 *     The  first parameter can take the values MagmaVec,'V' or MagmaNoVec,'N' and answers the question whether the eigenvectors are desired.
 *     If the eigenvectors are desired, it uses a divide and conquer algorithm.  The symmetric matrix A can be stored in lower (MagmaLower,'L')
 *     or upper  (MagmaUpper,'U') mode. If the eigenvectors are desired, then on exit A contains orthonormal eigenvectors.
 *     The eigenvalues are stored in an array w
 *
 *  STEP 5 :   Set eigenvalue limit
 *
 *  STEP 6 :   Write eigenvectors to V^T matrix
 *
 *  STEP 7 :   Write eigenvectors/eigenvalue to magma_h_VT1 if eigenvalue > limit
 *           Copy to magma_d_VT1
 *
 *  STEP 8 :   Compute M2 = VT1 VT. M2 is (AT A)^-1
 *
 *  STEP 9 :   Compute Ainv = M2 A. This is the pseudo inverse
 *
 * @note SVDeps^2 is applied as a limit to the eigenvectors of AT A, which are equal to the squares of the singular values of A, so this is equivalent to applying SVDeps as a limit on the singular values of A
 * @note When used to compute AO control matrix, N=number of actuators/modes, M=number of WFS elements
 * @note EIGENVALUES are good to about 1e-6 of peak eigenvalue in single precision, much better with double precision
 * @note 2.5x faster in single precision
 *
 * @note If provided with an additional data matrix named "", an additional Matrix Matrix product between Ainv and the provided matrix will be performed. This feature is used for predictive control and sensor fusion to create a control matrix.
 *
 * TEST MODE OUTPOUT
 *
 * non-QDWH mode:
 *
 * test_mA.fits               content of magmaf_h_A
 * test_mAtA.fits             content of transpose(A) x A = magmaf_d_AtA (output of STEP 3)
 * test_eigenv.dat            list of eigenvalues
 * test_SVDmodes.log          number of singular values kept
 * test_mM2.fits              matrix M2 (output of STEP 8)
 * test_mVT.fits              matrix transpose(V) = eigenvectors (output of step 6)
 * test_mAinv.fits            transpose of pseudoinverse
 * test_AinvA.fits            product of Ainv with A, should be close to identity matrix size NxN
 *
 *
 * QDWH mode:
 *
 * test_mA.QDWH.fits          content of magmaf_h_A
 * test_Aorig.QDWH.txt        content of magmaf_h_A prior to calling psinv function
 * test_sv.QDWH.dat           singular values after call to psinv function
 * test_SVDmodes.QDWH.log     number of singular values kept (note : independent form pseudo-inverse computation)
 * test_mAinv.QDWH.fits       transpose of pseudoinverse
 * test_AinvA.QDWH.fits       product of Ainv with A, should be close to identity matrix size NxN
 */

errno_t LINALGEBRA_magma_compute_SVDpseudoInverse(
    const char *ID_Rmatrix_name,
    const char *ID_Cmatrix_name,
    double      SVDeps,
    long        MaxNBmodes,
    const char *ID_VTmatrix_name,
    int         LOOPmode,
    int         testmode,
    int         precision,
    int         GPUdevice,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    svdpinv_ctx ctx = {0};
    ctx.MAGMAfloat = (precision == 32) ? 1 : 0;
    ctx.testmode   = testmode;
    ctx.LOOPmode   = LOOPmode;
    ctx.verbose    = 1;
    ctx.magmaXmode = 0;
    ctx.dAinvMODE  = 0;
    ctx.SVDeps     = SVDeps;
    ctx.MaxNBmodes = MaxNBmodes;

    clock_gettime(CLOCK_MILK, &ctx.t[0]);

    svdpinv_read_dims(&ctx, ID_Rmatrix_name);
    svdpinv_init_device(&ctx, GPUdevice);
    svdpinv_alloc(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[1]);

    svdpinv_load_input(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[2]);

    svdpinv_compute_AtA(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[3]);

    /* sets ctx.t[4] internally */
    svdpinv_eigendecomp(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[5]);

    svdpinv_select_modes(&ctx);
    svdpinv_build_VT(&ctx, ID_VTmatrix_name);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[6]);

    svdpinv_compute_M2(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[7]);

    svdpinv_compute_Ainv(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[8]);

    svdpinv_retrieve_Ainv(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[9]);

    svdpinv_create_output(
        &ctx, ID_Cmatrix_name);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[10]);

    svdpinv_fill_output(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[11]);

    svdpinv_test_AinvA(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[12]);

    svdpinv_optional_PFgemm(&ctx);

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &ctx.t[13]);

    svdpinv_free(&ctx);
    svdpinv_print_timing(&ctx);

    if (LOOPmode == 1)
    {
        MAGMAloop_iter++;
    }

    list_image_ID();

    if (outID != NULL)
    {
        *outID = ctx.ID_Cmatrix;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#endif

#endif

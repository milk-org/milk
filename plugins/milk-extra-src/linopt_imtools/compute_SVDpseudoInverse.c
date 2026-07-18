#include <math.h>
#include <stdlib.h>
#include <string.h>

// MILK_CMAKE_MANDATE_LAPACKE
// MILK_CMAKE_MANDATE_BLAS

#ifdef HAVE_MKL
#    include "mkl.h"
#    include "mkl_lapacke.h"
#else
#    ifdef HAVE_OPENBLAS
#        include <cblas.h>
#    endif
#    include <lapacke.h> // OpenBLAS OR lapacke standalone
#endif

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_iofits/savefits.h"
#include "timeutils.h"
#include "linalgebra/linalgebra.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = { .fps_name    = "impsinvsvd",
                                     .cmdkey      = "impsinvsvd",
                                     .description = "compute pseudoinverse",
                                     .description_long =
                                         "Compute the pseudo-inverse of an image-to-mode matrix "
                                         "using SVD with configurable singular value truncation." };


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   *inimname       = NULL;
static char   *outimname      = NULL;
static double *SVD_epsilon    = NULL;
static long   *max_NBmodes    = NULL;
static char   *outimVTmatname = NULL;
static long   *useGPU         = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".inim", &inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")         \
    X(".outim", &outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")          \
    X(".svdeps", &SVD_epsilon, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "SVD cutoff")        \
    X(".maxNBmode", &max_NBmodes, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "Maximum NB modes") \
    X(".outimVT", &outimVTmatname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output VT matrix")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/**
 * @brief Compute pseudoinverse via eigenvalue
 *        decomposition of D^T * D
 *
 * Uses LAPACKE dsyev for eigenvalue decomposition
 * and CBLAS dgemm for matrix multiplication.
 *
 * Conventions:
 *   m: number of actuators (= NB_MODES)
 *   n: number of sensors  (= # of pixels)
 *
 * Efficient when n >> m, as D^T*D is m x m.
 */
errno_t linopt_compute_SVDpseudoInverse(const char *ID_Rmatrix_name,
                                        const char *ID_Cmatrix_name,
                                        double      SVDeps,
                                        long        MaxNBmodes,
                                        const char *ID_VTmatrix_name,
                                        imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    FILE   *fp;
    char    fname[200];
    long    m, n;
    double  egvlim;
    long    nbmodesremoved;
    uint8_t datatype;
    long    MaxNBmodes1, mode;

    int             timing = 1;
    struct timespec t0, t1, t2, t3, t4;
    struct timespec t5, t6, t7;
    double          t01d, t12d, t23d, t34d;
    double          t45d, t56d, t67d;
    struct timespec tdiff;

    int testmode = 0;

    printf("[CPU (lapack) SVD start]");
    fflush(stdout);

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t0);
    }

    IMGID imgin = imgid_make_from_name(ID_Rmatrix_name);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

    datatype = imgin.md->datatype;
    if (imgin.md->naxis == 3)
    {
        n = imgin.md->size[0] * imgin.md->size[1];
        m = imgin.md->size[2];
        printf("3D image -> %ld %ld\n", n, m);
    }
    else
    {
        n = imgin.md->size[0];
        m = imgin.md->size[1];
        printf("2D image -> %ld %ld\n", n, m);
    }
    fflush(stdout);

    printf("m = %ld , n = %ld \n", m, n);
    fflush(stdout);

    /* Allocate double work arrays */
    double *D      = calloc((size_t) n * m, sizeof(double));
    double *Ds     = calloc((size_t) m * n, sizeof(double));
    double *DtD    = calloc((size_t) m * m, sizeof(double));
    double *DtDinv = calloc((size_t) m * m, sizeof(double));
    double *eval   = calloc((size_t) m, sizeof(double));
    double *tmp1   = calloc((size_t) m * m, sizeof(double));
    double *tmp2   = calloc((size_t) m * m, sizeof(double));

    /* Fill D column-major */
    if (datatype == _DATATYPE_FLOAT)
    {
        for (long k = 0; k < m; k++)
        {
            for (long ii = 0; ii < n; ii++)
            {
                D[ii + k * n] = imgin.im->array.F[k * n + ii];
            }
        }
    }
    else
    {
        for (long k = 0; k < m; k++)
        {
            for (long ii = 0; ii < n; ii++)
            {
                D[ii + k * n] = imgin.im->array.D[k * n + ii];
            }
        }
    }

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t1);
    }

    /* DtD = D^T * D  (m x m) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) m, (int) m, (int) n, 1.0, D, (int) n,
                D, (int) n, 0.0, DtD, (int) m);

    if (testmode == 1)
    {
        IMGID imgAtA       = imgid_make_from_name_2D("AtA", m, m);
        imgAtA.mdt->shared = 0;
        imgAtA.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgAtA);

        for (long ii = 0; ii < m; ii++)
        {
            for (long jj = 0; jj < m; jj++)
            {
                imgAtA.im->array.F[jj * m + ii] = (float) DtD[ii + jj * m];
            }
        }
        save_fits("AtA", "test_AtA.fits");
    }

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t2);
    }

    /* Eigenvalue decomposition */
    {
        int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', (int) m, DtD, (int) m, eval);
        if (info != 0)
        {
            printf("LAPACKE_dsyev"
                   " failed: %d\n",
                   info);
        }
    }

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t3);
    }

    /* Reverse to descending order */
    for (long i = 0; i < m / 2; i++)
    {
        double t        = eval[i];
        eval[i]         = eval[m - 1 - i];
        eval[m - 1 - i] = t;
    }
    for (long i = 0; i < m / 2; i++)
    {
        for (long j = 0; j < m; j++)
        {
            double t                 = DtD[j + i * m];
            DtD[j + i * m]           = DtD[j + (m - 1 - i) * m];
            DtD[j + (m - 1 - i) * m] = t;
        }
    }

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t4);
    }

    /* Write eigenvalues to file */
    snprintf(fname, sizeof(fname), "eigenv.dat");
    fp = fopen(fname, "w");
    if (fp == NULL)
    {
        printf("ERROR: cannot create"
               " \"%s\"\n",
               fname);
        exit(0);
    }
    for (long k = 0; k < m; k++)
    {
        fprintf(fp, "%ld %g %g\n", k, sqrt(eval[k]), eval[k]);
    }
    fclose(fp);

    egvlim      = SVDeps * SVDeps * eval[0];
    MaxNBmodes1 = MaxNBmodes;
    if (MaxNBmodes1 > m)
    {
        MaxNBmodes1 = m;
    }
    if (MaxNBmodes1 > n)
    {
        MaxNBmodes1 = n;
    }
    mode = 0;
    while ((mode < MaxNBmodes1) && (eval[mode] > egvlim))
    {
        mode++;
    }
    printf("Keeping %ld modes  "
           "(SVDeps = %g-> %g, "
           "MaxNBmodes = %ld -> %ld)\n",
           mode, SVDeps, egvlim, MaxNBmodes, MaxNBmodes1);
    MaxNBmodes1 = mode;

    /* Write rotation matrix VT */
    IMGID imgVT         = imgid_make_from_name_2D(ID_VTmatrix_name, m, m);
    imgVT.mdt->datatype = datatype;
    imgVT.mdt->shared   = 0;
    imgVT.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgVT);

    if (datatype == _DATATYPE_FLOAT)
    {
        for (long ii = 0; ii < m; ii++)
        {
            for (long k = 0; k < m; k++)
            {
                imgVT.im->array.F[k * m + ii] = (float) DtD[k + ii * m];
            }
        }
    }
    else
    {
        for (long ii = 0; ii < m; ii++)
        {
            for (long k = 0; k < m; k++)
            {
                imgVT.im->array.D[k * m + ii] = DtD[k + ii * m];
            }
        }
    }

    if (testmode == 1)
    {
        save_fits(ID_VTmatrix_name, "test_VT.fits");
    }

    /* Build diagonal inverse */
    nbmodesremoved = 0;
    memset(tmp1, 0, (size_t) m * m * sizeof(double));
    for (long ii = 0; ii < m; ii++)
    {
        if (ii > MaxNBmodes1 - 1)
        {
            nbmodesremoved++;
        }
        else
        {
            tmp1[ii + ii * m] = 1.0 / eval[ii];
        }
    }

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t5);
    }

    /* DtDinv = evec * diag^-1
     * * evec^T */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) m, (int) m, 1.0, DtD,
                (int) m, tmp1, (int) m, 0.0, tmp2, (int) m);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, (int) m, (int) m, (int) m, 1.0, tmp2,
                (int) m, DtD, (int) m, 0.0, DtDinv, (int) m);

    if (testmode == 1)
    {
        IMGID imgM2       = imgid_make_from_name_2D("M2", m, m);
        imgM2.mdt->shared = 0;
        imgM2.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgM2);

        for (long ii = 0; ii < m; ii++)
        {
            for (long jj = 0; jj < m; jj++)
            {
                imgM2.im->array.F[jj * m + ii] = (float) DtDinv[ii + jj * m];
            }
        }
        save_fits("M2", "test_M2.fits");
    }

    /* Ds = DtDinv * D^T  (m x n) */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, (int) m, (int) n, (int) m, 1.0, DtDinv,
                (int) m, D, (int) n, 0.0, Ds, (int) m);

    IMGID imgC = imgid_make_from_name(ID_Cmatrix_name);
    if (imgin.md->naxis == 3)
    {
        imgC.mdt->naxis   = 3;
        imgC.mdt->size[0] = imgin.md->size[0];
        imgC.mdt->size[1] = imgin.md->size[1];
        imgC.mdt->size[2] = m;
    }
    else
    {
        imgC.mdt->naxis   = 2;
        imgC.mdt->size[0] = n;
        imgC.mdt->size[1] = m;
    }
    imgC.mdt->datatype = datatype;
    imgC.mdt->shared   = 0;
    imgC.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgC);

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t6);
    }

    /* Write result */
    if (datatype == _DATATYPE_FLOAT)
    {
        for (long ii = 0; ii < n; ii++)
        {
            for (long k = 0; k < m; k++)
            {
                imgC.im->array.F[k * n + ii] = (float) Ds[k + ii * m];
            }
        }
    }
    else
    {
        for (long ii = 0; ii < n; ii++)
        {
            for (long k = 0; k < m; k++)
            {
                imgC.im->array.D[k * n + ii] = Ds[k + ii * m];
            }
        }
    }

    if (testmode == 1)
    {
        save_fits(ID_Cmatrix_name, "test_Ainv.fits");
    }

    if (timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t7);
    }

    free(eval);
    free(D);
    free(Ds);
    free(DtD);
    free(DtDinv);
    free(tmp1);
    free(tmp2);

    printf("[CPU pseudo-inverse done]\n");
    fflush(stdout);

    if (timing == 1)
    {
        tdiff = timespec_diff(t0, t1);
        t01d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        tdiff = timespec_diff(t1, t2);
        t12d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        tdiff = timespec_diff(t2, t3);
        t23d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        tdiff = timespec_diff(t3, t4);
        t34d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        tdiff = timespec_diff(t4, t5);
        t45d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        tdiff = timespec_diff(t5, t6);
        t56d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        tdiff = timespec_diff(t6, t7);
        t67d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        printf("Timing info: \n");
        printf("  0-1\t%12.3f ms\n", t01d * 1000.0);
        printf("  1-2\t%12.3f ms\n", t12d * 1000.0);
        printf("  2-3\t%12.3f ms\n", t23d * 1000.0);
        printf("  3-4\t%12.3f ms\n", t34d * 1000.0);
        printf("  4-5\t%12.3f ms\n", t45d * 1000.0);
        printf("  5-6\t%12.3f ms\n", t56d * 1000.0);
        printf("  6-7\t%12.3f ms\n", t67d * 1000.0);
    }

    if (outID != NULL)
    {
        *outID = imgC.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    if (*useGPU == 0)
    {
        printf("==== CPU =====\n");
        linopt_compute_SVDpseudoInverse(inimname, outimname, *SVD_epsilon, *max_NBmodes,
                                        outimVTmatname, NULL);
    }
    else
    {
        printf("==== GPU =====\n");
#ifdef HAVE_MAGMA
        LINALGEBRA_magma_compute_SVDpseudoInverse(inimname, outimname, *SVD_epsilon, *max_NBmodes,
                                                  outimVTmatname, 0, 1, 64, 0, /* GPU device */
                                                  NULL);
#endif
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linopt_imtools__compute_SVDpseudoinverse()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif

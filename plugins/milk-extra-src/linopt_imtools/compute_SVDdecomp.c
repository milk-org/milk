#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MKL
#    include "mkl_lapacke.h"
#else
#    include <cblas.h>
#    include <lapacke.h>
#endif

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imsvd",
    .cmdkey           = "imsvd",
    .description      = "Singular values decomposition",
    .description_long = "Decompose image data using Singular Value Decomposition. Extracts "
                        "principal modes and their singular values from an image cube."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *imcinname    = NULL;
static char *outimname    = NULL;
static char *outcoeffname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".inc", &imcinname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input 3D cube") \
    X(".outm", &outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output modes")     \
    X(".outcoeff", &outcoeffname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output coeffs")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/**
 * @brief SVD via eigenvalue decomposition
 *        of D^T * D (LAPACK dsyev + CBLAS dgemm)
 *
 * Rotation matrix written as SVD_VTm.
 */
errno_t linopt_compute_SVDdecomp(const char *IDin_name,
                                 const char *IDout_name,
                                 const char *IDcoeff_name,
                                 imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    printf("[SVD start]");
    fflush(stdout);

    IMGID imgin = imgid_make_from_name(IDin_name);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

    long n = imgin.md->size[0] * imgin.md->size[1];
    long m = imgin.md->size[2];

    /* Allocate work arrays */
    double *D    = calloc((size_t) n * m, sizeof(double));
    double *DtD  = calloc((size_t) m * m, sizeof(double));
    double *eval = calloc((size_t) m, sizeof(double));

    /* Fill D column-major */
    for (long k = 0; k < m; k++)
    {
        for (long ii = 0; ii < n; ii++)
        {
            D[ii + k * n] = imgin.im->array.F[k * n + ii];
        }
    }

    /* DtD = D^T * D  (m x m) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) m, (int) m, (int) n, 1.0, D, (int) n,
                D, (int) n, 0.0, DtD, (int) m);

    /* Eigenvalue decomposition */
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', (int) m, DtD, (int) m, eval);
    if (info != 0)
    {
        printf("LAPACKE_dsyev failed: %d\n", info);
    }

    /* Reverse to descending order */
    for (long i = 0; i < m / 2; i++)
    {
        double tmp      = eval[i];
        eval[i]         = eval[m - 1 - i];
        eval[m - 1 - i] = tmp;
    }
    for (long i = 0; i < m / 2; i++)
    {
        for (long j = 0; j < m; j++)
        {
            double tmp               = DtD[j + i * m];
            DtD[j + i * m]           = DtD[j + (m - 1 - i) * m];
            DtD[j + (m - 1 - i) * m] = tmp;
        }
    }

    /* Write eigenvalues */
    IMGID imgcoeff       = imgid_make_from_name_2D(IDcoeff_name, m, 1);
    imgcoeff.mdt->shared = 0;
    imgcoeff.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgcoeff);

    for (long k = 0; k < m; k++)
    {
        imgcoeff.im->array.F[k] = (float) eval[k];
    }

    /* Write rotation matrix VT */
    {
        imageID oldVT = image_ID("SVD_VTm", dcimg, dcnimg);
        if (oldVT != -1)
        {
            delete_image_ID("SVD_VTm", DELETE_IMAGE_ERRMODE_WARNING);
        }
    }

    IMGID imgVT       = imgid_make_from_name_2D("SVD_VTm", m, m);
    imgVT.mdt->shared = 0;
    imgVT.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgVT);

    for (long ii = 0; ii < m; ii++)
    {
        for (long k = 0; k < m; k++)
        {
            imgVT.im->array.F[k * m + ii] = (float) DtD[k + ii * m];
        }
    }

    /* Compute SVD modes */
    IMGID imgout       = imgid_make_from_name_3D(IDout_name, imgin.md->size[0], imgin.md->size[1],
                                                 imgin.md->size[2]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (long kk = 0; kk < m; kk++)
    {
        for (long kk1 = 0; kk1 < m; kk1++)
        {
            for (long ii = 0; ii < n; ii++)
            {
                imgout.im->array.F[kk * n + ii] +=
                    imgVT.im->array.F[kk1 * m + kk] * imgin.im->array.F[kk1 * n + ii];
            }
        }
    }

    free(D);
    free(DtD);
    free(eval);

    printf("[SVD done]\n");
    fflush(stdout);

    if (outID != NULL)
    {
        *outID = imgout.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_compute_SVDdecomp(imcinname, outimname, outcoeffname, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linopt_imtools__compute_SVDdecomp()
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

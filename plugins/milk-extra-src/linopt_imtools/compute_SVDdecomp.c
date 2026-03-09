#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MKL
#include "mkl_lapacke.h"
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "CLIcore.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imsvd",
    .cmdkey      = "imsvd",
    .description = "Singular values decomposition"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * imcinname = NULL;
static char * outimname = NULL;
static char * outcoeffname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".inc", &imcinname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input 3D cube") \
    X(".outm", &outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output modes") \
    X(".outcoeff", &outcoeffname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output coeffs")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/**
 * @brief SVD via eigenvalue decomposition
 *        of D^T * D (LAPACK dsyev + CBLAS dgemm)
 *
 * Rotation matrix written as SVD_VTm.
 */
errno_t linopt_compute_SVDdecomp(
    const char *IDin_name,
    const char *IDout_name,
    const char *IDcoeff_name,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID    IDin;
    imageID    IDout;
    imageID    IDcoeff;
    imageID    ID_VTmatrix;
    long       m;
    long       n;
    uint32_t  *arraysizetmp;

    arraysizetmp =
        (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if (arraysizetmp == NULL) {
        FUNC_RETURN_FAILURE(
            "malloc returns NULL pointer");
    }

    printf("[SVD start]");
    fflush(stdout);

    IDin = image_ID(IDin_name, dcimg, dcnimg);
    n = dcimg[IDin].md[0].size[0]
        * dcimg[IDin].md[0].size[1];
    m = dcimg[IDin].md[0].size[2];

    /* Allocate work arrays */
    double *D =
        calloc((size_t) n * m, sizeof(double));
    double *DtD =
        calloc((size_t) m * m, sizeof(double));
    double *eval =
        calloc((size_t) m, sizeof(double));

    /* Fill D column-major: D[ii + k*n] */
    for (long k = 0; k < m; k++) {
        for (long ii = 0; ii < n; ii++) {
            D[ii + k * n] =
                dcimg[IDin].array.F[k * n + ii];
        }
    }

    /* DtD = D^T * D  (m x m) */
    cblas_dgemm(CblasColMajor,
                CblasTrans, CblasNoTrans,
                (int) m, (int) m, (int) n,
                1.0, D, (int) n,
                D, (int) n,
                0.0, DtD, (int) m);

    /* Eigenvalue decomposition: DtD overwritten
     * with eigenvectors (columns) */
    int info = LAPACKE_dsyev(
        LAPACK_COL_MAJOR, 'V', 'U',
        (int) m, DtD, (int) m, eval);
    if (info != 0) {
        printf("LAPACKE_dsyev failed: %d\n",
               info);
    }

    /* LAPACK returns eigenvalues in ascending
     * order. Reverse to descending. */
    /* Reverse eval */
    for (long i = 0; i < m / 2; i++) {
        double tmp = eval[i];
        eval[i] = eval[m - 1 - i];
        eval[m - 1 - i] = tmp;
    }
    /* Reverse eigenvector columns */
    for (long i = 0; i < m / 2; i++) {
        for (long j = 0; j < m; j++) {
            double tmp = DtD[j + i * m];
            DtD[j + i * m] =
                DtD[j + (m - 1 - i) * m];
            DtD[j + (m - 1 - i) * m] = tmp;
        }
    }

    /* Write eigenvalues */
    create_2Dimage_ID(
        IDcoeff_name, m, 1, &IDcoeff);
    for (long k = 0; k < m; k++) {
        dcimg[IDcoeff].array.F[k] =
            (float) eval[k];
    }

    /* Write rotation matrix VT */
    arraysizetmp[0] = m;
    arraysizetmp[1] = m;
    ID_VTmatrix =
        image_ID("SVD_VTm", dcimg, dcnimg);
    if (ID_VTmatrix != -1) {
        delete_image_ID(
            "SVD_VTm",
            DELETE_IMAGE_ERRMODE_WARNING);
    }
    create_image_ID("SVD_VTm", 2,
                    arraysizetmp,
                    _DATATYPE_FLOAT,
                    0, 0, 0,
                    &ID_VTmatrix);
    for (long ii = 0; ii < m; ii++) {
        for (long k = 0; k < m; k++) {
            dcimg[ID_VTmatrix].array.F[
                k * m + ii] =
                (float) DtD[k + ii * m];
        }
    }

    /* Compute SVD modes: out = VT^T * in */
    FUNC_CHECK_RETURN(
        create_3Dimage_ID(
            IDout_name,
            dcimg[IDin].md[0].size[0],
            dcimg[IDin].md[0].size[1],
            dcimg[IDin].md[0].size[2],
            &IDout));

    for (long kk = 0; kk < m; kk++) {
        for (long kk1 = 0; kk1 < m; kk1++) {
            for (long ii = 0; ii < n; ii++) {
                dcimg[IDout].array.F[
                    kk * n + ii] +=
                    dcimg[ID_VTmatrix].array.F[
                        kk1 * m + kk]
                    * dcimg[IDin].array.F[
                        kk1 * n + ii];
            }
        }
    }

    free(arraysizetmp);
    free(D);
    free(DtD);
    free(eval);

    printf("[SVD done]\n");
    fflush(stdout);

    if (outID != NULL) {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_compute_SVDdecomp(
        imcinname, outimname,
        outcoeffname, NULL);

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
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_linopt_imtools__compute_SVDdecomp()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif

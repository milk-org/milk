/** @file MatMatMult_testPseudoInverse.c
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>

#ifdef HAVE_MAGMA
#include "magma_lapack.h"
#include "magma_v2.h"
extern int           INIT_MAGMA;
extern magma_queue_t magmaqueue;

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
#include "COREMOD_memory/COREMOD_memory.h"

#include "linalgebra_types.h"

// ==========================================
// Forward declaration(s)
// ==========================================

long LINALGEBRA_MatMatMult_testPseudoInverse(
    const char *IDmatA_name,
    const char *IDmatAinv_name,
    const char *IDmatOut_name);

// ==========================================
// Gen 4 V2 CLI command: cudatestpsinv
// ==========================================

static char tp_a[FUNCTION_PARAMETER_STRMAXLEN]
    = "matA";
static char tp_ai[FUNCTION_PARAMETER_STRMAXLEN]
    = "matAinv";
static char tp_o[FUNCTION_PARAMETER_STRMAXLEN]
    = "matOut";
static FPS_APP_INFO FPS_app_info_tp = {
    .fps_name = "cudatestpsinv",
    .cmdkey   = "cudatestpsinv",
    .description = "test pseudo inverse",
    .description_long =
        "Test pseudo-inverse computation using MAGMA GPU library. Validates matrix inversion accuracy and performance."
};
#define FPS_PARAMS_TP(X) \
    X(".matA", tp_a, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "matA") \
    X(".matAinv", tp_ai, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "matAinv") \
    X(".out_name", tp_o, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")
#include "fps.h"
static FPS_CLI_BINDING tp_b[] = {
    FPS_PARAMS_TP(FPS_X_BINDING) };
static const int tp_nb =
    sizeof(tp_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS_TP(FPS_X_FARG) };
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS tp_cms = {0};
static __attribute__((constructor))
void init_tp(void) {
    strncpy(CLIcmddata.key,
        FPS_app_info_tp.cmdkey,
        sizeof(CLIcmddata.key)-1);
    strncpy(CLIcmddata.description,
        FPS_app_info_tp.description,
        sizeof(CLIcmddata.description)-1);
    CLIcmddata.nbarg =
        sizeof(farg)/sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags = CLICMDFLAG_FPS;
    if(!CLIcmddata.cmdsettings)
        CLIcmddata.cmdsettings = &tp_cms;
}
static errno_t tp_compute(void) {
    LINALGEBRA_MatMatMult_testPseudoInverse(
        tp_a, tp_ai, tp_o);
    return RETURN_SUCCESS;
}
static errno_t CLIfunction(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_tp, farg,
        &CLIcmddata,
        tp_b, tp_nb, tp_compute);
}

errno_t MatMatMult_testPseudoInverse_addCLIcmd()
{
    safe_fps_fill_farg_examples(
        farg, tp_b, tp_nb);
    INSERT_STD_CLIREGISTERFUNC;

    return RETURN_SUCCESS;
}

/** @brief Test pseudo inverse
 *
 */

long LINALGEBRA_MatMatMult_testPseudoInverse(
    const char *IDmatA_name,
    const char *IDmatAinv_name,
    const char *IDmatOut_name)
{
    imageID IDmatA;
    imageID IDmatAinv;
    imageID IDmatOut;

    float *magmaf_h_A;
    float *magmaf_d_A;

    float *magmaf_h_Ainv;
    float *magmaf_d_Ainv;

    long   ii;
    float *magmaf_d_AinvA;
    float *magmaf_h_AinvA;

    uint32_t   *arraysizetmp;
    magma_int_t M, N;

    /**
     *
     * IDmatA is an image loaded as a M x N matrix
     * IDmatAinv is an image loaded as a M x M matrix, representing the transpose of the pseudo inverse of IDmatA
     *
     * The input matrices can be 2D or a 3D images
     *
     * If 2D image :
     *   IDmatA    M = xsize
     *   IDmatA    N = ysize
     *
     * If 3D image :
     *   IDmatA M = xsize*ysize
     *   IDmatA N = ysize
     *
     *
     */

    ///
    /// MAGMA uses column-major matrices. For matrix A with dimension (M,N), element A(i,j) is A[ j*M + i]
    /// i = 0 ... M
    /// j = 0 ... N
    ///

    arraysizetmp = (uint32_t *) malloc(sizeof(uint32_t) * 3);

    IDmatA    = image_ID(IDmatA_name, dcimg, dcnimg);
    IDmatAinv = image_ID(IDmatAinv_name, dcimg, dcnimg);

    if(dcimg[IDmatA].md[0].naxis == 3)
    {
        /// each column (N=cst) of A is a z=cst slice of image Rmatrix
        M = dcimg[IDmatA].md[0].size[0] * dcimg[IDmatA].md[0].size[1];
        N = dcimg[IDmatA].md[0].size[2];
    }
    else
    {
        /// each column (N=cst) of A is a line (y=cst) of Rmatrix (90 deg rotation)
        M = dcimg[IDmatA].md[0].size[0];
        N = dcimg[IDmatA].md[0].size[1];
    }

    /// Initialize MAGAM if needed
    if(INIT_MAGMA == 0)
    {
        magma_init();
        magma_print_environment();

        INIT_MAGMA = 1;
    }
    magma_queue_create(0, &magmaqueue);

    TESTING_SMALLOC_CPU(magmaf_h_A, M * N);
    TESTING_SMALLOC_DEV(magmaf_d_A, M * N);

    TESTING_SMALLOC_CPU(magmaf_h_Ainv, M * N);
    TESTING_SMALLOC_DEV(magmaf_d_Ainv, M * N);

    TESTING_SMALLOC_CPU(magmaf_h_AinvA, N * N);
    TESTING_SMALLOC_DEV(magmaf_d_AinvA, N * N);

    /// load matA in h_A -> d_A
    for(ii = 0; ii < M * N; ii++)
    {
        magmaf_h_A[ii] = dcimg[IDmatA].array.F[ii];
    }
    magma_ssetmatrix(M, N, magmaf_h_A, M, magmaf_d_A, M, magmaqueue);

    /// load matAinv in h_Ainv -> d_Ainv
    for(ii = 0; ii < M * N; ii++)
    {
        magmaf_h_Ainv[ii] = dcimg[IDmatAinv].array.F[ii];
    }
    magma_ssetmatrix(M, N, magmaf_h_Ainv, M, magmaf_d_Ainv, M, magmaqueue);

    magma_sgemm(MagmaTrans,
                MagmaNoTrans,
                N,
                N,
                M,
                1.0,
                magmaf_d_Ainv,
                M,
                magmaf_d_A,
                M,
                0.0,
                magmaf_d_AinvA,
                N,
                magmaqueue);

    magma_sgetmatrix(N, N, magmaf_d_AinvA, N, magmaf_h_AinvA, N, magmaqueue);

    arraysizetmp[0] = N;
    arraysizetmp[1] = N;
    {
        IMGID imgout =
            imgid_make_from_name(
                IDmatOut_name);
        imgout.mdt->naxis = 2;
        imgout.mdt->size[0] =
            arraysizetmp[0];
        imgout.mdt->size[1] =
            arraysizetmp[1];
        imgout.mdt->datatype =
            _DATATYPE_FLOAT;
        imgout.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgout);
        IDmatOut = imgout.ID;
    }

    for(ii = 0; ii < N * N; ii++)
    {
        dcimg[IDmatOut].array.F[ii] = magmaf_h_AinvA[ii];
    }

    TESTING_FREE_CPU(magmaf_h_AinvA);
    TESTING_FREE_DEV(magmaf_d_AinvA);

    TESTING_FREE_DEV(magmaf_d_A);
    TESTING_FREE_CPU(magmaf_h_A);

    TESTING_FREE_DEV(magmaf_d_Ainv);
    TESTING_FREE_CPU(magmaf_h_Ainv);

    free(arraysizetmp);

    magma_queue_destroy(magmaqueue);
    magma_finalize(); //  finalize  Magma

    return IDmatOut;
}

#endif

#endif

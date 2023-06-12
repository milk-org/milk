/** @file MatMatMult_testPseudoInverse.c
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>

#ifdef HAVE_MAGMA
#include "magma_lapack.h"
#include "magma_v2.h"
extern int           INIT_MAGMA;
extern magma_queue_t magmaqueue;

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "linalgebra_types.h"

// ==========================================
// Forward declaration(s)
// ==========================================

long LINALGEBRA_MatMatMult_testPseudoInverse(const char *IDmatA_name,
        const char *IDmatAinv_name,
        const char *IDmatOut_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t LINALGEBRA_MatMatMult_testPseudoInverse_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 3) == 0)
    {
        LINALGEBRA_MatMatMult_testPseudoInverse(data.cmdargtoken[1].val.string,
                                                data.cmdargtoken[2].val.string,
                                                data.cmdargtoken[3].val.string);

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

errno_t MatMatMult_testPseudoInverse_addCLIcmd()
{

    RegisterCLIcommand("cudatestpsinv",
                       __FILE__,
                       LINALGEBRA_MatMatMult_testPseudoInverse_cli,
                       "test pseudo inverse",
                       "<matA> <matAinv> <matOut>",
                       "cudatestpsinv matA matAinv matOut",
                       "long LINALGEBRA_MatMatMult_testPseudoInverse(const char "
                       "*IDmatA_name, const char "
                       "*IDmatAinv_name, const char *IDmatOut_name)");

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

    IDmatA    = image_ID(IDmatA_name);
    IDmatAinv = image_ID(IDmatAinv_name);

    if(data.image[IDmatA].md[0].naxis == 3)
    {
        /// each column (N=cst) of A is a z=cst slice of image Rmatrix
        M = data.image[IDmatA].md[0].size[0] * data.image[IDmatA].md[0].size[1];
        N = data.image[IDmatA].md[0].size[2];
    }
    else
    {
        /// each column (N=cst) of A is a line (y=cst) of Rmatrix (90 deg rotation)
        M = data.image[IDmatA].md[0].size[0];
        N = data.image[IDmatA].md[0].size[1];
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
        magmaf_h_A[ii] = data.image[IDmatA].array.F[ii];
    }
    magma_ssetmatrix(M, N, magmaf_h_A, M, magmaf_d_A, M, magmaqueue);

    /// load matAinv in h_Ainv -> d_Ainv
    for(ii = 0; ii < M * N; ii++)
    {
        magmaf_h_Ainv[ii] = data.image[IDmatAinv].array.F[ii];
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
    create_image_ID(IDmatOut_name,
                    2,
                    arraysizetmp,
                    _DATATYPE_FLOAT,
                    0,
                    0,
                    0,
                    &IDmatOut);

    for(ii = 0; ii < N * N; ii++)
    {
        data.image[IDmatOut].array.F[ii] = magmaf_h_AinvA[ii];
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

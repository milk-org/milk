/**
 * @file    linalgebra.c
 * @brief   Linear Algebra functions wrapper
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "linalg"
#define MODULE_DESCRIPTION       "linear algebra"

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#endif

#ifdef HAVE_MAGMA
#include "magma_lapack.h"
#include "magma_v2.h"
#endif

#include "CommandLineInterface/CLIcore.h"

#include "linalgebra_types.h"

#include "basis_rotate_match.h"
#include "cublas_Coeff2Map_Loop.h"
#include "GramSchmidt.h"
#include "MVMextractModes.h"
#include "magma_MatMatMult_testPseudoInverse.h"
#include "cublas_linalgebra_MVMextractModesLoop.h"
#include "linalgebrainit.h"
#include "cublas_linalgebratest.h"
#include "magma_compute_SVDpseudoInverse.h"
#include "magma_compute_SVDpseudoInverse_SVD.h"

#include "PCAmatch.h"

#include "SingularValueDecomp.h"
#include "SingularValueDecomp_mkU.h"
#include "SingularValueDecomp_mkM.h"
#include "SGEMM.h"

#include "modalremap.h"


#include "cublas_PCA.h"

// globals

imageID IDtimerinit = 0;
imageID IDtiming    = -1; // index to image where timing should be written

#ifdef HAVE_CUDA
int cuda_deviceCount;
GPUMATMULTCONF
gpumatmultconf[20]; // supports up to 20 configurations per process
float cublasSgemv_alpha = 1.0;
float cublasSgemv_beta  = 0.0;
#endif

#ifdef HAVE_MAGMA
int           INIT_MAGMA = 0;
magma_queue_t magmaqueue;
#endif

INIT_MODULE_LIB(linalgebra)

static void __attribute__((constructor)) libinit_linalgebra_printinfo()
{
#ifdef HAVE_CUDA
    if(!getenv("MILK_QUIET"))
    {
        printf("[CUDA %d]", data.quiet);
    }

#endif

#ifdef HAVE_MAGMA
    if(!getenv("MILK_QUIET"))
    {
        printf("[MAGMA]");
    }
#endif
}

static errno_t init_module_CLI()
{
#ifdef HAVE_CUDA
    //    printf("HAVE_CUDA defined\n");
    for(int i = 0; i < 20; i++)
    {
        gpumatmultconf[i].init  = 0;
        gpumatmultconf[i].alloc = 0;
    }

    linalgebrainit_addCLIcmd();
    linalgebratest_addCLIcmd();

    CLIADDCMD_linalgebra__basis_rotate_match();
    CLIADDCMD_linalgebra__GramSchmidt();

    CLIADDCMD_linalgebra__PCAdecomp();

#ifdef HAVE_MAGMA
    MatMatMult_testPseudoInverse_addCLIcmd();
    magma_compute_SVDpseudoInverse_addCLIcmd();
    magma_compute_SVDpseudoInverse_SVD_addCLIcmd();
#endif

    Coeff2Map_Loop_addCLIcmd();
    linalgebra_MVMextractModesLoop_addCLIcmd();
#endif

    CLIADDCMD_linalgebra__MVMextractModes();

    CLIADDCMD_linalgebra__PCAmatch();

    CLIADDCMD_linalgebra__compSVD();
    CLIADDCMD_linalgebra__compSVDU();
    CLIADDCMD_linalgebra__SVDmkM();

    CLIADDCMD_linalgebra__SGEMM();

    CLIADDCMD_linalgebra__ModalRemap();

    // add atexit functions here

    return RETURN_SUCCESS;
}

#ifdef HAVE_CUDA


#endif

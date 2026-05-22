/**
 * @file    im_complex.c
 * @brief   Complex number per-pixel image operations
 */

#include <math.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "libmilkcommon/milk_compiler.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "imgid_arith_helpers.h"

#include "imfunctions.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

// complex float (CF), complex float (CF) -> complex float (CF)
errno_t arith_image_function_CF_CF__CF(const char *ID_name1,
                                       const char *ID_name2,
                                       const char *ID_out,
                                       complex_float (*pt2function)(complex_float, complex_float))
{
    IMGID img1 = imgid_make_from_name(ID_name1);
    resolveIMGID(&img1, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID img2 = imgid_make_from_name(ID_name2);
    resolveIMGID(&img2, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID imgout      = imgid_make_from_name(ID_out);
    imgout.mdt->naxis = img1.md->naxis;
    for (uint8_t i = 0; i < img1.md->naxis; i++)
    {
        imgout.mdt->size[i] = img1.md->size[i];
    }
    imgout.mdt->datatype = img1.md->datatype;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    uint64_t nelement = img1.md->nelement;

#ifdef _OPENMP
#    pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#    pragma omp for simd
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout.im->array.CF[ii] = pt2function(img1.im->array.CF[ii], img2.im->array.CF[ii]);
        }
#ifdef _OPENMP
    }
#endif
    RegisterIMGID(&imgout, dcimg, dcnimg);
    imgid_free(&img1);
    imgid_free(&img2);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}

// complex double (CD), complex double (CD) -> complex double (CD)
errno_t arith_image_function_CD_CD__CD(const char *ID_name1,
                                       const char *ID_name2,
                                       const char *ID_out,
                                       complex_double (*pt2function)(complex_double,
                                                                     complex_double))
{
    IMGID img1 = imgid_make_from_name(ID_name1);
    resolveIMGID(&img1, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID img2 = imgid_make_from_name(ID_name2);
    resolveIMGID(&img2, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID imgout      = imgid_make_from_name(ID_out);
    imgout.mdt->naxis = img1.md->naxis;
    for (uint8_t i = 0; i < img1.md->naxis; i++)
    {
        imgout.mdt->size[i] = img1.md->size[i];
    }
    imgout.mdt->datatype = img1.md->datatype;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    uint64_t nelement = img1.md->nelement;

#ifdef _OPENMP
#    pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#    pragma omp for simd
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout.im->array.CD[ii] = pt2function(img1.im->array.CD[ii], img2.im->array.CD[ii]);
        }
#ifdef _OPENMP
    }
#endif
    RegisterIMGID(&imgout, dcimg, dcnimg);
    imgid_free(&img1);
    imgid_free(&img2);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}

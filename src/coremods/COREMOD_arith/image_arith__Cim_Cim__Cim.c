/**
 * @file    image_arith__Cim_Cim__Cim.c
 * @brief   Complex-image arithmetic (add/sub/mult/div)
 *
 * input : complex image, complex image
 * output: complex image
 *
 * All four operations generated via X-macro table.
 */


#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"


/* -------------------------------------------------------
 * X-macro table: (name, CF func ptr, CD func ptr)
 * ------------------------------------------------------- */
#define COMPLEX_OPS(X)                        \
    X(Cadd,  CPadd_CF_CF,  CPadd_CD_CD)       \
    X(Csub,  CPsub_CF_CF,  CPsub_CD_CD)       \
    X(Cmult, CPmult_CF_CF, CPmult_CD_CD)      \
    X(Cdiv,  CPdiv_CF_CF,  CPdiv_CD_CD)


/* -------------------------------------------------------
 * Stamp macro — one function per operation
 * ------------------------------------------------------- */
#define DEFINE_COMPLEX_OP(name, cf_fn, cd_fn)       \
errno_t arith_image_##name(                          \
    const char *ID1_name,                            \
    const char *ID2_name,                            \
    const char *ID_out)                              \
{                                                    \
    IMGID img1 =                                     \
        imgid_make_from_name(ID1_name);              \
    resolveIMGID(                                    \
        &img1, ERRMODE_ABORT, dcimg, dcnimg);        \
    IMGID img2 =                                     \
        imgid_make_from_name(ID2_name);              \
    resolveIMGID(                                    \
        &img2, ERRMODE_ABORT, dcimg, dcnimg);        \
                                                     \
    uint8_t dt1 = img1.md->datatype;                 \
    uint8_t dt2 = img2.md->datatype;                 \
                                                     \
    imgid_free(&img1);                               \
    imgid_free(&img2);                               \
                                                     \
    if (dt1 == _DATATYPE_COMPLEX_FLOAT               \
        && dt2 == _DATATYPE_COMPLEX_FLOAT)           \
    {                                                \
        arith_image_function_CF_CF__CF(              \
            ID1_name, ID2_name,                      \
            ID_out, &cf_fn);                         \
        return RETURN_SUCCESS;                       \
    }                                                \
    if (dt1 == _DATATYPE_COMPLEX_DOUBLE              \
        && dt2 == _DATATYPE_COMPLEX_DOUBLE)          \
    {                                                \
        arith_image_function_CD_CD__CD(              \
            ID1_name, ID2_name,                      \
            ID_out, &cd_fn);                         \
        return RETURN_SUCCESS;                       \
    }                                                \
    PRINT_ERROR("data types do not match");          \
    return RETURN_FAILURE;                           \
}

COMPLEX_OPS(DEFINE_COMPLEX_OP)
#undef DEFINE_COMPLEX_OP

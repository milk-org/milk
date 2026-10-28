// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__Cim_Cim__Cim.c
 * @brief   Complex-image arithmetic (add/sub/mult/div)
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "imgid_arith_helpers.h"
#include "image_arith__Cim_Cim__Cim.h"

/** Allocate and register a complex output image shaped like @img1 and run `arith_complex_binop_im`. */
static errno_t run_complex_named_op(int         op,
                                    const char *ID1_name,
                                    const char *ID2_name,
                                    const char *ID_out)
{
    IMGID img1 = imgid_make_from_name(ID1_name);
    resolveIMGID(&img1, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID img2 = imgid_make_from_name(ID2_name);
    resolveIMGID(&img2, ERRMODE_ABORT, dcimg, dcnimg);

    uint8_t dt1 = img1.md->datatype;
    uint8_t dt2 = img2.md->datatype;
    if (!arith_is_complex_dt(dt1) || dt1 != dt2)
    {
        imgid_free(&img1);
        imgid_free(&img2);
        PRINT_ERROR("data types do not match");
        return RETURN_FAILURE;
    }

    IMGID imgout      = imgid_make_from_name(ID_out);
    imgout.mdt->naxis = img1.md->naxis;
    for (uint8_t i = 0; i < img1.md->naxis; i++)
    {
        imgout.mdt->size[i] = img1.md->size[i];
    }
    imgout.mdt->datatype = dt1;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    arith_complex_binop_im(op, img1.im, 0.0, img2.im, 0.0, imgout.im);

    RegisterIMGID(&imgout, dcimg, dcnimg);
    imgid_free(&img1);
    imgid_free(&img2);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}

#define COMPLEX_OPS(X)       \
    X(Cadd, ARITH_CPLX_ADD)  \
    X(Csub, ARITH_CPLX_SUB)  \
    X(Cmult, ARITH_CPLX_MUL) \
    X(Cdiv, ARITH_CPLX_DIV)

#define DEFINE_COMPLEX_OP(name, op_code)                                                       \
    errno_t arith_image_##name(const char *ID1_name, const char *ID2_name, const char *ID_out) \
    {                                                                                          \
        return run_complex_named_op(op_code, ID1_name, ID2_name, ID_out);                      \
    }

COMPLEX_OPS(DEFINE_COMPLEX_OP)
#undef DEFINE_COMPLEX_OP

/**
 * @file  image_arith__im_im__im.h
 * @brief Binary image×image arithmetic prototypes (img,img → img).
 *
 * All prototypes are generated from X-macro op tables.
 * Adding a new operation requires only one edit:
 * add a row to the appropriate table below.
 *
 * Up to three call surfaces per operation:
 *   arith_image_<op>_IMGID(in1, in2, out)
 *   arith_image_<op>(name1, name2, name_out)
 *   arith_image_<op>_inplace(name1, name2)  [FULL ops only]
 */

#ifndef IMAGE_ARITH_IM_IM_IM_H
#define IMAGE_ARITH_IM_IM_IM_H

#include <libfps/IMGID.h>

double Ptrunc(double a, double b, double c);


/* ==========================================================
 * X-macro operation tables.
 *
 * MILK_BINARY_OPS_FULL — ops with all call surfaces
 *   (IMGID, string, inplace)
 *
 * MILK_BINARY_OPS_NOIP — ops with IMGID + string only
 *   (no inplace variants in the public API)
 *
 * X(op, fptr)
 *   op   — operation suffix
 *   fptr — function-pointer (for inplace dispatch)
 * ========================================================== */

#define MILK_BINARY_OPS_FULL(X) \
    X(fmod, Pfmod)              \
    X(pow, Ppow)                \
    X(add, Padd)                \
    X(sub, Psub)                \
    X(mult, Pmult)              \
    X(div, Pdiv)                \
    X(minv, Pminv)              \
    X(maxv, Pmaxv)              \
    X(testlt, Ptestlt)          \
    X(testmt, Ptestmt)

#define MILK_BINARY_OPS_NOIP(X) \
    X(teste, Pteste)            \
    X(testne, Ptestne)          \
    X(testle, Ptestle)          \
    X(testge, Ptestge)          \
    X(and, Pand)                \
    X(or, Por)

#define MILK_BINARY_OPS_ALL(X) \
    MILK_BINARY_OPS_FULL(X)    \
    MILK_BINARY_OPS_NOIP(X)


/* ----------------------------------------------------------
 * Generated prototypes — IMGID and string APIs (all ops)
 * ---------------------------------------------------------- */

#define MILK_DECLARE_BINARY(op, fptr)                                          \
    int arith_image_##op##_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout); \
    int arith_image_##op(const char *ID1_name, const char *ID2_name, const char *ID_out);

MILK_BINARY_OPS_ALL(MILK_DECLARE_BINARY)
#undef MILK_DECLARE_BINARY


/* ----------------------------------------------------------
 * Generated prototypes — inplace API (FULL ops only)
 * ---------------------------------------------------------- */

#define MILK_DECLARE_BINARY_INPLACE(op, fptr) \
    int arith_image_##op##_inplace(const char *ID1_name, const char *ID2_name);

MILK_BINARY_OPS_FULL(MILK_DECLARE_BINARY_INPLACE)
#undef MILK_DECLARE_BINARY_INPLACE

#endif /* IMAGE_ARITH_IM_IM_IM_H */

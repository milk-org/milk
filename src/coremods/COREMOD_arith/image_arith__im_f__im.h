/**
 * @file  image_arith__im_f__im.h
 * @brief Image + scalar arithmetic prototypes (img,float → img).
 *
 * All prototypes are generated from X-macro op tables.
 * Adding a new operation requires only one edit:
 * add a row to the appropriate table below.
 *
 * Three call surfaces per operation:
 *   arith_image_cst<op>_IMGID(imgin, f1, imgout)
 *   arith_image_cst<op>(name, f1, name_out)
 *   arith_image_cst<op>_inplace(name, f1)   [FULL ops only]
 */

#ifndef IMAGE_ARITH_IM_F_IM_H
#define IMAGE_ARITH_IM_F_IM_H

#include <libfps/IMGID.h>


/* ==========================================================
 * X-macro operation tables.
 *
 * MILK_CST_OPS_OPT_FULL  — optimized dispatch, with inplace
 * MILK_CST_OPS_FPTR_FULL — function-pointer dispatch, with inplace
 * MILK_CST_OPS_OPT_NOIP  — optimized dispatch, no inplace
 *
 * X(op, tag_or_fptr)
 * ========================================================== */

/** Optimized dispatch + inplace variants */
#define MILK_CST_OPS_OPT_FULL(X) \
    X(add, optimized)            \
    X(sub, optimized)            \
    X(mult, optimized)           \
    X(div, optimized)            \
    X(pow, optimized)            \
    X(testlt, optimized)         \
    X(testmt, optimized)

/** Function-pointer dispatch + inplace variants */
#define MILK_CST_OPS_FPTR_FULL(X) \
    X(fmod, Pfmod)                \
    X(subm, Psubm)                \
    X(div1, Pdiv1)                \
    X(maxv, Pmaxv)                \
    X(minv, Pminv)

/** Optimized dispatch, no inplace variant */
#define MILK_CST_OPS_OPT_NOIP(X) \
    X(teste, optimized)          \
    X(testne, optimized)         \
    X(testle, optimized)         \
    X(testge, optimized)         \
    X(and, optimized)            \
    X(or, optimized)

/** Expand all three tables for macros that apply to all */
#define MILK_CST_OPS_ALL(X)   \
    MILK_CST_OPS_OPT_FULL(X)  \
    MILK_CST_OPS_FPTR_FULL(X) \
    MILK_CST_OPS_OPT_NOIP(X)


/* ----------------------------------------------------------
 * Generated prototypes — IMGID and string APIs (all ops)
 * ---------------------------------------------------------- */

#define IMAGE_ARITH_DECLARE_CST(op, tag)                                     \
    int arith_image_cst##op##_IMGID(IMGID *imgin, double f1, IMGID *imgout); \
    int arith_image_cst##op(const char *ID_name, double f1, const char *ID_out);

MILK_CST_OPS_ALL(IMAGE_ARITH_DECLARE_CST)
#undef IMAGE_ARITH_DECLARE_CST


/* ----------------------------------------------------------
 * Generated prototypes — inplace API (FULL ops only)
 * ---------------------------------------------------------- */

#define IMAGE_ARITH_DECLARE_CST_INPLACE(op, tag) \
    int arith_image_cst##op##_inplace(const char *ID_name, double f1);

MILK_CST_OPS_OPT_FULL(IMAGE_ARITH_DECLARE_CST_INPLACE)
MILK_CST_OPS_FPTR_FULL(IMAGE_ARITH_DECLARE_CST_INPLACE)
#undef IMAGE_ARITH_DECLARE_CST_INPLACE

#endif /* IMAGE_ARITH_IM_F_IM_H */

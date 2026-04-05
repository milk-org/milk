/**
 * @file    image_arith__im_f__im.c
 * @brief   Image + scalar arithmetic (img,float → img)
 *
 * Applies a binary math function to every pixel of
 * an input image combined with a scalar constant,
 * producing an output image.
 *
 * All call surfaces are generated from X-macro tables:
 *
 *  - arith_image_cst<op>_IMGID(imgin, f1, imgout)
 *  - arith_image_cst<op>(name, f1, name_out)
 *  - arith_image_cst<op>_inplace(name, f1)
 */


#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__im_f__im.h"


/* ==========================================================
 * X-macro operation tables.
 *
 * Table groups used below:
 *   - CST_OPS_OPT_FULL  : optimized-dispatch ops with
 *                         inplace variants
 *   - CST_OPS_FPTR_FULL : function-pointer-dispatch ops with
 *                         inplace variants
 *   - CST_OPS_OPT_NOIP  : optimized-dispatch ops without
 *                         inplace variants
 *
 * All tables use the form:
 *   X(op, dispatch)
 *
 * For optimized-dispatch tables, the 2nd column is a
 * placeholder token kept for a consistent X-macro shape and
 * is not used by the optimized wrapper macros.
 *
 * For function-pointer tables, the 2nd column is the
 * function-pointer variable passed to the generic dispatcher.
 * ========================================================== */

/**
 * Ops dispatched to optimized (macro-stamped) IMGID
 * functions AND that have inplace variants.
 */
#define CST_OPS_OPT_FULL(X) \
    X(add,    optimized)     \
    X(sub,    optimized)     \
    X(mult,   optimized)     \
    X(div,    optimized)     \
    X(pow,    optimized)     \
    X(testlt, optimized)     \
    X(testmt, optimized)

/**
 * Ops dispatched via function-pointer AND that have
 * inplace variants.
 */
#define CST_OPS_FPTR_FULL(X) \
    X(fmod,   Pfmod)         \
    X(subm,   Psubm)         \
    X(div1,   Pdiv1)         \
    X(maxv,   Pmaxv)         \
    X(minv,   Pminv)

/**
 * Ops dispatched to optimized IMGID, NO inplace variants.
 */
#define CST_OPS_OPT_NOIP(X) \
    X(teste,  optimized)     \
    X(testne, optimized)     \
    X(testle, optimized)     \
    X(testge, optimized)     \
    X(and,    optimized)     \
    X(or,     optimized)


/* ----------------------------------------------------------
 * 1. IMGID wrappers — optimized dispatch
 * ---------------------------------------------------------- */

#define DEFINE_IMGID_OPT(op, tag) \
int arith_image_cst##op##_IMGID(  \
    IMGID *imgin,                 \
    double f1,                    \
    IMGID *imgout)                \
{                                 \
    return arith_image_cst##op##_optimized_IMGID( \
        imgin, f1, imgout);       \
}

CST_OPS_OPT_FULL(DEFINE_IMGID_OPT)
CST_OPS_OPT_NOIP(DEFINE_IMGID_OPT)
#undef DEFINE_IMGID_OPT


/* ----------------------------------------------------------
 * 1b. IMGID wrappers — function-pointer dispatch
 * ---------------------------------------------------------- */

#define DEFINE_IMGID_FPTR(op, fptr) \
int arith_image_cst##op##_IMGID(    \
    IMGID *imgin,                   \
    double f1,                      \
    IMGID *imgout)                  \
{                                   \
    return arith_image_function_1f_1_IMGID( \
        imgin, f1, imgout, &fptr);  \
}

CST_OPS_FPTR_FULL(DEFINE_IMGID_FPTR)
#undef DEFINE_IMGID_FPTR


/* ----------------------------------------------------------
 * 2. String-based wrapper macro
 * ---------------------------------------------------------- */

#define DEFINE_CST_STRING(op, tag) \
int arith_image_cst##op(                     \
    const char *ID_name,                     \
    double f1,                               \
    const char *ID_out)                      \
{                                            \
    IMGID imgin =                            \
        imgid_make_from_name(ID_name);       \
    IMGID imgout =                           \
        imgid_make_from_name(ID_out);        \
                                             \
    resolveIMGID(&imgin,                     \
        ERRMODE_ABORT, dcimg, dcnimg);       \
    resolveIMGID(&imgout,                    \
        ERRMODE_NULL, dcimg, dcnimg);        \
                                             \
    if (imgout.ID == -1) {                   \
        imgout.mdt->shared = dcshareddft;    \
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;  \
    }                                        \
                                             \
    int ret = arith_image_cst##op##_IMGID(   \
        &imgin, f1, &imgout);               \
                                             \
    if (imgout.ID == -1                      \
        && imgout.im != NULL) {              \
        RegisterIMGID(                       \
            &imgout, dcimg, dcnimg);         \
    }                                        \
    imgid_free(&imgin);                      \
    imgid_free(&imgout);                     \
    return ret;                              \
}

CST_OPS_OPT_FULL(DEFINE_CST_STRING)
CST_OPS_FPTR_FULL(DEFINE_CST_STRING)
CST_OPS_OPT_NOIP(DEFINE_CST_STRING)
#undef DEFINE_CST_STRING


/* ----------------------------------------------------------
 * 3. In-place wrappers  (name, f1 → name modified)
 *
 * Function-pointer table for inplace dispatch.
 * ---------------------------------------------------------- */

/**
 * Unified inplace function-pointer table.
 * Maps each operation to its math function pointer.
 */
#define CST_INPLACE_OPS(X) \
    X(fmod,   Pfmod)       \
    X(add,    Padd)        \
    X(sub,    Psub)        \
    X(subm,   Psubm)       \
    X(mult,   Pmult)       \
    X(div,    Pdiv)        \
    X(div1,   Pdiv1)       \
    X(pow,    Ppow)        \
    X(maxv,   Pmaxv)       \
    X(minv,   Pminv)       \
    X(testlt, Ptestlt)     \
    X(testmt, Ptestmt)

#define DEFINE_CST_INPLACE(op, fptr) \
int arith_image_cst##op##_inplace(   \
    const char *ID_name,             \
    double f1)                       \
{                                    \
    arith_image_function_1f_1_inplace( \
        ID_name, f1, &fptr);         \
    return 0;                        \
}

CST_INPLACE_OPS(DEFINE_CST_INPLACE)
#undef DEFINE_CST_INPLACE

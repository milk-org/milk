/**
 * @file    image_arith__im_f__im.c
 * @brief   arith functions
 *
 * input : image, float
 * output: image
 *
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

#define ARITH_IMAGE_CST_WRAPPER(name) \
int arith_image_cst##name(const char *ID_name, double f1, const char *ID_out) \
{ \
    IMGID imgin  = imgid_make_from_name(ID_name); \
    IMGID imgout = imgid_make_from_name(ID_out); \
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg); \
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg); \
    if (imgout.ID == -1) { \
        imgout.mdt->shared = dcshareddft; \
        imgout.mdt->NBkw = NB_KEYWNODE_MAX; \
    } \
    int ret = arith_image_cst##name##_IMGID(&imgin, f1, &imgout); \
    if (imgout.ID == -1 && imgout.im != NULL) { \
        RegisterIMGID(&imgout, dcimg, dcnimg); \
    } \
    imgid_free(&imgin); \
    imgid_free(&imgout); \
    return ret; \
}



int arith_image_cstfmod_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pfmod);
}

ARITH_IMAGE_CST_WRAPPER(fmod)

int arith_image_cstadd_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstadd_optimized_IMGID(imgin, f1, imgout);
}

ARITH_IMAGE_CST_WRAPPER(add)

int arith_image_cstsub_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstsub_optimized_IMGID(imgin, f1, imgout);
}

ARITH_IMAGE_CST_WRAPPER(sub)

int arith_image_cstsubm_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Psubm);
}

ARITH_IMAGE_CST_WRAPPER(subm)

int arith_image_cstmult_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstmult_optimized_IMGID(imgin, f1, imgout);
}

ARITH_IMAGE_CST_WRAPPER(mult)

int arith_image_cstdiv_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstdiv_optimized_IMGID(imgin, f1, imgout);
}

ARITH_IMAGE_CST_WRAPPER(div)

int arith_image_cstdiv1_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pdiv1);
}

ARITH_IMAGE_CST_WRAPPER(div1)

int arith_image_cstpow_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstpow_optimized_IMGID(imgin, f1, imgout);
}

ARITH_IMAGE_CST_WRAPPER(pow)

int arith_image_cstmaxv_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pmaxv);
}

ARITH_IMAGE_CST_WRAPPER(maxv)

int arith_image_cstminv_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pminv);
}

ARITH_IMAGE_CST_WRAPPER(minv)

int arith_image_csttestlt_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ptestlt);
}

ARITH_IMAGE_CST_WRAPPER(testlt)

int arith_image_csttestmt_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ptestmt);
}

ARITH_IMAGE_CST_WRAPPER(testmt)

int arith_image_cstfmod_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pfmod);
    return (0);
}

int arith_image_cstadd_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Padd);
    return (0);
}

int arith_image_cstsub_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Psub);
    return (0);
}

int arith_image_cstmult_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pmult);
    return (0);
}

int arith_image_cstdiv_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pdiv);
    return (0);
}

int arith_image_cstdiv1_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pdiv1);
    return (0);
}

int arith_image_cstpow_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Ppow);
    return (0);
}

int arith_image_cstmaxv_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pmaxv);
    return (0);
}

int arith_image_cstminv_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pminv);
    return (0);
}

int arith_image_csttestlt_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Ptestlt);
    return (0);
}

int arith_image_csttestmt_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Ptestmt);
    return (0);
}

int arith_image_cstfmod_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pfmod);
    return (0);
}

int arith_image_cstadd_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Padd);
    return (0);
}

int arith_image_cstsub_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Psub);
    return (0);
}

int arith_image_cstmult_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pmult);
    return (0);
}

int arith_image_cstdiv_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pdiv);
    return (0);
}

int arith_image_cstdiv1_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pdiv1);
    return (0);
}

int arith_image_cstpow_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Ppow);
    return (0);
}

int arith_image_cstmaxv_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pmaxv);
    return (0);
}

int arith_image_cstminv_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pminv);
    return (0);
}

int arith_image_csttestlt_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Ptestlt);
    return (0);
}

int arith_image_csttestmt_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Ptestmt);
    return (0);
}

int arith_image_cstteste_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pteste);
}
ARITH_IMAGE_CST_WRAPPER(teste)

int arith_image_csttestne_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ptestne);
}
ARITH_IMAGE_CST_WRAPPER(testne)

int arith_image_csttestle_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ptestle);
}
ARITH_IMAGE_CST_WRAPPER(testle)

int arith_image_csttestge_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ptestge);
}
ARITH_IMAGE_CST_WRAPPER(testge)

int arith_image_cstand_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pand);
}
ARITH_IMAGE_CST_WRAPPER(and)

int arith_image_cstor_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Por);
}
ARITH_IMAGE_CST_WRAPPER(or)

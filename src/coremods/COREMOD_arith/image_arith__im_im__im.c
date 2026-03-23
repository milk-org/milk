/**
 * @file    image_arith__im_im__im.c
 * @brief   arith functions
 *
 * input : image, image
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
#include "image_arith__im_im__im.h"

int arith_image_fmod_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_fmod_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_fmod(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_fmod_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_pow_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_pow_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_pow(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_pow_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_add_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_add_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_add(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_add_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_sub_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_sub_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_sub(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_sub_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_mult_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_mult_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_mult(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_mult_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_div_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_div_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_div(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_div_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_minv_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_minv_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_minv(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_minv_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_maxv_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_maxv_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_maxv(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_maxv_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_testlt_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_testlt_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_testlt(const char *ID1_name,
                       const char *ID2_name,
                       const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_testlt_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_testmt_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_testmt_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_testmt(const char *ID1_name,
                       const char *ID2_name,
                       const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);

    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);

    if (imgout.ID == -1) {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    }

    int ret = arith_image_testmt_IMGID(&imgin1, &imgin2, &imgout);

    if (imgout.ID == -1 && imgout.im != NULL) {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin1);
    imgid_free(&imgin2);
    imgid_free(&imgout);
    return ret;
}

int arith_image_fmod_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pfmod);
    return (0);
}

int arith_image_pow_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Ppow);
    return (0);
}

int arith_image_add_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Padd);
    return (0);
}

int arith_image_sub_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Psub);
    return (0);
}

int arith_image_mult_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pmult);
    return (0);
}

int arith_image_div_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pdiv);
    return (0);
}

int arith_image_minv_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pminv);
    return (0);
}

int arith_image_maxv_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pmaxv);
    return (0);
}

int arith_image_testlt_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Ptestlt);
    return (0);
}

int arith_image_testmt_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Ptestmt);
    return (0);
}

int arith_image_fmod_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pfmod);
    return (0);
}

int arith_image_pow_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Ppow);
    return (0);
}

int arith_image_add_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Padd);
    return (0);
}

int arith_image_sub_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Psub);
    return (0);
}

int arith_image_mult_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pmult);
    return (0);
}
int arith_image_div_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pdiv);
    return (0);
}

int arith_image_minv_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pminv);
    return (0);
}

int arith_image_maxv_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pmaxv);
    return (0);
}

int arith_image_testlt_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Ptestlt);
    return (0);
}

int arith_image_testmt_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Ptestmt);
    return (0);
}

int arith_image_teste_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_teste_optimized_IMGID(imgin1, imgin2, imgout);
}
int arith_image_teste(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);
    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgout.ID == -1) { imgout.mdt->shared = dcshareddft; imgout.mdt->NBkw = NB_KEYWNODE_MAX; }
    int ret = arith_image_teste_IMGID(&imgin1, &imgin2, &imgout);
    if (imgout.ID == -1 && imgout.im != NULL) { RegisterIMGID(&imgout, dcimg, dcnimg); }
    imgid_free(&imgin1); imgid_free(&imgin2); imgid_free(&imgout);
    return ret;
}

int arith_image_testne_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_testne_optimized_IMGID(imgin1, imgin2, imgout);
}
int arith_image_testne(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);
    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgout.ID == -1) { imgout.mdt->shared = dcshareddft; imgout.mdt->NBkw = NB_KEYWNODE_MAX; }
    int ret = arith_image_testne_IMGID(&imgin1, &imgin2, &imgout);
    if (imgout.ID == -1 && imgout.im != NULL) { RegisterIMGID(&imgout, dcimg, dcnimg); }
    imgid_free(&imgin1); imgid_free(&imgin2); imgid_free(&imgout);
    return ret;
}

int arith_image_testle_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_testle_optimized_IMGID(imgin1, imgin2, imgout);
}
int arith_image_testle(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);
    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgout.ID == -1) { imgout.mdt->shared = dcshareddft; imgout.mdt->NBkw = NB_KEYWNODE_MAX; }
    int ret = arith_image_testle_IMGID(&imgin1, &imgin2, &imgout);
    if (imgout.ID == -1 && imgout.im != NULL) { RegisterIMGID(&imgout, dcimg, dcnimg); }
    imgid_free(&imgin1); imgid_free(&imgin2); imgid_free(&imgout);
    return ret;
}

int arith_image_testge_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_testge_optimized_IMGID(imgin1, imgin2, imgout);
}
int arith_image_testge(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);
    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgout.ID == -1) { imgout.mdt->shared = dcshareddft; imgout.mdt->NBkw = NB_KEYWNODE_MAX; }
    int ret = arith_image_testge_IMGID(&imgin1, &imgin2, &imgout);
    if (imgout.ID == -1 && imgout.im != NULL) { RegisterIMGID(&imgout, dcimg, dcnimg); }
    imgid_free(&imgin1); imgid_free(&imgin2); imgid_free(&imgout);
    return ret;
}

int arith_image_and_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_and_optimized_IMGID(imgin1, imgin2, imgout);
}
int arith_image_and(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);
    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgout.ID == -1) { imgout.mdt->shared = dcshareddft; imgout.mdt->NBkw = NB_KEYWNODE_MAX; }
    int ret = arith_image_and_IMGID(&imgin1, &imgin2, &imgout);
    if (imgout.ID == -1 && imgout.im != NULL) { RegisterIMGID(&imgout, dcimg, dcnimg); }
    imgid_free(&imgin1); imgid_free(&imgin2); imgid_free(&imgout);
    return ret;
}

int arith_image_or_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_or_optimized_IMGID(imgin1, imgin2, imgout);
}
int arith_image_or(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = imgid_make_from_name(ID1_name);
    IMGID imgin2 = imgid_make_from_name(ID2_name);
    IMGID imgout = imgid_make_from_name(ID_out);
    resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgout.ID == -1) { imgout.mdt->shared = dcshareddft; imgout.mdt->NBkw = NB_KEYWNODE_MAX; }
    int ret = arith_image_or_IMGID(&imgin1, &imgin2, &imgout);
    if (imgout.ID == -1 && imgout.im != NULL) { RegisterIMGID(&imgout, dcimg, dcnimg); }
    imgid_free(&imgin1); imgid_free(&imgin2); imgid_free(&imgout);
    return ret;
}

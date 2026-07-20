// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im_im__im.c
 * @brief   arith functions
 *
 * input : image, image
 * output: image
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "image_arith__im_im__im.h"
#include "imfunctions.h"
#include "mathfuncs.h"

int arith_image_fmod_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &Pfmod);
}

int arith_image_fmod(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_fmod_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_pow_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &Ppow);
}

int arith_image_pow(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_pow_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_add_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_add_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_add(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_add_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_sub_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_sub_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_sub(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_sub_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_mult_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_mult_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_mult(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_mult_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_div_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_div_optimized_IMGID(imgin1, imgin2, imgout);
}

int arith_image_div(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_div_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_minv_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &Pminv);
}

int arith_image_minv(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_minv_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_maxv_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &Pmaxv);
}

int arith_image_maxv(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_maxv_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_testlt_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &Ptestlt);
}

int arith_image_testlt(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_testlt_IMGID(&imgin1, &imgin2, &imgout);
}

int arith_image_testmt_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)
{
    return arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &Ptestmt);
}

int arith_image_testmt(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    IMGID imgin1 = mkIMGID_from_name(ID1_name);
    IMGID imgin2 = mkIMGID_from_name(ID2_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_testmt_IMGID(&imgin1, &imgin2, &imgout);
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

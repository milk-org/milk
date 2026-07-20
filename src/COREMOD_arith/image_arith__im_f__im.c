// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im_f__im.c
 * @brief   arith functions
 *
 * input : image, float
 * output: image
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__im_f__im.h"


int arith_image_cstfmod_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pfmod);
}

int arith_image_cstfmod(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstfmod_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstadd_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstadd_optimized_IMGID(imgin, f1, imgout);
}

int arith_image_cstadd(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstadd_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstsub_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstsub_optimized_IMGID(imgin, f1, imgout);
}

int arith_image_cstsub(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstsub_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstsubm_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Psubm);
}

int arith_image_cstsubm(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstsubm_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstmult_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstmult_optimized_IMGID(imgin, f1, imgout);
}

int arith_image_cstmult(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstmult_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstdiv_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_cstdiv_optimized_IMGID(imgin, f1, imgout);
}

int arith_image_cstdiv(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstdiv_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstdiv1_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pdiv1);
}

int arith_image_cstdiv1(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstdiv1_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstpow_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ppow);
}

int arith_image_cstpow(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstpow_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstmaxv_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pmaxv);
}

int arith_image_cstmaxv(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstmaxv_IMGID(&imgin, f1, &imgout);
}

int arith_image_cstminv_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Pminv);
}

int arith_image_cstminv(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_cstminv_IMGID(&imgin, f1, &imgout);
}

int arith_image_csttestlt_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ptestlt);
}

int arith_image_csttestlt(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_csttestlt_IMGID(&imgin, f1, &imgout);
}

int arith_image_csttestmt_IMGID(IMGID *imgin, double f1, IMGID *imgout)
{
    return arith_image_function_1f_1_IMGID(imgin, f1, imgout, &Ptestmt);
}

int arith_image_csttestmt(const char *ID_name, double f1, const char *ID_out)
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_csttestmt_IMGID(&imgin, f1, &imgout);
}

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

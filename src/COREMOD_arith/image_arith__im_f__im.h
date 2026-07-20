// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im_f__im.h
 *
 */

int arith_image_cstfmod(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstfmod_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstadd(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstadd_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstsub(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstsub_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstsubm(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstsubm_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstmult(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstmult_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstdiv(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstdiv_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstdiv1(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstdiv1_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstpow(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstpow_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstmaxv(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstmaxv_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstminv(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstminv_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_csttestlt(const char *ID_name, double f1, const char *ID_out);
int arith_image_csttestlt_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_csttestmt(const char *ID_name, double f1, const char *ID_out);
int arith_image_csttestmt_IMGID(IMGID *imgin, double f1, IMGID *imgout);

int arith_image_cstfmod_inplace(const char *ID_name, double f1);

int arith_image_cstadd_inplace(const char *ID_name, double f1);

int arith_image_cstsub_inplace(const char *ID_name, double f1);

int arith_image_cstmult_inplace(const char *ID_name, double f1);

int arith_image_cstdiv_inplace(const char *ID_name, double f1);

int arith_image_cstdiv1_inplace(const char *ID_name, double f1);

int arith_image_cstpow_inplace(const char *ID_name, double f1);

int arith_image_cstmaxv_inplace(const char *ID_name, double f1);

int arith_image_cstminv_inplace(const char *ID_name, double f1);

int arith_image_csttestlt_inplace(const char *ID_name, double f1);

int arith_image_csttestmt_inplace(const char *ID_name, double f1);

int arith_image_cstfmod_inplace_byID(long ID, double f1);

int arith_image_cstadd_inplace_byID(long ID, double f1);

int arith_image_cstsub_inplace_byID(long ID, double f1);

int arith_image_cstmult_inplace_byID(long ID, double f1);

int arith_image_cstdiv_inplace_byID(long ID, double f1);

int arith_image_cstdiv1_inplace_byID(long ID, double f1);

int arith_image_cstpow_inplace_byID(long ID, double f1);

int arith_image_cstmaxv_inplace_byID(long ID, double f1);

int arith_image_cstminv_inplace_byID(long ID, double f1);

int arith_image_csttestlt_inplace_byID(long ID, double f1);

int arith_image_csttestmt_inplace_byID(long ID, double f1);

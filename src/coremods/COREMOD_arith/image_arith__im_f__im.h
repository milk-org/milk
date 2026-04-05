/**
 * @file image_arith__im_f__im.h
 * @brief Image arith  im f  im module
 */

/**
 * @file    image_arith__im_f__im.h
 *
 */

#include <libfps/IMGID.h>

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

int arith_image_cstteste(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstteste_IMGID(IMGID *imgin, double f1, IMGID *imgout);
int arith_image_csttestne(const char *ID_name, double f1, const char *ID_out);
int arith_image_csttestne_IMGID(IMGID *imgin, double f1, IMGID *imgout);
int arith_image_csttestle(const char *ID_name, double f1, const char *ID_out);
int arith_image_csttestle_IMGID(IMGID *imgin, double f1, IMGID *imgout);
int arith_image_csttestge(const char *ID_name, double f1, const char *ID_out);
int arith_image_csttestge_IMGID(IMGID *imgin, double f1, IMGID *imgout);
int arith_image_cstand(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstand_IMGID(IMGID *imgin, double f1, IMGID *imgout);
int arith_image_cstor(const char *ID_name, double f1, const char *ID_out);
int arith_image_cstor_IMGID(IMGID *imgin, double f1, IMGID *imgout);

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























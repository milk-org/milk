/**
 * @file imfunctions.h
 * @brief Functions for bison / flex
 */

/**
 * @file    imfunctions.c
 *
 *
 */

#include <libfps/IMGID.h>

/* Functions for bison / flex    */

errno_t arith_image_function_im_im__d_d(
    const char *ID_name,
    const char *ID_out,
    double (*pt2function)(double));

errno_t arith_image_cstadd_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_cstsub_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_cstmult_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_cstdiv_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);

errno_t arith_image_add_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);

errno_t arith_image_sub_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);

errno_t arith_image_mult_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);

errno_t arith_image_div_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);

errno_t arith_image_pow_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_fmod_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_minv_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_maxv_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_cstpow_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);

errno_t arith_image_acos_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_asin_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_atan_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_ceil_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_cos_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_cosh_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_exp_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_fabs_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_floor_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_ln_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_log_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_sqrt_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_sin_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_sinh_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_tan_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_tanh_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);
errno_t arith_image_positive_optimized_IMGID(
    IMGID *imgin,
    IMGID *imgout);

errno_t arith_image_testlt_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_testmt_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_teste_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_testne_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_testle_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_testge_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_and_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);
errno_t arith_image_or_optimized_IMGID(
    IMGID *imgin1,
    IMGID *imgin2,
    IMGID *imgout);

errno_t arith_image_csttestlt_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_csttestmt_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_cstteste_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_csttestne_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_csttestle_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_csttestge_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_cstand_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);
errno_t arith_image_cstor_optimized_IMGID(
    IMGID  *imgin,
    double f1,
    IMGID  *imgout);



errno_t arith_image_function_im_im__d_d_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    double (*pt2function)(double));

errno_t arith_image_function_imd_im__dd_d(
    const char                    *ID_name,
    double                        v0,
    const char                    *ID_out,
    double (*pt2function)(double, double));

errno_t arith_image_function_imd_im__dd_d_IMGID(
    IMGID                         *imgin,
    double                        v0,
    IMGID                         *imgout,
    double (*pt2function)(double, double));

errno_t arith_image_function_imdd_im__ddd_d(
    const char                    *ID_name,
    double                        v0,
    double                        v1,
    const char                    *ID_out,
    double (*pt2function)(double, double, double));

errno_t arith_image_function_imdd_im__ddd_d_IMGID(
    IMGID                         *imgin,
    double                        v0,
    double                        v1,
    IMGID                         *imgout,
    double (*pt2function)(double, double, double));



errno_t arith_image_function_1_1(
    const char *ID_name,
    const char *ID_out,
    double (*pt2function)(double));

errno_t arith_image_function_1_1_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    double (*pt2function)(double));

// imagein -> imagein (in place)
errno_t arith_image_function_1_1_inplace_IMGID(
    IMGID *imgin,
    double (*pt2function)(double));

// imagein -> imagein (in place)
errno_t arith_image_function_1_1_inplace(
    const char *ID_name,
    double (*pt2function)(double));

errno_t arith_image_function_2_1(
    const char                    *ID_name1,
    const char                    *ID_name2,
    const char                    *ID_out,
    double (*pt2function)(double, double));

errno_t arith_image_function_2_1_IMGID(
    IMGID                         *imgin1,
    IMGID                         *imgin2,
    IMGID                         *imgout,
    double (*pt2function)(double, double));

errno_t arith_img_function_2_1(
    IMGID                         inimg1,
    IMGID                         inimg2,
    IMGID                         *outimg,
    double (*pt2function)(double, double));

errno_t arith_image_function_2_1_inplace(
    const char                    *ID_name1,
    const char                    *ID_name2,
    double (*pt2function)(double, double));

errno_t arith_image_function_CF_CF__CF(
    const char                                 *ID_name1,
    const char                                 *ID_name2,
    const char                                 *ID_out,
    complex_float(*pt2function)(complex_float, complex_float));

errno_t arith_image_function_CD_CD__CD(
    const char                                   *ID_name1,
    const char                                   *ID_name2,
    const char                                   *ID_out,
    complex_double(*pt2function)(complex_double, complex_double));

int arith_image_function_1f_1(
    const char                    *ID_name,
    double                        f1,
    const char                    *ID_out,
    double (*pt2function)(double, double));

int arith_image_function_1f_1_IMGID(
    IMGID                         *imgin,
    double                        f1,
    IMGID                         *imgout,
    double (*pt2function)(double, double));

int arith_image_function_1f_1_inplace_IMGID(
    IMGID                         *imgin,
    double                        f1,
    double (*pt2function)(double, double));

int arith_image_function_1f_1_inplace(
    const char                    *ID_name,
    double                        f1,
    double (*pt2function)(double, double));

int arith_image_function_1ff_1(
    const char                    *ID_name,
    double                        f1,
    double                        f2,
    const char                    *ID_out,
    double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_IMGID(
    IMGID                         *imgin,
    double                        f1,
    double                        f2,
    IMGID                         *imgout,
    double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_inplace(
    const char                    *ID_name,
    double                        f1,
    double                        f2,
    double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_inplace_IMGID(
    IMGID                         *imgin,
    double                        f1,
    double                        f2,
    double (*pt2function)(double, double, double));

/**
 * @file    image_arith__im__im.h
 *
 */

double Ptrunc(double a, double b, double c);

/* ------------------------------------------------------------------------- */
/* predefined functions    image, image  -> image                                                    */
/* ------------------------------------------------------------------------- */

int arith_image_fmod_byID(long ID1, long ID2, long IDout);
int arith_image_pow_byID(long ID1, long ID2, const char *IDout);
int arith_image_add_byID(long ID1, long ID2, long IDout);
int arith_image_sub_byID(long ID1, long ID2, long IDout);
int arith_image_mult_byID(long ID1, long ID2, long IDout);
int arith_image_div_byID(long ID1, long ID2, long IDout);
int arith_image_minv_byID(long ID1, long ID2, long IDout);
int arith_image_maxv_byID(long ID1, long ID2, long IDout);

int arith_image_fmod(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out);
int arith_image_pow(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out);
int arith_image_add(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out);
int arith_image_sub(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out);
int arith_image_mult(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out);
int arith_image_div(const char *ID1_name,
                    const char *ID2_name,
                    const char *ID_out);
int arith_image_minv(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out);
int arith_image_maxv(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out);
int arith_image_testlt(const char *ID1_name,
                       const char *ID2_name,
                       const char *ID_out);
int arith_image_testmt(const char *ID1_name,
                       const char *ID2_name,
                       const char *ID_out);

int arith_image_fmod_inplace_byID(long ID1, long ID2);
int arith_image_pow_inplace_byID(long ID1, long ID2);
int arith_image_add_inplace_byID(long ID1, long ID2);
int arith_image_sub_inplace_byID(long ID1, long ID2);
int arith_image_mult_inplace_byID(long ID1, long ID2);
int arith_image_div_inplace_byID(long ID1, long ID2);
int arith_image_minv_inplace_byID(long ID1, long ID2);
int arith_image_maxv_inplace_byID(long ID1, long ID2);

int arith_image_fmod_inplace(const char *ID1_name,
                             const char *ID2_name); // ID1 is output
int arith_image_pow_inplace(const char *ID1_name, const char *ID2_name);
int arith_image_add_inplace(const char *ID1_name, const char *ID2_name);
int arith_image_sub_inplace(const char *ID1_name, const char *ID2_name);
int arith_image_mult_inplace(const char *ID1_name, const char *ID2_name);
int arith_image_div_inplace(const char *ID1_name, const char *ID2_name);
int arith_image_minv_inplace(const char *ID1_name, const char *ID2_name);
int arith_image_maxv_inplace(const char *ID1_name, const char *ID2_name);

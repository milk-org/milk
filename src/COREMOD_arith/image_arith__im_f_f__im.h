/**
 * @file    image_arith__im_f_f__im.h
 */

errno_t image_arith__im_f_f__im_addCLIcmd();

int arith_image_trunc_byID(long ID, double f1, double f2, long IDout);

int arith_image_trunc_inplace_byID(long IDname, double f1, double f2);

int arith_image_trunc(const char *ID_name,
                      double      f1,
                      double      f2,
                      const char *ID_out);

int arith_image_trunc_inplace(const char *ID_name, double f1, double f2);

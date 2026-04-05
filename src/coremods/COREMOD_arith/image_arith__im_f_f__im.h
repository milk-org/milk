/**
 * @file image_arith__im_f_f__im.h
 * @brief Image arith  im f f  im module
 */

/**
 * @file    image_arith__im_f_f__im.h
 */

#include <libfps/IMGID.h>

errno_t image_arith__im_f_f__im_addCLIcmd();





int arith_image_trunc(const char *ID_name,
                      double      f1,
                      double      f2,
                      const char *ID_out);

int arith_image_trunc_IMGID(IMGID  *imgin,
                            double  f1,
                            double  f2,
                            IMGID  *imgout);

int arith_image_trunc_inplace(const char *ID_name, double f1, double f2);

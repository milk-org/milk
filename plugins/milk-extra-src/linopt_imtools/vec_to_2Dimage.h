#ifndef LINOPT_IMTOOLS__VEC_TO_2DIMAGE_H
#define LINOPT_IMTOOLS__VEC_TO_2DIMAGE_H

errno_t CLIADDCMD_linopt_imtools__vec_to_2DImage();

errno_t linopt_imtools_vec_to_2DImage(const char *IDvec_name,
                                      const char *IDpixindex_name,
                                      const char *IDpixmult_name,
                                      const char *ID_name,
                                      long        xsize,
                                      long        ysize,
                                      imageID    *outID);

#endif

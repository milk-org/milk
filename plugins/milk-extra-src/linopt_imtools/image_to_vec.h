#ifndef LINOPT_IMTOOLS__IMAGE_TO_VEC_H
#define LINOPT_IMTOOLS__IMAGE_TO_VEC_H

errno_t CLIADDCMD_linopt_imtools__image_to_vec();

errno_t linopt_imtools_image_to_vec(const char *ID_name,
                                    const char *IDpixindex_name,
                                    const char *IDpixmult_name,
                                    const char *IDvec_name,
                                    imageID    *outID);

#endif

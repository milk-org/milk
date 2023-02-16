#ifndef LINOPT_IMTOOLS__IMAGE_FITMODES_H
#define LINOPT_IMTOOLS__IMAGE_FITMODES_H

errno_t CLIADDCMD_linopt_imtools__image_fitModes();

errno_t linopt_imtools_image_fitModes(const char *ID_name,
                                      const char *IDmodes_name,
                                      const char *IDmask_name,
                                      double      SVDeps,
                                      const char *IDcoeff_name,
                                      int         reuse,
                                      imageID    *outIDcoeff);

#endif

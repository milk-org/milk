#ifndef IMAGE_FORMAT_MKBMPIMAGE_H
#define IMAGE_FORMAT_MKBMPIMAGE_H

errno_t CLIADDCMD_image_format__mkBMPimage();

errno_t image_writeBMP(const char *__restrict IDnameR,
                       const char *__restrict IDnameG,
                       const char *__restrict IDnameB,
                       char *__restrict outname);

#endif

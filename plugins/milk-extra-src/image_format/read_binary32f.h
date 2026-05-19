/**
 * @file read_binary32f.h
 * @brief Read 32-bit float RAW image
 */

errno_t CLIADDCMD_image_format__read_binary32f();

imageID IMAGE_FORMAT_read_binary32f(
    const char *__restrict fname,
    long xsize,
    long ysize,
    const char *__restrict IDname);

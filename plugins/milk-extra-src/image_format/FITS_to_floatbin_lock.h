/**
 * @file FITS_to_floatbin_lock.h
 * @brief Write float binary with file locking
 */

errno_t CLIADDCMD_image_format__floatbin_lock();

imageID IMAGE_FORMAT_FITS_to_floatbin_lock(const char *__restrict IDname,
                                           const char *__restrict fname);

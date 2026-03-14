/**
 * @file FITS_to_ushortintbin_lock.h
 * @brief Write ushort binary with file locking
 */

errno_t
CLIADDCMD_image_format__ushortintbin_lock();

imageID IMAGE_FORMAT_FITS_to_ushortintbin_lock(
    const char *__restrict IDname,
    const char *__restrict fname);

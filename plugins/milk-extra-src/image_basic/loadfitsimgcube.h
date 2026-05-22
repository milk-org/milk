/**
 * @file loadfitsimgcube.h
 * @brief Loadfitsimgcube module
 */

/** @file loadfitsimgcube.h
 */

errno_t __attribute__((cold)) CLIADDCMD_image_basic__loadfitsimgcube();

long load_fitsimages_cube(const char *__restrict strfilter, const char *__restrict ID_out_name);

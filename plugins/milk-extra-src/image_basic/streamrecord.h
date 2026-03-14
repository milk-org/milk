/**
 * @file streamrecord.h
 * @brief Streamrecord module
 */

/** @file streamrecord.h
 */

errno_t __attribute__((cold)) CLIADDCMD_image_basic__streamrecord();

imageID IMAGE_BASIC_streamrecord(const char *__restrict streamname,
                                 long NBframes,
                                 const char *__restrict IDname);

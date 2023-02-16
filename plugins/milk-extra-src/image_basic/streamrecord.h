/** @file streamrecord.h
 */

errno_t __attribute__((cold)) streamrecord_addCLIcmd();

imageID IMAGE_BASIC_streamrecord(const char *__restrict streamname,
                                 long NBframes,
                                 const char *__restrict IDname);

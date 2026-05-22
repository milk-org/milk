/**
 * @file stream_pixmapdecode.h
 * @brief Decode image stream via pixel map
 */

errno_t CLIADDCMD_COREMOD_memory__stream_pixmapdecode();

imageID COREMOD_MEMORY_PixMapDecode_U(const char *inputstream_name,
                                      uint32_t    xsizeim,
                                      uint32_t    ysizeim,
                                      const char *NBpix_fname,
                                      const char *IDmap_name,
                                      const char *IDout_name,
                                      const char *IDout_pixslice_fname,
                                      uint32_t    reverse);

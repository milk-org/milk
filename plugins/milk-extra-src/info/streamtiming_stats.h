/**
 * @file streamtiming_stats.h
 * @brief Streamtiming stats module
 */

/** @file streamtiming_stats.h
 */


errno_t info_image_streamtiming_stats(imageID ID,
                                      int     sem,
                                      long    NBsamplesmax,
                                      float   samplestimeout,
                                      int     buffinit);

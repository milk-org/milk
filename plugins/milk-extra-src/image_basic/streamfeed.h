/**
 * @file streamfeed.h
 * @brief Streamfeed module
 */

/** @file streamfeed.h
 */

errno_t __attribute__((cold)) CLIADDCMD_image_basic__streamfeed();

long IMAGE_BASIC_streamfeed(const char *__restrict IDname,
                            const char *__restrict streamname,
                            float frequ);

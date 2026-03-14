/**
 * @file indexmap.h
 * @brief Indexmap module
 */

/** @file indexmap.h
 */

errno_t __attribute__((cold)) CLIADDCMD_image_basic__indexmap();

imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name);

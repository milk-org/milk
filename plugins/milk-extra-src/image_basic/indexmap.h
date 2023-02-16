/** @file indexmap.h
 */

errno_t __attribute__((cold)) indexmap_addCLIcmd();

imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name);

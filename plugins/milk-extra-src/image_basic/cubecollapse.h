/** @file cubecollapse.h
 */

errno_t __attribute__((cold)) cubecollapse_addCLIcmd();

imageID cube_collapse(const char *__restrict ID_in_name,
                      const char *__restrict ID_out_name);

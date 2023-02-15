/** @file imrotate.h
 */

errno_t imrotate_addCLIcmd();

imageID basic_rotate(const char *__restrict ID_name,
                     const char *__restrict IDout_name,
                     float angle);

imageID basic_rotate90(const char *__restrict ID_name,
                       const char *__restrict ID_out_name);

imageID basic_rotate_int(const char *__restrict ID_name,
                         const char *__restrict ID_out_name,
                         long nbstep);

imageID basic_rotate2(const char *__restrict ID_name_in,
                      const char *__restrict ID_name_out,
                      float angle);

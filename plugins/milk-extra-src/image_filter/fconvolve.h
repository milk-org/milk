/** @file fconvolve.h
 *
 */

errno_t fconvolve_addCLIcmd();

imageID fconvolve(const char *__restrict name_in,
                  const char *__restrict name_ke,
                  const char *__restrict name_out);

imageID fconvolve_padd(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       long paddsize,
                       const char *__restrict name_out);

imageID fconvolve_1(const char *__restrict name_in,
                    const char *__restrict kefft,
                    const char *__restrict name_out);

imageID fconvolveblock(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       const char *__restrict name_out,
                       long blocksize);

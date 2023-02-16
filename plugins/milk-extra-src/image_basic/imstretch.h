/** @file imstretch.h
 */

imageID basic_stretch(const char *__restrict name_in,
                      const char *__restrict name_out,
                      float coeff,
                      long  Xcenter,
                      long  Ycenter);

imageID basic_stretch_range(const char *__restrict name_in,
                            const char *__restrict name_out,
                            float coeff1,
                            float coeff2,
                            long  Xcenter,
                            long  Ycenter,
                            long  NBstep,
                            float ApoCoeff);

imageID basic_stretchc(const char *__restrict name_in,
                       const char *__restrict name_out,
                       float coeff);

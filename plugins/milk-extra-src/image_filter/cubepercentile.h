/** @file cubepercentile.h
 */

imageID filter_CubePercentile(const char *__restrict IDcin_name,
                              float perc,
                              const char *__restrict IDout_name);

imageID filter_CubePercentileLimit(const char *__restrict IDcin_name,
                                   float perc,
                                   float limit,
                                   const char *__restrict IDout_name);

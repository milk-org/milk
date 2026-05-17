#ifndef OPTIMIZELYOT_H
#define OPTIMIZELYOT_H

errno_t optimizeLyotStop(const char *__restrict__ IDamp_name,
                         const char *__restrict__ IDpha_name,
                         const char *__restrict__ Dincohc_name,
                         float   zmin,
                         float   zmax,
                         double  throughput,
                         long    NBz,
                         long    NBmasks,
                         double *outratioval);

#endif

#ifndef PIAACMCSIMUL_LYOTSTOP_MKLYOTMASK_H
#define PIAACMCSIMUL_LYOTSTOP_MKLYOTMASK_H

errno_t mkLyotMask(const char *__restrict__ IDincoh_name,
                   const char *__restrict__ IDmc_name,
                   const char *__restrict__ IDzone_name,
                   double throughput,
                   const char *__restrict__ IDout_name,
                   imageID *outID);

#endif

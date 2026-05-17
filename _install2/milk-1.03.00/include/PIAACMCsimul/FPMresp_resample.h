#ifndef PIAACMCSIMUL_FOCALPLANEMASK_FPMRESP_RESAMPLE_H
#define PIAACMCSIMUL_FOCALPLANEMASK_FPMRESP_RESAMPLE_H

errno_t CLIADDCMD_PIAACMCsimul__FPMresp_resample();

errno_t PIAACMC_FPMresp_resample(const char *__restrict__ FPMresp_in_name,
                                 const char *__restrict__ FPMresp_out_name,
                                 long     NBlambda,
                                 long     PTstep,
                                 imageID *outID);

#endif

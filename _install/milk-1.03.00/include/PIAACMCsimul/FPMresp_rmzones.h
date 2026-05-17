#ifndef PIAACMCSIMUL_FOCALPLANEMASK_FPMRESP_RMZONES_H
#define PIAACMCSIMUL_FOCALPLANEMASK_FPMRESP_RMZONES_H

errno_t CLIADDCMD_PIAACMCsimul__FPMresp_rmzones();

errno_t FPMresp_rmzones(const char *__restrict__ FPMresp_in_name,
                        const char *__restrict__ FPMresp_out_name,
                        long     NBzones,
                        imageID *outID);

#endif

#ifndef PIAACMCSIMUL_FOCALPLANEMASK_FPM_PROCESS_H
#define PIAACMCSIMUL_FOCALPLANEMASK_FPM_PROCESS_H

errno_t CLIADDCMD_PIAACMCsimul__FPM_process();

errno_t PIAACMC_FPM_process(const char *__restrict__ FPMsag_name,
                            const char *__restrict__ zonescoord_name,
                            long NBexp,
                            const char *__restrict__ outname);

#endif

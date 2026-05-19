#ifndef PIAACMCSIMUL_FOCALPLANEMASK_RING2SECTORS_H
#define PIAACMCSIMUL_FOCALPLANEMASK_RING2SECTORS_H

errno_t CLIADDCMD_PIAACMCsimul__ring2sectors();

errno_t rings2sectors(const char *IDin_name,
                      const char *sectfname,
                      const char *IDout_name,
                      imageID    *outID);

#endif

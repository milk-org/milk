#ifndef LINOPT_IMTOOLS__COMPUTE_SVDDECOMP_H
#define LINOPT_IMTOOLS__COMPUTE_SVDDECOMP_H

errno_t CLIADDCMD_linopt_imtools__compute_SVDdecomp();

errno_t linopt_compute_SVDdecomp(const char *IDin_name,
                                 const char *IDout_name,
                                 const char *IDcoeff_name,
                                 imageID    *outID);

#endif

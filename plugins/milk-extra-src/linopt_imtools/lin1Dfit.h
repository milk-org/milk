#ifndef LINOPT_IMTOOLS__LIN1DFIT_H
#define LINOPT_IMTOOLS__LIN1DFIT_H

errno_t CLIADDCMD_linopt_imtools__lin1Dfits();

errno_t linopt_compute_1Dfit(const char *fnamein,
                             long        NBpt,
                             long        MaxOrder,
                             const char *fnameout,
                             int         MODE,
                             imageID    *outID);

#endif

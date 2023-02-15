#ifndef LINOPT_IMTOOLS__MAKECOSRADMODES_H
#define LINOPT_IMTOOLS__MAKECOSRADMODES_H

errno_t CLIADDCMD_linopt_imtools__makeCosRadModes();

errno_t linopt_imtools_makeCosRadModes(const char *ID_name,
                                       long        size,
                                       long        kmax,
                                       float       radius,
                                       float       radfactlim,
                                       imageID    *outID);

#endif

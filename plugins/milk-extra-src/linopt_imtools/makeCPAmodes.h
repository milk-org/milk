#ifndef LINOPT_IMTOOLS__MAKECPAMODES_H
#define LINOPT_IMTOOLS__MAKECPAMODES_H

errno_t CLIADDCMD_linopt_imtools__makeCPAmodes();

errno_t linopt_imtools_makeCPAmodes(const char *ID_name,
                                    long        size,
                                    float       rCPAmin,
                                    float       rCPAmax,
                                    float       CPAmax,
                                    float       deltaCPA,
                                    float       radius,
                                    float       radfactlim,
                                    int         writeMfile,
                                    long       *outNBmax);

#endif

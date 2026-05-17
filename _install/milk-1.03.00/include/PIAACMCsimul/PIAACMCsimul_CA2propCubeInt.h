#ifndef PIAACMCSIMUL_CA2PROPCUBEINT_H
#define PIAACMCSIMUL_CA2PROPCUBEINT_H

errno_t PIAACMCsimul_CA2propCubeInt(const char *__restrict IDamp_name,
                                    const char *__restrict IDpha_name,
                                    float zmin,
                                    float zmax,
                                    long  NBz,
                                    const char *__restrict IDout_name,
                                    imageID *outID);

#endif

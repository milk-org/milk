#ifndef LINOPT_IMTOOLS__COMPUTE_SVDPSEUDOINVERSE_H
#define LINOPT_IMTOOLS__COMPUTE_SVDPSEUDOINVERSE_H

errno_t CLIADDCMD_linopt_imtools__compute_SVDpseudoinverse();

errno_t linopt_compute_SVDpseudoInverse(const char *ID_Rmatrix_name,
                                        const char *ID_Cmatrix_name,
                                        double      SVDeps,
                                        long        MaxNBmodes,
                                        const char *ID_VTmatrix_name,
                                        imageID    *outID);

#endif

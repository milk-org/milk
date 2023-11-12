#ifndef LINOPT_IMTOOLS__MAKECPAMODES_H
#define LINOPT_IMTOOLS__MAKECPAMODES_H

errno_t CLIADDCMD_linopt_imtools__makeCPAmodes();

errno_t linopt_imtools_makeCPAmodes(IMGID *imgoutm,
                                    long        size,
                                    float       rCPAmin,
                                    float       rCPAmax,
                                    float       CPAmax,
                                    float       deltaCPA,
                                    float       radius,
                                    float       radfactlim,
                                    int         writeMfile,
                                    long       *outNBmax,
                                    IMGID       imgmask,
                                    float       extrfactor,
                                    float       extroffset
                                   );

#endif

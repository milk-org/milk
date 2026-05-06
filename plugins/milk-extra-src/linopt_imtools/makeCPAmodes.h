/**
 * @file makeCPAmodes.h
 * @brief Makecpamodes module
 */

#ifndef LINOPT_IMTOOLS__MAKECPAMODES_H
#define LINOPT_IMTOOLS__MAKECPAMODES_H

errno_t CLIADDCMD_linopt_imtools__makeCPAmodes();

errno_t linopt_imtools_makeCPAmodes(
    IMGID          *imgoutm,
    uint32_t        sizex,
    uint32_t        sizey,
    float           xcenter,
    float           ycenter,
    float           rCPAmin,
    float           rCPAmax,
    float           CPAmax,
    float           deltaCPA,
    float           radius,
    float           radfactlim,
    float           fpowerlaw,
    float           fpowerlaw_minf,
    float           fpowerlaw_maxf,
    uint32_t        writeMfile,
    long           *outNBmax,
    IMGID           imgmask,
    float           extrfactor,
    float           extroffset
);

#endif

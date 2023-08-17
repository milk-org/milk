/**
 * @brief Matrix Matrix mulitply
 * 
 */


#ifndef LINALGEBRA_SGEMM_H
#define LINALGEBRA_SGEMM_H


errno_t computeSGEMM(
    IMGID imginA,
    IMGID imginB,
    IMGID *outimg,
    int TranspA,
    int TranspB,
    int GPUdev
);

errno_t CLIADDCMD_linalgebra__SGEMM();

#endif

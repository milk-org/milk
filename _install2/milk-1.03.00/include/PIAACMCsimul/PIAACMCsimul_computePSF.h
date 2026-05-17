#ifndef PIAACMCSIMUL_COMPUTEPSF_H
#define PIAACMCSIMUL_COMPUTEPSF_H

errno_t PIAACMCsimul_computePSF(
    float xld, /// @param[in]   xld         float: Source X position [l/D]
    float yld, /// @param[in]   yld         float: Source Y position [l/D]
    long
    startelem, /// @param[in]   startelem   long : First element in propagation
    long
    endelem, /// @param[in]   endelem     long : Last element in propagation
    int savepsf, /// @param[in]   savepsf     int  : Save PSF flag
    int sourcesize, /// @param[in]   sourcezise  int  : Source size (10x log10)
    int extmode,    /// @param[in]   extmode     int  : Source extended type
    int outsave,    /// @param[in]   outsave     int  : Save output flag
    double *contrastval /// @param[out]  contrastval *double : contrast value
);

#endif

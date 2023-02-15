/**
 * @file    linARfilterPred.h
 * @brief   Function prototypes for linear autoregressive prediction
 *
 * Implements Empirical Orthogonal Functions
 *
 */

#ifndef _LINARFILTERPRED_H
#define _LINARFILTERPRED_H

void __attribute__((constructor)) libinit_linARfilterPred();

/* =============================================================================================== */
/* =============================================================================================== */
/** @name 1. INITIALIZATION, configurations
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name 2. I/O TOOLS
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

long LINARFILTERPRED_LoadASCIIfiles(
    double tstart, double dt, long NBpt, long NBfr, const char *IDoutname);

imageID LINARFILTERPRED_SelectBlock(const char *IDin_name,
                                    const char *IDblknb_name,
                                    long        blkNB,
                                    const char *IDout_name);

///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name 3. BUILD PREDICTIVE FILTER
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

imageID linARfilterPred_repeat_shift_X(const char *IDin_name,
                                       long        NBstep,
                                       const char *IDout_name);

/** @brief Build predictive filter
 *
 *
 * Optional pixel masks select input and output variables: "inmask" and "outmask"
 *
 *
 *
 * if LOOPmode = 1, operate in a loop, and re-run filter computation everytime IDin_name changes
 *
 * @note if atmospheric wavefronts, data should be piston-free
 *
 * @return output filter image index
 */

imageID LINARFILTERPRED_Build_LinPredictor(
    const char *IDin_name, ///< [in]  Input telemetry, a 2D or 3D image
    long        PForder,   ///< [in]  Number of time steps in output filter
    float
    PFlag, ///< [in]  Time lag between last measurement and prediction, unit: sampling period
    double
    SVDeps, ///< [in]  Singular value cutoff limit. Ratio between strongest singular value and limit
    double      RegLambda,    ///< [in]  Regularization paramater
    const char *IDoutPF_name, ///< [in]  Output predictive filter name
    int
    outMode, ///< [in]  Output mode. 0: do not write individual files, 1: write individual files (note: output filter cube is always written)
    int LOOPmode, ///< [in]  1 if running in infinite loop waiting for input telemetry
    float
    LOOPgain, ///< [in]  If running in loop, mixing coefficient between previous and current filter
    int testmode);

///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name 4. APPLY PREDICTIVE FILTER
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

imageID LINARFILTERPRED_Apply_LinPredictor_RT(const char *IDfilt_name,
        const char *IDin_name,
        const char *IDout_name);

imageID LINARFILTERPRED_Apply_LinPredictor(const char *IDfilt_name,
        const char *IDin_name,
        float       PFlag,
        const char *IDout_name);

imageID LINARFILTERPRED_PF_updatePFmatrix(const char *IDPF_name,
        const char *IDPFM_name,
        float       alpha);

imageID LINARFILTERPRED_PF_RealTimeApply(const char *IDmodevalIN_name,
        long        IndexOffset,
        int         semtrig,
        const char *IDPFM_name,
        long        NBPFstep,
        const char *IDPFout_name,
        int         nbGPU,
        long        loop,
        long        NBiter,
        int         SAVEMODE,
        float       tlag,
        long        PFindex);

///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name 5. MISC TOOLS, DIAGNOSTICS
 *
 */
///@{
/* =============================================================================================== */
/* =============================================================================================== */

float LINARFILTERPRED_ScanGain(char *IDin_name, float multfact, float framelag);

#endif

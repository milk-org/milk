#ifndef PIAACMCSIMUL_H
#define PIAACMCSIMUL_H

#define ApoFitCosFact 1.0

#include <stdint.h>

#include "CLIcore.h"
#include "OptSystProp/OptSystProp.h"

//#include "PIAAshape/mkPIAAMshapes_from_RadSag.h"

//
// *****************************************************************************************************
// -------------------------- structure holding global variables ---------------------------------------
// *****************************************************************************************************

typedef struct
{

    char piaacmcconfdir
    [STRINGMAXLEN_DIRNAME]; ///  Current configuration directory
    int optsystinit;

    int FORCE_CREATE_Cmodes;
    int CREATE_Cmodes;
    int FORCE_CREATE_Fmodes;
    int CREATE_Fmodes;

    int FORCE_CREATE_fpmzmap;
    int CREATE_fpmzmap;
    int FORCE_CREATE_fpmzt;
    int CREATE_fpmzt;

    int FORCE_CREATE_fpmza;
    int CREATE_fpmza;

    int FORCE_MAKE_PIAA0shape;
    int MAKE_PIAA0shape;
    int FORCE_MAKE_PIAA1shape;
    int MAKE_PIAA1shape;

    int focmMode; // if != -1, compute only impulse response to corresponding zone
    int PIAACMC_FPMsectors;

    double FPMSCALEFACTOR; /// undersize mask in array to avoid edge clipping

    double LAMBDASTART;
    double LAMBDAEND;

    float PIAACMC_MASKRADLD;

    // for minimization
    double *fpmresp_array;
    double *zonez_array;
    double *zonez0_array;
    double *zonez1_array;
    double *zonezbest_array;
    double *dphadz_array;
    double *outtmp_array;
    long    LOOPCNT;

    long vsize;

    double CnormFactor; // for contrast normalization

    int computePSF_FAST_FPMresp;
    int computePSF_ResolvedTarget; // source size = 1e-{0.1*computePSF_ResolvedTarget}
    int computePSF_ResolvedTarget_mode; // 0: source is simulated as 3 points, 1: source is simulated as 6 points
    int PIAACMC_FPM_FASTDERIVATIVES;

    double SCORINGTOTAL;
    double MODampl;
    int    SCORINGMASKTYPE;
    int    PIAACMC_save;
    //	float PIAACMC_MASKregcoeff;
    int PIAACMC_fpmtype; // 0 for idealized PIAACMC focal plane mask, 1 for physical focal plane mask

    long PIAACMC_FPMresp_mp;
    long PIAACMC_FPMresp_thread;

    int WRITE_OK;

    double PIAACMCSIMUL_VAL;
    double PIAACMCSIMUL_VAL0;
    double PIAACMCSIMUL_VALREF;

    // Linear Optimization
    int     LINOPT;              // 1 if linear optimization should be started
    long    linopt_number_param; // number of optimization paramters
    int     linopt_paramtype[10000]; // _DATATYPE_FLOAT or _DATATYPE_DOUBLE
    float  *linopt_paramvalf[10000]; // array of pointers, float
    double *linopt_paramval[10000];  // array of pointers, double
    double  linopt_paramrefval[10000];

    double linopt_paramdelta[10000];
    double linopt_paramdeltaval[10000];
    double linopt_parammaxstep[10000]; // maximum single iteration step
    double linopt_parammin[10000];     // minimum value
    double linopt_parammax[10000];     // maximum value

    int   linopt_REGPIAASHAPES;
    float linopt_piaa0C_regcoeff;
    float linopt_piaa1C_regcoeff;
    float linopt_piaa0C_regcoeff_alpha;
    float linopt_piaa1C_regcoeff_alpha;

    float linopt_piaa0F_regcoeff;
    float linopt_piaa1F_regcoeff;
    float linopt_piaa0F_regcoeff_alpha;
    float linopt_piaa1F_regcoeff_alpha;

    int linopt_REGFPMSAG;

    long linopt_NBiter;

    char fnamedescr
    [STRINGMAXLEN_FILENAME]; // File name descriptor for focal plane mask, inserted inside output file names
    char fnamedescr_conf
    [STRINGMAXLEN_FILENAME]; // File name descriptor for focal plane mask configuration, inserted inside output file names

} PIAACMCSIMUL_PARAMS;

//
// *****************************************************************************************************
// -------------------------- structure defining a reflective PIAACMC system ---------------------------
// *****************************************************************************************************

#define STRINGMAXLEN_PIAACMCSIMUL_MATERIALNAME 10

//
// this structure holds parameters to be optimized in the PIAACMC diffractive design
//
typedef struct
{

    // ======= SEED RADIAL PIAACMC PARAMETERS ======

    double centObs0; /**< input central obstruction */
    double centObs1; /**< output central obstruction */
    double r0lim;    /**< outer radius after extrapolation, piaa mirror 0 */
    double r1lim;    /**< outer radius after extrapolation, piaa mirror 1 */
    long
    NBradpts; /**< number of points for common r0, r1, piaa sags 1D table */

    // Wavelength
    int    nblambda;
    double lambda;  // central wavelength [m]
    double lambdaB; // spectral bandwidth [%]
    double lambdaarray
    [2000]; // [m]  lambdaarray is also defined in OptSystProp structure

    // ====== Overall OPTICAL Geometry ===============

    float beamrad; // [m]
    long  size;
    float pixscale; // [m/pix]

    int PIAAmode;
    // 0: no PIAA,
    // 1: PIAA

    // conjugation (z) of first PIAA surface [m]
    float PIAA0pos;

    // separation between PIAA surfaces [m]
    float PIAAsep;

    // 1 if mask before PIAA surface 0
    int prePIAA0mask;

    // position of mask before PIAA surface 0 [m]
    float prePIAA0maskpos;

    // 1 if mask after PIAA surface 0
    int postPIAA0mask;

    // position of mask after PIAA surface 0 [m]
    float postPIAA0maskpos;

    // fraction of apodization done by PIAA
    float PIAAcoeff;

    // 0: no inv PIAA, 1: inv PIAA after Lyot stops, 2: inv PIAA before Lyot stops
    int invPIAAmode;

    float LyotZmin;
    float LyotZmax;

    // output pupil mask radius (scaled to pupil radius)
    float pupoutmaskrad;

    // ========== WAVEFRONT CONTROL ==================
    int    nbDM;      // number of deformable mirrors (10 max)
    double DMpos[10]; // DM conjugation in collimated space
    long   ID_DM[10]; // DM image identifier

    // ========= LYOT STOPS ============
    long   NBLyotStop; /**< Number of Lyot stops */
    long   IDLyotStop[10];
    double LyotStop_zpos[10];

    // ======= Optics shapes modes ============
    char PIAAmaterial_name[10];
    int  PIAAmaterial_code;
    long CmodesID; // Cosine radial mode
    long Cmsize;   // cosine modes size
    long NBCmodes;
    long
    piaaNBCmodesmax; // maximum number of radial cosine modes for PIAA optics

    long  FmodesID; // Fourier 2D modes
    long  Fmsize;
    long  NBFmodes;
    float piaaCPAmax; // maximum spatial frequency (CPA) for PIAA optics

    long piaa0CmodesID;
    long piaa0FmodesID;
    long piaa1CmodesID;
    long piaa1FmodesID;

    // PSF flux calib
    float peakPSF;

    // ========= Focal Plane Mask ============

    double
    fpmaskradld; // mask radius [l/d] for the idealized PIAACMC starting point
    long   focmNBzone; // number of zones
    double Fratio;     // beam Fratio at focal plane
    long
    zonezID; // focm zone material thickness, double precision image, named fpmzt / fpm_zonez.fits
    double fpmaskamptransm; // mask amplitude transmission (normally 1.0)
    long
    zoneaID; // focm zone amplitude transmission, double precision image, named fpmza / fpm_zonea.fits
    double fpzfactor; // focal plane mask DFT zoom factor

    double fpmRad; // outer radius of physical focal plane mask [m]

    long   NBrings;   // number of rings
    double fpmminsag; // [m]
    double fpmmaxsag; // [m]
    double fpmsagreg_coeff;
    double fpmsagreg_alpha;
    long   NBringCentCone;    // number of rings that the central cone occupies
    double fpmCentConeRad;    // [m]
    double fpmCentConeZ;      // peak sag of central cone [m]
    double fpmOuterConeZ;     // outer sag offset [m]
    double fpmOuterConeRadld; // outer radius (end of outer cone) [lambda/D]
    double fpmOuterConeRad;   // [m]
    long   fpmarraysize;
    char   fpmmaterial_name[STRINGMAXLEN_PIAACMCSIMUL_MATERIALNAME];
    int    fpmmaterial_code;

    // Mask description
    //
    // CENTRAL CONE
    // inner zone (optional) is a cone, covering the central NBringCentCone rings (set to zero for no central cone)
    // The outer edge of the central cone is at sag = 0, and the central peak of the cone is at sag fpmCentConeZ
    //
    // SECTORS
    // The next zone outwards consists of sectors arranged in rings. Each sector has its own sag
    // The zones are between sag fpmminsag and fpmmaxsag
    //
    // OUTER CONE
    // The outer cone starts at the outer edge of the sectors, where it has sag=0
    // Its outer is at sag fpmOuterZ
    // outside of the outer cone, the sag is constant at fpmOuterZ
    //

} OPTPIAACMCDESIGN;

extern PIAACMCSIMUL_PARAMS piaacmcparams;
extern OPTPIAACMCDESIGN    piaacmcopticaldesign;
extern OPTSYST             piaacmcopticalsystem;

void __attribute__((constructor)) libinit_PIAACMCsimul();

errno_t init_PIAACMCsimul();

void PIAACMCsimul_free(void);

#endif

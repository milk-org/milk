/**
 * @file    linARfilterPred.c
 * @brief   linear auto-regressive predictive filter
 *
 * Implements Empirical Orthogonal Functions
 *
 *
 */

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "larpf"

// Module short description
#define MODULE_DESCRIPTION "Linear auto-regressive predictive filters"

#include <assert.h>
#include <ctype.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <fitsio.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <time.h>

#include "CLIcore.h"
#include "timeutils.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "info/info.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"

#include "linARfilterPred/linARfilterPred.h"
#include "linARfilterPred_internal.h"
\
#include "build_linPF.h"
#include "applyPF.h"



#ifdef HAVE_CUDA
#include "linalgebra/linalgebra.h"
#endif

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(linARfilterPred)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/* ===== Command: pfloadascii ===== */
long LINARFILTERPRED_LoadASCIIfiles(
    double tstart, double dt,
    long NBpt, long NBfr,
    const char *IDoutname);

static double la_tstart = 200.0;
static double la_dt = 0.001;
static int64_t la_nbpt = 10000;
static int64_t la_nbfr = 4;
static char la_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "pfin";
static FPS_APP_INFO FPS_app_info_la = {
    .fps_name = "pfloadascii",
    .cmdkey   = "pfloadascii",
    .description =
        "load ascii files to PF input",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_LA(X) \
    X(".tstart", &la_tstart, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "tstart") \
    X(".dt", &la_dt, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "dt") \
    X(".nbpt", &la_nbpt, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb points") \
    X(".nbfr", &la_nbfr, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb frames") \
    X(".out_name", la_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")

#include "fps.h"

static FPS_CLI_BINDING la_b[] = {
    FPS_PARAMS_LA(FPS_X_BINDING) };
static const int la_nb =
    sizeof(la_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS_LA(FPS_X_FARG) };
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS la_cms = {0};
static __attribute__((constructor))
void init_la(void) {
    strncpy(CLIcmddata.key,
        FPS_app_info_la.cmdkey,
        sizeof(CLIcmddata.key)-1);
    strncpy(CLIcmddata.description,
        FPS_app_info_la.description,
        sizeof(CLIcmddata.description)-1);
    CLIcmddata.nbarg =
        sizeof(farg)/sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags = CLICMDFLAG_FPS;
    if(!CLIcmddata.cmdsettings)
        CLIcmddata.cmdsettings = &la_cms;
}
static errno_t la_compute(void) {
    LINARFILTERPRED_LoadASCIIfiles(
        la_tstart, la_dt,
        (long)la_nbpt, (long)la_nbfr, la_out);
    return RETURN_SUCCESS;
}
static errno_t CLIfunction(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_la, farg, &CLIcmddata,
        la_b, la_nb, la_compute);
}

/* ===== Command: mselblock ===== */
imageID LINARFILTERPRED_SelectBlock(
    const char *IDin_name,
    const char *IDblknb_name,
    long blkNB,
    const char *IDout_name);

static char sb_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "modevals";
static char sb_bm[FUNCTION_PARAMETER_STRMAXLEN]
    = "blockmap";
static int64_t sb_blk = 23;
static char sb_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "blk23modevals";
static FPS_APP_INFO FPS_app_info_sb = {
    .fps_name = "mselblock",
    .cmdkey   = "mselblock",
    .description =
        "select modes belonging to block",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_SB(X) \
    X(".in_name", sb_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input modes") \
    X(".bm_name", sb_bm, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "block map") \
    X(".blk", &sb_blk, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "block number") \
    X(".out_name", sb_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")
static FPS_CLI_BINDING sb_b[] = {
    FPS_PARAMS_SB(FPS_X_BINDING) };
static const int sb_nb =
    sizeof(sb_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF sb_farg[] = {
    FPS_PARAMS_SB(FPS_X_FARG) };
static CLICMDDATA sb_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS sb_cms = {0};
static __attribute__((constructor))
void init_sb(void) {
    strncpy(sb_d.key,
        FPS_app_info_sb.cmdkey,
        sizeof(sb_d.key)-1);
    strncpy(sb_d.description,
        FPS_app_info_sb.description,
        sizeof(sb_d.description)-1);
    sb_d.nbarg =
        sizeof(sb_farg)/sizeof(CLICMDARGDEF);
    sb_d.funcfpscliarg = sb_farg;
    sb_d.flags = CLICMDFLAG_FPS;
    if(!sb_d.cmdsettings)
        sb_d.cmdsettings = &sb_cms;
}
static errno_t sb_compute(void) {
    LINARFILTERPRED_SelectBlock(
        sb_in, sb_bm, (long)sb_blk, sb_out);
    return RETURN_SUCCESS;
}
static errno_t sb_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_sb, sb_farg, &sb_d,
        sb_b, sb_nb, sb_compute);
}

/* ===== Command: imrepshiftx ===== */
imageID linARfilterPred_repeat_shift_X(
    const char *IDin_name,
    long NBstep,
    const char *IDout_name);

static char rs_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "imin";
static int64_t rs_nb = 5;
static char rs_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "imout";
static FPS_APP_INFO FPS_app_info_rs = {
    .fps_name = "imrepshiftx",
    .cmdkey   = "imrepshiftx",
    .description =
        "repeat and shift image along X",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_RS(X) \
    X(".in_name", rs_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input") \
    X(".nbstep", &rs_nb, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb steps") \
    X(".out_name", rs_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")
static FPS_CLI_BINDING rs_b[] = {
    FPS_PARAMS_RS(FPS_X_BINDING) };
static const int rs_nbb =
    sizeof(rs_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF rs_farg[] = {
    FPS_PARAMS_RS(FPS_X_FARG) };
static CLICMDDATA rs_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS rs_cms = {0};
static __attribute__((constructor))
void init_rs(void) {
    strncpy(rs_d.key,
        FPS_app_info_rs.cmdkey,
        sizeof(rs_d.key)-1);
    strncpy(rs_d.description,
        FPS_app_info_rs.description,
        sizeof(rs_d.description)-1);
    rs_d.nbarg =
        sizeof(rs_farg)/sizeof(CLICMDARGDEF);
    rs_d.funcfpscliarg = rs_farg;
    rs_d.flags = CLICMDFLAG_FPS;
    if(!rs_d.cmdsettings)
        rs_d.cmdsettings = &rs_cms;
}
static errno_t rs_compute(void) {
    linARfilterPred_repeat_shift_X(
        rs_in, (long)rs_nb, rs_out);
    return RETURN_SUCCESS;
}
static errno_t rs_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_rs, rs_farg, &rs_d,
        rs_b, rs_nbb, rs_compute);
}

/* ===== Command: mkARpfilt ===== */
imageID LINARFILTERPRED_Build_LinPredictor(
    const char *IDin_name,
    long PForder, float PFlag,
    double SVDeps, double RegLambda,
    const char *IDoutPF_name,
    int outMode, int LOOPmode,
    float LOOPgain, int testmode);

static char mk_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "indata";
static int64_t mk_ord = 5;
static double mk_lag = 2.4;
static double mk_svd = 0.0001;
static double mk_reg = 0.0;
static char mk_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outPF";
static int64_t mk_loop = 0;
static double mk_gain = 0.1;
static int64_t mk_test = 1;
static FPS_APP_INFO FPS_app_info_mk = {
    .fps_name = "mkARpfilt",
    .cmdkey   = "mkARpfilt",
    .description =
        "make linear AR filter",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_MK(X) \
    X(".in_name", mk_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input data") \
    X(".pforder", &mk_ord, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "PF order") \
    X(".pflag", &mk_lag, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "PF lag") \
    X(".svdeps", &mk_svd, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "SVD eps") \
    X(".reglambda", &mk_reg, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "reg param") \
    X(".out_name", mk_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output PF") \
    X(".loopmode", &mk_loop, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "loop mode") \
    X(".loopgain", &mk_gain, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "loop gain") \
    X(".testmode", &mk_test, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "test mode")
static FPS_CLI_BINDING mk_b[] = {
    FPS_PARAMS_MK(FPS_X_BINDING) };
static const int mk_nbb =
    sizeof(mk_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF mk_farg[] = {
    FPS_PARAMS_MK(FPS_X_FARG) };
static CLICMDDATA mk_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS mk_cms = {0};
static __attribute__((constructor))
void init_mk(void) {
    strncpy(mk_d.key,
        FPS_app_info_mk.cmdkey,
        sizeof(mk_d.key)-1);
    strncpy(mk_d.description,
        FPS_app_info_mk.description,
        sizeof(mk_d.description)-1);
    mk_d.nbarg =
        sizeof(mk_farg)/sizeof(CLICMDARGDEF);
    mk_d.funcfpscliarg = mk_farg;
    mk_d.flags = CLICMDFLAG_FPS;
    if(!mk_d.cmdsettings)
        mk_d.cmdsettings = &mk_cms;
}
static errno_t mk_compute(void) {
    LINARFILTERPRED_Build_LinPredictor(
        mk_in, (long)mk_ord,
        (float)mk_lag, mk_svd, mk_reg,
        mk_out, 1,
        (int)mk_loop,
        (float)mk_gain, (int)mk_test);
    return RETURN_SUCCESS;
}
static errno_t mk_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_mk, mk_farg, &mk_d,
        mk_b, mk_nbb, mk_compute);
}

/* ===== Command: applyARpfilt ===== */
long LINARFILTERPRED_Apply_LinPredictor(
    const char *IDfilt_name,
    const char *IDin_name,
    float PFlag,
    const char *IDout_name);

static char ap_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "indata";
static char ap_filt[FUNCTION_PARAMETER_STRMAXLEN]
    = "Pfilt";
static double ap_lag = 2.4;
static char ap_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outPF";
static FPS_APP_INFO FPS_app_info_ap = {
    .fps_name = "applyARpfilt",
    .cmdkey   = "applyARpfilt",
    .description =
        "apply linear AR filter",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_AP(X) \
    X(".in_name", ap_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input data") \
    X(".filt_name", ap_filt, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "predictor") \
    X(".pflag", &ap_lag, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "PF lag") \
    X(".out_name", ap_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")
static FPS_CLI_BINDING ap_b[] = {
    FPS_PARAMS_AP(FPS_X_BINDING) };
static const int ap_nbb =
    sizeof(ap_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF ap_farg[] = {
    FPS_PARAMS_AP(FPS_X_FARG) };
static CLICMDDATA ap_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS ap_cms = {0};
static __attribute__((constructor))
void init_ap(void) {
    strncpy(ap_d.key,
        FPS_app_info_ap.cmdkey,
        sizeof(ap_d.key)-1);
    strncpy(ap_d.description,
        FPS_app_info_ap.description,
        sizeof(ap_d.description)-1);
    ap_d.nbarg =
        sizeof(ap_farg)/sizeof(CLICMDARGDEF);
    ap_d.funcfpscliarg = ap_farg;
    ap_d.flags = CLICMDFLAG_FPS;
    if(!ap_d.cmdsettings)
        ap_d.cmdsettings = &ap_cms;
}
static errno_t ap_compute(void) {
    LINARFILTERPRED_Apply_LinPredictor(
        ap_in, ap_filt,
        (float)ap_lag, ap_out);
    return RETURN_SUCCESS;
}
static errno_t ap_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_ap, ap_farg, &ap_d,
        ap_b, ap_nbb, ap_compute);
}

/* ===== Command: mscangain ===== */
float LINARFILTERPRED_ScanGain(
    char *IDin_name,
    float multfact,
    float framelag);

static char sg_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "olwfsmeas";
static double sg_mf = 0.98;
static double sg_fl = 2.65;
static FPS_APP_INFO FPS_app_info_sg = {
    .fps_name = "mscangain",
    .cmdkey   = "mscangain",
    .description = "scan gain",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_SG(X) \
    X(".in_name", sg_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "mode vals") \
    X(".multfact", &sg_mf, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "mult factor") \
    X(".framelag", &sg_fl, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "frame lag")
static FPS_CLI_BINDING sg_b[] = {
    FPS_PARAMS_SG(FPS_X_BINDING) };
static const int sg_nbb =
    sizeof(sg_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF sg_farg[] = {
    FPS_PARAMS_SG(FPS_X_FARG) };
static CLICMDDATA sg_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS sg_cms = {0};
static __attribute__((constructor))
void init_sg(void) {
    strncpy(sg_d.key,
        FPS_app_info_sg.cmdkey,
        sizeof(sg_d.key)-1);
    strncpy(sg_d.description,
        FPS_app_info_sg.description,
        sizeof(sg_d.description)-1);
    sg_d.nbarg =
        sizeof(sg_farg)/sizeof(CLICMDARGDEF);
    sg_d.funcfpscliarg = sg_farg;
    sg_d.flags = CLICMDFLAG_FPS;
    if(!sg_d.cmdsettings)
        sg_d.cmdsettings = &sg_cms;
}
static errno_t sg_compute(void) {
    LINARFILTERPRED_ScanGain(
        sg_in, (float)sg_mf, (float)sg_fl);
    return RETURN_SUCCESS;
}
static errno_t sg_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_sg, sg_farg, &sg_d,
        sg_b, sg_nbb, sg_compute);
}

/* ===== Command: linARPFMupdate ===== */
long LINARFILTERPRED_PF_updatePFmatrix(
    const char *IDPF_name,
    const char *IDPFM_name,
    float alpha);

static char pu_pf[FUNCTION_PARAMETER_STRMAXLEN]
    = "outPF";
static char pu_pfm[FUNCTION_PARAMETER_STRMAXLEN]
    = "PFMat";
static double pu_alpha = 0.1;
static FPS_APP_INFO FPS_app_info_pu = {
    .fps_name = "linARPFMupdate",
    .cmdkey   = "linARPFMupdate",
    .description =
        "update predictive filter matrix",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_PU(X) \
    X(".pf_name", pu_pf, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "3D predictor") \
    X(".pfm_name", pu_pfm, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "2D matrix") \
    X(".alpha", &pu_alpha, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "update coeff")
static FPS_CLI_BINDING pu_b[] = {
    FPS_PARAMS_PU(FPS_X_BINDING) };
static const int pu_nbb =
    sizeof(pu_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF pu_farg[] = {
    FPS_PARAMS_PU(FPS_X_FARG) };
static CLICMDDATA pu_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS pu_cms = {0};
static __attribute__((constructor))
void init_pu(void) {
    strncpy(pu_d.key,
        FPS_app_info_pu.cmdkey,
        sizeof(pu_d.key)-1);
    strncpy(pu_d.description,
        FPS_app_info_pu.description,
        sizeof(pu_d.description)-1);
    pu_d.nbarg =
        sizeof(pu_farg)/sizeof(CLICMDARGDEF);
    pu_d.funcfpscliarg = pu_farg;
    pu_d.flags = CLICMDFLAG_FPS;
    if(!pu_d.cmdsettings)
        pu_d.cmdsettings = &pu_cms;
}
static errno_t pu_compute(void) {
    LINARFILTERPRED_PF_updatePFmatrix(
        pu_pf, pu_pfm, (float)pu_alpha);
    return RETURN_SUCCESS;
}
static errno_t pu_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_pu, pu_farg, &pu_d,
        pu_b, pu_nbb, pu_compute);
}

/* ===== Command: linARapplyRT ===== */
long LINARFILTERPRED_PF_RealTimeApply(
    const char *IDmodevalOL_name,
    long IndexOffset, int semtrig,
    const char *IDPFM_name,
    long NBPFstep,
    const char *IDPFout_name,
    int nbGPU, long loop,
    long NBiter, int SAVEMODE,
    float tlag, long PFindex);

static char rt_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "modevalOL";
static int64_t rt_off = 0;
static int64_t rt_sem = 2;
static char rt_pfm[FUNCTION_PARAMETER_STRMAXLEN]
    = "PFmat";
static int64_t rt_ord = 5;
static char rt_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outPFmodeval";
static int64_t rt_gpu = 0;
static int64_t rt_loop = 0;
static int64_t rt_nbit = 0;
static int64_t rt_save = 0;
static double rt_tlag = 1.8;
static int64_t rt_pfi = 0;
static FPS_APP_INFO FPS_app_info_rt = {
    .fps_name = "linARapplyRT",
    .cmdkey   = "linARapplyRT",
    .description =
        "RT apply predictive filter",
    .description_long =
        "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};
#define FPS_PARAMS_RT(X) \
    X(".in_name", rt_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "OL coeffs") \
    X(".offset", &rt_off, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "index off") \
    X(".semtrig", &rt_sem, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "sem trig") \
    X(".pfm_name", rt_pfm, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "PF matrix") \
    X(".pforder", &rt_ord, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "filter order") \
    X(".out_name", rt_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".nbgpu", &rt_gpu, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb GPUs") \
    X(".loop", &rt_loop, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "loop flag") \
    X(".nbiter", &rt_nbit, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb iter") \
    X(".savemode", &rt_save, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "save mode") \
    X(".tlag", &rt_tlag, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "time lag") \
    X(".pfindex", &rt_pfi, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "PF index")
static FPS_CLI_BINDING rt_b[] = {
    FPS_PARAMS_RT(FPS_X_BINDING) };
static const int rt_nbb =
    sizeof(rt_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF rt_farg[] = {
    FPS_PARAMS_RT(FPS_X_FARG) };
static CLICMDDATA rt_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS rt_cms = {0};
static __attribute__((constructor))
void init_rt(void) {
    strncpy(rt_d.key,
        FPS_app_info_rt.cmdkey,
        sizeof(rt_d.key)-1);
    strncpy(rt_d.description,
        FPS_app_info_rt.description,
        sizeof(rt_d.description)-1);
    rt_d.nbarg =
        sizeof(rt_farg)/sizeof(CLICMDARGDEF);
    rt_d.funcfpscliarg = rt_farg;
    rt_d.flags = CLICMDFLAG_FPS;
    if(!rt_d.cmdsettings)
        rt_d.cmdsettings = &rt_cms;
}
static errno_t rt_compute(void) {
    LINARFILTERPRED_PF_RealTimeApply(
        rt_in, (long)rt_off, (int)rt_sem,
        rt_pfm, (long)rt_ord, rt_out,
        (int)rt_gpu, (long)rt_loop,
        (long)rt_nbit, (int)rt_save,
        (float)rt_tlag, (long)rt_pfi);
    return RETURN_SUCCESS;
}
static errno_t rt_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_rt, rt_farg, &rt_d,
        rt_b, rt_nbb, rt_compute);
}

/* ===== Module init ===== */

static errno_t init_module_CLI()
{
    {
        safe_fps_fill_farg_examples(
            farg, la_b, la_nb);
        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            sb_farg, sb_b, sb_nb);
        int cmdi = RegisterCLIcmd(
            sb_d, sb_CLIfunc);
        sb_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            rs_farg, rs_b, rs_nbb);
        int cmdi = RegisterCLIcmd(
            rs_d, rs_CLIfunc);
        rs_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            mk_farg, mk_b, mk_nbb);
        int cmdi = RegisterCLIcmd(
            mk_d, mk_CLIfunc);
        mk_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            ap_farg, ap_b, ap_nbb);
        int cmdi = RegisterCLIcmd(
            ap_d, ap_CLIfunc);
        ap_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            sg_farg, sg_b, sg_nbb);
        int cmdi = RegisterCLIcmd(
            sg_d, sg_CLIfunc);
        sg_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            pu_farg, pu_b, pu_nbb);
        int cmdi = RegisterCLIcmd(
            pu_d, pu_CLIfunc);
        pu_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            rt_farg, rt_b, rt_nbb);
        int cmdi = RegisterCLIcmd(
            rt_d, rt_CLIfunc);
        rt_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    CLIADDCMD_LinARfilterPred__build_linPF();
    CLIADDCMD_LinARfilterPred__applyPF();

    // add atexit functions here

    return RETURN_SUCCESS;
}

/* =============================================================================================== */

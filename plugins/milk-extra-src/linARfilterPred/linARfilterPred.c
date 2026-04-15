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
        "load ascii files to PF input"
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
        "select modes belonging to block"
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
        "repeat and shift image along X"
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
        "make linear AR filter"
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
        "apply linear AR filter"
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
    .description = "scan gain"
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
        "update predictive filter matrix"
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
        "RT apply predictive filter"
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
/* =============================================================================================== */
/*                                                                                                 */
/* 1. INITIALIZATION                                                                               */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 2. I/O TOOLS                                                                                    */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

int NBwords(const char sentence[])
{
    int counted = 0; // result

    // state:
    const char *it     = sentence;
    int         inword = 0;

    do
        switch(*it)
        {
            case '\0':
            case ' ':
            case '\t':
            case '\n':
            case '\r':
                if(inword)
                {
                    inword = 0;
                    counted++;
                }
                break;
            default:
                inword = 1;
        }
    while(*it++);

    return counted;
}

/**
 * @brief load ascii file(s) into image cube
 *
 *  resamples sequence(s) of data points
 * INPUT FILES HAVE TO BE NAMED seq000.dat, seq001.dat etc...
 *
 * file starts at tstart, sampling = dt
 * NBpt per file
 * NBfr files
*/

long LINARFILTERPRED_LoadASCIIfiles(
    double tstart, double dt, long NBpt, long NBfr, const char *IDoutname)
{
    FILE       *fp;
    long        NBfiles;
    double      runtime;
    char        fname[200];
    struct stat fstat;
    int         fOK;
    long        NBvarin[200];
    long        fcnt;
    FILE       *fparray[200];
    long        kk;
    size_t      linesiz = 0;
    char       *linebuf = 0;
    //ssize_t linelen=0;
    //int     ret;
    long    vcnt;
    double  ftime0[200];
    double  var0[200][200];
    double  ftime1[200];
    double  var1[200][200];
    double  varC[200][200];
    float   alpha;
    long    nbvar;
    long    fr;
    char    imoutname[200];
    FILE   *fpout;
    imageID IDout[200];
    //int     HPfilt = 1; // high pass filter
    float HPgain = 0.005;

    long ii;
    long kkpt, kkfr;

    runtime = tstart;

    fOK     = 1;
    NBfiles = 0;
    nbvar   = 0;
    while(fOK == 1)
    {
        snprintf(fname, sizeof(fname),
                 "seq%03ld.dat", NBfiles);
        if(stat(fname, &fstat) == 0)
        {
            printf("Found file %s\n", fname);
            fflush(stdout);
            fp = fopen(fname, "r");
            //linelen =
            if(getline(&linebuf, &linesiz, fp) == -1)
            {
                PRINT_ERROR("getline error");
            }
            fclose(fp);
            NBvarin[NBfiles] = NBwords(linebuf) - 1;
            free(linebuf);
            linebuf = NULL;
            printf("   NB variables = %ld\n", NBvarin[NBfiles]);
            nbvar += NBvarin[NBfiles];
            NBfiles++;
        }
        else
        {
            printf("No more files\n");
            fflush(stdout);
            fOK = 0;
        }
    }
    printf("NBfiles = %ld\n", NBfiles);

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        snprintf(fname, sizeof(fname),
                 "seq%03ld.dat", fcnt);
        printf("   %03ld  OPENING FILE %s\n", fcnt, fname);
        fflush(stdout);
        fparray[fcnt] = fopen(fname, "r");
    }

    kk      = 0; // time
    runtime = tstart;

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        if(fscanf(fparray[fcnt], "%lf", &ftime0[fcnt]) != 1)
        {
            PRINT_ERROR("fscanf error");
        }

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            if(fscanf(fparray[fcnt], "%lf", &var0[fcnt][vcnt]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
        }
        if(fscanf(fparray[fcnt], "\n") != 0)
        {
            PRINT_ERROR("fscanf error");
        }

        if(fscanf(fparray[fcnt], "%lf", &ftime1[fcnt]) != 1)
        {
            PRINT_ERROR("fscanf error");
        }

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            if(fscanf(fparray[fcnt], "%lf", &var1[fcnt][vcnt]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
        }
        if(fscanf(fparray[fcnt], "\n") != 0)
        {
            PRINT_ERROR("fscanf error");
        }

        printf("FILE %ld :  \n", fcnt);
        printf(" time :    %20f  %20f\n", ftime0[fcnt], ftime1[fcnt]);
        fflush(stdout);

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            printf("    variable %3ld   :   %20f  %20f\n",
                   vcnt,
                   var0[fcnt][vcnt],
                   var1[fcnt][vcnt]);
            varC[fcnt][vcnt] = var0[fcnt][vcnt];
        }
        printf("\n");
    }

    for(fr = 0; fr < NBfr; fr++)
    {
        snprintf(imoutname, sizeof(imoutname),
                 "%s_%03ld", IDoutname, fr);
        create_3Dimage_ID(imoutname, nbvar, 1, NBpt, &(IDout[fr]));
    }

    fpout = fopen("out.txt", "w");

    kk   = 0;
    kkpt = 0;
    kkfr = 0;
    while(kkfr < NBfr)
    {
        fprintf(fpout, "%20f", runtime);

        ii = 0;
        for(fcnt = 0; fcnt < NBfiles; fcnt++)
        {
            while(ftime1[fcnt] < runtime)
            {
                ftime0[fcnt] = ftime1[fcnt];
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    var0[fcnt][vcnt] = var1[fcnt][vcnt];
                }

                if(fscanf(fparray[fcnt], "%lf", &ftime1[fcnt]) != 1)
                {
                    PRINT_ERROR("fscanf error");
                }
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    if(fscanf(fparray[fcnt], "%lf", &var1[fcnt][vcnt]) != 1)
                    {
                        PRINT_ERROR("fscanf error");
                    }
                }
                if(fscanf(fparray[fcnt], "\n") != 0)
                {
                    PRINT_ERROR("fscanf error");
                }
            }
            if(kk == 0)
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    varC[fcnt][vcnt] = var0[fcnt][vcnt];
                }

            alpha = (runtime - ftime0[fcnt]) / (ftime1[fcnt] - ftime0[fcnt]);
            for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
            {
                fprintf(fpout,
                        " %20f",
                        (1.0 - alpha) * var0[fcnt][vcnt] +
                        alpha * var1[fcnt][vcnt] - varC[fcnt][vcnt]);
                varC[fcnt][vcnt] = (1.0 - HPgain) * varC[fcnt][vcnt] +
                                   HPgain * ((1.0 - alpha) * var0[fcnt][vcnt] +
                                             alpha * var1[fcnt][vcnt]);

                dcimg[IDout[kkfr]].array.F[kkpt * nbvar + ii] =
                    (1.0 - alpha) * var0[fcnt][vcnt] +
                    alpha * var1[fcnt][vcnt] - varC[fcnt][vcnt];
                ii++;
            }
        }

        fprintf(fpout, "\n");

        kk++;
        kkpt++;
        runtime += dt;
        if(kkpt == NBpt)
        {
            kkpt = 0;
            kkfr++;
        }
    }

    fclose(fpout);

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        fclose(fparray[fcnt]);
    }

    return (NBfiles);
}

// select block on first dimension
imageID LINARFILTERPRED_SelectBlock(const char *IDin_name,
                                    const char *IDblknb_name,
                                    long        blkNB,
                                    const char *IDout_name)
{
    imageID IDin;
    imageID IDblknb;
    uint8_t naxis;

    long          m;
    long          NBmodes1;
    uint32_t     *sizearray;
    uint32_t      xsize, ysize, zsize;
    unsigned long cnt;
    imageID       IDout;
    //char imname[200];
    long mmax;

    printf("Selecting block %ld ...\n", blkNB);
    fflush(stdout);

    IDin    = image_ID(IDin_name, dcimg, dcnimg);
    IDblknb = image_ID(IDblknb_name, dcimg, dcnimg);
    naxis   = dcimg[IDin].md[0].naxis;
    mmax    = dcimg[IDblknb].md[0].size[0];

    if(dcimg[IDin].md[0].size[0] != dcimg[IDblknb].md[0].size[0])
    {
        printf(
            "WARNING: block index file and telemetry have different sizes\n");
        fflush(stdout);
        mmax = dcimg[IDin].md[0].size[0];
        if(dcimg[IDblknb].md[0].size[0] < mmax)
        {
            mmax = dcimg[IDblknb].md[0].size[0];
        }
    }

    NBmodes1 = 0;
    for(m = 0; m < mmax; m++)
    {
        if(dcimg[IDblknb].array.UI16[m] == blkNB)
        {
            NBmodes1++;
        }
    }

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(uint8_t axis = 0; axis < naxis; axis++)
    {
        sizearray[axis] = dcimg[IDin].md[0].size[axis];
    }
    sizearray[0] = NBmodes1;

    {
        IMGID imgout_tmp =
            imgid_make_from_name(
                IDout_name);
        imgout_tmp.mdt->naxis = naxis;
        for(uint8_t a = 0; a < naxis;
            a++)
        {
            imgout_tmp.mdt->size[a] =
                sizearray[a];
        }
        imgout_tmp.mdt->datatype =
            _DATATYPE_FLOAT;
        imgout_tmp.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }

    xsize = dcimg[IDin].md[0].size[0];
    if(naxis > 1)
    {
        ysize = dcimg[IDin].md[0].size[1];
    }
    else
    {
        ysize = 1;
    }
    if(naxis > 2)
    {
        zsize = dcimg[IDin].md[0].size[2];
    }
    else
    {
        zsize = 1;
    }

    cnt = 0;

    for(uint32_t jj = 0; jj < ysize; jj++)
        for(uint32_t kk = 0; kk < zsize; kk++)
            for(uint32_t ii = 0; ii < mmax; ii++)
                if(dcimg[IDblknb].array.UI16[ii] == blkNB)
                {
                    //printf("%ld / %ld   cnt = %8ld / %ld\n", ii, xsize, cnt, NBmodes1*ysize*zsize);
                    //fflush(stdout);
                    dcimg[IDout].array.F[cnt] =
                        dcimg[IDin]
                        .array.F[kk * xsize * ysize + jj * ysize + ii];
                    cnt++;
                }

    free(sizearray);

    return (IDout);
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 3. BUILD PREDICTIVE FILTER                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Expand 2D image/matrix in X direction by repeat and shift
 *
 */
imageID linARfilterPred_repeat_shift_X(const char *IDin_name,
                                       long        NBstep,
                                       const char *IDout_name)
{
    imageID  IDin;
    uint32_t xsize, ysize;

    imageID  IDout;
    uint32_t xsizeout, ysizeout;

    uint32_t *imsizeout;

    IDin     = image_ID(IDin_name, dcimg, dcnimg);
    xsize    = dcimg[IDin].md[0].size[0];
    ysize    = dcimg[IDin].md[0].size[1];
    xsizeout = xsize * NBstep;
    ysizeout = ysize - NBstep;

    imsizeout = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsizeout == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imsizeout[0] = xsizeout;
    imsizeout[1] = ysizeout;
    {
        IMGID imgout_tmp =
            imgid_make_from_name(
                IDout_name);
        imgout_tmp.mdt->naxis = 2;
        imgout_tmp.mdt->size[0] =
            imsizeout[0];
        imgout_tmp.mdt->size[1] =
            imsizeout[1];
        imgout_tmp.mdt->datatype =
            _DATATYPE_FLOAT;
        imgout_tmp.mdt->shared = 1;
        imgout_tmp.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }
    free(imsizeout);

    long step;
    for(step = 0; step < NBstep; step++)
    {
        for(uint32_t ii = 0; ii < xsize; ii++)
        {
            for(uint32_t jjout = 0; jjout < ysize - NBstep; jjout++)
            {
                dcimg[IDout]
                .array.F[jjout * xsizeout + step * xsize + ii] =
                    dcimg[IDin]
                    .array.F[(jjout + NBstep - step - 1) * xsize + ii];
            }
        }
    }

    return IDout;
}

/** ## Purpose
 *
 * Build predictive filter from real-time AO telemetry
 *
 *
 * ## Masking
 *
 *  Optional input and output pixel masks select active input & output
 *
 *
 * ## Loop mode
 *
 * If LOOPmode = 1, operate in a loop, and re-run filter computation everytime IDin_name changes
 *
 *
 * ## Input parameters: dynamic mode
 *
 * if <IFoutPF_name>_PFparam image exist, read parameters from it: PFlag, SVDeps, RegLambda, LOOPgain
 * create it in shared memory by default
 *
 *
 * @return If testmode=2, write 3D output filter
 * @return output filter image indentifier
 *
   */

imageID LINARFILTERPRED_Build_LinPredictor(const char *IDin_name,
        long        PForder,
        float       PFlag,
        double      SVDeps,
        double      RegLambda,
        const char *IDoutPF_name,
        __attribute__((unused)) int outMode,
        int                         LOOPmode,
        float                       LOOPgain,
        int                         testmode)
{
    /// ---
    /// # Code Description

    imageID IDin;
    imageID IDmatA;
    //imageID IDout;
    imageID IDinmask;
    imageID IDoutmask;
    long    nbspl; // Number of samples
    long    NBpixin, NBpixout;
    long    NBmvec, NBmvec1;
    long    mvecsize;
    long    xsize, ysize;
    long   *pixarray_x;
    long   *pixarray_y;
    long   *pixarray_xy;

    long *outpixarray_x;
    long *outpixarray_y;
    long *outpixarray_xy;

    double *ave_inarray;
    int     REG = 0; // 1 if regularization
    long    m, pix, k0, dt;
    int     Save = 0;
    long    xysize;
    long    IDmatC;
    //int use_magma = 1;                         // use MAGMA library if available
    //int magmacomp = 0;

    //imageID IDfiltC;
    // float *valfarray;
    float alpha;
    long  PFpix;
    //char filtname[200];
    //char filtfname[200];
    //imageID ID_Pfilt;
    float   val, val0;
    long    ind1;
    imageID IDoutPF2D;    // averaged with previous filters
    imageID IDoutPF2Draw; // individual filter
    char    IDoutPF_name_raw[200];
    //  long IDoutPF3D;
    //  char IDoutPF_name3D[500];

    long NB_SVD_Modes;

    int DC_MODE = 0; // 1 if average value of each mode is removed

    long      NBiter, iter;
    long      semtrig = 2;
    uint32_t *imsizearray;

    //char fname[200];

    //time_t t;
    //struct tm *uttime;
    //struct timespec timenow;

    struct timespec t0;
    struct timespec t1;
    struct timespec t2;
    struct timespec tdiff;
    double          tdiffv01; // waiting time
    double          tdiffv12; // computing time

    imageID IDPFparam; // parameters in shared memory (optional)
    char    imname[200];
    int     ExternalPFparam;

    float PFlag_run;
    float SVDeps_run;
    float RegLambda_run;
    float LOOPgain_run;
    float gain;

    uint32_t *imsize;
    long      IDincp;
    long      inNBelem;

    list_variable_ID(NULL);

    int  PSINV_MODE = 0;
    long IDv;
    if((IDv = variable_ID("_SVD_PSINV")) != -1)
    {
        PSINV_MODE = (int)(dcvar[IDv].value.f + 0.1);
        printf("PSINV_MODE = %d\n", PSINV_MODE);
    }

    float PSINV_s = 1.0e-6;
    if((IDv = variable_ID("_SVD_s")) != -1)
    {
        PSINV_s = dcvar[IDv].value.f;
        printf("PSINV_s = %f\n", PSINV_s);
    }

    float PSINV_tol = 1.0;
    if((IDv = variable_ID("_SVD_tol")) != -1)
    {
        PSINV_tol = dcvar[IDv].value.f;
        printf("PSINV_tol = %f\n", PSINV_tol);
    }

    /// ## Reading Parameters from Image

    /// If image named <IDoutPF_name>_PFparam exists, the predictive filter
    /// parameters are read from it instead of the function arguments. \n
    /// This mode is particularly useful in LOOP mode if the user needs
    /// to change the parameters between LOOP iterations.\n

    snprintf(imname, sizeof(imname),
             "%s_PFparam", IDoutPF_name);
    imsize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsize == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }
    imsize[0] = 4;
    imsize[1] = 1;
    {
        IMGID imgparam =
            imgid_make_from_name(imname);
        imgparam.mdt->naxis = 2;
        imgparam.mdt->size[0] =
            imsize[0];
        imgparam.mdt->size[1] =
            imsize[1];
        imgparam.mdt->datatype =
            _DATATYPE_FLOAT;
        imgparam.mdt->shared = 1;
        imgparam.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgparam);
        IDPFparam = imgparam.ID;
    }
    free(imsize);

    if((IDPFparam = image_ID(imname, dcimg, dcnimg)) != -1)
    {
        ExternalPFparam                  = 1;
        dcimg[IDPFparam].array.F[0] = PFlag;
        dcimg[IDPFparam].array.F[1] = SVDeps;
        dcimg[IDPFparam].array.F[2] = RegLambda;
        dcimg[IDPFparam].array.F[3] = LOOPgain;
    }
    else
    {
        ExternalPFparam = 0;
    }

    LOOPgain_run = LOOPgain;
    if(LOOPmode == 0)
    {
        LOOPgain_run = 1.0;
        NBiter       = 1;
    }
    else
    {
        NBiter = 100000000;
    }

    //sprintf(IDoutPF_name3D, "%s_3D", IDoutPF_name);

    /// ## Selecting input values

    /// The goal of this function is to build a linear link between
    /// input and output variables. \n
    /// Input variables values are provided by the input telemetry image
    /// which is first read to measure dimensions, and allocate memory.\n
    /// Note that an optional variable selection step allows only a
    /// subset of the telemetry variables to be considered.

    /// ### Read input telemetry image IDin_name to measure xsize, ysize and number of samples
    IDin = image_ID(IDin_name, dcimg, dcnimg);

    switch(dcimg[IDin].md[0].naxis)
    {

        case 2:
            /// If 2D image:
            /// - xysize <- size[0] is number of variables
            /// - nbspl <- size[1] is number of samples
            nbspl = dcimg[IDin].md[0].size[1];
            xsize = dcimg[IDin].md[0].size[0];
            ysize = 1;
            // copy of image to avoid input change during computation
            create_2Dimage_ID("PFin_cp",
                              dcimg[IDin].md[0].size[0],
                              dcimg[IDin].md[0].size[1],
                              &IDincp);
            inNBelem =
                dcimg[IDin].md[0].size[0] * dcimg[IDin].md[0].size[1];
            break;

        case 3:
            /// If 3D image
            /// - xysize <- size[0] * size[1] is number of variables
            /// - nbspl <- size[2] is number of samples
            nbspl = dcimg[IDin].md[0].size[2];
            xsize = dcimg[IDin].md[0].size[0];
            ysize = dcimg[IDin].md[0].size[1];
            create_3Dimage_ID("PFin_copy",
                              dcimg[IDin].md[0].size[0],
                              dcimg[IDin].md[0].size[1],
                              dcimg[IDin].md[0].size[2],
                              &IDincp);

            inNBelem = dcimg[IDin].md[0].size[0] *
                       dcimg[IDin].md[0].size[1] *
                       dcimg[IDin].md[0].size[2];
            break;

        default:
            printf("Invalid image size\n");
            break;
    }
    xysize = xsize * ysize;
    printf("xysize = %ld\n", xysize);

    /// Once input telemetry size measured, arrays are created:
    /// - pixarray_x  : x coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_y  : y coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_xy : combined index (avoids re-computing index frequently)
    /// - ave_inarray : time averaged value, useful because the predictive filter often needs average to be zero, so we will remove it

    pixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    pixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    pixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ave_inarray = (double *) malloc(sizeof(double) * xsize * ysize);
    if(ave_inarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    /// ### Select input variables from mask (optional)
    /// If image "inmask" exists, use it to select which variables are active.
    /// Otherwise, all variables are active\n
    /// The number of active input variables is stored in NBpixin.

    IDinmask = image_ID("inmask", dcimg, dcnimg);
    if(IDinmask == -1)
    {
        NBpixin = 0; //xsize*ysize;

        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                pixarray_x[NBpixin]  = ii;
                pixarray_y[NBpixin]  = jj;
                pixarray_xy[NBpixin] = jj * xsize + ii;
                NBpixin++;
            }
    }
    else
    {
        NBpixin = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(dcimg[IDinmask].array.F[jj * xsize + ii] > 0.5f)
                {
                    pixarray_x[NBpixin]  = ii;
                    pixarray_y[NBpixin]  = jj;
                    pixarray_xy[NBpixin] = jj * xsize + ii;
                    NBpixin++;
                }
    }
    printf("NBpixin = %ld\n", NBpixin);

    /// ## Selecting Output Variables
    /// By default, the output variables are the same as the input variables,
    /// so the prediction is performed on the same variables as the input.\n
    ///
    /// With inmask and outmask, input AND output variables can be
    /// selected amond the telemetry.

    /// Arrays are created:
    /// - outpixarray_x  : x coordinate of each output variable (useful to keep track of spatial coordinates)
    /// - outpixarray_y  : y coordinate of each output variable (useful to keep track of spatial coordinates)
    /// - outpixarray_xy : combined output index (avoids re-computing index frequently)

    outpixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outpixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outpixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IDoutmask = image_ID("outmask", dcimg, dcnimg);
    if(IDoutmask == -1)
    {
        NBpixout = 0; //xsize*ysize;

        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                outpixarray_x[NBpixout]  = ii;
                outpixarray_y[NBpixout]  = jj;
                outpixarray_xy[NBpixout] = jj * xsize + ii;
                NBpixout++;
            }
    }
    else
    {
        NBpixout = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(dcimg[IDoutmask].array.F[jj * xsize + ii] > 0.5f)
                {
                    outpixarray_x[NBpixout]  = ii;
                    outpixarray_y[NBpixout]  = jj;
                    outpixarray_xy[NBpixout] = jj * xsize + ii;
                    NBpixout++;
                }
    }

    /// ## Reading PFlag from image (optional)
    /// PFlag_run needs to be read before entering the loop as some
    /// array sizes depend on its value.
    if(ExternalPFparam == 1)
    {
        PFlag_run = dcimg[IDPFparam].array.F[0];
    }
    else
    {
        PFlag_run = PFlag;
    }

    /// ## Build Empty Data Matrix
    ///
    /// Note: column / row description follows FITS file viewing conventions.\n
    /// The data matrix is build from the telemetry. Each column (= time sample) of the
    /// data matrix consists of consecutives columns (= time sample) of the input telemetry.\n
    ///
    /// Variable naming:
    /// - NBmvec is the number of telemetry vectors (each corresponding to a different time) in the data matrix.
    /// - mvecsize is the size of each vector, equal to NBpixin times PForder
    ///
    /// Data matrix is stored as image of size NBmvec x mvecsize, to be fed to routine compute_SVDpseudoInverse 
    // in linopt_imtools (CPU mode) or in linalgebra (GPU mode)\n
    ///
    NBmvec =
        nbspl - PForder -
        (int)(PFlag_run) -
        2; // could put "-1", but "-2" allows user to change PFlag_run by up to 1 frame without reading out of array
    mvecsize =
        NBpixin *
        PForder; // size of each sample vector for AR filter, excluding regularization

    /// Regularization can be added to penalize strong coefficients in the predictive filter.
    /// It is optionally implemented by adding extra columns at the end of the data matrix.\n
    if(REG == 0)  // no regularization
    {
        printf("NBmvec   = %ld  -> %ld \n", NBmvec, NBmvec);
        NBmvec1 = NBmvec;
        create_2Dimage_ID("PFmatD", NBmvec, mvecsize, &IDmatA);
    }
    else // with regularization
    {
        printf("NBmvec   = %ld  -> %ld \n", NBmvec, NBmvec + mvecsize);
        NBmvec1 = NBmvec + mvecsize;
        create_2Dimage_ID("PFmatD", NBmvec + mvecsize, mvecsize, &IDmatA);
    }

    IDmatA = image_ID("PFmatD", dcimg, dcnimg);

    /// Data matrix conventions :
    /// - each column (ii = cst) is a measurement
    /// - m index is measurement
    /// - dt*NBpixin+pix index is pixel

    printf("mvecsize = %ld  (%ld x %ld)\n", mvecsize, PForder, NBpixin);
    printf("NBpixin = %ld\n", NBpixin);
    printf("NBpixout = %ld\n", NBpixout);
    printf("NBmvec1 = %ld\n", NBmvec1);
    printf("PForder = %ld\n", PForder);

    printf("xysize = %ld\n", xysize);
    printf("IDin = %ld\n\n", IDin);
    list_image_ID();

    /// ## Predictive Filter Computation
    ///
    /// In LOOP mode, LOOP STARTS HERE \n
    ///

    if(LOOPmode == 1)
    {
        COREMOD_MEMORY_image_set_semflush(IDin_name, semtrig);
    }

    for(iter = 0; iter < NBiter; iter++)
    {

        /// ### Prepare data matrix PFmatD

        /// *STEP: Read parameters from external image (optional)*\n
        if(ExternalPFparam == 1)
        {
            PFlag_run     = dcimg[IDPFparam].array.F[0];
            SVDeps_run    = dcimg[IDPFparam].array.F[1];
            RegLambda_run = dcimg[IDPFparam].array.F[2];
            LOOPgain_run  = dcimg[IDPFparam].array.F[3];
        }
        else
        {
            PFlag_run     = PFlag;
            SVDeps_run    = SVDeps;
            RegLambda_run = RegLambda;
            LOOPgain_run  = LOOPgain;
        }

        printf(
            "=========== LOOP ITERATION %6ld ======= [ExternalPFparam = %d ]\n",
            iter,
            ExternalPFparam);
        printf(" parameters read from %s\n", dcimg[IDPFparam].md[0].name);
        printf("  PFlag     = %20f      ", PFlag_run);
        printf("  SVDeps    = %20f\n", SVDeps_run);
        printf("  RegLambda = %20f      ", RegLambda_run);
        printf("  LOOPgain  = %20f\n", LOOPgain_run);
        printf("\n");

        gain = 1.0 / (iter + 1);
        if(gain < LOOPgain_run)
        {
            gain = LOOPgain_run;
        }

        /// *STEP: In loop mode, wait for input data to arrive*

        printf("WAITING FOR INPUT DATA ...... \n");
        clock_gettime(CLOCK_MILK, &t0);
        if(LOOPmode == 1)
        {
            ImageStreamIO_semwait(dcimg+IDin, semtrig);
        }

        /// *STEP: Copy IDin to IDincp*
        ///
        /// Necessary as input may be continuously changing between consecutive loop iterations.
        ///
        IDincp = image_ID("PFin_copy", dcimg, dcnimg);
        memcpy(dcimg[IDincp].array.F,
               dcimg[IDin].array.F,
               sizeof(float) * inNBelem);

        //save_fits("PFin_copy", "test_PFin_copy.fits");
        //save_fits(IDin_name, "test_PFin.fits");

        clock_gettime(CLOCK_MILK, &t1);

        /// *STEP: if DC_MODE==1, compute average value from each variable*
        if(DC_MODE == 1)  // remove average
        {
            for(pix = 0; pix < NBpixin; pix++)
            {
                ave_inarray[pix] = 0.0;
                for(m = 0; m < nbspl; m++)
                {
                    ave_inarray[pix] +=
                        dcimg[IDincp]
                        .array.F[m * xysize + pixarray_xy[pix]];
                }
                ave_inarray[pix] /= nbspl;
            }
        }
        else
        {
            for(pix = 0; pix < NBpixin; pix++)
            {
                ave_inarray[pix] = 0.0;
            }
        }

        ///
        /// *STEP: Fill up data matrix PFmatD from input telemetry*
        ///
        for(m = 0; m < NBmvec1; m++)
        {
            k0 = m + PForder - 1; // dt=0 index
            for(pix = 0; pix < NBpixin; pix++)
                for(dt = 0; dt < PForder; dt++)
                {
                    dcimg[IDmatA]
                    .array.F[(NBpixin * dt + pix) * NBmvec1 + m] =
                        dcimg[IDincp]
                        .array.F[(k0 - dt) * xysize + pixarray_xy[pix]] -
                        ave_inarray[pix];
                }
        }

        if(LOOPmode == 0)
        {
            free(ave_inarray); // No need to hold on to array
        }

        ///
        /// *STEP: Write regularization coefficients (optional)*
        ///
        if(REG == 1)
        {
            for(m = 0; m < mvecsize; m++)
            {
                //m1 = NBmvec + m;
                dcimg[IDmatA].array.F[(m) *NBmvec1 + (NBmvec + m)] =
                    RegLambda_run;
            }
        }

        if(Save == 1)
        {
            save_fits("PFmatD", "PFmatD.fits");
        }
        //list_image_ID();

        /// ### Compute pseudo-inverse of PFmatD
        ///
        /// *STEP: Compute Pseudo-Inverse of PFmatD*
        ///
        printf("Assembling pseudoinverse\n");
        fflush(stdout);

        // Assemble future measured data matrix
        imageID IDfm;
        create_2Dimage_ID("PFfmdat", NBmvec, NBpixout, &IDfm);

        alpha = PFlag_run - ((long) PFlag_run);
        for(PFpix = 0; PFpix < NBpixout; PFpix++)
            for(m = 0; m < NBmvec; m++)
            {
                k0 = m + PForder - 1;
                k0 += (long) PFlag_run;

                dcimg[IDfm].array.F[PFpix * NBmvec + m] =
                    (1.0 - alpha) *
                    dcimg[IDincp]
                    .array.F[(k0) * xysize + outpixarray_xy[PFpix]] +
                    alpha *
                    dcimg[IDincp]
                    .array.F[(k0 + 1) * xysize + outpixarray_xy[PFpix]];
            }
        save_fits("PFfmdat", "PFfmdat.fits");

        /// If using MAGMA, call function LINALGEBRA_magma_compute_SVDpseudoInverse()\n
        /// Otherwise, call function linopt_compute_SVDpseudoInverse()\n

        NB_SVD_Modes = 10000;

#ifdef HAVE_MAGMA
        printf("Using magma ...\n");
        LINALGEBRA_magma_compute_SVDpseudoInverse("PFmatD",
                                                "PFmatC",
                                                SVDeps_run,
                                                NB_SVD_Modes,
                                                "PF_VTmat",
                                                LOOPmode,
                                                testmode,
                                                64,
                                                0, // GPU device
                                                NULL);
#else
        printf("Not using magma ...\n");
        linopt_compute_SVDpseudoInverse("PFmatD",
                                        "PFmatC",
                                        SVDeps_run,
                                        NB_SVD_Modes,
                                        "PF_VTmat",
                                        NULL);
#endif

        /// Result (pseudoinverse) is stored in image PFmatC\n
        printf("Done assembling pseudoinverse\n");
        fflush(stdout);

        if(Save == 1)
        {
            save_fits("PF_VTmat", "PF_VTmat.fits");
            save_fits("PFmatC", "PFmatC.fits");
        }
        IDmatC = image_ID("PFmatC", dcimg, dcnimg);

        ///
        /// ### Assemble Predictive Filter
        ///
        printf("Compute filters\n");
        fflush(stdout);

        if(system("mkdir -p pixfilters") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }

        // 3D FILTER MATRIX - contains all pixels
        // axis 0 [ii] : input mode
        // axis 1 [jj] : reconstructed mode
        // axis 2 [kk] : time step

        // 2D Filter - contains only used input and output
        // axis 0 [ii1] : input mode x time step
        // axis 1 [jj1] : output mode

        if(LOOPmode == 0)
        {
            create_2Dimage_ID(IDoutPF_name,
                              NBpixin * PForder,
                              NBpixout,
                              &IDoutPF2D);
        }

        else
        {
            if(iter == 0)  // create 2D and 3D filters as shared memory
            {
                imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
                if(imsizearray == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                imsizearray[0] = NBpixin * PForder;
                imsizearray[1] = NBpixout;
                snprintf(IDoutPF_name_raw,
                         sizeof(IDoutPF_name_raw),
                         "%s_raw", IDoutPF_name);

                {
                    IMGID imgpf =
                        imgid_make_from_name(
                            IDoutPF_name);
                    imgpf.mdt->naxis = 2;
                    imgpf.mdt->size[0] =
                        imsizearray[0];
                    imgpf.mdt->size[1] =
                        imsizearray[1];
                    imgpf.mdt->datatype =
                        _DATATYPE_FLOAT;
                    imgpf.mdt->shared = 1;
                    imgpf.mdt->NBkw = 1;
                    imgpf.im =
                        (IMAGE *) calloc(
                            1,
                            sizeof(IMAGE));
                    imgid_mkimage(&imgpf);
                    IDoutPF2D = imgpf.ID;
                }
                {
                    IMGID imgpfr =
                        imgid_make_from_name(
                            IDoutPF_name_raw);
                    imgpfr.mdt->naxis = 2;
                    imgpfr.mdt->size[0] =
                        imsizearray[0];
                    imgpfr.mdt->size[1] =
                        imsizearray[1];
                    imgpfr.mdt->datatype =
                        _DATATYPE_FLOAT;
                    imgpfr.mdt->shared = 1;
                    imgpfr.mdt->NBkw = 1;
                    imgpfr.im =
                        (IMAGE *) calloc(
                            1,
                            sizeof(IMAGE));
                    imgid_mkimage(&imgpfr);
                    IDoutPF2Draw =
                        imgpfr.ID;
                }
                free(imsizearray);
                COREMOD_MEMORY_image_set_semflush(IDoutPF_name, -1);
                COREMOD_MEMORY_image_set_semflush(IDoutPF_name_raw, -1);
            }
            else
            {
                IDoutPF2D = image_ID(IDoutPF_name, dcimg, dcnimg);
            }
        }

        IDoutmask = image_ID("outmask", dcimg, dcnimg);

        printf("===========================================================\n");
        printf("ASSEMBLING OUTPUT\n");
        printf("  NBpixout = %ld\n", NBpixout);
        printf("  NBmvec   = %ld\n", NBmvec);
        printf("  NBmvec1  = %ld\n", NBmvec1);
        printf("  NBpixin  = %ld\n", NBpixin);
        printf("  PForder  = %ld\n", PForder);
        printf("===========================================================\n");

        long IDoutPF2Dn = image_ID("psinvPFmat", dcimg, dcnimg);
        if(IDoutPF2Dn == -1)
        {
            printf("------------------- CPU computing PF matrix\n");

            create_2Dimage_ID("psinvPFmat",
                              NBpixin * PForder,
                              NBpixout,
                              &IDoutPF2Dn);
            for(
                PFpix = 0; PFpix < NBpixout;
                PFpix++) // PFpix is the pixel for which the filter is created (axis 1 in cube, jj)
            {

                // loop on input values
                for(pix = 0; pix < NBpixin; pix++)
                {
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val  = 0.0;
                        ind1 = (NBpixin * dt + pix) * NBmvec1;
                        for(m = 0; m < NBmvec; m++)
                        {
                            val += dcimg[IDmatC].array.F[ind1 + m] *
                                   dcimg[IDfm].array.F[PFpix * NBmvec + m];
                        }

                        dcimg[IDoutPF2Dn]
                        .array.F[PFpix * (PForder * NBpixin) +
                                       dt * NBpixin + pix] = val;
                    }
                }
            }
        }
        else
        {
            printf("------------------- Using GPU-computed PF matrix\n");
        }
        delete_image_ID("PFfmdat", DELETE_IMAGE_ERRMODE_WARNING);

        if(LOOPmode == 1)
        {
            SHMIM_WRITE_ACQUIRE(&dcimg[IDoutPF2Draw].md[0]);
            memcpy(dcimg[IDoutPF2Draw].array.F,
                   dcimg[IDoutPF2Dn].array.F,
                   sizeof(float) * NBpixout * NBpixin * PForder);
            COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2Draw, -1);
            SHMIM_CNT0_INCREMENT(&dcimg[IDoutPF2Draw].md[0]);
            SHMIM_WRITE_RELEASE(&dcimg[IDoutPF2Draw].md[0]);
        }

        // Mix current PF with last one
        SHMIM_WRITE_ACQUIRE(&dcimg[IDoutPF2D].md[0]);
        if(LOOPmode == 0)
        {
            memcpy(dcimg[IDoutPF2D].array.F,
                   dcimg[IDoutPF2Dn].array.F,
                   sizeof(float) * NBpixout * NBpixin * PForder);
            save_fits(IDoutPF_name, "_outPF.fits");
        }
        else
        {
            printf("Mixing PF matrix with gain = %f ....", gain);
            fflush(stdout);
            for(PFpix = 0; PFpix < NBpixout; PFpix++)
                for(pix = 0; pix < NBpixin; pix++)
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val0 = dcimg[IDoutPF2D]
                               .array.F[PFpix * (PForder * NBpixin) +
                                              dt * NBpixin + pix]; // Previous
                        val = dcimg[IDoutPF2Dn]
                              .array.F[PFpix * (PForder * NBpixin) +
                                             dt * NBpixin + pix]; // New
                        dcimg[IDoutPF2D]
                        .array.F[PFpix * (PForder * NBpixin) +
                                       dt * NBpixin + pix] =
                                     (1.0 - gain) * val0 + gain * val;
                    }
            printf(" done\n");
            fflush(stdout);
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2D, -1);
        SHMIM_CNT0_INCREMENT(&dcimg[IDoutPF2D].md[0]);
        SHMIM_WRITE_RELEASE(&dcimg[IDoutPF2D].md[0]);

        if(testmode == 2)
        {
            printf("Prepare 3D output \n");

            imageID IDoutPF3D;
            create_3Dimage_ID("outPF3D",
                              NBpixin,
                              NBpixout,
                              PForder,
                              &IDoutPF3D);

            for(pix = 0; pix < NBpixin; pix++)
                for(PFpix = 0; PFpix < NBpixout; PFpix++)
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val = dcimg[IDoutPF2D]
                              .array.F[PFpix * (PForder * NBpixin) +
                                             dt * NBpixin + pix];
                        dcimg[IDoutPF3D].array.F[NBpixout * NBpixin * dt +
                                                      NBpixin * PFpix + pix] =
                                                          val;
                    }
            save_fits("outPF3D", "_outPF3D.fits");
        }

        printf("DONE\n");
        fflush(stdout);
        clock_gettime(CLOCK_MILK, &t2);

        tdiff    = timespec_diff(t0, t1);
        tdiffv01 = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff    = timespec_diff(t1, t2);
        tdiffv12 = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        printf("Computing time = %5.3f s / %5.3f s -> fraction = %8.6f\n",
               tdiffv12,
               tdiffv01 + tdiffv12,
               tdiffv12 / (tdiffv01 + tdiffv12));
    }
    ///
    /// In LOOP mode, LOOP ENDS HERE \n
    ///

    // free(valfarray);

    free(pixarray_x);
    free(pixarray_y);
    free(pixarray_xy);

    free(outpixarray_x);
    free(outpixarray_y);
    free(outpixarray_xy);

    ///
    /// ---
    ///

    return IDoutPF2D;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 4. APPLY PREDICTIVE FILTER                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

//
// real-time apply predictive filter
//
// filter can be smaller than input telemetry but needs to include contiguous pixels at the beginning of the input telemetry
//
imageID LINARFILTERPRED_Apply_LinPredictor_RT(const char *IDfilt_name,
        const char *IDin_name,
        const char *IDout_name)
{
    imageID   IDout;
    imageID   IDin;
    imageID   IDfilt;
    long      PForder;
    long      NBpix_in;
    long      NBpix_out;
    uint32_t *imsizearray;
    int       semtrig = 7;

    float *inarray;
    float *outarray;

    //    long ii; // input index
    //    long jj; // output index
    //    long kk; // time step index

    IDfilt = image_ID(IDfilt_name, dcimg, dcnimg);
    IDin   = image_ID(IDin_name, dcimg, dcnimg);

    PForder   = dcimg[IDfilt].md[0].size[2];
    NBpix_in  = dcimg[IDfilt].md[0].size[0];
    NBpix_out = dcimg[IDfilt].md[0].size[1];

    list_image_ID();

    if(dcimg[IDin].md[0].size[0] * dcimg[IDin].md[0].size[1] !=
            NBpix_in)
    {
        printf(
            "ERROR: lin predictor engine: filter input size does not match "
            "input telemetry\n");
        exit(0);
    }

    printf("Create prediction output %s\n", IDout_name);
    fflush(stdout);
    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imsizearray[0] = NBpix_out;
    imsizearray[1] = 1;
    {
        IMGID imgout_tmp =
            imgid_make_from_name(
                IDout_name);
        imgout_tmp.mdt->naxis = 2;
        imgout_tmp.mdt->size[0] =
            imsizearray[0];
        imgout_tmp.mdt->size[1] =
            imsizearray[1];
        imgout_tmp.mdt->datatype =
            _DATATYPE_FLOAT;
        imgout_tmp.mdt->shared = 1;
        imgout_tmp.mdt->NBkw = 1;
        imgout_tmp.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }
    free(imsizearray);
    COREMOD_MEMORY_image_set_semflush(IDout_name, -1);
    printf("Done\n");
    fflush(stdout);

    inarray = (float *) malloc(sizeof(float) * NBpix_in * PForder);
    if(inarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outarray = (float *) malloc(sizeof(float) * NBpix_out);
    if(outarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    while(ImageStreamIO_semtrywait(dcimg+IDin, semtrig) == 0)
    {
    }
    while(1)
    {
        // initialize output array to zero
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            outarray[jj] = 0.0;
        }

        // shift input buffer entries back one time step
        for(uint32_t kk = PForder - 1; kk > 0; kk--)
            for(uint32_t ii = 0; ii < NBpix_in; ii++)
            {
                inarray[kk * NBpix_in + ii] = inarray[(kk - 1) * NBpix_in + ii];
            }

        // multiply input by prediction matrix .. except for measurement yet to come
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
            for(uint32_t ii = 0; ii < NBpix_in; ii++)
                for(uint32_t kk = 1; kk < PForder; kk++)
                {
                    outarray[jj] +=
                        dcimg[IDfilt].array.F[kk * NBpix_in * NBpix_out +
                                                   jj * NBpix_in + ii] *
                        inarray[kk * NBpix_in + ii];
                }

        ImageStreamIO_semwait(dcimg+IDin, semtrig);

        // write new input in inarray vector
        for(uint32_t ii = 0; ii < NBpix_in; ii++)
        {
            inarray[ii] = dcimg[IDin].array.F[ii];
        }

        // multiply input by prediction matrix
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
            for(uint32_t ii = 0; ii < NBpix_in; ii++)
            {
                outarray[jj] += dcimg[IDfilt].array.F[jj * NBpix_in + ii] *
                                inarray[ii];
            }

        SHMIM_WRITE_ACQUIRE(&dcimg[IDout].md[0]);
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            dcimg[IDout].array.F[jj] = outarray[jj];
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        SHMIM_CNT0_INCREMENT(&dcimg[IDout].md[0]);
        SHMIM_WRITE_RELEASE(&dcimg[IDout].md[0]);
    }

    free(inarray);
    free(outarray);

    return IDout;
}

//
//
// out : prediction
//
// ADDITIONAL OUTPUTS:
// outf : time-shifted measurement
//

imageID LINARFILTERPRED_Apply_LinPredictor(const char *IDfilt_name,
        const char *IDin_name,
        float       PFlag,
        const char *IDout_name)
{
    imageID  IDout;
    imageID  IDin;
    imageID  IDfilt;
    uint32_t xsize;
    uint32_t ysize;
    uint64_t xysize;

    long  nbspl;
    long  PForder;
    long  step;
    long  kk;
    float alpha;
    long  PFlagl;
    float valp, valf;

    imageID IDoutf;

    IDin   = image_ID(IDin_name, dcimg, dcnimg);
    IDfilt = image_ID(IDfilt_name, dcimg, dcnimg);

    switch(dcimg[IDin].md[0].naxis)
    {

        case 2:
            nbspl = dcimg[IDin].md[0].size[1];
            xsize = dcimg[IDin].md[0].size[0];
            ysize = 1;
            create_2Dimage_ID(IDout_name, xsize, nbspl, &IDout);
            create_2Dimage_ID("outf", xsize, nbspl, &IDoutf);
            break;

        case 3:
            nbspl = dcimg[IDin].md[0].size[2];
            xsize = dcimg[IDin].md[0].size[0];
            ysize = dcimg[IDin].md[0].size[1];
            create_3Dimage_ID(IDout_name, xsize, ysize, nbspl, &IDout);
            create_3Dimage_ID("outf", xsize, ysize, nbspl, &IDoutf);
            break;

        default:
            printf("Invalid image size\n");
            break;
    }
    xysize = xsize * ysize;

    PForder = dcimg[IDfilt].md[0].size[2];

    if((dcimg[IDfilt].md[0].size[0] != xysize) ||
            (dcimg[IDfilt].md[0].size[1] != xysize))
    {
        printf("ERROR: filter \"%s\" size is incorrect\n", IDfilt_name);
        exit(0);
    }

    alpha  = PFlag - ((long) PFlag);
    PFlagl = (long) PFlag;

    for(kk = PForder; kk < nbspl; kk++)  // time step
    {
        for(uint32_t iip = 0; iip < xysize; iip++)  // predicted variable
        {
            valp = 0.0; // prediction
            for(step = 0; step < PForder; step++)
            {
                for(uint32_t ii = 0; ii < xsize * ysize;
                        ii++) // input variable
                {
                    valp += dcimg[IDfilt].array.F[xysize * xysize * step +
                                                       iip * xysize + ii] *
                            dcimg[IDin].array.F[(kk - step) * xysize + ii];
                }
            }
            dcimg[IDout].array.F[kk * xysize + iip] = valp;

            valf = 0.0;
            if(kk + PFlag + 1 < nbspl)
            {
                valf =
                    (1.0 - alpha) *
                    dcimg[IDin].array.F[(kk + PFlagl) * xysize + iip] +
                    alpha * dcimg[IDin]
                    .array.F[(kk + PFlagl + 1) * xysize + iip];
            }
            dcimg[IDoutf].array.F[kk * xysize + iip] = valf;
        }
    }

    return IDout;
}

//
// IDPF_name and IDPFM_name should be pre-loaded
//
imageID LINARFILTERPRED_PF_updatePFmatrix(const char *IDPF_name,
        const char *IDPFM_name,
        float       alpha)
{
    imageID IDPF;
    imageID IDPFM;
    long    inmode, NBmode, outmode, NBmode2;
    long    tstep, NBtstep;

    uint32_t *sizearray;
    uint8_t   naxis;

    // IDPF should be square
    IDPF    = image_ID(IDPF_name, dcimg, dcnimg);
    NBmode  = dcimg[IDPF].md[0].size[0];
    NBmode2 = NBmode * NBmode;
    assert(dcimg[IDPF].md[0].size[0] == dcimg[IDPF].md[0].size[1]);
    NBtstep = dcimg[IDPF].md[0].size[2];

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = NBmode * NBtstep;
    sizearray[1] = NBmode;
    naxis        = 2;

    IDPFM = image_ID(IDPFM_name, dcimg, dcnimg);

    if(IDPFM == -1)
    {
        printf("Creating shared mem image %s  [ %ld  x  %ld ]\n",
               IDPFM_name,
               (long) sizearray[0],
               (long) sizearray[1]);
        fflush(stdout);
        {
            IMGID imgpfm =
                imgid_make_from_name(
                    IDPFM_name);
            imgpfm.mdt->naxis = naxis;
            imgpfm.mdt->size[0] =
                sizearray[0];
            imgpfm.mdt->size[1] =
                sizearray[1];
            imgpfm.mdt->datatype =
                _DATATYPE_FLOAT;
            imgpfm.mdt->shared = 1;
            imgpfm.im =
                (IMAGE *) calloc(
                    1, sizeof(IMAGE));
            imgid_mkimage(&imgpfm);
            IDPFM = imgpfm.ID;
        }
    }
    free(sizearray);

    SHMIM_WRITE_ACQUIRE(&dcimg[IDPFM].md[0]);
    for(outmode = 0; outmode < NBmode; outmode++)
    {
        for(tstep = 0; tstep < NBtstep; tstep++)
            for(inmode = 0; inmode < NBmode; inmode++)
                dcimg[IDPFM].array.F[outmode * (NBmode * NBtstep) +
                                          tstep * NBmode + inmode] =
                                              (1.0 - alpha) *
                                              dcimg[IDPFM].array.F[outmode * (NBmode * NBtstep) +
                                                      tstep * NBmode + inmode] +
                                              alpha * dcimg[IDPF].array.F[tstep * NBmode2 +
                                                      outmode * NBmode + inmode];
    }
    COREMOD_MEMORY_image_set_sempost_byID(IDPFM, -1);
    SHMIM_WRITE_RELEASE(&dcimg[IDPFM].md[0]);
    SHMIM_CNT0_INCREMENT(&dcimg[IDPFM].md[0]);

    return IDPFM;
}

//
// IDmodevalIN_name : open loop modal coefficients
// IndexOffset      : predicted mode start at this input index
// semtrig          : semaphore trigger index in input input
// IDPFM_name       : predictive filter matrix
// IDPFout_name     : prediction
//
//  NBiter: run for fixed number of iteration
//  SAVEMODE:   0 no file output
//  			1	write txt and FITS output
//				2	write FITS telemetry with prediction: replace output measurements with predictions
//
//	tlag is only used if SAVEMODE = 2
//  used outmask to identify outputs
//
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
        long        PFindex)
{
    imageID IDmodevalIN;
    long    NBmodeIN, NBmodeIN0, NBmodeOUT, mode;
    imageID IDPFM;

    imageID   IDINbuff;
    long      tstep;
    uint32_t *sizearray;
    uint8_t   naxis;

    imageID IDPFout;

    int *GPUsetPF;
    char GPUsetfname[200];
    int  gpuindex;

#ifdef HAVE_CUDA
    int status;
    int GPUstatus[100];
    int GPUMATMULTCONFindex = 2;
#endif

    FILE *fp;

    //time_t t;
    //struct tm *uttime;
    struct timespec timenow;
    double          timesec, timesec0;
    long            IDsave;

    FILE *fpout;
    long  iter;
    long  kk;

    imageID IDinmask;
    long   *inmaskindex;
    long    NBinmaskpix;

    long  tlag0;
    float tlagalpha = 0.0;

    imageID IDoutmask;
    long   *outmaskindex;
    long    NBoutmaskpix;
    long    kk0, kk1;
    float   val, val0, val1;
    long    ii0, ii1;

    long IDmasterout;
    char imname[200];

    IDmodevalIN = image_ID(IDmodevalIN_name, dcimg, dcnimg);
    NBmodeIN0   = dcimg[IDmodevalIN].md[0].size[0];

    IDPFM     = image_ID(IDPFM_name, dcimg, dcnimg);
    NBmodeOUT = dcimg[IDPFM].md[0].size[1];

    snprintf(imname, sizeof(imname),
             "aol%ld_modevalPF", loop);
    IDmasterout = image_ID(imname, dcimg, dcnimg);

    IDinmask = image_ID("inmask", dcimg, dcnimg);
    if(IDinmask != -1)
    {
        NBinmaskpix = 0;
        for(uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
            if(dcimg[IDinmask].array.F[ii] > 0.5f)
            {
                NBinmaskpix++;
            }

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);
        if(inmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBinmaskpix = 0;
        for(uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
            if(dcimg[IDinmask].array.F[ii] > 0.5f)
            {
                inmaskindex[NBinmaskpix] = ii;
                NBinmaskpix++;
            }
        //printf("Number of active input modes  = %ld\n", NBinmaskpix);
    }
    else
    {
        NBinmaskpix = NBmodeIN0;
        printf("no input mask -> assuming NBinmaskpix = %ld\n", NBinmaskpix);
        create_2Dimage_ID("inmask", NBinmaskpix, 1, &IDinmask);
        for(uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
        {
            dcimg[IDinmask].array.F[ii] = 1.0f;
        }

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);
        if(inmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for(uint32_t ii = 0; ii < dcimg[IDinmask].md[0].size[0]; ii++)
        {
            inmaskindex[NBinmaskpix] = ii;
        }
    }
    NBmodeIN = NBinmaskpix;

    NBPFstep = dcimg[IDPFM].md[0].size[0] / NBmodeIN;

    printf("Number of input modes         = %ld\n", NBmodeIN0);
    printf("Number of active input modes  = %ld\n", NBmodeIN);
    printf("Number of output modes        = %ld\n", NBmodeOUT);
    printf("Number of time steps          = %ld\n", NBPFstep);
    if(IDmasterout != -1)
    {
        printf("Writing result in master output stream %s  (%ld)\n",
               imname,
               IDmasterout);
    }

    if((SAVEMODE > 0) || (IDmasterout != -1))
    {
        IDoutmask = image_ID("outmask", dcimg, dcnimg);
        if(IDoutmask == -1)
        {
            printf("ERROR: outmask image required\n");
            exit(0);
        }
        NBoutmaskpix = 0;
        for(uint32_t ii = 0; ii < dcimg[IDoutmask].md[0].size[0]; ii++)
            if(dcimg[IDoutmask].array.F[ii] > 0.5f)
            {
                NBoutmaskpix++;
            }

        outmaskindex = (long *) malloc(sizeof(long) * NBoutmaskpix);
        if(outmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBoutmaskpix = 0;
        for(uint32_t ii = 0; ii < dcimg[IDoutmask].md[0].size[0]; ii++)
            if(dcimg[IDoutmask].array.F[ii] > 0.5f)
            {
                outmaskindex[NBoutmaskpix] = ii;
                NBoutmaskpix++;
            }
        if(NBoutmaskpix != NBmodeOUT)
        {
            printf("ERROR: NBoutmaskpix (%ld)   !=   NBmodeOUT (%ld)\n",
                   NBoutmaskpix,
                   NBmodeOUT);
            list_image_ID();
            exit(0);
        }
    }

    create_2Dimage_ID("INbuffer", NBmodeIN, NBPFstep, &IDINbuff);

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = NBmodeOUT;
    sizearray[1] = 1;
    naxis        = 2;
    IDPFout      = image_ID(IDPFout_name, dcimg, dcnimg);

    if(IDPFout == -1)
    {
        {
            IMGID imgpfout =
                imgid_make_from_name(
                    IDPFout_name);
            imgpfout.mdt->naxis = naxis;
            imgpfout.mdt->size[0] =
                sizearray[0];
            imgpfout.mdt->size[1] =
                sizearray[1];
            imgpfout.mdt->datatype =
                _DATATYPE_FLOAT;
            imgpfout.mdt->shared = 1;
            imgpfout.im =
                (IMAGE *) calloc(
                    1, sizeof(IMAGE));
            imgid_mkimage(&imgpfout);
            IDPFout = imgpfout.ID;
        }
    }
    free(sizearray);

    if(nbGPU > 0)
    {
        GPUsetPF = (int *) malloc(sizeof(int) * nbGPU);
        if(GPUsetPF == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for(gpuindex = 0; gpuindex < nbGPU; gpuindex++)
        {
            snprintf(GPUsetfname, sizeof(GPUsetfname),
                     "./conf/param_PFb%ldGPU%ddevice.txt",
                     PFindex,
                     gpuindex);
            fp = fopen(GPUsetfname, "r");
            if(fp == NULL)
            {
                printf("ERROR: file %s not found\n", GPUsetfname);
                exit(0);
            }
            if(fscanf(fp, "%d", &GPUsetPF[gpuindex]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
            fclose(fp);
        }
        printf("USING %d GPUs: ", nbGPU);
        for(gpuindex = 0; gpuindex < nbGPU; gpuindex++)
        {
            printf(" %d", GPUsetPF[gpuindex]);
        }
        printf("\n\n");
    }
    else
    {
        printf("Using CPU\n");
    }

    iter = 0;
    if(SAVEMODE > 0)
        if(NBiter > 50000)
        {
            NBiter = 50000;
        }

    if(SAVEMODE == 1)
    {
        create_2Dimage_ID("testPFsave",
                          1 + NBmodeIN0 + NBmodeOUT,
                          NBiter,
                          &IDsave);
    }
    if(SAVEMODE == 2)
    {
        create_3Dimage_ID("testPFTout", NBmodeIN0, 1, NBiter, &IDsave);
    }

    //	t = time(NULL);
    //    uttime = gmtime(&t);
    //	clock_gettime(CLOCK_MILK, &timenow);
    //	timesec0 = 3600.0*uttime->tm_hour  + 60.0*uttime->tm_min + 1.0*(timenow.tv_sec % 60) + 1.0e-9*timenow.tv_nsec;

    printf("Running on semaphore trigger %d of image %s\n",
           semtrig,
           dcimg[IDmodevalIN].md[0].name);

    while(iter != NBiter)
    {
        //	printf("iter %5ld / %5ld", iter, NBiter);
        //	fflush(stdout);

        ImageStreamIO_semwait(dcimg+IDmodevalIN, semtrig);
        //	printf("\n");
        //	fflush(stdout);

        // fill in buffer
        for(mode = 0; mode < NBmodeIN; mode++)
        {
            dcimg[IDINbuff].array.F[mode] =
                dcimg[IDmodevalIN]
                .array.F[IndexOffset + inmaskindex[mode]];
        }

        //
        // Main matrix multiplication is done here
        // input vector contains recent history of mode coefficients
        // output vector contains the predicted mode coefficients
        //
        if(nbGPU > 0)  // if using GPU
        {

#ifdef HAVE_CUDA
            if(iter == 0)
            {
                printf("INITIALIZE GPU(s)\n\n");
                fflush(stdout);

                GPU_loop_MultMat_setup(GPUMATMULTCONFindex,
                                       IDPFM_name,
                                       "INbuffer",
                                       IDPFout_name,
                                       nbGPU,
                                       GPUsetPF,
                                       0,
                                       1,
                                       1,
                                       loop);

                printf("INITIALIZATION DONE\n\n");
                fflush(stdout);
            }
            GPU_loop_MultMat_execute(GPUMATMULTCONFindex,
                                     &status,
                                     &GPUstatus[100],
                                     1.0,
                                     0.0,
                                     0,
                                     0);
#endif
        }
        else // if using CPU
        {
            // compute output : matrix vector mult with a CPU-based loop
            SHMIM_WRITE_ACQUIRE(&dcimg[IDPFout].md[0]);
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDPFout].array.F[mode] = 0.0f;
                for(uint32_t ii = 0; ii < NBmodeIN * NBPFstep; ii++)
                {
                    dcimg[IDPFout].array.F[mode] +=
                        dcimg[IDINbuff].array.F[ii] *
                        dcimg[IDPFM]
                        .array
                        .F[mode * dcimg[IDPFM].md[0].size[0] + ii];
                }
            }
            COREMOD_MEMORY_image_set_sempost_byID(IDPFout, -1);
            SHMIM_WRITE_RELEASE(&dcimg[IDPFout].md[0]);
            SHMIM_CNT0_INCREMENT(&dcimg[IDPFout].md[0]);
        }

        if(iter == 0)
        {
            /// measure time
            //t = time(NULL);
            //uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, &timenow);
            timesec0 = 1.0 * timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

            // fprintf(fp, "%02d:%02d:%02ld.%09ld ", uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
        }

        if(SAVEMODE == 1)
        {
            //		printf("	Saving step (mode = 1) ...");
            //		fflush(stdout);

            //t = time(NULL);
            //uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, &timenow);
            timesec = 1.0 * timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

            kk = 0;
            dcimg[IDsave].array.F[iter * (1 + NBmodeIN0 + NBmodeOUT)] =
                (float)(timesec - timesec0);
            //printf(" [%f] ", dcimg[IDsave].array.F[iter*(1+NBmodeIN0+NBmodeOUT)]);
            kk++;
            for(mode = 0; mode < NBmodeIN0; mode++)
            {
                dcimg[IDsave]
                .array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk] =
                    dcimg[IDmodevalIN].array.F[IndexOffset + mode];
                kk++;
            }
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDsave]
                .array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk] =
                    dcimg[IDPFout].array.F[mode];
                kk++;
            }
            //	printf(" done\n");
            //	fflush(stdout);
        }
        if(SAVEMODE == 2)
        {
            //	printf("	Saving step (mode = 2) ...");
            //	fflush(stdout);

            for(mode = 0; mode < NBmodeIN0; mode++)
            {
                dcimg[IDsave].array.F[iter * NBmodeIN0 + mode] =
                    dcimg[IDmodevalIN].array.F[IndexOffset + mode];
            }
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDsave]
                .array.F[iter * NBmodeIN0 + outmaskindex[mode]] =
                    dcimg[IDPFout].array.F[mode];
            }
            //	printf(" done\n");
            //	fflush(stdout);
        }

        if(IDmasterout != -1)
        {
            SHMIM_WRITE_ACQUIRE(&dcimg[IDmasterout].md[0]);
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                dcimg[IDmasterout].array.F[outmaskindex[mode]] =
                    dcimg[IDPFout].array.F[mode];
            }
            COREMOD_MEMORY_image_set_sempost_byID(IDmasterout, -1);
            SHMIM_WRITE_RELEASE(&dcimg[IDmasterout].md[0]);
            SHMIM_CNT0_INCREMENT(&dcimg[IDmasterout].md[0]);
        }

        iter++;

        if(iter != NBiter)
        {
            // do this now to save time when semaphore is posted
            for(tstep = NBPFstep - 1; tstep > 0; tstep--)
            {
                // tstep-1 -> tstep
                for(mode = 0; mode < NBmodeIN; mode++)
                {
                    dcimg[IDINbuff].array.F[NBmodeIN * tstep + mode] =
                        dcimg[IDINbuff]
                        .array.F[NBmodeIN * (tstep - 1) + mode];
                }
            }
        }
    }
    printf("LOOP done\n");
    fflush(stdout);

    // output ASCII file
    if(SAVEMODE == 1)
    {
        printf("SAVING DATA [1] ...");
        fflush(stdout);

        printf("IDsave = %ld     %ld  %ld\n",
               IDsave,
               1 + NBmodeIN0 + NBmodeOUT,
               NBmodeOUT);
        list_image_ID();

        //	for(mode=0;mode<NBmodeOUT;mode++)
        //	printf("output %4ld -> %5ld\n", outmaskindex[mode]);

        fpout = fopen("testPFsave.dat", "w");
        for(iter = 0; iter < NBiter; iter++)
        {
            fprintf(fpout, "%5ld ", iter);
            for(kk = 0; kk < (1 + NBmodeIN0 + NBmodeOUT); kk++)
            {
                fprintf(fpout,
                        "%10f ",
                        dcimg[IDsave]
                        .array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk]);
            }

            tlag0     = (long) tlag;
            tlagalpha = tlag - tlag0;

            ii0 = iter - (tlag0 + 1);
            ii1 = iter - (tlag0);

            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                if(ii0 > -1)
                {
                    val0 = dcimg[IDsave]
                           .array.F[ii0 * (1 + NBmodeIN0 + NBmodeOUT) + 1 +
                                        NBmodeIN0 + mode];
                    val1 = dcimg[IDsave]
                           .array.F[ii1 * (1 + NBmodeIN0 + NBmodeOUT) + 1 +
                                        NBmodeIN0 + mode];
                }
                val = tlagalpha * val0 + (1.0 - tlagalpha) * val1;
                fprintf(fpout, "%10f ", val);
            }
            fprintf(fpout, "\n");
        }
        fclose(fpout);

        printf(" done\n");
        fflush(stdout);
    }

    free(inmaskindex);

    if(SAVEMODE == 2)  // time shift predicted output into FITS output
    {
        tlag0     = (long) tlag;
        tlagalpha = tlag - tlag0;
        for(kk = NBiter - 1; kk > tlag0; kk--)
        {
            kk0 = kk - (tlag0 + 1);
            kk1 = kk - (tlag0);

            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                val0 = dcimg[IDmodevalIN]
                       .array.F[kk0 * NBmodeIN0 + outmaskindex[mode]];
                val1 = dcimg[IDmodevalIN]
                       .array.F[kk1 * NBmodeIN0 + outmaskindex[mode]];
                val = tlagalpha * val0 + (1.0 - tlagalpha) * val1;

                dcimg[IDsave]
                .array.F[kk * NBmodeIN0 + outmaskindex[mode]] = val;
            }
        }

        save_fits("testPFTout", "testPFTout.fits");
    }

    if(SAVEMODE > 0)
    {
        free(outmaskindex);
    }

    return IDPFout;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 5. MISC TOOLS, DIAGNOSTICS                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

//
// IDin_name is a 2 or 3D image, open-loop disturbance
// last axis is time (step)
// this optimization asssumes no correlation in noise
//
float LINARFILTERPRED_ScanGain(char *IDin_name, float multfact, float framelag)
{
    float   gain;
    float   gainmax = 1.1;
    float   optgainblock;
    float   residualblock;
    float   residualblock0;
    float   gainstep = 0.01;
    imageID IDin;

    long nbstep;
    long step, step0, step1;

    long  framelag0;
    long  framelag1;
    float alpha;

    float *actval_array; // actuator value
    float  actval;

    long nbvar;
    long axis, naxis;

    double *errval;
    double  errvaltot;
    long    cnt;

    FILE *fp;
    char  fname[200];
    float mval;
    long  ii;
    float tmpv;

    int   TEST       = 0;
    float TESTperiod = 20.0;

    // results
    float *optgain;
    float *optres;
    float *res0;
    int    optinit = 0;

    if(framelag < 1.00000001)
    {
        printf("ERROR: framelag should be be > 1\n");
        exit(0);
    }

    IDin  = image_ID(IDin_name, dcimg, dcnimg);
    naxis = dcimg[IDin].md[0].naxis;

    nbvar = 1;
    for(axis = 0; axis < naxis - 1; axis++)
    {
        nbvar *= dcimg[IDin].md[0].size[axis];
    }

    errval = (double *) malloc(sizeof(double) * nbvar);
    if(errval == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nbstep = dcimg[IDin].md[0].size[naxis - 1];

    framelag0 = (long) framelag;
    framelag1 = framelag0 + 1;
    alpha     = framelag - framelag0;

    printf("alpha = %f    nbvar = %ld\n", alpha, nbvar);

    list_image_ID();
    if(TEST == 1)
    {
        for(ii = 0; ii < nbvar; ii++)
            for(step = 0; step < nbstep; step++)
            {
                dcimg[IDin].array.F[step * nbvar + ii] =
                    1.0 * sin(2.0 * M_PI * step / TESTperiod);
            }
    }

    actval_array = (float *) malloc(sizeof(float) * nbstep);
    if(actval_array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    optgain = (float *) malloc(sizeof(float) * nbvar);
    if(optgain == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    optres = (float *) malloc(sizeof(float) * nbvar);
    if(optres == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    res0 = (float *) malloc(sizeof(float) * nbvar);
    if(res0 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    snprintf(fname, sizeof(fname), "gainscan.txt");

    gain          = 0.2;
    ii            = 0;
    fp            = fopen(fname, "w");
    residualblock = 1.0e20;
    optgainblock  = 0.0;
    for(gain = 0; gain < gainmax; gain += gainstep)
    {
        fprintf(fp, "%5.3f", gain);

        errvaltot = 0.0;
        for(ii = 0; ii < nbvar; ii++)
        {
            errval[ii] = 0.0;
            cnt        = 0.0;
            for(step = 0; step < framelag1 + 2; step++)
            {
                actval_array[step] = 0.0;
            }
            for(step = framelag1; step < nbstep; step++)
            {
                step0 = step - framelag0;
                step1 = step - framelag1;

                actval = (1.0 - alpha) * actval_array[step0] +
                         alpha * actval_array[step1];
                mval = ((1.0 - alpha) *
                        dcimg[IDin].array.F[step0 * nbvar + ii] +
                        alpha * dcimg[IDin].array.F[step1 * nbvar + ii]) -
                       actval;
                actval_array[step] =
                    multfact * (actval_array[step - 1] + gain * mval);
                tmpv = dcimg[IDin].array.F[step * nbvar + ii] -
                       actval_array[step];
                errval[ii] += tmpv * tmpv;
                cnt++;
            }
            errval[ii] = sqrt(errval[ii] / cnt);
            fprintf(fp, " %10f", errval[ii]);
            errvaltot += errval[ii] * errval[ii];

            if(optinit == 0)
            {
                optgain[ii] = gain;
                optres[ii]  = errval[ii];
                res0[ii]    = errval[ii];
            }
            else
            {
                if(errval[ii] < optres[ii])
                {
                    optres[ii]  = errval[ii];
                    optgain[ii] = gain;
                }
            }
        }

        if(optinit == 0)
        {
            residualblock0 = errvaltot;
        }

        optinit = 1;
        fprintf(fp, "%10f\n", errvaltot);

        if(errvaltot < residualblock)
        {
            residualblock = errvaltot;
            optgainblock  = gain;
        }
    }
    fclose(fp);

    free(actval_array);
    free(errval);

    for(ii = 0; ii < nbvar; ii++)
    {
        printf(
            "MODE %4ld    optimal gain = %5.2f     residual = %.6f -> %.6f \n",
            ii,
            optgain[ii],
            res0[ii],
            optres[ii]);
    }

    printf("\noptimal block gain = %f     residual = %.6f -> %.6f\n\n",
           optgainblock,
           sqrt(residualblock0),
           sqrt(residualblock));
    printf("RMS per mode = %f -> %f\n",
           sqrt(residualblock0 / nbvar),
           sqrt(residualblock / nbvar));

    free(optgain);
    free(optres);
    free(res0);

    return (optgainblock);
}

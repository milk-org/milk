/**
 * @file    image_gen.c
 * @brief   Generate frequently used image(s)
 *
 * Creates images for misc applications
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
#define MODULE_SHORTNAME_DEFAULT "imgen"

// Module short description
#define MODULE_DESCRIPTION                                                     \
    "Creating images (shapes, useful functions and patterns)"

#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CFITSIO
#include <fitsio.h> /* required by every program that uses CFITSIO  */
#endif

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

#include "COREMOD_arith/COREMOD_arith.h"
#ifdef USE_CFITSIO
#include "COREMOD_iofits/COREMOD_iofits.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

#include "statistic/statistic.h"

#ifndef MILK_NO_CLI
#include "image_gen/image_gen.h"

#include "mkdisk.h"
#include "mkpolygon.h"
#include "mkrandomim.h"
#include "mkspdisk.h"
#include "voronoi.h"

#define OMP_NELEMENT_LIMIT 1000000

#define SWAP(x, y)                                                             \
    tmp = (x);                                                                 \
    x   = (y);                                                                 \
    y   = tmp;

#define PI 3.14159265358979323846264338328

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(image_gen)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/* Placeholder for CLICMD_FIELDS_DEFAULTS macro which
 * hardcodes 'farg'. The constructor init_xx() functions
 * below overwrite nbarg and funcfpscliarg at runtime. */
static CLICMDARGDEF farg[] = {
    {CLIARG_FLOAT64, "", "", "", 0, NULL, NULL}
};

#include "fps.h"

/* ===== Command: mkgauss ===== */
imageID make_gauss(const char *ID_name,
    uint32_t l1, uint32_t l2,
    double a, double A);

static char gs_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imgauss";
static int64_t gs_xs = 512;
static int64_t gs_ys = 512;
static double gs_a = 12.0;
static double gs_A = 1.0;
static FPS_APP_INFO FPS_app_info_gs = {
    .fps_name = "mkgauss",
    .cmdkey   = "mkgauss",
    .description =
        "make gaussian spot image"
};
#define FPS_PARAMS_GS(X) \
    X(".out_name", gs_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &gs_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &gs_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".a", &gs_a, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "width param") \
    X(".amp", &gs_A, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "amplitude")
static FPS_CLI_BINDING gs_b[] = {
    FPS_PARAMS_GS(FPS_X_BINDING) };
static const int gs_nb =
    sizeof(gs_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF gs_farg[] = {
    FPS_PARAMS_GS(FPS_X_FARG) };
static CLICMDDATA gs_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS gs_cms = {0};
static __attribute__((constructor))
void init_gs(void) {
    strncpy(gs_d.key,
        FPS_app_info_gs.cmdkey,
        sizeof(gs_d.key)-1);
    strncpy(gs_d.description,
        FPS_app_info_gs.description,
        sizeof(gs_d.description)-1);
    gs_d.nbarg =
        sizeof(gs_farg)/sizeof(CLICMDARGDEF);
    gs_d.funcfpscliarg = gs_farg;
    gs_d.flags = CLICMDFLAG_FPS;
    if(!gs_d.cmdsettings)
        gs_d.cmdsettings = &gs_cms;
}
static errno_t gs_compute(void) {
    make_gauss(gs_n,
        (uint32_t)gs_xs, (uint32_t)gs_ys,
        gs_a, gs_A);
    return RETURN_SUCCESS;
}
static errno_t gs_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_gs, gs_farg, &gs_d,
        gs_b, gs_nb, gs_compute);
}

/* ===== Command: mkfiberclpoverlap ===== */
imageID make_FiberCouplingOverlap(
    const char *ID_name);

static char fc_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imdisk";
static FPS_APP_INFO FPS_app_info_fc = {
    .fps_name = "mkfiberclpoverlap",
    .cmdkey   = "mkfiberclpoverlap",
    .description =
        "fiber coupling overlap integral"
};
#define FPS_PARAMS_FC(X) \
    X(".out_name", fc_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")
static FPS_CLI_BINDING fc_b[] = {
    FPS_PARAMS_FC(FPS_X_BINDING) };
static const int fc_nb =
    sizeof(fc_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF fc_farg[] = {
    FPS_PARAMS_FC(FPS_X_FARG) };
static CLICMDDATA fc_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS fc_cms = {0};
static __attribute__((constructor))
void init_fc(void) {
    strncpy(fc_d.key,
        FPS_app_info_fc.cmdkey,
        sizeof(fc_d.key)-1);
    strncpy(fc_d.description,
        FPS_app_info_fc.description,
        sizeof(fc_d.description)-1);
    fc_d.nbarg =
        sizeof(fc_farg)/sizeof(CLICMDARGDEF);
    fc_d.funcfpscliarg = fc_farg;
    fc_d.flags = CLICMDFLAG_FPS;
    if(!fc_d.cmdsettings)
        fc_d.cmdsettings = &fc_cms;
}
static errno_t fc_compute(void) {
    make_FiberCouplingOverlap(fc_n);
    return RETURN_SUCCESS;
}
static errno_t fc_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_fc, fc_farg, &fc_d,
        fc_b, fc_nb, fc_compute);
}

/* ===== Command: mkslopexy ===== */
imageID make_slopexy(const char *ID_name,
    uint32_t l1, uint32_t l2,
    double sx, double sy);

static char sl_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imslope";
static int64_t sl_xs = 512;
static int64_t sl_ys = 512;
static double sl_sx = 1.2;
static double sl_sy = 1.0;
static FPS_APP_INFO FPS_app_info_sl = {
    .fps_name = "mkslopexy",
    .cmdkey   = "mkslopexy",
    .description = "make slope image"
};
#define FPS_PARAMS_SL(X) \
    X(".out_name", sl_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &sl_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &sl_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".slopex", &sl_sx, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "slope x") \
    X(".slopey", &sl_sy, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "slope y")
static FPS_CLI_BINDING sl_b[] = {
    FPS_PARAMS_SL(FPS_X_BINDING) };
static const int sl_nb =
    sizeof(sl_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF sl_farg[] = {
    FPS_PARAMS_SL(FPS_X_FARG) };
static CLICMDDATA sl_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS sl_cms = {0};
static __attribute__((constructor))
void init_sl(void) {
    strncpy(sl_d.key,
        FPS_app_info_sl.cmdkey,
        sizeof(sl_d.key)-1);
    strncpy(sl_d.description,
        FPS_app_info_sl.description,
        sizeof(sl_d.description)-1);
    sl_d.nbarg =
        sizeof(sl_farg)/sizeof(CLICMDARGDEF);
    sl_d.funcfpscliarg = sl_farg;
    sl_d.flags = CLICMDFLAG_FPS;
    if(!sl_d.cmdsettings)
        sl_d.cmdsettings = &sl_cms;
}
static errno_t sl_compute(void) {
    make_slopexy(sl_n,
        (uint32_t)sl_xs, (uint32_t)sl_ys,
        sl_sx, sl_sy);
    return RETURN_SUCCESS;
}
static errno_t sl_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_sl, sl_farg, &sl_d,
        sl_b, sl_nb, sl_compute);
}

/* ===== Command: mkdist ===== */
imageID make_dist(const char *ID_name,
    uint32_t l1, uint32_t l2,
    double f1, double f2);

static char di_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imdist";
static int64_t di_xs = 512;
static int64_t di_ys = 512;
static double di_cx = 256.0;
static double di_cy = 256.0;
static FPS_APP_INFO FPS_app_info_di = {
    .fps_name = "mkdist",
    .cmdkey   = "mkdist",
    .description =
        "make distance from point image"
};
#define FPS_PARAMS_DI(X) \
    X(".out_name", di_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &di_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &di_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".centerx", &di_cx, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "center x") \
    X(".centery", &di_cy, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "center y")
static FPS_CLI_BINDING di_b[] = {
    FPS_PARAMS_DI(FPS_X_BINDING) };
static const int di_nb =
    sizeof(di_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF di_farg[] = {
    FPS_PARAMS_DI(FPS_X_FARG) };
static CLICMDDATA di_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS di_cms = {0};
static __attribute__((constructor))
void init_di(void) {
    strncpy(di_d.key,
        FPS_app_info_di.cmdkey,
        sizeof(di_d.key)-1);
    strncpy(di_d.description,
        FPS_app_info_di.description,
        sizeof(di_d.description)-1);
    di_d.nbarg =
        sizeof(di_farg)/sizeof(CLICMDARGDEF);
    di_d.funcfpscliarg = di_farg;
    di_d.flags = CLICMDFLAG_FPS;
    if(!di_d.cmdsettings)
        di_d.cmdsettings = &di_cms;
}
static errno_t di_compute(void) {
    make_dist(di_n,
        (uint32_t)di_xs, (uint32_t)di_ys,
        di_cx, di_cy);
    return RETURN_SUCCESS;
}
static errno_t di_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_di, di_farg, &di_d,
        di_b, di_nb, di_compute);
}

/* ===== Command: mkhexsegpup ===== */
imageID make_hexsegpupil(
    const char *IDname, uint32_t size,
    double radius, double gap,
    double step);

static char hx_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imhex";
static int64_t hx_sz = 4096;
static double hx_r = 200.0;
static double hx_g = 2.0;
static double hx_s = 46.3;
static FPS_APP_INFO FPS_app_info_hx = {
    .fps_name = "mkhexsegpup",
    .cmdkey   = "mkhexsegpup",
    .description = "make hex seg pupil"
};
#define FPS_PARAMS_HX(X) \
    X(".out_name", hx_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".size", &hx_sz, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "size") \
    X(".radius", &hx_r, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "radius") \
    X(".gap", &hx_g, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "gap") \
    X(".step", &hx_s, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "step")
static FPS_CLI_BINDING hx_b[] = {
    FPS_PARAMS_HX(FPS_X_BINDING) };
static const int hx_nb =
    sizeof(hx_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF hx_farg[] = {
    FPS_PARAMS_HX(FPS_X_FARG) };
static CLICMDDATA hx_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS hx_cms = {0};
static __attribute__((constructor))
void init_hx(void) {
    strncpy(hx_d.key,
        FPS_app_info_hx.cmdkey,
        sizeof(hx_d.key)-1);
    strncpy(hx_d.description,
        FPS_app_info_hx.description,
        sizeof(hx_d.description)-1);
    hx_d.nbarg =
        sizeof(hx_farg)/sizeof(CLICMDARGDEF);
    hx_d.funcfpscliarg = hx_farg;
    hx_d.flags = CLICMDFLAG_FPS;
    if(!hx_d.cmdsettings)
        hx_d.cmdsettings = &hx_cms;
}
static errno_t hx_compute(void) {
    make_hexsegpupil(hx_n,
        (uint32_t)hx_sz, hx_r, hx_g, hx_s);
    return RETURN_SUCCESS;
}
static errno_t hx_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_hx, hx_farg, &hx_d,
        hx_b, hx_nb, hx_compute);
}

/* ===== Command: segs2wfmodes ===== */
long IMAGE_gen_segments2WFmodes(
    const char *prefix, long ndigit,
    const char *IDout);

static char sw_pfx[FUNCTION_PARAMETER_STRMAXLEN]
    = "segim";
static int64_t sw_nd = 2;
static char sw_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "WFmodes";
static FPS_APP_INFO FPS_app_info_sw = {
    .fps_name = "segs2wfmodes",
    .cmdkey   = "segs2wfmodes",
    .description =
        "segments to WF modes"
};
#define FPS_PARAMS_SW(X) \
    X(".prefix", sw_pfx, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "seg prefix") \
    X(".ndigit", &sw_nd, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb digits") \
    X(".out_name", sw_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")
static FPS_CLI_BINDING sw_b[] = {
    FPS_PARAMS_SW(FPS_X_BINDING) };
static const int sw_nb =
    sizeof(sw_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF sw_farg[] = {
    FPS_PARAMS_SW(FPS_X_FARG) };
static CLICMDDATA sw_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS sw_cms = {0};
static __attribute__((constructor))
void init_sw(void) {
    strncpy(sw_d.key,
        FPS_app_info_sw.cmdkey,
        sizeof(sw_d.key)-1);
    strncpy(sw_d.description,
        FPS_app_info_sw.description,
        sizeof(sw_d.description)-1);
    sw_d.nbarg =
        sizeof(sw_farg)/sizeof(CLICMDARGDEF);
    sw_d.funcfpscliarg = sw_farg;
    sw_d.flags = CLICMDFLAG_FPS;
    if(!sw_d.cmdsettings)
        sw_d.cmdsettings = &sw_cms;
}
static errno_t sw_compute(void) {
    IMAGE_gen_segments2WFmodes(
        sw_pfx, (long)sw_nd, sw_out);
    return RETURN_SUCCESS;
}
static errno_t sw_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_sw, sw_farg, &sw_d,
        sw_b, sw_nb, sw_compute);
}

/* ===== Command: mkrect ===== */
imageID make_rectangle(const char *ID_name,
    uint32_t l1, uint32_t l2,
    double xc, double yc,
    double r1, double r2);

static char rc_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imrect";
static int64_t rc_xs = 512;
static int64_t rc_ys = 512;
static double rc_xc = 256.0;
static double rc_yc = 256.0;
static double rc_r1 = 100.0;
static double rc_r2 = 200.0;
static FPS_APP_INFO FPS_app_info_rc = {
    .fps_name = "mkrect",
    .cmdkey   = "mkrect",
    .description = "make rectangle"
};
#define FPS_PARAMS_RC(X) \
    X(".out_name", rc_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &rc_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &rc_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".xcenter", &rc_xc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &rc_yc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y center") \
    X(".radius1", &rc_r1, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "radius 1") \
    X(".radius2", &rc_r2, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "radius 2")
static FPS_CLI_BINDING rc_b[] = {
    FPS_PARAMS_RC(FPS_X_BINDING) };
static const int rc_nb =
    sizeof(rc_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF rc_farg[] = {
    FPS_PARAMS_RC(FPS_X_FARG) };
static CLICMDDATA rc_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS rc_cms = {0};
static __attribute__((constructor))
void init_rc(void) {
    strncpy(rc_d.key,
        FPS_app_info_rc.cmdkey,
        sizeof(rc_d.key)-1);
    strncpy(rc_d.description,
        FPS_app_info_rc.description,
        sizeof(rc_d.description)-1);
    rc_d.nbarg =
        sizeof(rc_farg)/sizeof(CLICMDARGDEF);
    rc_d.funcfpscliarg = rc_farg;
    rc_d.flags = CLICMDFLAG_FPS;
    if(!rc_d.cmdsettings)
        rc_d.cmdsettings = &rc_cms;
}
static errno_t rc_compute(void) {
    make_rectangle(rc_n,
        (uint32_t)rc_xs, (uint32_t)rc_ys,
        rc_xc, rc_yc, rc_r1, rc_r2);
    return RETURN_SUCCESS;
}
static errno_t rc_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_rc, rc_farg, &rc_d,
        rc_b, rc_nb, rc_compute);
}

/* ===== Command: mkline ===== */
imageID make_line(const char *IDname,
    uint32_t l1, uint32_t l2,
    double x1, double y1,
    double x2, double y2, double t);

static char ln_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imline";
static int64_t ln_xs = 512;
static int64_t ln_ys = 512;
static double ln_x1 = 256.0;
static double ln_y1 = 256.0;
static double ln_x2 = 100.0;
static double ln_y2 = 200.0;
static double ln_t = 3.0;
static FPS_APP_INFO FPS_app_info_ln = {
    .fps_name = "mkline",
    .cmdkey   = "mkline",
    .description = "make line"
};
#define FPS_PARAMS_LN(X) \
    X(".out_name", ln_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &ln_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &ln_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".x1", &ln_x1, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x1") \
    X(".y1", &ln_y1, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y1") \
    X(".x2", &ln_x2, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x2") \
    X(".y2", &ln_y2, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y2") \
    X(".thickness", &ln_t, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "thickness")
static FPS_CLI_BINDING ln_b[] = {
    FPS_PARAMS_LN(FPS_X_BINDING) };
static const int ln_nb =
    sizeof(ln_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF ln_farg[] = {
    FPS_PARAMS_LN(FPS_X_FARG) };
static CLICMDDATA ln_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS ln_cms = {0};
static __attribute__((constructor))
void init_ln(void) {
    strncpy(ln_d.key,
        FPS_app_info_ln.cmdkey,
        sizeof(ln_d.key)-1);
    strncpy(ln_d.description,
        FPS_app_info_ln.description,
        sizeof(ln_d.description)-1);
    ln_d.nbarg =
        sizeof(ln_farg)/sizeof(CLICMDARGDEF);
    ln_d.funcfpscliarg = ln_farg;
    ln_d.flags = CLICMDFLAG_FPS;
    if(!ln_d.cmdsettings)
        ln_d.cmdsettings = &ln_cms;
}
static errno_t ln_compute(void) {
    make_line(ln_n,
        (uint32_t)ln_xs, (uint32_t)ln_ys,
        ln_x1, ln_y1, ln_x2, ln_y2, ln_t);
    return RETURN_SUCCESS;
}
static errno_t ln_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_ln, ln_farg, &ln_d,
        ln_b, ln_nb, ln_compute);
}

/* ===== Command: mklincoord ===== */
imageID make_lincoordinate(
    const char *IDname,
    uint32_t l1, uint32_t l2,
    double xc, double yc, double angle);

static char lc_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imlincoord";
static int64_t lc_xs = 512;
static int64_t lc_ys = 512;
static double lc_xc = 256.0;
static double lc_yc = 256.0;
static double lc_a = 1.42;
static FPS_APP_INFO FPS_app_info_lc = {
    .fps_name = "mklincoord",
    .cmdkey   = "mklincoord",
    .description =
        "make linear coordinate"
};
#define FPS_PARAMS_LC(X) \
    X(".out_name", lc_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &lc_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &lc_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".xcenter", &lc_xc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &lc_yc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y center") \
    X(".angle", &lc_a, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "angle")
static FPS_CLI_BINDING lc_b[] = {
    FPS_PARAMS_LC(FPS_X_BINDING) };
static const int lc_nb =
    sizeof(lc_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF lc_farg[] = {
    FPS_PARAMS_LC(FPS_X_FARG) };
static CLICMDDATA lc_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS lc_cms = {0};
static __attribute__((constructor))
void init_lc(void) {
    strncpy(lc_d.key,
        FPS_app_info_lc.cmdkey,
        sizeof(lc_d.key)-1);
    strncpy(lc_d.description,
        FPS_app_info_lc.description,
        sizeof(lc_d.description)-1);
    lc_d.nbarg =
        sizeof(lc_farg)/sizeof(CLICMDARGDEF);
    lc_d.funcfpscliarg = lc_farg;
    lc_d.flags = CLICMDFLAG_FPS;
    if(!lc_d.cmdsettings)
        lc_d.cmdsettings = &lc_cms;
}
static errno_t lc_compute(void) {
    make_lincoordinate(lc_n,
        (uint32_t)lc_xs, (uint32_t)lc_ys,
        lc_xc, lc_yc, lc_a);
    return RETURN_SUCCESS;
}
static errno_t lc_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_lc, lc_farg, &lc_d,
        lc_b, lc_nb, lc_compute);
}

/* ===== Command: mkgridpix ===== */
imageID make_2Dgridpix(const char *IDname,
    uint32_t xsize, uint32_t ysize,
    double pitchx, double pitchy,
    double offsetx, double offsety);

static char gp_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "impgrid";
static int64_t gp_xs = 512;
static int64_t gp_ys = 512;
static double gp_px = 10.0;
static double gp_py = 10.0;
static double gp_ox = 4.5;
static double gp_oy = 2.8;
static FPS_APP_INFO FPS_app_info_gp = {
    .fps_name = "mkgridpix",
    .cmdkey   = "mkgridpix",
    .description = "make regular grid"
};
#define FPS_PARAMS_GP(X) \
    X(".out_name", gp_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &gp_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &gp_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".pitchx", &gp_px, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x pitch") \
    X(".pitchy", &gp_py, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y pitch") \
    X(".offsetx", &gp_ox, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x offset") \
    X(".offsety", &gp_oy, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y offset")
static FPS_CLI_BINDING gp_b[] = {
    FPS_PARAMS_GP(FPS_X_BINDING) };
static const int gp_nb =
    sizeof(gp_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF gp_farg[] = {
    FPS_PARAMS_GP(FPS_X_FARG) };
static CLICMDDATA gp_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS gp_cms = {0};
static __attribute__((constructor))
void init_gp(void) {
    strncpy(gp_d.key,
        FPS_app_info_gp.cmdkey,
        sizeof(gp_d.key)-1);
    strncpy(gp_d.description,
        FPS_app_info_gp.description,
        sizeof(gp_d.description)-1);
    gp_d.nbarg =
        sizeof(gp_farg)/sizeof(CLICMDARGDEF);
    gp_d.funcfpscliarg = gp_farg;
    gp_d.flags = CLICMDFLAG_FPS;
    if(!gp_d.cmdsettings)
        gp_d.cmdsettings = &gp_cms;
}
static errno_t gp_compute(void) {
    make_2Dgridpix(gp_n,
        (uint32_t)gp_xs, (uint32_t)gp_ys,
        gp_px, gp_py, gp_ox, gp_oy);
    return RETURN_SUCCESS;
}
static errno_t gp_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_gp, gp_farg, &gp_d,
        gp_b, gp_nb, gp_compute);
}

/* ===== Command: mkrndim ===== */
imageID make_rnd(const char *ID_name,
    uint32_t l1, uint32_t l2,
    const char *options);

static char ri_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imrnd";
static int64_t ri_xs = 512;
static int64_t ri_ys = 512;
static FPS_APP_INFO FPS_app_info_ri = {
    .fps_name = "mkrndim",
    .cmdkey   = "mkrndim",
    .description = "make random image"
};
#define FPS_PARAMS_RI(X) \
    X(".out_name", ri_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &ri_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &ri_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize")
static FPS_CLI_BINDING ri_b[] = {
    FPS_PARAMS_RI(FPS_X_BINDING) };
static const int ri_nb =
    sizeof(ri_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF ri_farg[] = {
    FPS_PARAMS_RI(FPS_X_FARG) };
static CLICMDDATA ri_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS ri_cms = {0};
static __attribute__((constructor))
void init_ri(void) {
    strncpy(ri_d.key,
        FPS_app_info_ri.cmdkey,
        sizeof(ri_d.key)-1);
    strncpy(ri_d.description,
        FPS_app_info_ri.description,
        sizeof(ri_d.description)-1);
    ri_d.nbarg =
        sizeof(ri_farg)/sizeof(CLICMDARGDEF);
    ri_d.funcfpscliarg = ri_farg;
    ri_d.flags = CLICMDFLAG_FPS;
    if(!ri_d.cmdsettings)
        ri_d.cmdsettings = &ri_cms;
}
static errno_t ri_compute(void) {
    make_rnd(ri_n,
        (uint32_t)ri_xs, (uint32_t)ri_ys,
        "");
    return RETURN_SUCCESS;
}
static errno_t ri_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_ri, ri_farg, &ri_d,
        ri_b, ri_nb, ri_compute);
}

/* ===== Command: mkrndgim ===== */
static char rg_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imrndg";
static int64_t rg_xs = 512;
static int64_t rg_ys = 512;
static FPS_APP_INFO FPS_app_info_rg = {
    .fps_name = "mkrndgim",
    .cmdkey   = "mkrndgim",
    .description =
        "make random gaussian image"
};
#define FPS_PARAMS_RG(X) \
    X(".out_name", rg_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &rg_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &rg_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize")
static FPS_CLI_BINDING rg_b[] = {
    FPS_PARAMS_RG(FPS_X_BINDING) };
static const int rg_nb =
    sizeof(rg_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF rg_farg[] = {
    FPS_PARAMS_RG(FPS_X_FARG) };
static CLICMDDATA rg_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS rg_cms = {0};
static __attribute__((constructor))
void init_rg(void) {
    strncpy(rg_d.key,
        FPS_app_info_rg.cmdkey,
        sizeof(rg_d.key)-1);
    strncpy(rg_d.description,
        FPS_app_info_rg.description,
        sizeof(rg_d.description)-1);
    rg_d.nbarg =
        sizeof(rg_farg)/sizeof(CLICMDARGDEF);
    rg_d.funcfpscliarg = rg_farg;
    rg_d.flags = CLICMDFLAG_FPS;
    if(!rg_d.cmdsettings)
        rg_d.cmdsettings = &rg_cms;
}
static errno_t rg_compute(void) {
    make_rnd(rg_n,
        (uint32_t)rg_xs, (uint32_t)rg_ys,
        "gauss");
    return RETURN_SUCCESS;
}
static errno_t rg_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_rg, rg_farg, &rg_d,
        rg_b, rg_nb, rg_compute);
}

/* ===== Command: im2coord ===== */
imageID image_gen_im2coord(
    const char *IDin_name,
    uint8_t axis,
    const char *IDout_name);

static char ic_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "imin";
static int64_t ic_ax = 1;
static char ic_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "imy";
static FPS_APP_INFO FPS_app_info_ic = {
    .fps_name = "im2coord",
    .cmdkey   = "im2coord",
    .description =
        "make coordinate image"
};
#define FPS_PARAMS_IC(X) \
    X(".in_name", ic_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input") \
    X(".axis", &ic_ax, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "axis") \
    X(".out_name", ic_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output")
static FPS_CLI_BINDING ic_b[] = {
    FPS_PARAMS_IC(FPS_X_BINDING) };
static const int ic_nb =
    sizeof(ic_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF ic_farg[] = {
    FPS_PARAMS_IC(FPS_X_FARG) };
static CLICMDDATA ic_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS ic_cms = {0};
static __attribute__((constructor))
void init_ic(void) {
    strncpy(ic_d.key,
        FPS_app_info_ic.cmdkey,
        sizeof(ic_d.key)-1);
    strncpy(ic_d.description,
        FPS_app_info_ic.description,
        sizeof(ic_d.description)-1);
    ic_d.nbarg =
        sizeof(ic_farg)/sizeof(CLICMDARGDEF);
    ic_d.funcfpscliarg = ic_farg;
    ic_d.flags = CLICMDFLAG_FPS;
    if(!ic_d.cmdsettings)
        ic_d.cmdsettings = &ic_cms;
}
static errno_t ic_compute(void) {
    image_gen_im2coord(
        ic_in, (uint8_t)ic_ax, ic_out);
    return RETURN_SUCCESS;
}
static errno_t ic_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_ic, ic_farg, &ic_d,
        ic_b, ic_nb, ic_compute);
}


/* ===== Module init ===== */

static errno_t init_module_CLI()
{
    init_gs();
    init_fc();
    init_sl();
    init_di();
    init_hx();
    init_sw();
    init_rc();
    init_ln();
    init_lc();
    init_gp();
    init_ri();
    init_rg();
    init_ic();

    CLIADDCMD_image_gen__mkdisk();
    CLIADDCMD_image_gen__mkpolygon();
    CLIADDCMD_image_gen__mkspdisk();
    {
        safe_fps_fill_farg_examples(
            gs_farg, gs_b, gs_nb);
        int cmdi = RegisterCLIcmd(
            gs_d, gs_CLIfunc);
        gs_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            fc_farg, fc_b, fc_nb);
        int cmdi = RegisterCLIcmd(
            fc_d, fc_CLIfunc);
        fc_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            sl_farg, sl_b, sl_nb);
        int cmdi = RegisterCLIcmd(
            sl_d, sl_CLIfunc);
        sl_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            di_farg, di_b, di_nb);
        int cmdi = RegisterCLIcmd(
            di_d, di_CLIfunc);
        di_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            hx_farg, hx_b, hx_nb);
        int cmdi = RegisterCLIcmd(
            hx_d, hx_CLIfunc);
        hx_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            sw_farg, sw_b, sw_nb);
        int cmdi = RegisterCLIcmd(
            sw_d, sw_CLIfunc);
        sw_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            rc_farg, rc_b, rc_nb);
        int cmdi = RegisterCLIcmd(
            rc_d, rc_CLIfunc);
        rc_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            ln_farg, ln_b, ln_nb);
        int cmdi = RegisterCLIcmd(
            ln_d, ln_CLIfunc);
        ln_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            lc_farg, lc_b, lc_nb);
        int cmdi = RegisterCLIcmd(
            lc_d, lc_CLIfunc);
        lc_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            gp_farg, gp_b, gp_nb);
        int cmdi = RegisterCLIcmd(
            gp_d, gp_CLIfunc);
        gp_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            ri_farg, ri_b, ri_nb);
        int cmdi = RegisterCLIcmd(
            ri_d, ri_CLIfunc);
        ri_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            rg_farg, rg_b, rg_nb);
        int cmdi = RegisterCLIcmd(
            rg_d, rg_CLIfunc);
        rg_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            ic_farg, ic_b, ic_nb);
        int cmdi = RegisterCLIcmd(
            ic_d, ic_CLIfunc);
        ic_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    CLIADDCMD_image_gen__voronoi();
    CLIADDCMD_image_gen__mkrandomim();

    // add atexit functions here

    return RETURN_SUCCESS;
}

#endif /* MILK_NO_CLI */

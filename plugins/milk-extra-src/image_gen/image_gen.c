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

#include <fitsio.h> /* required by every program that uses CFITSIO  */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "statistic/statistic.h"

#ifndef MILK_NO_CLI
#include "image_gen/image_gen.h"

#include "mkrandomim.h"
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

/* ===== Command: mkdisk ===== */
imageID make_disk(const char *ID_name,
    uint32_t l1, uint32_t l2,
    double x_center, double y_center,
    double radius);

static char dk_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imdisk";
static int64_t dk_xs = 512;
static int64_t dk_ys = 512;
static double dk_xc = 256.0;
static double dk_yc = 256.0;
static double dk_r = 100.0;
static FPS_APP_INFO FPS_app_info_dk = {
    .fps_name = "mkdisk",
    .cmdkey   = "mkdisk",
    .description = "make disk image"
};
#define FPS_PARAMS_DK(X) \
    X(".out_name", dk_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &dk_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &dk_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".xcenter", &dk_xc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &dk_yc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y center") \
    X(".radius", &dk_r, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "radius")

#include "fps.h"

static FPS_CLI_BINDING dk_b[] = {
    FPS_PARAMS_DK(FPS_X_BINDING) };
static const int dk_nb =
    sizeof(dk_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS_DK(FPS_X_FARG) };
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS dk_cms = {0};
static __attribute__((constructor))
void init_dk(void) {
    strncpy(CLIcmddata.key,
        FPS_app_info_dk.cmdkey,
        sizeof(CLIcmddata.key)-1);
    strncpy(CLIcmddata.description,
        FPS_app_info_dk.description,
        sizeof(CLIcmddata.description)-1);
    CLIcmddata.nbarg =
        sizeof(farg)/sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags = CLICMDFLAG_FPS;
    if(!CLIcmddata.cmdsettings)
        CLIcmddata.cmdsettings = &dk_cms;
}
static errno_t dk_compute(void) {
    make_disk(dk_n,
        (uint32_t)dk_xs, (uint32_t)dk_ys,
        dk_xc, dk_yc, dk_r);
    return RETURN_SUCCESS;
}
static errno_t CLIfunction(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_dk, farg,
        &CLIcmddata,
        dk_b, dk_nb, dk_compute);
}

/* ===== Command: mkspdisk ===== */
imageID make_subpixdisk(const char *ID_name,
    uint32_t l1, uint32_t l2,
    double x_center, double y_center,
    double radius);

static char sd_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imdisk";
static int64_t sd_xs = 512;
static int64_t sd_ys = 512;
static double sd_xc = 256.0;
static double sd_yc = 256.0;
static double sd_r = 100.0;
static FPS_APP_INFO FPS_app_info_sd = {
    .fps_name = "mkspdisk",
    .cmdkey   = "mkspdisk",
    .description =
        "make subpixel disk image"
};
#define FPS_PARAMS_SD(X) \
    X(".out_name", sd_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &sd_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &sd_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".xcenter", &sd_xc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &sd_yc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y center") \
    X(".radius", &sd_r, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "radius")
static FPS_CLI_BINDING sd_b[] = {
    FPS_PARAMS_SD(FPS_X_BINDING) };
static const int sd_nb =
    sizeof(sd_b)/sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF sd_farg[] = {
    FPS_PARAMS_SD(FPS_X_FARG) };
static CLICMDDATA sd_d = {
    "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS sd_cms = {0};
static __attribute__((constructor))
void init_sd(void) {
    strncpy(sd_d.key,
        FPS_app_info_sd.cmdkey,
        sizeof(sd_d.key)-1);
    strncpy(sd_d.description,
        FPS_app_info_sd.description,
        sizeof(sd_d.description)-1);
    sd_d.nbarg =
        sizeof(sd_farg)/sizeof(CLICMDARGDEF);
    sd_d.funcfpscliarg = sd_farg;
    sd_d.flags = CLICMDFLAG_FPS;
    if(!sd_d.cmdsettings)
        sd_d.cmdsettings = &sd_cms;
}
static errno_t sd_compute(void) {
    make_subpixdisk(sd_n,
        (uint32_t)sd_xs, (uint32_t)sd_ys,
        sd_xc, sd_yc, sd_r);
    return RETURN_SUCCESS;
}
static errno_t sd_CLIfunc(void) {
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_sd, sd_farg, &sd_d,
        sd_b, sd_nb, sd_compute);
}

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
    {
        safe_fps_fill_farg_examples(
            farg, dk_b, dk_nb);
        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        safe_fps_fill_farg_examples(
            sd_farg, sd_b, sd_nb);
        int cmdi = RegisterCLIcmd(
            sd_d, sd_CLIfunc);
        sd_d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
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

/** @brief creates a double star */
imageID make_double_star(const char *ID_name,
                         uint32_t    l1,
                         uint32_t    l2,
                         double      intensity_1,
                         double      intensity_2,
                         double      separation,
                         double      position_angle)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    dcimg[ID]
    .array.F[((int)(naxes[1] / 2)) * naxes[0] + ((int)(naxes[0] / 2))] =
        intensity_1;
    dcimg[ID]
    .array.F[((int)(naxes[1] / 2 + separation * cos(position_angle))) *
                             naxes[0] +
                             ((int)(naxes[0] / 2 + separation * sin(position_angle)))] =
                 intensity_2;

    return (ID);
}

/** @brief creates a disk */
imageID make_disk(const char *ID_name,
                  uint32_t    l1,
                  uint32_t    l2,
                  double      x_center,
                  double      y_center,
                  double      radius)
{
    imageID  ID;
    uint32_t ii, jj;
    uint32_t naxes[2];
    long     x1, x2, y1, y2;
    long     x1i, x2i, y1i, y2i;
    double   r2;
    /*
      int i,j;
      double r;
      double tot;
      int subgrid=100;
      double x,y;
    */

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1  = (long)(x_center - radius - 2);
    x2  = (long)(x_center + radius + 2);
    y1  = (long)(y_center - radius - 2);
    y2  = (long)(y_center + radius + 2);
    x1i = (long)(x_center - 0.707106781 * radius + 2);
    x2i = (long)(x_center + 0.707106781 * radius - 2);
    y1i = (long)(y_center - 0.707106781 * radius + 2);
    y2i = (long)(y_center + 0.707106781 * radius - 2);

    if(x1 < 0)
    {
        x1 = 0;
    }
    if(x1 > naxes[0])
    {
        x1 = naxes[0];
    }

    if(x2 < 0)
    {
        x2 = 0;
    }
    if(x2 > naxes[0])
    {
        x2 = naxes[0];
    }

    if(y1 < 0)
    {
        y1 = 0;
    }
    if(y1 > naxes[1])
    {
        y1 = naxes[1];
    }

    if(y2 > naxes[1])
    {
        y2 = naxes[1];
    }

    if(x1i < 0)
    {
        x1i = 0;
    }
    if(x1i > naxes[0])
    {
        x1i = naxes[0];
    }

    if(x2i < 0)
    {
        x2i = 0;
    }
    if(x2i > naxes[0])
    {
        x2i = naxes[0];
    }

    if(y1i < 0)
    {
        y1i = 0;
    }
    if(y1i > naxes[1])
    {
        y1i = naxes[1];
    }

    if(y2i < 0)
    {
        y2i = 0;
    }
    if(y2i > naxes[1])
    {
        y2i = naxes[1];
    }

    r2 = radius * radius;

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1i; jj < y2i; jj++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
        }

    for(ii = x1; ii < x1i; ii++)
        for(jj = y1; jj < y2; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    for(ii = x2i; ii < x2; ii++)
        for(jj = y1; jj < y2; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1; jj < y1i; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y2i; jj < y2; jj++)
            if(((ii - x_center) * (ii - x_center) +
                    (jj - y_center) * (jj - y_center)) < r2)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }

    /*
    for (jj = x1; jj < x2; jj++)
      for (ii = y1; ii < y2; ii++)
        {
    if (((ii-x_center)*(ii-x_center)+(jj-y_center)*(jj-y_center))<r2)
      dcimg[ID].array.F[jj*naxes[0]+ii] = 1;
        }
    */
    /*
      for (jj = 0; jj < naxes[1]; jj++)
      for (ii = 0; ii < naxes[0]; ii++)
      {
      r = sqrt(((ii-x_center)*(ii-x_center)+(jj-y_center)*(jj-y_center)));

      if (r<radius)
      dcimg[ID].array.F[jj*naxes[0]+ii] = 1.0f;
      else
      dcimg[ID].array.F[jj*naxes[0]+ii] = 0.0f;

      if(((radius-r)*(radius-r))<1.5)
      {
      tot = 0;
      for (j = 0; j < subgrid; j++)
      for (i = 0; i < subgrid; i++)
      {
      x = 1.0*ii-0.5+0.5/subgrid+1.0*i/subgrid;
      y = 1.0*jj-0.5+0.5/subgrid+1.0*j/subgrid;
      r = sqrt((x-1.0*x_center)*(x-1.0*x_center)+(y-1.0*y_center)*(y-1.0*y_center));
      if (r < radius)
      tot = tot + 1.0;
      else
      tot = tot + 0.0;
      }
      tot = tot/subgrid/subgrid;
      dcimg[ID].array.F[jj*naxes[0]+ii] = tot;
      }
      }
    */
    return (ID);
}

imageID make_subpixdisk(const char *ID_name,
                        uint32_t    l1,
                        uint32_t    l2,
                        double      x_center,
                        double      y_center,
                        double      radius)
{
    imageID  ID;
    uint32_t ii, jj;
    uint32_t naxes[2];
    int      i, j;
    double   r;
    double   tot;
    int      subgrid = 55;
    double   grid[55]; // same number of points as subgrid
    double   x, y;
    long     x1, x2, y1, y2;
    long     x1i, x2i, y1i, y2i;
    double   r2, r2ref;
    double   xdiff, ydiff;
    double   subgrid2;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1  = (long)(x_center - radius - 2);
    x2  = (long)(x_center + radius + 2);
    y1  = (long)(y_center - radius - 2);
    y2  = (long)(y_center + radius + 2);
    x1i = (long)(x_center - 0.707106781 * radius + 2);
    x2i = (long)(x_center + 0.707106781 * radius - 2);
    y1i = (long)(y_center - 0.707106781 * radius + 2);
    y2i = (long)(y_center + 0.707106781 * radius - 2);

    if(x1 < 0)
    {
        x1 = 0;
    }
    if(x1 > naxes[0])
    {
        x1 = naxes[0];
    }
    if(x2 < 0)
    {
        x2 = 0;
    }
    if(x2 > naxes[0])
    {
        x2 = naxes[0];
    }

    if(y1 < 0)
    {
        y1 = 0;
    }
    if(y1 > naxes[1])
    {
        y1 = naxes[1];
    }
    if(y2 < 0)
    {
        y2 = 0;
    }
    if(y2 > naxes[1])
    {
        y2 = naxes[1];
    }

    if(x1i < 0)
    {
        x1i = 0;
    }
    if(x1i > naxes[0] - 1)
    {
        x1i = naxes[0] - 1;
    }
    if(x2i < 0)
    {
        x2i = 0;
    }
    if(x2i > naxes[0] - 1)
    {
        x2i = naxes[0] - 1;
    }

    if(y1i < 0)
    {
        y1i = 0;
    }
    if(y1i > naxes[1] - 1)
    {
        y1i = naxes[1] - 1;
    }
    if(y2i < 0)
    {
        y2i = 0;
    }
    if(y2i > naxes[1] - 1)
    {
        y2i = naxes[1] - 1;
    }

    r2ref    = radius * radius;
    subgrid2 = subgrid * subgrid;

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1i; jj < y2i; jj++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
        }

    for(i = 0; i < subgrid; i++)
    {
        grid[i] = (0.5 - 0.5 / subgrid - 1.0 * i / subgrid);
    }

    for(ii = x1; ii < x1i; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x2i; ii < x2; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1; jj < y1i; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y2i; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            r2    = xdiff * xdiff + ydiff * ydiff;
            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x = xdiff + grid[i];
                        y = ydiff + grid[j];
                        r = x * x + y * y;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    return (ID);
}

// creates a shape with contour described by sum of sine waves
//
// r = radius + SUM[ ra[i] * cos( ka[i]*PA/2.0/PI + pa[i]) ]

imageID make_subpixdisk_perturb(const char *ID_name,
                                uint32_t    l1,
                                uint32_t    l2,
                                double      x_center,
                                double      y_center,
                                double      radius,
                                long        n,
                                double     *ra,
                                double     *ka,
                                double     *pa)
{
    imageID  ID;
    uint32_t ii, jj;
    uint32_t naxes[2];
    int      i, j;
    double   r;
    double   tot;
    int      subgrid = 55;
    double   grid[55]; // same number of points as subgrid
    double   x, y;
    long     x1, x2, y1, y2;
    long     x1i, x2i, y1i, y2i;
    double   r2, r2ref;
    double   xdiff, ydiff;
    double   subgrid2;
    double   PA;
    double   v0;
    long     k;

    double radius1, radius2;

    radius1 = radius;
    radius2 = radius;
    for(k = 0; k < n; k++)
    {
        radius1 += radius * fabs(ra[k]);
    }
    for(k = 0; k < n; k++)
    {
        radius2 -= radius * fabs(ra[k]);
    }
    if(radius2 < 0.0)
    {
        radius2 = 0.0;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    x1  = (long)(x_center - radius1 - 2);
    x2  = (long)(x_center + radius1 + 2);
    y1  = (long)(y_center - radius1 - 2);
    y2  = (long)(y_center + radius1 + 2);
    x1i = (long)(x_center - 0.707106781 * radius2 + 2);
    x2i = (long)(x_center + 0.707106781 * radius2 - 2);
    y1i = (long)(y_center - 0.707106781 * radius2 + 2);
    y2i = (long)(y_center + 0.707106781 * radius2 - 2);

    if(x1 < 0)
    {
        x1 = 0;
    }
    if(x1 > naxes[0])
    {
        x1 = naxes[0];
    }
    if(x2 < 0)
    {
        x2 = 0;
    }
    if(x2 > naxes[0])
    {
        x2 = naxes[0];
    }

    if(y1 < 0)
    {
        y1 = 0;
    }
    if(y1 > naxes[1])
    {
        y1 = naxes[1];
    }
    if(y2 < 0)
    {
        y2 = 0;
    }
    if(y2 > naxes[1])
    {
        y2 = naxes[1];
    }

    if(x1i < 0)
    {
        x1i = 0;
    }
    if(x1i > naxes[0] - 1)
    {
        x1i = naxes[0] - 1;
    }
    if(x2i < 0)
    {
        x2i = 0;
    }
    if(x2i > naxes[0] - 1)
    {
        x2i = naxes[0] - 1;
    }

    if(y1i < 0)
    {
        y1i = 0;
    }
    if(y1i > naxes[1] - 1)
    {
        y1i = naxes[1] - 1;
    }
    if(y2i < 0)
    {
        y2i = 0;
    }
    if(y2i > naxes[1] - 1)
    {
        y2i = naxes[1] - 1;
    }

    r2ref    = radius * radius;
    subgrid2 = subgrid * subgrid;

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1i; jj < y2i; jj++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
        }

    for(i = 0; i < subgrid; i++)
    {
        grid[i] = (0.5 - 0.5 / subgrid - 1.0 * i / subgrid);
    }

    for(ii = x1; ii < x1i; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;

            v0 = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;

                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;

                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x2i; ii < x2; ii++)
        for(jj = y1; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;

            v0 = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        r  = x * x + y * y;
                        PA = atan2(y, x);
                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;

                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y1; jj < y1i; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;
            v0    = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;
                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    for(ii = x1i; ii < x2i; ii++)
        for(jj = y2i; jj < y2; jj++)
        {
            xdiff = x_center - ii;
            ydiff = y_center - jj;
            PA    = atan2(ydiff, xdiff);
            r2    = xdiff * xdiff + ydiff * ydiff;
            v0    = radius;
            for(k = 0; k < n; k++)
            {
                v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
            }
            r2ref = v0 * v0;

            if(r2 < r2ref)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            if(fabs(sqrt(r2) - sqrt(r2ref)) < 1.5)
            {
                tot = 0;
                for(j = 0; j < subgrid; j++)
                    for(i = 0; i < subgrid; i++)
                    {
                        x  = xdiff + grid[i];
                        y  = ydiff + grid[j];
                        PA = atan2(y, x);
                        r  = x * x + y * y;
                        v0 = radius;
                        for(k = 0; k < n; k++)
                        {
                            v0 += radius * ra[k] * cos(ka[k] * PA + pa[k]);
                        }
                        r2ref = v0 * v0;
                        if(r < r2ref)
                        {
                            tot += 1.0;
                        }
                    }
                tot                                        = tot / subgrid2;
                dcimg[ID].array.F[jj * naxes[0] + ii] = tot;
            }
        }

    return (ID);
}

/* creates a square */
imageID make_square(const char *ID_name,
                    uint32_t    l1,
                    uint32_t    l2,
                    double      x_center,
                    double      y_center,
                    double      radius)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if((((ii - x_center) * (ii - x_center)) < (radius * radius)) &&
                    (((jj - y_center) * (jj - y_center)) < (radius * radius)))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }
        }

    return (ID);
}

imageID make_rectangle(const char *ID_name,
                       uint32_t    l1,
                       uint32_t    l2,
                       double      x_center,
                       double      y_center,
                       double      radius1,
                       double      radius2)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if((((ii - x_center) * (ii - x_center)) < (radius1 * radius1)) &&
                    (((jj - y_center) * (jj - y_center)) < (radius2 * radius2)))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }
        }

    return (ID);
}

// line of thickness t from (x1,y1) to (x2,y2)
imageID make_line(const char *IDname,
                  uint32_t    l1,
                  uint32_t    l2,
                  double      x1,
                  double      y1,
                  double      x2,
                  double      y2,
                  double      t)
{
    imageID  ID;
    double   x, y, xr, yr, r0;
    double   PA0;
    uint32_t naxes[2];

    create_2Dimage_ID(IDname, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    r0  = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    PA0 = atan2((y2 - y1), (x2 - x1));
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            x = 1.0 * ii;
            y = 1.0 * jj;
            x -= x1;
            y -= y1;
            xr = x * cos(PA0) + y * sin(PA0);
            yr = -x * sin(PA0) + y * cos(PA0);
            //r=sqrt(xr*xr+yr*yr);
            xr /= r0;
            yr /= r0;
            if((xr > 0) && (xr < 1.0) && (yr < 0.5 * t / r0) &&
                    (yr > -0.5 * t / r0))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 0.0f;
            }
        }

    return (ID);
}

// draw line crossing point xc, yc with angle, pixel value is coordinate axis perp to line
imageID make_lincoordinate(const char *IDname,
                           uint32_t    l1,
                           uint32_t    l2,
                           double      x_center,
                           double      y_center,
                           double      angle)
{
    imageID  ID;
    uint32_t naxes[2];
    double   x, y, x1;

    create_2Dimage_ID(IDname, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            x  = 1.0 * ii - x_center;
            y  = 1.0 * jj - y_center;
            x1 = x * cos(angle) + y * sin(angle);
            //y1 = -x*sin(angle) + y*cos(angle);
            dcimg[ID].array.F[jj * naxes[0] + ii] = x1;
        }

    return (ID);
}

imageID make_hexagon(const char *IDname,
                     uint32_t    l1,
                     uint32_t    l2,
                     double      x_center,
                     double      y_center,
                     double      radius)
{
    imageID  ID;
    uint32_t ii, jj;
    uint32_t naxes[2];
    float    x, y, r;
    float    value;

    long  iimin, iimax, jjmin, jjmax;
    float radius1, radius0sq;

    radius1   = radius * 2.0 / sqrt(3.0);
    radius0sq = radius * radius;

    printf("Making hexagon at %f x %f\n", x_center, y_center);

    create_2Dimage_ID(IDname, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    iimin = (long)(x_center - radius1 - 1.0);
    if(iimin < 0)
    {
        iimin = 0;
    }
    if(iimin > l1 - 1)
    {
        iimin = l1 - 1;
    }

    iimax = (long)(x_center + radius1 + 1.0);
    if(iimax < 0)
    {
        iimax = 0;
    }
    if(iimax > l1 - 1)
    {
        iimax = l1 - 1;
    }

    jjmin = (long)(y_center - radius1 - 1.0);
    if(jjmin < 0)
    {
        jjmin = 0;
    }
    if(jjmin > l2 - 1)
    {
        jjmin = l2 - 1;
    }

    jjmax = (long)(y_center + radius1 + 1.0);
    if(jjmax < 0)
    {
        jjmax = 0;
    }
    if(jjmax > l2 - 1)
    {
        jjmax = l2 - 1;
    }

#ifdef HAVE_LIBGOMP
    #pragma omp parallel default(shared) private(ii, jj, value, x, y, r)
    {
        #pragma omp for simd
#endif

        for(jj = jjmin; jj < jjmax; jj++)
            for(ii = iimin; ii < iimax; ii++)
            {
                value = 1.0;
                x     = 1.0 * ii - x_center;
                y     = 1.0 * jj - y_center;

                if(x * x + y * y > radius0sq)
                {
                    r = y;
                    if(fabs(r) > radius)
                    {
                        value = 0.0;
                    }
                    else
                    {
                        r = cos(PI / 6.0) * x + sin(PI / 6.0) * y;
                        if(fabs(r) > radius)
                        {
                            value = 0.0;
                        }
                        else
                        {
                            r = cos(-PI / 6.0) * x + sin(-PI / 6.0) * y;
                            if(fabs(r) > radius)
                            {
                                value = 0.0;
                            }
                        }
                    }
                }
                dcimg[ID].array.F[jj * naxes[0] + ii] = value;
            }
#ifdef HAVE_LIBGOMP
    }
#endif

    return (ID);
}

imageID IMAGE_gen_segments2WFmodes(const char *prefix,
                                   long        ndigit,
                                   const char *IDout_name)
{
    imageID IDout = -1;
    long    NBseg;
    long    seg;
    int     OK;
    char    imname[200];
    imageID IDarray[10000];
    long    ii, jj, kk, xsize, ysize, xysize;
    double  x, y;
    imageID IDmask;
    double *segxc;
    double *segyc;
    double *segsum;

    seg = 0;
    OK  = 1;
    while(OK == 1)
    {
        switch(ndigit)
        {

            case 1:
                sprintf(imname, "%s%01ld", prefix, seg);
                break;
            case 2:
                sprintf(imname, "%s%02ld", prefix, seg);
                break;
            case 3:
                sprintf(imname, "%s%03ld", prefix, seg);
                break;
            case 4:
                sprintf(imname, "%s%04ld", prefix, seg);
                break;
            case 5:
                sprintf(imname, "%s%05ld", prefix, seg);
                break;
            case 6:
                sprintf(imname, "%s%06ld", prefix, seg);
                break;

            default:
                printf("ERROR: Invalid number of didits\n");
                exit(0);
        }
        IDarray[seg] = image_ID(imname, dcimg, dcnimg);
        if(IDarray[seg] != -1)
        {
            seg++;
        }
        else
        {
            OK = 0;
        }
    }
    NBseg = seg;
    printf("Processing %ld segments\n", NBseg);
    if(NBseg > 0)
    {
        xsize  = dcimg[IDarray[0]].md[0].size[0];
        ysize  = dcimg[IDarray[0]].md[0].size[1];
        xysize = xsize * ysize;

        segxc = (double *) malloc(sizeof(double) * NBseg);
        if(segxc == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        segyc = (double *) malloc(sizeof(double) * NBseg);
        if(segyc == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        segsum = (double *) malloc(sizeof(double) * NBseg);
        if(segsum == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        create_2Dimage_ID("_pupmask", xsize, ysize, &IDmask);

        for(seg = 0; seg < NBseg; seg++)
        {
            segxc[seg]  = 0.0;
            segyc[seg]  = 0.0;
            segsum[seg] = 0.0;

            for(ii = 0; ii < xsize; ii++)
                for(jj = 0; jj < ysize; jj++)
                {
                    x = 1.0 * ii;
                    y = 1.0 * jj;
                    segxc[seg] +=
                        x * dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                    segyc[seg] +=
                        y * dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                    segsum[seg] +=
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii];

                    dcimg[IDmask].array.F[jj * xsize + ii] +=
                        (1.0 + seg) *
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                }
            segxc[seg] /= segsum[seg];
            segyc[seg] /= segsum[seg];
        }

        save_fits("_pupmask", "_pupmask.fits");

        //IDtmp = create_2Dimage_ID("_seg2wfm_tmp", xsize, ysize);
        create_3Dimage_ID(IDout_name, xsize, ysize, 3 * NBseg, &IDout);
        kk = 0;
        for(seg = 0; seg < NBseg; seg++)  // create modes one at a time
        {
            // piston seg
            for(ii = 0; ii < xsize; ii++)
                for(jj = 0; jj < xsize; jj++)
                {
                    dcimg[IDout].array.F[kk * xysize + jj * xsize + ii] =
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii];
                }
            kk++;

            // Tip
            for(ii = 0; ii < xsize; ii++)
                for(jj = 0; jj < xsize; jj++)
                {
                    dcimg[IDout].array.F[kk * xysize + jj * xsize + ii] =
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii] *
                        (1.0 * ii - segxc[seg]);
                }
            kk++;

            // Tilt
            for(ii = 0; ii < xsize; ii++)
                for(jj = 0; jj < xsize; jj++)
                {
                    dcimg[IDout].array.F[kk * xysize + jj * xsize + ii] =
                        dcimg[IDarray[seg]].array.F[jj * xsize + ii] *
                        (1.0 * jj - segyc[seg]);
                }
            kk++;
        }

        //delete_image_ID("_seg2wfm_tmp", DELETE_IMAGE_ERRMODE_WARNING);

        free(segxc);
        free(segyc);
        free(segsum);
    }

    return (IDout);
}

imageID make_hexsegpupil(
    const char *IDname, uint32_t size, double radius, double gap, double step)
{
    imageID  ID, ID1, IDp;
    long     x1, y1;
    double   x2, y2;
    imageID  IDdisk;
    uint32_t ii;
    double   tot = 0.0;
    long     size2;

    int    PISTONerr   = 0;
    int    errSEGindex = -1;
    double pampl;
    double piston;
    long   SEGcnt = 0;

    int   mkInfluenceFunctions = 1;
    long  IDif;
    int   seg;
    long  kk, jj;
    float xc, yc, tc;

    int    WriteCIF = 0;
    FILE  *fpmlevel;
    FILE  *fp       = NULL;
    FILE  *fp1      = NULL;
    double pixscale = 1.0;
    long   vID;
    double x, y;
    int    pt;

    long   IDmap1;
    long   index;
    double mapscalefactor = 1.037;
    long   size1;

    long *seglevel;
    long  i;
    long  tmpl1, tmpl2;
    int   segi;
    float segf;
    int   k;

    int *bitval;       // 0 or 1
    int  bitindex = 4; // 0 = MSB

    double vx, vy, rmsx, rmsy;

    if(WriteCIF == 1)
    {
        fp  = fopen("hexcoord.txt", "w");
        fp1 = fopen("hexcoord_pt.txt", "w");

        fprintf(fp, "DS 1 1 1;\n");
    }

    if((vID = variable_ID("pixscale")) != -1)
    {
        pixscale = dcvar[vID].value.f;
        printf("pixscale = %f\n", pixscale);
    }

    SEGcnt = 100;
    if((vID = variable_ID("SEGcnt")) != -1)
    {
        SEGcnt = (long)(0.1 + dcvar[vID].value.f);
        printf("SEGcnt = %ld\n", SEGcnt);
    }

    seglevel = (long *) malloc(sizeof(long) * SEGcnt);
    if(seglevel == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    bitval = (int *) malloc(sizeof(int) * SEGcnt);
    if(bitval == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    fpmlevel = fopen("fpm_level.txt", "r");
    if(fpmlevel != NULL)
    {
        for(i = 0; i < SEGcnt; i++)
        {
            int fscanfcnt = fscanf(fpmlevel, "%ld %ld\n", &tmpl1, &tmpl2);
            if(fscanfcnt == EOF)
            {
                if(ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr,
                            "Error: fscanf reached end of file, no matching "
                            "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if(fscanfcnt != 2)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i "
                        "input items, 2 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }

            seglevel[tmpl1 - 1] = tmpl2 + 15;
        }
        fclose(fpmlevel);
    }

    // SINGLE BIT
    for(i = 0; i < SEGcnt; i++)
    {
        printf("%5ld %5ld   ", i + 1, seglevel[i]);
        segf = 1.0 * seglevel[i] / 16.0;
        for(k = 0; k < 5; k++)
        {
            segi = (int) segf;
            printf(" %d", segi);
            segf -= segi;
            segf *= 2;

            if(k == bitindex)
            {
                bitval[i] = segi;
            }
        }
        printf("\n");
    }

    IDmap1 = image_ID("indexmap", dcimg, dcnimg);
    size1  = dcimg[IDmap1].md[0].size[0];

    size2 = size * size;

    ID = variable_ID("hexpupnoif");
    if(ID != -1)
    {
        mkInfluenceFunctions = 0;
    }

    ID = variable_ID("HEXPISTONerr");
    if(ID != -1)
    {
        PISTONerr = 1;
        pampl     = dcvar[ID].value.f;
        printf("Piston error = %f\n", pampl);
    }
    else
    {
        pampl = 0.0;
    }

    ID = variable_ID("HEXPISTONindex");
    if(ID != -1)
    {
        errSEGindex = (long)(dcvar[ID].value.f + 0.01);
        printf("SEGMENT INDEX = %ld\n", (long) errSEGindex);
    }

    create_2Dimage_ID(IDname, size, size, &ID);
    if(PISTONerr == 1)
    {
        create_2Dimage_ID("hexpupPha", size, size, &IDp);
    }

    IDdisk = make_disk("_TMPdisk", size, size, size / 2, size / 2, radius);
    for(ii = 0; ii < size2; ii++)
    {
        dcimg[IDdisk].array.F[ii] = 1.0f - dcimg[IDdisk].array.F[ii];
    }

    SEGcnt = 0;
    for(x1 = -(long)(2 * size / step); x1 < (long)(2 * size / step); x1++)
        for(y1 = -(long)(2 * size / step); y1 < (long)(2 * size / step);
                y1++)
        {
            x2 = step * x1 * 3;
            y2 = step * sqrt(3.0) * y1;

            if(sqrt(x2 * x2 + y2 * y2) < radius)
            {
                if(errSEGindex == -1)
                {
                    piston = pampl * (1.0 - 2.0 * ran1());
                }
                else
                {
                    if(errSEGindex == SEGcnt)
                    {
                        piston = pampl;
                    }
                    else
                    {
                        piston = 0.0;
                    }
                }
                printf("Hexagon %ld: ", SEGcnt);
                ID1 = make_hexagon("_TMPhex",
                                   size,
                                   size,
                                   0.5 * size + x2,
                                   0.5 * size + y2,
                                   (step - gap) * (sqrt(3.0) / 2.0));

                tot = 0.0;
                for(ii = 0; ii < size2; ii++)
                {
                    tot += dcimg[ID1].array.F[ii] *
                           dcimg[IDdisk].array.F[ii];
                }
                if(tot < 0.1)
                {
                    SEGcnt++;
                    if(WriteCIF == 1)
                    {
                        ii = (long)(0.5 * size1 + x2 * (0.5 * size1 / radius) *
                                    mapscalefactor);
                        jj = (long)(0.5 * size1 + y2 * (0.5 * size1 / radius) *
                                    mapscalefactor);
                        index = 0;
                        if(IDmap1 != -1)
                        {
                            index =
                                dcimg[IDmap1].array.UI16[jj * size1 + ii];
                        }

                        //  fprintf(fp, "# hex%03ld     index%03ld   [ %f %f ] -> [ %f %f ]     [%4ld %4ld] %f\n", SEGcnt, index, x2, y2, 0.5*size+x2, 0.5*size+y2, ii, jj, radius);
                        if(bitval[index - 1] == 1)
                        {
                            fprintf(fp, "L %ld;\n", seglevel[index - 1]);
                            fprintf(fp, "P");
                            for(pt = 0; pt < 6; pt++)
                            {
                                x = pixscale *
                                    (x2 + 1.0 * cos(2.0 * M_PI * pt / 6) *
                                     (step - gap));
                                y = pixscale *
                                    (y2 + 1.0 * sin(2.0 * M_PI * pt / 6) *
                                     (step - gap));
                                fprintf(fp,
                                        " %ld,%ld",
                                        (long)(100.0 * x),
                                        (long)(100.0 * y));
                                fprintf(fp1,
                                        "%ld %ld\n",
                                        (long)(100.0 * x),
                                        (long)(100.0 * y));
                            }
                            fprintf(fp, ";\n");
                        }
                    }

                    if(PISTONerr == 1)
                    {
                        for(ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] +=
                                dcimg[ID1].array.F[ii];
                        }
                    }
                    else
                    {
                        for(ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] +=
                                1.0f * SEGcnt * dcimg[ID1].array.F[ii];
                        }
                    }

                    if(PISTONerr == 1)
                    {
                        for(ii = 0; ii < size2; ii++)
                        {
                            dcimg[IDp].array.F[ii] +=
                                dcimg[ID1].array.F[ii] * piston;
                        }
                    }
                }
                delete_image_ID("_TMPhex", DELETE_IMAGE_ERRMODE_WARNING);
            }

            x2 += step * 1.5;
            y2 += step * sqrt(3.0) / 2.0;
            if(sqrt(x2 * x2 + y2 * y2) < radius)
            {
                // piston = pampl*(1.0-2.0*ran1());
                if(errSEGindex == -1)
                {
                    piston = pampl * (1.0 - 2.0 * ran1());
                }
                else
                {
                    if(errSEGindex == SEGcnt)
                    {
                        piston = pampl;
                    }
                    else
                    {
                        piston = 0.0;
                    }
                }
                printf("Hexagon %ld: ", SEGcnt);
                ID1 = make_hexagon("_TMPhex",
                                   size,
                                   size,
                                   0.5 * size + x2,
                                   0.5 * size + y2,
                                   (step - gap) * (sqrt(3.0) / 2.0));
                tot = 0.0;
                for(ii = 0; ii < size2; ii++)
                {
                    tot += dcimg[ID1].array.F[ii] *
                           dcimg[IDdisk].array.F[ii];
                }
                if(tot < 0.1)
                {
                    SEGcnt++;

                    if(WriteCIF == 1)
                    {
                        ii = (long)(0.5 * size1 + x2 * (0.5 * size1 / radius) *
                                    mapscalefactor);
                        jj = (long)(0.5 * size1 + y2 * (0.5 * size1 / radius) *
                                    mapscalefactor);
                        index = 0;
                        if(IDmap1 != -1)
                        {
                            index =
                                dcimg[IDmap1].array.UI16[jj * size1 + ii];
                        }

                        // fprintf(fp, "# hex%03ld     index%03ld   [ %f %f ] -> [ %f %f ]   [%4ld %4ld] %f\n", SEGcnt, index, x2, y2, 0.5*size+x2, 0.5*size+y2, ii, jj, radius);

                        if(bitval[index - 1] == 1)
                        {
                            fprintf(fp, "L %ld;\n", seglevel[index - 1]);
                            fprintf(fp, "P");
                            for(pt = 0; pt < 6; pt++)
                            {
                                x = pixscale *
                                    (x2 + 1.0 * cos(2.0 * M_PI * pt / 6) *
                                     (step - gap));
                                y = pixscale *
                                    (y2 + 1.0 * sin(2.0 * M_PI * pt / 6) *
                                     (step - gap));
                                fprintf(fp,
                                        " %ld,%ld",
                                        (long)(100.0 * x),
                                        (long)(100.0 * y));
                                fprintf(fp1,
                                        "%ld %ld\n",
                                        (long)(100.0 * x),
                                        (long)(100.0 * y));
                            }
                            fprintf(fp, ";\n");
                        }
                    }

                    if(PISTONerr == 1)
                    {
                        for(ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] +=
                                dcimg[ID1].array.F[ii];
                        }
                    }
                    else
                        for(ii = 0; ii < size2; ii++)
                        {
                            dcimg[ID].array.F[ii] +=
                                1.0f * SEGcnt * dcimg[ID1].array.F[ii];
                        }

                    if(PISTONerr == 1)
                    {
                        for(ii = 0; ii < size2; ii++)
                        {
                            dcimg[IDp].array.F[ii] +=
                                dcimg[ID1].array.F[ii] * piston;
                        }
                    }
                }
                delete_image_ID("_TMPhex", DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
    delete_image_ID("_TMPdisk", DELETE_IMAGE_ERRMODE_WARNING);

    printf("%ld segments\n", SEGcnt);

    if(WriteCIF == 1)
    {
        fprintf(fp, "DF;\n");
        fprintf(fp, "E\n");

        fclose(fp);
        fclose(fp1);
    }
    free(seglevel);
    free(bitval);

    if(mkInfluenceFunctions == 1)  // TT and focus for each segment
    {

        create_3Dimage_ID("hexpupif", size, size, 3 * SEGcnt, &IDif);
        for(seg = 0; seg < SEGcnt; seg++)
        {

            // piston
            kk = 3 * seg;
            xc = 0.0;
            yc = 0.0;
            tc = 0.0;
            for(ii = 0; ii < size; ii++)
                for(jj = 0; jj < size; jj++)
                {
                    if(fabs(dcimg[ID].array.F[jj * size + ii] -
                            (seg + 1.0)) < 0.01)
                    {
                        dcimg[IDif].array.F[kk * size2 + jj * size + ii] =
                            1.0;
                        xc += 1.0 * ii;
                        yc += 1.0 * jj;
                        tc += 1.0;
                    }
                }
            xc /= tc;
            yc /= tc;

            // tip and tilt
            rmsx = 0.0;
            rmsy = 0.0;
            for(ii = 0; ii < size; ii++)
                for(jj = 0; jj < size; jj++)
                {
                    if(fabs(dcimg[ID].array.F[jj * size + ii] -
                            (seg + 1.0)) < 0.01)
                    {
                        vx = 1.0 * ii - xc;
                        dcimg[IDif]
                        .array.F[(kk + 1) * size2 + jj * size + ii] = vx;
                        rmsx += vx * vx;

                        vy = 1.0 * jj - yc;
                        dcimg[IDif]
                        .array.F[(kk + 2) * size2 + jj * size + ii] = vy;
                        rmsy += vy * vy;
                    }
                }
            for(ii = 0; ii < size2; ii++)
            {
                dcimg[IDif].array.F[(kk + 1) * size2 + ii] *=
                    sqrt(tc / rmsx);
                dcimg[IDif].array.F[(kk + 2) * size2 + ii] *=
                    sqrt(tc / rmsy);
            }
        }
    }

    return (ID);
}

imageID make_jacquinot_pupil(const char *ID_name,
                             uint32_t    l1,
                             uint32_t    l2,
                             double      x_center,
                             double      y_center,
                             double      width,
                             double      height)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if((fabs(jj - y_center) / height) <
                    exp(-((ii - x_center) * (ii - x_center) / width / width)))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1;
            }
        }

    return (ID);
}

imageID make_sectors(const char *ID_name,
                     uint32_t    l1,
                     uint32_t    l2,
                     double      x_center,
                     double      y_center,
                     double      step,
                     long        NB_sectors)
{
    imageID  ID;
    uint32_t naxes[2];
    double   theta;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            theta = atan2((ii - x_center), (jj - y_center));
            if(theta < 0.0)
            {
                theta += 2.0 * PI;
            }
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                step * ((long)(theta / 2.0 / PI * NB_sectors));
        }

    return (ID);
}

imageID
make_rnd(const char *ID_name, uint32_t l1, uint32_t l2, const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    int      distrib;
    uint64_t nelement;

    distrib = 0; /* uniform */
    if(strstr(options, "gauss") != NULL)
    {
        distrib = 1; /* gauss */
        printf("gaussian distribution\n");
    }

    if(strstr(options, "trgauss") != NULL)
    {
        distrib = 2; /* truncated gauss */
        printf("truncated gaussian distribution\n");
    }

    if(dcdebug > 1)
    {
        fprintf(stdout, "Image size = %u %u\n", l1, l2);
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    nelement = naxes[0] * naxes[1];

    // openMP is slow when calling gsl random number generator : do not use openMP here
    if(distrib == 0)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.F[ii] = (double) ran1();
        }
    }
    if(distrib == 1)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.F[ii] = (double) gauss();
        }
    }
    if(distrib == 2)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.F[ii] = (double) gauss_trc();
        }
    }

    return (ID);
}

imageID make_rnd_double(const char *ID_name,
                        uint32_t    l1,
                        uint32_t    l2,
                        const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    int      distrib;
    uint64_t nelement;

    distrib = 0; /* uniform */
    if(strstr(options, "gauss") != NULL)
    {
        distrib = 1; /* gauss */
        printf("gaussian distribution\n");
    }

    if(strstr(options, "trgauss") != NULL)
    {
        distrib = 2; /* truncated gauss */
        printf("truncated gaussian distribution\n");
    }

    if(dcdebug > 1)
    {
        fprintf(stdout, "Image size = %u %u\n", l1, l2);
    }

    create_2Dimage_ID_double(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    nelement = naxes[0] * naxes[1];

    // openMP is slow when calling gsl random number generator : do not use openMP here
    if(distrib == 0)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.D[ii] = (double) ran1();
        }
    }
    if(distrib == 1)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.D[ii] = (double) gauss();
        }
    }
    if(distrib == 2)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[ID].array.D[ii] = (double) gauss_trc();
        }
    }

    return (ID);
}

/*
int make_rnd1(const char *ID_name, long l1, long l2, const char *options)
{
  int ID;
  long naxes[2];
  int distrib;
  long nelements;
  struct prng *g;

  distrib = 0;
  if (strstr(options,"-gauss")!=NULL)
    {
      distrib = 1;
    }

  if (strstr(options,"-trgauss")!=NULL)
    {
      distrib = 2;
      printf("truncated gaussian distribution\n");
   }

  g = prng_new("eicg(2147483647,111,1,0)");

   if (g == NULL)
   {
      fprintf(stderr,"Initialisation of generator failed.\n");
      exit (-1);
   }

   printf("Short name: %s\n",prng_short_name(g));

   printf("Expanded name: %s\n",prng_long_name(g));


   create_2Dimage_ID(ID_name,l1,l2);
   ID = image_ID(ID_name, dcimg, dcnimg);
   naxes[0] = dcimg[ID].md[0].size[0];
   naxes[1] = dcimg[ID].md[0].size[1];
   nelements=naxes[0]*naxes[1];

   prng_get_array(g,dcimg[ID].array.F,nelements);
   prng_reset(g);
   prng_free(g);


   return(0);
}
*/

imageID
make_gauss(const char *ID_name, uint32_t l1, uint32_t l2, double a, double A)
{
    imageID  ID;
    uint32_t naxes[2];
    double   distsq;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            distsq = (ii - naxes[0] / 2) * (ii - naxes[0] / 2) +
                     (jj - naxes[1] / 2) * (jj - naxes[1] / 2);
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (double) A * exp(-distsq / a / a);
        }
    /*  printf("FWHM = %f\n",2.0*a*sqrt(log(2)));*/
    return (ID);
}

imageID make_FiberCouplingOverlap(const char *ID_name)
{
    imageID  ID;
    uint32_t naxes[2];
    uint32_t size = 128;

    create_2Dimage_ID(ID_name, size, size, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    float TTcoeff = 0.2;

    float puprad = 0.1 * size;
    float xcent  = 1.32;
    float ycent  = 0.0;

    // compute TEM00 map
    imageID IDtem00;
    create_2Dimage_ID("tem00", size, size, &IDtem00);

    double fluxtot = 0.0;
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            float x     = 1.0 * (1.0 * ii - 0.5 * naxes[0]) / puprad;
            float y     = 1.0 * (1.0 * jj - 0.5 * naxes[1]) / puprad;
            float r0    = sqrt(x * x + y * y);
            float TEM00 = exp(-r0 * r0);

            fluxtot += TEM00 * TEM00;
            dcimg[IDtem00].array.F[jj * naxes[0] + ii] = TEM00;
        }
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[IDtem00].array.F[jj * naxes[0] + ii] /= sqrt(fluxtot);
        }
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            double totre = 0.0;
            double totim = 0.0;

            float TTx = 1.0 * (1.0 * ii - 0.5 * naxes[0]) * TTcoeff;
            float TTy = 1.0 * (1.0 * jj - 0.5 * naxes[1]) * TTcoeff;

            fluxtot = 0.0;
            for(uint32_t jj0 = 0; jj0 < naxes[1]; jj0++)
            {
                for(uint32_t ii0 = 0; ii0 < naxes[0]; ii0++)
                {
                    float pup_ampl;
                    float pup_pha;

                    // pupil coord x, y

                    float x  = 1.0 * (1.0 * ii0 - 0.5 * naxes[0]) / puprad;
                    float y  = 1.0 * (1.0 * jj0 - 0.5 * naxes[1]) / puprad;
                    float dx = x - xcent;
                    float dy = y - ycent;

                    float r = sqrt(dx * dx + dy * dy);

                    float TEM00 =
                        dcimg[IDtem00].array.F[jj0 * naxes[0] + ii0];

                    //dcimg[ID].array.F[jj * naxes[0] + ii] = -r;

                    if((r < 1.0) && (r > 0.3))
                    {
                        pup_ampl =
                            1.0f; //dcimg[IDtem00].array.F[jj0 * naxes[0] + ii0];
                        pup_pha = x * TTx + y * TTy;

                        fluxtot += pup_ampl * pup_ampl;

                        totre += TEM00 * (pup_ampl * cos(pup_pha));
                        totim += TEM00 * (pup_ampl * sin(pup_pha));
                    }
                    else
                    {
                        dcimg[ID].array.F[jj * naxes[0] + ii] = 0.0f;
                        pup_ampl                                   = 0.0;
                    }
                }
            }

            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (totre * totre + totim * totim) / sqrt(fluxtot);
        }
    }

    return ID;
}

imageID make_2axis_gauss(const char *ID_name,
                         uint32_t    l1,
                         uint32_t    l2,
                         double      a,
                         double      A,
                         double      E,
                         double      PA)
{
    imageID  ID;
    uint32_t naxes[2];
    double   distsq;
    double   iin, jjn;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            iin = 1.0 * (ii - naxes[0] / 2) * cos(PA) +
                  1.0 * (jj - naxes[1] / 2) * sin(PA);
            jjn = 1.0 * (jj - naxes[1] / 2) * cos(PA) -
                  1.0 * (ii - naxes[0] / 2) * sin(PA);
            distsq = iin * iin + (1.0 / (1.0 + E)) * jjn * jjn;
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (double) A * exp(-distsq / a / a);
        }

    return (ID);
}

imageID
make_cluster(const char *ID_name, uint32_t l1, uint32_t l2, const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    long     nb_star       = 3000;
    double   cluster_size  = 0.1; /* relative to the FOV */
    double   concentration = 1.0;
    long     i;
    double   tmp, dist, angle;
    char     input[50];
    int      str_pos;
    int      sim = 0;
    long     lii, ljj, hii, hjj;

    if(strstr(options, "-nbstars ") != NULL)
    {
        str_pos = strstr(options, "-nbstars ") - options;
        str_pos = str_pos + strlen("-nbstars ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i] = '\0';
        nb_star  = atol(input);
        printf("number of stars is %ld\n", nb_star);
    }

    if(strstr(options, "-conc ") != NULL)
    {
        str_pos = strstr(options, "-conc ") - options;
        str_pos = str_pos + strlen("-conc ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]      = '\0';
        concentration = atof(input);
        printf("concentration is %f\n", concentration);
    }

    if(strstr(options, "-size ") != NULL)
    {
        str_pos = strstr(options, "-size ") - options;
        str_pos = str_pos + strlen("-size ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]     = '\0';
        cluster_size = atof(input);
        printf("cluster size is %f\n", cluster_size);
    }

    if(strstr(options, "-sim") != NULL)
    {
        printf("all sources in the central half array \n");
        sim = 1;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    if(sim == 0)
    {
        lii = 0;
        ljj = 0;
        hii = naxes[0];
        hjj = naxes[1];
    }
    else
    {
        lii = naxes[0] / 4;
        ljj = naxes[1] / 4;
        hii = 3 * naxes[0] / 4;
        hjj = 3 * naxes[1] / 4;
    }

    i = 0;
    while(i < nb_star)
    {
        dist        = gauss();
        dist        = sqrt(sqrt(dist * dist));
        dist        = powf(dist, concentration);
        angle       = 2 * PI * ran1();
        uint32_t ii = (uint32_t)(naxes[0] / 2 + (cluster_size * naxes[0] / 2) *
                                 dist * cos(angle));
        uint32_t jj = (uint32_t)(naxes[1] / 2 + (cluster_size * naxes[1] / 2) *
                                 dist * sin(angle));

        if((ii > lii) && (jj > ljj) && (ii < hii) && (jj < hjj))
        {
            tmp = gauss();
            dcimg[ID].array.F[jj * naxes[0] + ii] += tmp * tmp;
            i++;
        }
    }

    return (ID);
}

imageID make_galaxy(const char *ID_name,
                    uint32_t    l1,
                    uint32_t    l2,
                    double      S_radius,
                    double      S_L0,
                    double      S_ell,
                    double      S_PA,
                    double      E_radius,
                    double      E_L0,
                    double      E_ell,
                    double      E_PA)
{
    imageID  ID;
    uint32_t naxes[2];
    double   x, y, r;
    double   aob, boa; /* a over b and b over a */
    double   total = 0;

    /* E = 1-b/a */

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = l1;
    naxes[1] = l2;

    /* Spiral component */
    aob = 1.0 / (1.0 - S_ell);
    boa = 1.0 - S_ell;

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1] / 2 + 1; jj++)
        {
            x = cos(S_PA) * (ii - naxes[0] / 2) +
                sin(S_PA) * (jj - naxes[1] / 2);
            y = -sin(S_PA) * (ii - naxes[0] / 2) +
                cos(S_PA) * (jj - naxes[1] / 2);
            r = sqrt(aob * x * x + boa * y * y);
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                S_L0 * exp(-r / S_radius);
        }

    /* Elliptical component */
    aob = 1.0 / (1.0 - E_ell);
    boa = 1.0 - E_ell;

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1] / 2 + 1; jj++)
        {
            x = cos(E_PA) * (ii - naxes[0] / 2) +
                sin(E_PA) * (jj - naxes[1] / 2);
            y = -sin(E_PA) * (ii - naxes[0] / 2) +
                cos(E_PA) * (jj - naxes[1] / 2);
            r = sqrt(aob * x * x + boa * y * y);
            dcimg[ID].array.F[jj * naxes[0] + ii] +=
                E_L0 * powf(10.0f, (-3.3307f * (powf((r / E_radius), 0.25f) - 1.0f)));
        }

    /* filling other half */
    for(uint32_t ii = 1; ii < naxes[0]; ii++)
        for(uint32_t jj = 1; jj < naxes[1] / 2; jj++)
        {
            dcimg[ID]
            .array.F[(naxes[1] - jj) * naxes[0] + (naxes[0] - ii)] =
                dcimg[ID].array.F[jj * naxes[0] + ii];
        }
    uint32_t ii = 0;
    for(uint32_t jj = naxes[1] / 2; jj < naxes[1]; jj++)
    {
        aob = 1.0 / (1.0 - S_ell);
        boa = 1.0 - S_ell;
        x   = cos(S_PA) * (ii - naxes[0] / 2) + sin(S_PA) * (jj - naxes[1] / 2);
        y = -sin(S_PA) * (ii - naxes[0] / 2) + cos(S_PA) * (jj - naxes[1] / 2);
        r = sqrt(aob * x * x + boa * y * y);
        dcimg[ID].array.F[jj * naxes[0] + ii] = S_L0 * exp(-r / S_radius);
        aob                                        = 1.0 / (1.0 - E_ell);
        boa                                        = 1.0 - E_ell;
        x = cos(E_PA) * (ii - naxes[0] / 2) + sin(E_PA) * (jj - naxes[1] / 2);
        y = -sin(E_PA) * (ii - naxes[0] / 2) + cos(E_PA) * (jj - naxes[1] / 2);
        r = sqrt(aob * x * x + boa * y * y);
        dcimg[ID].array.F[jj * naxes[0] + ii] +=
            E_L0 * powf(10.0f, (-3.3307f * (powf((r / E_radius), 0.25f) - 1.0f)));
    }

    total = 2.0 * PI * S_L0 * S_radius * S_radius +
            23.02 * E_L0 * E_radius * E_radius;
    printf("total : %f (%f)\n", arith_image_total(ID_name), total);

    return (ID);
}

imageID
make_Egalaxy(const char *ID_name, uint32_t l1, uint32_t l2, const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    double   galaxy_size   = 0.1; /* relative to the FOV */
    double   concentration = 1.0;
    long     i;
    double   PA   = 0;
    double   E    = 0.3; /* position angle and ellipticity */
    double   peak = 1;   /* maximum value */
    char     input[50];
    int      str_pos;
    int      sim = 0;
    long     lii, ljj, hii, hjj;
    double   x, y, xcenter, ycenter, distsq;

    if(strstr(options, "-conc ") != NULL)
    {
        str_pos = strstr(options, "-conc ") - options;
        str_pos = str_pos + strlen("-conc ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]      = '\0';
        concentration = atof(input);
        printf("concentration is %f\n", concentration);
    }

    if(strstr(options, "-size ") != NULL)
    {
        str_pos = strstr(options, "-size ") - options;
        str_pos = str_pos + strlen("-size ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i]    = '\0';
        galaxy_size = atof(input);
        printf("size is %f\n", galaxy_size);
    }

    if(strstr(options, "-pa ") != NULL)
    {
        str_pos = strstr(options, "-pa ") - options;
        str_pos = str_pos + strlen("-pa ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i] = '\0';
        PA       = atof(input);
        printf("galaxy pa size is %f radians \n", PA);
    }

    if(strstr(options, "-e ") != NULL)
    {
        str_pos = strstr(options, "-e ") - options;
        str_pos = str_pos + strlen("-e ");
        i       = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            input[i] = options[i + str_pos];
            i++;
        }
        input[i] = '\0';
        E        = atof(input);
        printf("galaxy elipticity is %f \n", E);
    }

    if(strstr(options, "-sim") != NULL)
    {
        printf("all sources in the central half array \n");
        sim = 1;
    }

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    xcenter  = naxes[0] / 2;
    ycenter  = naxes[1] / 2;

    if(sim == 0)
    {
        lii = 0;
        ljj = 0;
        hii = naxes[0];
        hjj = naxes[1];
    }
    else
    {
        lii = naxes[0] / 4;
        ljj = naxes[1] / 4;
        hii = 3 * naxes[0] / 4;
        hjj = 3 * naxes[1] / 4;
    }

    for(uint32_t jj = ljj; jj < hjj; jj++)
        for(uint32_t ii = lii; ii < hii; ii++)
        {
            x = cos(PA) * (ii - xcenter) + sin(PA) * (jj - ycenter);
            y = -sin(PA) * (ii - xcenter) + cos(PA) * (jj - ycenter);
            /* E = sqrt(a*a-b*b)/a */
            /* a = 1 */
            x      = x;
            y      = y / sqrt(1 - E * E);
            distsq = (x * x + y * y) /
                     (naxes[0] * naxes[0] + naxes[1] * naxes[1]) / galaxy_size /
                     galaxy_size;
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                (double) peak * exp(-concentration * distsq);
        }

    return (ID);
}

// for sol system, index ~2.4 with local zodi
imageID gen_image_EZdisk(const char *ID_name,
                         uint32_t    size,
                         double      InnerEdge,
                         double      Index,
                         double      Incl)
{
    imageID ID;
    double  x, y, r, r0;
    double  value;

    create_2Dimage_ID(ID_name, size, size, &ID);
    r0 = 6.0;
    for(uint32_t ii = 0; ii < size; ii++)
        for(uint32_t jj = 0; jj < size; jj++)
        {
            x = 1.0 * (ii + 0.5) - size / 2;
            y = 1.0 * (jj + 0.5) - size / 2;
            y /= cos(Incl);
            r = sqrt(x * x + y * y);
            if(r < InnerEdge)
            {
                value = 0.0;
            }
            else
            {
                value = powf(r, -Index);
            }
            value /= cos(Incl);

            value += powf(r0, -Index);
            dcimg[ID].array.F[jj * size + ii] = value;
        }

    return (ID);
}

imageID make_slopexy(
    const char *ID_name, uint32_t l1, uint32_t l2, double sx, double sy)
{
    imageID  ID;
    uint32_t naxes[2];
    double   coeff;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    coeff = sx * (naxes[0] / 2) + sy * (naxes[1] / 2);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                sx * ii + sy * jj - coeff;
        }

    return (ID);
}

imageID
make_dist(const char *ID_name, uint32_t l1, uint32_t l2, double f1, double f2)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] =
                sqrt((f1 - ii) * (f1 - ii) + (f2 - jj) * (f2 - jj));
        }

    return (ID);
}

imageID make_PosAngle(
    const char *ID_name, uint32_t l1, uint32_t l2, double f1, double f2)
{
    imageID  ID;
    uint32_t naxes[2];

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            double x, y;
            x                                          = 1.0 * ii - f1;
            y                                          = 1.0 * jj - f2;
            dcimg[ID].array.F[jj * naxes[0] + ii] = atan2(y, x);
        }

    return (ID);
}

imageID make_psf_from_profile(const char *profile_name,
                              const char *ID_name,
                              uint32_t    l1,
                              uint32_t    l2)
{
    imageID  ID;
    uint32_t naxes[2];
    FILE    *fp;
    long     nb_lines;
    char     lstring[1000];
    char     line[200];
    double  *distarr;
    double  *valarr;
    long     i;
    double   dist;
    float    fl1, fl2;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    /* compute number of lines */
    sprintf(lstring, "wc -l %s > tmpcnt.txt", profile_name);
    if(system(lstring) == -1)
    {
        printf("ERROR: system(\"%s\"), %s line %d\n",
               lstring,
               __FILE__,
               __LINE__);
        exit(0);
    }
    if((fp = fopen("tmpcnt.txt", "r")) == NULL)
    {
        printf("error : can't open file \"tmpcnt.txt\"\n");
    }
    if(fgets(line, 200, fp) == NULL)
    {
        printf("ERROR: fgets, %s line %d\n", __FILE__, __LINE__);
        exit(0);
    }
    fclose(fp);
    printf("%s\n", line);
    fflush(stdout);
    sscanf(line, "%ld %s", &nb_lines, lstring);

    printf("%ld lines\n", nb_lines);

    distarr = (double *) malloc(sizeof(double) * nb_lines);
    if(distarr == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    valarr = (double *) malloc(sizeof(double) * nb_lines);
    if(valarr == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if((fp = fopen(profile_name, "r")) == NULL)
    {
        printf("error : can't open file \"%s\"\n", profile_name);
    }

    for(i = 0; i < nb_lines; i++)
    {
        if(fgets(line, 200, fp) == NULL)
        {
            printf("ERROR: fgets, %s line %d\n", __FILE__, __LINE__);
            exit(0);
        }
        sscanf(line, "%f %f", &fl1, &fl2);
        distarr[i] = fl1;
        valarr[i]  = fl2;
    }
    fclose(fp);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dist = sqrt((ii - naxes[0] / 2) * (ii - naxes[0] / 2) +
                        (jj - naxes[1] / 2) * (jj - naxes[1] / 2));
            i    = 0;
            while((distarr[i] < dist) && (i != nb_lines - 1))
            {
                i++;
            }
            if(i != 0)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] =
                    valarr[i - 1] + (valarr[i] - valarr[i - 1]) *
                    (dist - distarr[i - 1]) /
                    (distarr[i] - distarr[i - 1]);
            }
            else
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = valarr[0];
            }
        }

    free(distarr);
    free(valarr);

    return (ID);
}

imageID make_offsetHyperGaussian(
    uint32_t size, double a, double b, long n, const char *IDname)
{
    imageID ID;

    create_2Dimage_ID(IDname, size, size, &ID);
    for(uint32_t ii = 0; ii < size; ii++)
        for(uint32_t jj = 0; jj < size; jj++)
        {
            double x, y, dist;

            x    = 1.0 * ii - size / 2;
            y    = 1.0 * jj - size / 2;
            dist = sqrt(x * x + y * y);
            if(dist < a)
            {
                dcimg[ID].array.F[jj * size + ii] = 0.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * size + ii] =
                    1.0f - expf(-powf((dist - a) / b, n));
            }
        }

    return (ID);
}

imageID
make_cosapoedgePupil(uint32_t size, double a, double b, const char *IDname)
{
    imageID ID;

    create_2Dimage_ID(IDname, size, size, &ID);
    for(uint32_t ii = 0; ii < size; ii++)
        for(uint32_t jj = 0; jj < size; jj++)
        {
            double x, y, dist;

            x    = 1.0 * ii - size / 2;
            y    = 1.0 * jj - size / 2;
            dist = sqrt(x * x + y * y);
            if(dist < a)
            {
                dcimg[ID].array.F[jj * size + ii] = 1.0f;
            }
            else if(dist > b)
            {
                dcimg[ID].array.F[jj * size + ii] = 0.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * size + ii] =
                    0.5 * (cos(PI * (dist - a) / (b - a)) + 1.0);
            }
        }

    return ID;
}

// make square grid of pixels
imageID make_2Dgridpix(const char *IDname,
                       uint32_t    xsize,
                       uint32_t    ysize,
                       double      pitchx,
                       double      pitchy,
                       double      offsetx,
                       double      offsety)
{
    imageID ID;
    double  x, y;
    long    i, j;
    double  u, t;

    create_2Dimage_ID(IDname, xsize, ysize, &ID);
    for(x = offsetx; x < xsize - 1; x += pitchx)
        for(y = offsety; y < ysize - 1; y += pitchy)
        {
            i                                           = (long) x;
            j                                           = (long) y;
            u                                           = x - i;
            t                                           = y - j;
            dcimg[ID].array.F[j * xsize + i]       = (1.0f - u) * (1.0f - t);
            dcimg[ID].array.F[(j + 1) * xsize + i] = (1.0f - u) * t;
            dcimg[ID].array.F[j * xsize + i + 1]   = u * (1.0f - t);
            dcimg[ID].array.F[(j + 1) * xsize + i + 1] = u * t;
        }

    return (ID);
}

// make tile
imageID make_tile(const char *IDin_name, uint32_t size, const char *IDout_name)
{
    uint32_t sizex0, sizey0; // input
    imageID  IDin, IDout;

    create_2Dimage_ID(IDout_name, size, size, &IDout);
    IDin   = image_ID(IDin_name, dcimg, dcnimg);
    sizex0 = dcimg[IDin].md[0].size[0];
    sizey0 = dcimg[IDin].md[0].size[1];

    for(uint32_t ii = 0; ii < size; ii++)
        for(uint32_t jj = 0; jj < size; jj++)
        {
            uint32_t ii0 = ii % sizex0;
            uint32_t jj0 = jj % sizey0;
            dcimg[IDout].array.F[jj * size + ii] =
                dcimg[IDin].array.F[jj0 * sizex0 + ii0];
        }

    return (IDout);
}

// make image that is coordinate of input
// for example, if axis = 0
// value = 1.0 x ii
// if axis value is not one of the coordinates, write pixel index
//
imageID
image_gen_im2coord(const char *IDin_name, uint8_t axis, const char *IDout_name)
{
    uint8_t  naxis;
    int      OK = 1;
    imageID  IDin;
    imageID  IDout = -1;
    uint32_t xsize, ysize, zsize;

    IDin  = image_ID(IDin_name, dcimg, dcnimg);
    naxis = dcimg[IDin].md[0].naxis;

    if(axis > naxis - 1)
    {
        printf("Image has only %u axis, cannot access axis %u\n", naxis, axis);
        OK = 0;
    }

    if(naxis > 3)
    {
        printf("naxis should be 3 or less\n");
        OK = 0;
    }

    if(OK == 1)
    {

        if(naxis == 1)
        {
            printf("naxis = 1\n");
            fflush(stdout);
            xsize = dcimg[IDin].md[0].size[0];
            create_1Dimage_ID(IDout_name, xsize, &IDout);
            for(uint32_t ii = 0; ii < xsize; ii++)
            {
                dcimg[IDout].array.F[ii] = 1.0f * ii;
            }
        }

        if(naxis == 2)
        {
            printf("naxis = 2\n");
            fflush(stdout);
            xsize = dcimg[IDin].md[0].size[0];
            ysize = dcimg[IDin].md[0].size[1];
            create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);

            switch(axis)
            {
                case 0:
                    for(uint32_t ii = 0; ii < xsize; ii++)
                        for(uint32_t jj = 0; jj < ysize; jj++)
                        {
                            dcimg[IDout].array.F[jj * xsize + ii] = 1.0f * ii;
                        }
                    break;
                case 1:
                    for(uint32_t ii = 0; ii < xsize; ii++)
                        for(uint32_t jj = 0; jj < ysize; jj++)
                        {
                            dcimg[IDout].array.F[jj * xsize + ii] = 1.0f * jj;
                        }
                    break;
                default:
                    for(uint32_t ii = 0; ii < xsize; ii++)
                        for(uint32_t jj = 0; jj < ysize; jj++)
                        {
                            dcimg[IDout].array.F[jj * xsize + ii] =
                                1.0 * jj * xsize + ii;
                        }
            }
        }

        if(naxis == 3)
        {
            printf("naxis = 3\n");
            fflush(stdout);
            xsize = dcimg[IDin].md[0].size[0];
            ysize = dcimg[IDin].md[0].size[1];
            zsize = dcimg[IDin].md[0].size[2];
            create_3Dimage_ID(IDout_name, xsize, ysize, zsize, &IDout);

            switch(axis)
            {
                case 0:
                    for(uint32_t ii = 0; ii < xsize; ii++)
                        for(uint32_t jj = 0; jj < ysize; jj++)
                            for(uint32_t kk = 0; kk < zsize; kk++)
                            {
                                dcimg[IDout]
                                .array.F[kk * xsize * ysize + jj * xsize + ii] =
                                    1.0 * ii;
                            }
                    break;
                case 1:
                    for(uint32_t ii = 0; ii < xsize; ii++)
                        for(uint32_t jj = 0; jj < ysize; jj++)
                            for(uint32_t kk = 0; kk < zsize; kk++)
                            {
                                dcimg[IDout]
                                .array.F[kk * xsize * ysize + jj * xsize + ii] =
                                    1.0 * jj;
                            }
                    break;
                case 2:
                    for(uint32_t ii = 0; ii < xsize; ii++)
                        for(uint32_t jj = 0; jj < xsize; jj++)
                            for(uint32_t kk = 0; kk < zsize; kk++)
                            {
                                dcimg[IDout]
                                .array.F[kk * xsize * ysize + jj * xsize + ii] =
                                    1.0 * kk;
                            }
                    break;
                default:
                    for(uint32_t ii = 0; ii < xsize; ii++)
                        for(uint32_t jj = 0; jj < xsize; jj++)
                            for(uint32_t kk = 0; kk < zsize; kk++)
                            {
                                dcimg[IDout]
                                .array.F[kk * xsize * ysize + jj * xsize + ii] =
                                    1.0 * kk * xsize * ysize + jj * xsize + ii;
                            }
            }
        }
    }

    return (IDout);
}


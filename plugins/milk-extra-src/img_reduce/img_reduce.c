/**
 * @file    img_reduce.c
 * @brief   Image analysis functions
 *
 * Misc image analysis functions
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
#define MODULE_SHORTNAME_DEFAULT "imgred"

// Module short description
#define MODULE_DESCRIPTION "Image analysis/reduction routines"

#include "img_reduce_internal.h"
/** Image analysis/reduction routines for astronomy
 *
 *
 */

int    badpixclean_init = 0;
long   badpixclean_NBop;
long  *badpixclean_array_indexin;
long  *badpixclean_array_indexout;
float *badpixclean_array_coeff;

long  badpixclean_NBbadpix;
long *badpixclean_indexlist;

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(img_reduce)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/* ============================================
 * Command: rmbadpixfast
 * ============================================ */

imageID IMG_REDUCE_cleanbadpix_fast(
    const char *IDname,
    const char *IDbadpix_name,
    const char *IDoutname,
    int         streamMode);

static char bpf_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "im";
static char bpf_bp[FUNCTION_PARAMETER_STRMAXLEN]
    = "bpmap";
static char bpf_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outim";

static FPS_APP_INFO FPS_app_info_bpf = {
    .fps_name    = "rmbadpixfast",
    .cmdkey      = "rmbadpixfast",
    .description =
        "remove bad pixels (fast algo)",
    .description_long =
        "Reduce raw astronomical image data: apply dark subtraction, flat fielding, and bad pixel correction to image cubes."
};

#define FPS_PARAMS_BPF(X) \
    X(".in_name", bpf_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".bp_name", bpf_bp, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "bad pixel map") \
    X(".out_name", bpf_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image")

#include "fps.h"

static FPS_CLI_BINDING bpf_bindings[] = {
    FPS_PARAMS_BPF(FPS_X_BINDING)
};
static const int bpf_nb =
    sizeof(bpf_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS_BPF(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS bpf_cms = {0};

static __attribute__((constructor))
void init_bpf_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info_bpf.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info_bpf.description,
            sizeof(CLIcmddata.description)
            - 1);
    CLIcmddata.nbarg =
        sizeof(farg) / sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags = CLICMDFLAG_FPS;
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &bpf_cms;
    }
}

static errno_t bpf_compute(void)
{
    IMG_REDUCE_cleanbadpix_fast(
        bpf_in, bpf_bp, bpf_out, 0);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_bpf, farg,
        &CLIcmddata,
        bpf_bindings, bpf_nb,
        bpf_compute);
}

/* ============================================
 * Command: rmbadpixfasts
 * ============================================ */

static char bps_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "imstream";
static char bps_bp[FUNCTION_PARAMETER_STRMAXLEN]
    = "bpmap";
static char bps_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outimstream";

static FPS_APP_INFO FPS_app_info_bps = {
    .fps_name    = "rmbadpixfasts",
    .cmdkey      = "rmbadpixfasts",
    .description =
        "remove bad pixels (fast, stream)",
    .description_long =
        "Reduce raw astronomical image data: apply dark subtraction, flat fielding, and bad pixel correction to image cubes."
};

#define FPS_PARAMS_BPS(X) \
    X(".in_name", bps_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
    X(".bp_name", bps_bp, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "bad pixel map") \
    X(".out_name", bps_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output stream")

static FPS_CLI_BINDING bps_bindings[] = {
    FPS_PARAMS_BPS(FPS_X_BINDING)
};
static const int bps_nb =
    sizeof(bps_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF bps_farg[] = {
    FPS_PARAMS_BPS(FPS_X_FARG)
};
static CLICMDDATA bps_data = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS bps_cms = {0};

static __attribute__((constructor))
void init_bps_cms(void)
{
    strncpy(bps_data.key,
            FPS_app_info_bps.cmdkey,
            sizeof(bps_data.key) - 1);
    strncpy(bps_data.description,
            FPS_app_info_bps.description,
            sizeof(bps_data.description)
            - 1);
    bps_data.nbarg =
        sizeof(bps_farg) / sizeof(CLICMDARGDEF);
    bps_data.funcfpscliarg = bps_farg;
    bps_data.flags = CLICMDFLAG_FPS;
    if (bps_data.cmdsettings == NULL) {
        bps_data.cmdsettings = &bps_cms;
    }
}

static errno_t bps_compute(void)
{
    IMG_REDUCE_cleanbadpix_fast(
        bps_in, bps_bp, bps_out, 1);
    return RETURN_SUCCESS;
}

static errno_t bps_CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_bps, bps_farg,
        &bps_data,
        bps_bindings, bps_nb,
        bps_compute);
}

/* ============================================
 * Command: cubesimplestat
 * ============================================ */

imageID IMG_REDUCE_cubesimplestat(
    const char *IDin_name);

static char css_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "imcube";

static FPS_APP_INFO FPS_app_info_css = {
    .fps_name    = "cubesimplestat",
    .cmdkey      = "cubesimplestat",
    .description =
        "simple data cube stats",
    .description_long =
        "Reduce raw astronomical image data: apply dark subtraction, flat fielding, and bad pixel correction to image cubes."
};

#define FPS_PARAMS_CSS(X) \
    X(".in_name", css_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image cube")

static FPS_CLI_BINDING css_bindings[] = {
    FPS_PARAMS_CSS(FPS_X_BINDING)
};
static const int css_nb =
    sizeof(css_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF css_farg[] = {
    FPS_PARAMS_CSS(FPS_X_FARG)
};
static CLICMDDATA css_data = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS css_cms = {0};

static __attribute__((constructor))
void init_css_cms(void)
{
    strncpy(css_data.key,
            FPS_app_info_css.cmdkey,
            sizeof(css_data.key) - 1);
    strncpy(css_data.description,
            FPS_app_info_css.description,
            sizeof(css_data.description)
            - 1);
    css_data.nbarg =
        sizeof(css_farg) / sizeof(CLICMDARGDEF);
    css_data.funcfpscliarg = css_farg;
    css_data.flags = CLICMDFLAG_FPS;
    if (css_data.cmdsettings == NULL) {
        css_data.cmdsettings = &css_cms;
    }
}

static errno_t css_compute(void)
{
    IMG_REDUCE_cubesimplestat(css_in);
    return RETURN_SUCCESS;
}

static errno_t css_CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_css, css_farg,
        &css_data,
        css_bindings, css_nb,
        css_compute);
}

/* ============================================
 * Command: imcenternorm
 * ============================================ */

imageID IMG_REDUCE_centernormim(
    const char *IDin_name,
    const char *IDref_name,
    const char *IDout_name,
    long xcent0,
    long ycent0,
    long xcentsize,
    long ycentsize,
    int mode,
    int semtrig);

static char cn_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "imin";
static char cn_ref[FUNCTION_PARAMETER_STRMAXLEN]
    = "imref";
static char cn_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "imout";
static int64_t cn_xc0 = 100;
static int64_t cn_yc0 = 100;
static int64_t cn_xcs = 20;
static int64_t cn_ycs = 20;
static int64_t cn_mode = 0;
static int64_t cn_sem = 0;

static FPS_APP_INFO FPS_app_info_cn = {
    .fps_name    = "imcenternorm",
    .cmdkey      = "imcenternorm",
    .description =
        "recenter and normalize image",
    .description_long =
        "Reduce raw astronomical image data: apply dark subtraction, flat fielding, and bad pixel correction to image cubes."
};

#define FPS_PARAMS_CN(X) \
    X(".in_name", cn_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".ref_name", cn_ref, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "reference image") \
    X(".out_name", cn_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".xc0", &cn_xc0, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x centering start") \
    X(".yc0", &cn_yc0, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y centering start") \
    X(".xcs", &cn_xcs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x centering size") \
    X(".ycs", &cn_ycs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y centering size") \
    X(".mode", &cn_mode, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "shared mem mode") \
    X(".semtrig", &cn_sem, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "semaphore trigger")

static FPS_CLI_BINDING cn_bindings[] = {
    FPS_PARAMS_CN(FPS_X_BINDING)
};
static const int cn_nb =
    sizeof(cn_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF cn_farg[] = {
    FPS_PARAMS_CN(FPS_X_FARG)
};
static CLICMDDATA cn_data = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cn_cms = {0};

static __attribute__((constructor))
void init_cn_cms(void)
{
    strncpy(cn_data.key,
            FPS_app_info_cn.cmdkey,
            sizeof(cn_data.key) - 1);
    strncpy(cn_data.description,
            FPS_app_info_cn.description,
            sizeof(cn_data.description)
            - 1);
    cn_data.nbarg =
        sizeof(cn_farg) / sizeof(CLICMDARGDEF);
    cn_data.funcfpscliarg = cn_farg;
    cn_data.flags = CLICMDFLAG_FPS;
    if (cn_data.cmdsettings == NULL) {
        cn_data.cmdsettings = &cn_cms;
    }
}

static errno_t cn_compute(void)
{
    IMG_REDUCE_centernormim(
        cn_in, cn_ref, cn_out,
        (long) cn_xc0, (long) cn_yc0,
        (long) cn_xcs, (long) cn_ycs,
        (int) cn_mode, (int) cn_sem);
    return RETURN_SUCCESS;
}

static errno_t cn_CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_cn, cn_farg,
        &cn_data,
        cn_bindings, cn_nb,
        cn_compute);
}

/* ============================================
 * Command: imgcubeprocess
 * ============================================ */

int IMG_REDUCE_cubeprocess(
    const char *IDin_name);

static char cp_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "imcube";

static FPS_APP_INFO FPS_app_info_cp = {
    .fps_name    = "imgcubeprocess",
    .cmdkey      = "imgcubeprocess",
    .description =
        "data cube process",
    .description_long =
        "Reduce raw astronomical image data: apply dark subtraction, flat fielding, and bad pixel correction to image cubes."
};

#define FPS_PARAMS_CP(X) \
    X(".in_name", cp_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image cube")

static FPS_CLI_BINDING cp_bindings[] = {
    FPS_PARAMS_CP(FPS_X_BINDING)
};
static const int cp_nb =
    sizeof(cp_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF cp_farg[] = {
    FPS_PARAMS_CP(FPS_X_FARG)
};
static CLICMDDATA cp_data = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cp_cms = {0};

static __attribute__((constructor))
void init_cp_cms(void)
{
    strncpy(cp_data.key,
            FPS_app_info_cp.cmdkey,
            sizeof(cp_data.key) - 1);
    strncpy(cp_data.description,
            FPS_app_info_cp.description,
            sizeof(cp_data.description)
            - 1);
    cp_data.nbarg =
        sizeof(cp_farg) / sizeof(CLICMDARGDEF);
    cp_data.funcfpscliarg = cp_farg;
    cp_data.flags = CLICMDFLAG_FPS;
    if (cp_data.cmdsettings == NULL) {
        cp_data.cmdsettings = &cp_cms;
    }
}

static errno_t cp_compute(void)
{
    IMG_REDUCE_cubeprocess(cp_in);
    return RETURN_SUCCESS;
}

static errno_t cp_CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_cp, cp_farg,
        &cp_data,
        cp_bindings, cp_nb,
        cp_compute);
}

/* ============================================
 * Module init
 * ============================================ */

static errno_t init_module_CLI()
{
    /* rmbadpixfast */
    {
        safe_fps_fill_farg_examples(
            farg, bpf_bindings, bpf_nb);
        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    /* rmbadpixfasts */
    {
        safe_fps_fill_farg_examples(
            bps_farg, bps_bindings, bps_nb);
        int cmdi = RegisterCLIcmd(
            bps_data, bps_CLIfunction);
        bps_data.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    /* cubesimplestat */
    {
        safe_fps_fill_farg_examples(
            css_farg, css_bindings, css_nb);
        int cmdi = RegisterCLIcmd(
            css_data, css_CLIfunction);
        css_data.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    /* imcenternorm */
    {
        safe_fps_fill_farg_examples(
            cn_farg, cn_bindings, cn_nb);
        int cmdi = RegisterCLIcmd(
            cn_data, cn_CLIfunction);
        cn_data.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    /* imgcubeprocess */
    {
        safe_fps_fill_farg_examples(
            cp_farg, cp_bindings, cp_nb);
        int cmdi = RegisterCLIcmd(
            cp_data, cp_CLIfunction);
        cp_data.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    // add atexit functions here

    return RETURN_SUCCESS;
}


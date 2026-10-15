// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#include "milk_runnable.h"

int    badpixclean_init = 0;
long   badpixclean_NBop;
long  *badpixclean_array_indexin;
long  *badpixclean_array_indexout;
float *badpixclean_array_coeff;

long  badpixclean_NBbadpix;
long *badpixclean_indexlist;

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/* ============================================
 * Command: rmbadpixfast
 * ============================================ */

imageID IMG_REDUCE_cleanbadpix_fast(const char *IDname,
                                    const char *IDbadpix_name,
                                    const char *IDoutname,
                                    int         streamMode);

static char bpf_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im";
static char bpf_bp[FUNCTION_PARAMETER_STRMAXLEN]  = "bpmap";
static char bpf_out[FUNCTION_PARAMETER_STRMAXLEN] = "outim";

#define FPS_PARAMS_BPF(X)                                                              \
    X(".in_name", bpf_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")   \
    X(".bp_name", bpf_bp, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "bad pixel map") \
    X(".out_name", bpf_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")

MILK_PARAMS_DEFINE(bpf, FPS_PARAMS_BPF);

static MILK_HOT errno_t bpf_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMG_REDUCE_cleanbadpix_fast(bpf_in, bpf_bp, bpf_out, 0);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_img_reduce_bpf = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "rmbadpixfast",
             .cmdkey           = "rmbadpixfast",
             .description      = "remove bad pixels (fast algo)",
             .description_long = "Reduce raw astronomical image data: apply dark subtraction, flat "
                                 "fielding, and bad pixel correction to image cubes." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(bpf),
    .compute = bpf_compute,
};

/* ============================================
 * Command: rmbadpixfasts
 * ============================================ */

static char bps_in[FUNCTION_PARAMETER_STRMAXLEN]  = "imstream";
static char bps_bp[FUNCTION_PARAMETER_STRMAXLEN]  = "bpmap";
static char bps_out[FUNCTION_PARAMETER_STRMAXLEN] = "outimstream";

#define FPS_PARAMS_BPS(X)                                                              \
    X(".in_name", bps_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream")  \
    X(".bp_name", bps_bp, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "bad pixel map") \
    X(".out_name", bps_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output stream")

MILK_PARAMS_DEFINE(bps, FPS_PARAMS_BPS);

static MILK_HOT errno_t bps_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMG_REDUCE_cleanbadpix_fast(bps_in, bps_bp, bps_out, 1);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_img_reduce_bps = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "rmbadpixfasts",
             .cmdkey           = "rmbadpixfasts",
             .description      = "remove bad pixels (fast, stream)",
             .description_long = "Reduce raw astronomical image data: apply dark subtraction, flat "
                                 "fielding, and bad pixel correction to image cubes." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(bps),
    .compute = bps_compute,
};

/* ============================================
 * Command: cubesimplestat
 * ============================================ */

imageID IMG_REDUCE_cubesimplestat(const char *IDin_name);

static char css_in[FUNCTION_PARAMETER_STRMAXLEN] = "imcube";

#define FPS_PARAMS_CSS(X) \
    X(".in_name", css_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image cube")

MILK_PARAMS_DEFINE(css, FPS_PARAMS_CSS);

static MILK_HOT errno_t css_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMG_REDUCE_cubesimplestat(css_in);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_img_reduce_css = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "cubesimplestat",
             .cmdkey           = "cubesimplestat",
             .description      = "simple data cube stats",
             .description_long = "Reduce raw astronomical image data: apply dark subtraction, flat "
                                 "fielding, and bad pixel correction to image cubes." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(css),
    .compute = css_compute,
};

/* ============================================
 * Command: imcenternorm
 * ============================================ */

imageID IMG_REDUCE_centernormim(const char *IDin_name,
                                const char *IDref_name,
                                const char *IDout_name,
                                long        xcent0,
                                long        ycent0,
                                long        xcentsize,
                                long        ycentsize,
                                int         mode,
                                int         semtrig);

static char    cn_in[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static char    cn_ref[FUNCTION_PARAMETER_STRMAXLEN] = "imref";
static char    cn_out[FUNCTION_PARAMETER_STRMAXLEN] = "imout";
static int64_t cn_xc0                               = 100;
static int64_t cn_yc0                               = 100;
static int64_t cn_xcs                               = 20;
static int64_t cn_ycs                               = 20;
static int64_t cn_mode                              = 0;
static int64_t cn_sem                               = 0;

#define FPS_PARAMS_CN(X)                                                                  \
    X(".in_name", cn_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")       \
    X(".ref_name", cn_ref, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "reference image") \
    X(".out_name", cn_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")        \
    X(".xc0", &cn_xc0, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x centering start")        \
    X(".yc0", &cn_yc0, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y centering start")        \
    X(".xcs", &cn_xcs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x centering size")         \
    X(".ycs", &cn_ycs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y centering size")         \
    X(".mode", &cn_mode, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "shared mem mode")        \
    X(".semtrig", &cn_sem, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "semaphore trigger")

MILK_PARAMS_DEFINE(cn, FPS_PARAMS_CN);

static MILK_HOT errno_t cn_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMG_REDUCE_centernormim(cn_in, cn_ref, cn_out, (long) cn_xc0, (long) cn_yc0, (long) cn_xcs,
                            (long) cn_ycs, (int) cn_mode, (int) cn_sem);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_img_reduce_cn = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "imcenternorm",
             .cmdkey           = "imcenternorm",
             .description      = "recenter and normalize image",
             .description_long = "Reduce raw astronomical image data: apply dark subtraction, flat "
                                 "fielding, and bad pixel correction to image cubes." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cn),
    .compute = cn_compute,
};

/* ============================================
 * Command: imgcubeprocess
 * ============================================ */

int IMG_REDUCE_cubeprocess(const char *IDin_name);

static char cp_in[FUNCTION_PARAMETER_STRMAXLEN] = "imcube";

#define FPS_PARAMS_CP(X) \
    X(".in_name", cp_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image cube")

MILK_PARAMS_DEFINE(cp, FPS_PARAMS_CP);

static MILK_HOT errno_t cp_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMG_REDUCE_cubeprocess(cp_in);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_img_reduce_cp = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "imgcubeprocess",
             .cmdkey           = "imgcubeprocess",
             .description      = "data cube process",
             .description_long = "Reduce raw astronomical image data: apply dark subtraction, flat "
                                 "fielding, and bad pixel correction to image cubes." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cp),
    .compute = cp_compute,
};

/* ============================================
 * Module registry
 * ============================================ */

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_img_reduce_bpf, &runnable_img_reduce_bps, &runnable_img_reduce_css,
    &runnable_img_reduce_cn,  &runnable_img_reduce_cp,
};

MILK_MODULE_RUNNABLES(img_reduce, NULL, runnables);

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

#include <err.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <ncurses.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <fitsio.h>

#include "CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fft/fft.h"
#include "image_filter/image_filter.h"

#include "img_reduce/img_reduce.h"

#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
#endif

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
        "remove bad pixels (fast algo)"
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
        "remove bad pixels (fast, stream)"
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
        "simple data cube stats"
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
        "recenter and normalize image"
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
        "data cube process"
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

/** compute ave, RMS
 *
 */

imageID IMG_REDUCE_cubesimplestat(const char *IDin_name)
{
    imageID IDin;
    long    xsize, ysize, zsize;
    long    xysize;
    long    ii, kk;
    long    offset;
    double  tmpf;

    long IDave, IDrms;

    IDin = image_ID(IDin_name, dcimg, dcnimg);

    xsize = dcimg[IDin].md[0].size[0];
    ysize = dcimg[IDin].md[0].size[1];
    zsize = dcimg[IDin].md[0].size[2];

    xysize = xsize * ysize;

    create_2Dimage_ID("c_ave", xsize, ysize, &IDave);
    create_2Dimage_ID("c_rms", xsize, ysize, &IDrms);

    for(kk = 0; kk < zsize; kk++)
    {
        offset = kk * xysize;
        for(ii = 0; ii < xysize; ii++)
        {
            tmpf = dcimg[IDin].array.F[offset + ii];
            dcimg[IDave].array.F[ii] += tmpf;
            dcimg[IDrms].array.F[ii] += tmpf * tmpf;
        }
    }

    for(ii = 0; ii < xysize; ii++)
    {
        dcimg[IDave].array.F[ii] /= zsize;
        dcimg[IDrms].array.F[ii] /= zsize;
        dcimg[IDrms].array.F[ii] =
            sqrt(dcimg[IDrms].array.F[ii] -
                 dcimg[IDave].array.F[ii] * dcimg[IDave].array.F[ii]);
    }

    return IDin;
}

/// removes bad pixels in cube

errno_t clean_bad_pix(const char *IDin_name, const char *IDbadpix_name)
{
    long    ii, jj, kk;
    long    IDin, IDbadpix, IDbadpix1; //, IDouttmp;
    long    xsize, ysize, zsize;
    double *pix;
    double  bpix[3][3];
    long    i, j;
    double  sum_bpix, *sum_pix;
    long    left, fixed;
    long    xysize;

    IDin = image_ID(IDin_name, dcimg, dcnimg);

    xsize  = dcimg[IDin].md[0].size[0];
    ysize  = dcimg[IDin].md[0].size[1];
    zsize  = dcimg[IDin].md[0].size[2];
    xysize = xsize * ysize;

    sum_pix = (double *) malloc(sizeof(double) * zsize);
    if(sum_pix == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    pix = (double *) malloc(sizeof(double) * zsize * 3 * 3);
    if(pix == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    copy_image_ID(IDbadpix_name, "badpix_tmp", 0);
    IDbadpix = image_ID("badpix_tmp", dcimg, dcnimg);
    copy_image_ID("badpix_tmp", "newbadpix_tmp", 0);
    IDbadpix1 = image_ID("newbadpix_tmp", dcimg, dcnimg);

    //    copy_image_ID(IDin_name, "bpcleaned_tmp");
    //   IDouttmp = image_ID("bpcleaned_tmp", dcimg, dcnimg);

    left = 1;
    while(left != 0)
    {
        left  = 0;
        fixed = 0;

        for(jj = 1; jj < ysize - 1; jj++)
            for(ii = 1; ii < xsize - 1; ii++)
            {
                if(dcimg[IDbadpix].array.F[jj * xsize + ii] > 0.5f)
                {
                    sum_bpix = 0.0;
                    for(kk = 0; kk < zsize; kk++)
                    {
                        sum_pix[kk] = 0.0;
                    }

                    for(i = 0; i < 3; i++)
                        for(j = 0; j < 3; j++)
                        {
                            for(kk = 0; kk < zsize; kk++)
                            {
                                pix[kk * 9 + j * 3 + i] =
                                    dcimg[IDin]
                                    .array
                                    .F[kk * xysize + (jj - 1 + j) * xsize +
                                          (ii - 1 + i)];
                            }
                            bpix[i][j] =
                                dcimg[IDbadpix]
                                .array
                                .F[(jj - 1 + j) * xsize + (ii - 1 + i)];
                            sum_bpix += bpix[i][j];
                            for(kk = 0; kk < zsize; kk++)
                            {
                                sum_pix[kk] += (1.0 - bpix[i][j]) *
                                               pix[kk * 9 + j * 3 + i];
                            }
                        }
                    sum_bpix = 9.0 - sum_bpix;
                    if(sum_bpix > 2.1)
                    {
                        for(kk = 0; kk < zsize; kk++)
                        {
                            dcimg[IDin]
                            .array.F[kk * xysize + jj * xsize + ii] =
                                sum_pix[kk] / sum_bpix;
                        }
                        dcimg[IDbadpix1].array.F[jj * xsize + ii] = 0.0f;
                        fixed += 1;
                    }
                    else
                    {
                        left += 1;
                    }
                }
            }

        for(jj = 1; jj < ysize - 1; jj++)
            for(ii = 1; ii < xsize - 1; ii++)
            {
                dcimg[IDbadpix].array.F[jj * xsize + ii] =
                    dcimg[IDbadpix1].array.F[jj * xsize + ii];
            }

        printf(" %ld bad pixels cleaned. %ld pixels left\n", fixed, left);
    }
    delete_image_ID("badpix_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("newbadpix_tmp", DELETE_IMAGE_ERRMODE_WARNING);

    free(sum_pix);
    free(pix);

    return RETURN_SUCCESS;
}

// pre-compute operations to clean bad pixels
long IMG_REDUCE_cleanbadpix_fast_precompute(const char *IDmask_name)
{
    long NBop;
    long IDbadpix;
    long xsize, ysize;
    long xysize;
    long ii, jj;
    long ii1, jj1;
    long k;
    long distmax;
    long NBnearbypix;
    long bpcnt;

    long  *nearbypix_array_index;
    float *nearbypix_array_dist2;
    float *nearbypix_array_coeff;

    float coefftot;

    printf("Pre-computing bad pixel compensation operations\n");
    fflush(stdout);

    IDbadpix = image_ID(IDmask_name, dcimg, dcnimg);
    xsize    = dcimg[IDbadpix].md[0].size[0];
    ysize    = dcimg[IDbadpix].md[0].size[1];

    xysize = xsize * ysize;

    nearbypix_array_index = (long *) malloc(sizeof(long) * xysize);
    if(nearbypix_array_index == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nearbypix_array_dist2 = (float *) malloc(sizeof(float) * xysize);
    if(nearbypix_array_dist2 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nearbypix_array_coeff = (float *) malloc(sizeof(float) * xysize);
    if(nearbypix_array_coeff == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    badpixclean_init = 1;

    badpixclean_indexlist = (long *) malloc(sizeof(long) * xysize);
    if(badpixclean_indexlist == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    k = 0;
    for(ii = 0; ii < xysize; ii++)
    {
        if(dcimg[IDbadpix].array.F[ii] > 0.5f)
        {
            badpixclean_indexlist[k] = ii;
            k++;
        }
    }

    badpixclean_NBbadpix = k;

    badpixclean_array_indexin = (long *) malloc(sizeof(long) * xysize);
    if(badpixclean_array_indexin == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    badpixclean_array_indexout = (long *) malloc(sizeof(long) * xysize);
    if(badpixclean_array_indexout == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    badpixclean_array_coeff = (float *) malloc(sizeof(float) * xysize);
    if(badpixclean_array_coeff == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    printf("Computing operations...\n");
    fflush(stdout);
    NBop  = 0;
    bpcnt = 0;

    for(ii = 0; ii < xsize; ii++)
        for(jj = 0; jj < ysize; jj++)
        {
            if(dcimg[IDbadpix].array.F[jj * xsize + ii] > 0.5f)
            {
                // fill up array of nearby pixels
                k       = 0;
                distmax = 1;
                while(k < 4)
                {
                    k        = 0;
                    coefftot = 0.0;
                    for(ii1 = ii - distmax; ii1 < ii + distmax + 1; ii1++)
                        for(jj1 = jj - distmax; jj1 < jj + distmax + 1; jj1++)
                        {
                            if((ii1 > -1) && (ii1 < xsize) && (jj1 > -1) &&
                                    (jj1 < ysize) &&
                                    (dcimg[IDbadpix]
                                     .array.F[jj1 * xsize + ii1] < 0.5f))
                            {
                                if((ii1 != ii) || (jj1 != jj))
                                {
                                    nearbypix_array_index[k] =
                                        (long)(jj1 * xsize + ii1);
                                    nearbypix_array_dist2[k] =
                                        (float)(1.0 * (ii1 - ii) * (ii1 - ii) +
                                                1.0 * (jj1 - jj) * (jj1 - jj));
                                    nearbypix_array_coeff[k] =
                                        pow(1.0 / nearbypix_array_dist2[k],
                                            2.0);
                                    coefftot += nearbypix_array_coeff[k];
                                    k++;
                                    if(k > xysize - 1)
                                    {
                                        printf(
                                            "ERROR: too many nearby pixels\n");
                                        exit(0);
                                    }
                                }
                            }
                        }
                    distmax++;
                }
                NBnearbypix = k;
                //      printf("%ld  distmax = %ld  -> k = %ld / %ld\n", bpcnt, distmax, NBnearbypix, xysize);
                //    fflush(stdout);

                if(NBnearbypix > xysize)
                {
                    printf("ERROR: NBnearbypix>xysize\n");
                    exit(0);
                }

                for(k = 0; k < NBnearbypix; k++)
                {
                    nearbypix_array_coeff[k] /= coefftot;

                    badpixclean_array_indexin[NBop]  = nearbypix_array_index[k];
                    badpixclean_array_indexout[NBop] = jj * xsize + ii;
                    badpixclean_array_coeff[NBop]    = nearbypix_array_coeff[k];
                    NBop++;

                    if(NBop > xysize - 1)
                    {
                        printf(
                            "ERROR: TOO MANY BAD PIXELS .... sorry... you need "
                            "a better detector.\n");
                        exit(0);
                    }
                }

                bpcnt++;
            }
        }
    printf("%ld bad pixels\n", bpcnt);
    printf("%ld / %ld  operations\n", NBop, xysize);

    printf("free nearbypix_array_index ...\n");
    fflush(stdout);
    free(nearbypix_array_index);

    printf("free nearbypix_array_dist2 ...\n");
    fflush(stdout);
    free(nearbypix_array_dist2);

    printf("free nearbypix_array_coeff ...\n");
    fflush(stdout);
    free(nearbypix_array_coeff);

    return (NBop);
}

imageID IMG_REDUCE_cleanbadpix_fast(const char *IDname,
                                    const char *IDbadpix_name,
                                    const char *IDoutname,
                                    int         streamMode)
{
    imageID   ID;
    uint32_t *sizearray;
    long      k;
    long      xysize, zsize;
    imageID   IDout;
    imageID   IDdark;
    long      ii, kk;
    int       naxis;

    ID = image_ID(IDname, dcimg, dcnimg);

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = dcimg[ID].md[0].size[0];
    sizearray[1] = dcimg[ID].md[0].size[1];
    naxis        = 2;
    if(dcimg[ID].md[0].naxis == 3)
    {
        sizearray[2] = dcimg[ID].md[0].size[2];
        zsize        = sizearray[2];
        naxis        = 3;
    }
    else
    {
        zsize = 1;
    }

    xysize = sizearray[0] * sizearray[1];

    IDdark = image_ID("dark", dcimg, dcnimg); // use if it exists
    list_image_ID();

    IDout = image_ID(IDoutname, dcimg, dcnimg);
    if(IDout == -1)
    {
        printf("Creating output image\n");
        fflush(stdout);
        if(streamMode == 1)
        {
            IMGID imgout_tmp =
                imgid_make_from_name(
                    IDoutname);
            imgout_tmp.mdt->naxis =
                naxis;
            for(long i = 0; i < naxis;
                i++)
            {
                imgout_tmp.mdt->size[i]
                    = sizearray[i];
            }
            imgout_tmp.mdt->datatype =
                _DATATYPE_FLOAT;
            imgout_tmp.mdt->shared = 1;
            imgout_tmp.im =
                (IMAGE *) calloc(
                    1, sizeof(IMAGE));
            imgid_mkimage(&imgout_tmp);
            IDout = imgout_tmp.ID;
        }
        else
        {
            IMGID imgout_tmp =
                imgid_make_from_name(
                    IDoutname);
            imgout_tmp.mdt->naxis =
                naxis;
            for(long i = 0; i < naxis;
                i++)
            {
                imgout_tmp.mdt->size[i]
                    = sizearray[i];
            }
            imgout_tmp.mdt->datatype =
                _DATATYPE_FLOAT;
            imgout_tmp.im =
                (IMAGE *) calloc(
                    1, sizeof(IMAGE));
            imgid_mkimage(&imgout_tmp);
            IDout = imgout_tmp.ID;
        }
    }
    if(streamMode == 1)
    {
    }

    if(badpixclean_init == 0)
    {
        badpixclean_NBop =
            IMG_REDUCE_cleanbadpix_fast_precompute(IDbadpix_name);
    }

    int OKloop = 1;
    while(OKloop == 1)
    {
        if(streamMode == 1)
        {
            printf("Waiting for incoming image ... \n");
            fflush(stdout);
            if(dcimg[ID].md[0].sem > 0)
            {
                ImageStreamIO_semwait(dcimg+ID, 0);
            }
            else
            {
                printf("NO SEMAPHORE !!!\n");
                fflush(stdout);
                exit(0);
            }
        }
        else
        {
            OKloop = 0;
        }

        SHMIM_WRITE_ACQUIRE(&dcimg[IDout].md[0]);

        memcpy(dcimg[IDout].array.F,
               dcimg[ID].array.F,
               sizeof(float) * xysize * zsize);

        for(kk = 0; kk < zsize; kk++)
        {
            if(IDdark != -1)
                for(ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.F[kk * xysize + ii] -=
                        dcimg[IDdark].array.F[ii];
                }

            for(k = 0; k < badpixclean_NBbadpix; k++)
            {
                dcimg[IDout]
                .array.F[kk * xysize + badpixclean_indexlist[k]] = 0.0f;
            }

            for(k = 0; k < badpixclean_NBop; k++)
            {
                //    printf("Operation %ld / %ld    %ld x %f -> %ld", k, badpixclean_NBop, badpixclean_array_indexin[k], badpixclean_array_coeff[k], badpixclean_array_indexout[k]);
                //   fflush(stdout);
                dcimg[IDout]
                .array.F[kk * xysize + badpixclean_array_indexout[k]] +=
                    badpixclean_array_coeff[k] *
                    dcimg[IDout]
                    .array.F[kk * xysize + badpixclean_array_indexin[k]];
                //  printf("\n");
                //  fflush(stdout);
            }
        }

        if(streamMode == 1)
        {
            if(dcimg[IDout].md[0].sem > 0)
            {
                ImageStreamIO_sempost(dcimg+IDout, 0);
            }
        }
        SHMIM_WRITE_RELEASE(&dcimg[IDout].md[0]);
        SHMIM_CNT0_INCREMENT(&dcimg[IDout].md[0]);
    }

    free(sizearray);

    return IDout;
}

errno_t IMG_REDUCE_correlMatrix(const char *IDin_name,
                                const char *IDmask_name,
                                const char *IDout_name)
{
    imageID IDin, IDout;
    imageID IDmask;
    long    xsize, ysize, zsize;
    long    xysize;
    long    ii, kk1, kk2;
    double  v, tot;
    double  tot1, tot2;

    IDin = image_ID(IDin_name, dcimg, dcnimg);

    xsize = dcimg[IDin].md[0].size[0];
    ysize = dcimg[IDin].md[0].size[1];
    zsize = dcimg[IDin].md[0].size[2];

    xysize = xsize * ysize;

    IDmask = image_ID(IDmask_name, dcimg, dcnimg);
    create_2Dimage_ID(IDout_name, zsize, zsize, &IDout);

    for(kk1 = 0; kk1 < zsize; kk1++)
        for(kk2 = kk1 + 1; kk2 < zsize; kk2++)
        {
            tot  = 0.0;
            tot1 = 0.0;
            tot2 = 0.0;

            for(ii = 0; ii < xysize; ii++)
            {
                tot1 += dcimg[IDin].array.F[kk1 * xysize + ii];
                tot2 += dcimg[IDin].array.F[kk2 * xysize + ii];
            }

            for(ii = 0; ii < xysize; ii++)
            {
                v = (dcimg[IDin].array.F[kk1 * xysize + ii] / tot1) -
                    (dcimg[IDin].array.F[kk2 * xysize + ii] / tot2);
                tot += v * v * dcimg[IDmask].array.F[ii];
            }
            dcimg[IDout].array.F[kk2 * zsize + kk1] = tot;
        }

    return RETURN_SUCCESS;
}

/** Recenter and normalize image
 *
 * if mode = 1, shared memory loop
 *
 */

imageID IMG_REDUCE_centernormim(const char *IDin_name,
                                const char *IDref_name,
                                const char *IDout_name,
                                long        xcent0,
                                long        ycent0,
                                long        xcentsize,
                                long        ycentsize,
                                int         mode,
                                int         semtrig)
{
    imageID IDin, IDout, IDref, IDtin;
    imageID IDcent, IDcentref;
    long    xsize, ysize;
    float   peak;
    double  totx, toty;
    long    ii, jj, ii0, jj0;
    long    brad;
    float   v;
    int     loopOK = 1;
    double  tot;

    int   zfactor = 4;
    long  IDcorrz;
    long  xsizez, ysizez;
    float vmin, vlim;

    float centx, centy;
    long  IDtout;

    uint32_t *imsizearray;

    IDin  = image_ID(IDin_name, dcimg, dcnimg);
    xsize = dcimg[IDin].md[0].size[0];
    ysize = dcimg[IDin].md[0].size[1];

    brad = 2;

    IDref = image_ID(IDref_name, dcimg, dcnimg);

    IDout = image_ID(IDout_name, dcimg, dcnimg);
    if(IDout == -1)
    {
        if(mode == 0)
        {
            create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);
        }
        else
        {
            imsizearray = (uint32_t *)
                malloc(
                    sizeof(uint32_t) * 2);
            if(imsizearray == NULL)
            {
                PRINT_ERROR(
                    "malloc returns "
                    "NULL pointer");
                abort();
            }

            imsizearray[0] = xsize;
            imsizearray[1] = ysize;
            {
                IMGID imgout_tmp =
                    imgid_make_from_name(
                        IDout_name);
                imgout_tmp.mdt->naxis =
                    2;
                imgout_tmp.mdt->size[0]
                    = xsize;
                imgout_tmp.mdt->size[1]
                    = ysize;
                imgout_tmp.mdt
                    ->datatype =
                    _DATATYPE_FLOAT;
                imgout_tmp.mdt->shared
                    = 1;
                imgout_tmp.mdt->NBkw
                    = 1;
                imgout_tmp.im =
                    (IMAGE *) calloc(
                        1,
                        sizeof(IMAGE));
                imgid_mkimage(
                    &imgout_tmp);
                IDout = imgout_tmp.ID;
            }
            free(imsizearray);
        }
    }

    IDcent = image_ID("_tmp_centerim", dcimg, dcnimg);
    if(IDcent == -1)
    {
        create_2Dimage_ID("_tmp_centerim", xcentsize, ycentsize, &IDcent);
    }

    IDcentref = image_ID("_tmp_centerimref", dcimg, dcnimg);
    if(IDcentref == -1)
    {
        create_2Dimage_ID("_tmp_centerimref", xcentsize, ycentsize, &IDcentref);
        // Extract centering subimage
        for(ii = 0; ii < xcentsize; ii++)
            for(jj = 0; jj < ycentsize; jj++)
            {
                ii0 = ii + xcent0;
                jj0 = jj + ycent0;
                //		totim += dcimg[IDref].array.F[jj0*xsize + ii0];
                dcimg[IDcentref].array.F[jj * xcentsize + ii] =
                    dcimg[IDref].array.F[jj0 * xsize + ii0];
            }

        /*	for(ii=0; ii<xcentsize; ii++)
        		for(jj=0; jj<ycentsize; jj++)
        			dcimg[IDcentref].array.F[jj*xcentsize+ii] -= totim*xcentsize*ycentsize;*/
    }

    while(loopOK == 1)
    {
        if(mode == 1)  // wait for semaphore trigger
        {
            COREMOD_MEMORY_image_set_semwait(IDin_name, semtrig);
        }

        // Extract centering subimage
        for(ii = 0; ii < xcentsize; ii++)
            for(jj = 0; jj < ycentsize; jj++)
            {
                ii0 = ii + xcent0;
                jj0 = jj + ycent0;
                //		totim += dcimg[IDin].array.F[jj0*xsize + ii0];
                dcimg[IDcent].array.F[jj * xcentsize + ii] =
                    dcimg[IDin].array.F[jj0 * xsize + ii0];
            }

        /** compute offset */
        fft_correlation("_tmp_centerim", "_tmp_centerimref", "outcorr");
        //IDcorr = image_ID("outcorr", dcimg, dcnimg);
        fftzoom("outcorr", "outcorrz", zfactor);
        //            save_fits("outcorr", "outcorr0.fits");

        IDcorrz = image_ID("outcorrz", dcimg, dcnimg);
        xsizez  = dcimg[IDcorrz].md[0].size[0];
        ysizez  = dcimg[IDcorrz].md[0].size[1];

        peak = 0.0;
        for(ii = 0; ii < xsizez; ii++)
            for(jj = 0; jj < ysizez; jj++)
                if(dcimg[IDcorrz].array.F[jj * xsizez + ii] > peak)
                {
                    //peakx = ii;
                    //peaky = jj;
                    peak = dcimg[IDcorrz].array.F[jj * xsizez + ii];
                }

        for(ii = 0; ii < xsizez * ysizez; ii++)
        {
            dcimg[IDcorrz].array.F[ii] /= peak;
        }

        vmin = 1.0;
        for(ii = xsizez / 2 - brad * zfactor;
                ii < xsizez / 2 + brad * zfactor + 1;
                ii++)
            for(jj = ysizez / 2 - brad * zfactor;
                    jj < ysizez / 2 + brad * zfactor + 1;
                    jj++)
            {
                v = dcimg[IDcorrz].array.F[jj * xsizez + ii];
                if(v < vmin)
                {
                    vmin = v;
                }
            }
        vlim = (vmin + 1.0) / 2.0;

        for(ii = xsizez / 2 - brad * zfactor;
                ii < xsizez / 2 + brad * zfactor + 1;
                ii++)
            for(jj = ysizez / 2 - brad * zfactor;
                    jj < ysizez / 2 + brad * zfactor + 1;
                    jj++)
            {
                float val = dcimg[IDcorrz].array.F[jj * xsizez + ii];
                val -= vlim;
                val /= (1.0f - vlim);

                if(val < 0.0f)
                {
                    val = 0.0f;
                }
                dcimg[IDcorrz].array.F[jj * xsizez + ii] = val * val;
            }

        totx = 0.0;
        toty = 0.0;
        tot  = 0.0;
        for(ii = xsizez / 2 - brad * zfactor;
                ii < xsizez / 2 + brad * zfactor + 1;
                ii++)
            for(jj = ysizez / 2 - brad * zfactor;
                    jj < ysizez / 2 + brad * zfactor + 1;
                    jj++)
            {
                v = dcimg[IDcorrz].array.F[jj * xsizez + ii];

                totx += 1.0f * (ii - xsizez * 0.5f) * v;
                toty += 1.0f * (jj - ysizez * 0.5f) * v;
                tot += v;
            }
        totx /= tot;
        toty /= tot;

        centx = totx / zfactor;
        centy = toty / zfactor;

        // save_fits("outcorr", "outcorr.fits");
        //  save_fits("outcorrz", "outcorrz.fits");
        delete_image_ID("outcorr", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("outcorrz", DELETE_IMAGE_ERRMODE_WARNING);

        printf("translating %s\n", IDin_name);
        create_2Dimage_ID("tinim", xsize, ysize, &IDtin);
        memcpy(dcimg[IDtin].array.F,
               dcimg[IDin].array.F,
               sizeof(float) * xsize * ysize);
        fft_image_translate("tinim", "_translout", -centx, -centy);
        delete_image_ID("tinim", DELETE_IMAGE_ERRMODE_WARNING);
        IDtout = image_ID("_translout", dcimg, dcnimg);
        //save_fits("_translout","_translout.fits");

        printf("zsize = %ld   vmin = %10f   offset = %+8.3f %+8.3f\n",
               brad * zfactor,
               vmin,
               centx,
               centy);

        if(mode == 0)
        {
            loopOK = 0;
        }
        else
        {
            SHMIM_WRITE_ACQUIRE(&dcimg[IDout].md[0]);
            for(ii = 0; ii < xsize; ii++)
                for(jj = 0; jj < ysize; jj++)
                {
                    dcimg[IDout].array.F[jj * xsize + ii] =
                        dcimg[IDtout].array.F[jj * xsize + ii];
                }
            SHMIM_WRITE_RELEASE(&dcimg[IDout].md[0]);
            SHMIM_CNT0_INCREMENT(&dcimg[IDout].md[0]);
            dcimg[IDout].md[0].cnt1++;
            COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        }
        delete_image_ID("_translout", DELETE_IMAGE_ERRMODE_WARNING);
    }

    return IDout;
}

/** this is the main routine to pre-process a cube stream of images (PSFs) for high contrast imaging stability analysis
 *
 *
 * Optional inputs:
 * 	calib_darkim  (single frame or cube)
 *  calib_badpix (single frame)
 *  calib_flat
 *
 *
 */

errno_t IMG_REDUCE_cubeprocess(const char *IDin_name)
{
    imageID IDin;
    long    xsize, ysize, zsize;
    long    xysize;
    long    ii, jj, kk;

    imageID IDdark;
    long    zsized;
    long    kk1;

    double *xcent;
    double *ycent;
    double  xtot, ytot, tot;
    double  boxrad;
    double  v;
    long    xmin, xmax, ymin, ymax;

    long    xsize1, ysize1, xysize1; // crop size
    imageID ID1, ID2;
    long    ii1, jj1;
    imageID IDt1, IDt2;

    double x, y, r;
    long   ID;

    long    kk2;
    long    kk1min, kk2min;
    double  vmin;
    imageID IDdiff;

    IDin = image_ID(IDin_name, dcimg, dcnimg);

    xsize = dcimg[IDin].md[0].size[0];
    ysize = dcimg[IDin].md[0].size[1];
    zsize = dcimg[IDin].md[0].size[2];

    xysize = xsize * ysize;

    /// remove dark
    if((IDdark = image_ID("calib_dark", dcimg, dcnimg)) != -1)
    {
        printf("REMOVING DARK ...");
        fflush(stdout);
        zsized = dcimg[IDdark].md[0].size[2];

        list_image_ID();
        kk1 = 0;
        for(kk = 0; kk < zsize; kk++)
        {
            for(ii = 0; ii < xysize; ii++)
            {
                dcimg[IDin].array.F[kk * xysize + ii] -=
                    dcimg[IDdark].array.F[kk1 * xysize + ii];
            }
            kk1++;
            if(kk1 == zsized)
            {
                kk1 = 0;
            }
        }
        printf(" DONE\n");
        fflush(stdout);
    }

    /// remove bad pixels
    if(image_ID("calib_badpix", dcimg, dcnimg) != -1)
    {
        printf("REMOVING BAD PIXELS ...");
        fflush(stdout);
        clean_bad_pix(IDin_name, "calib_badpix");
        save_fits(IDin_name, "out1.fits");
        list_image_ID();
    }

    /// compute photocenter
    xcent = (double *) malloc(sizeof(double) * zsize);
    if(xcent == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ycent = (double *) malloc(sizeof(double) * zsize);
    if(ycent == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(kk = 0; kk < zsize; kk++)
    {
        xtot = 0.0;
        ytot = 0.0;
        tot  = 0.0;

        xcent[kk] = 0.5 * xsize;
        ycent[kk] = 0.5 * ysize;

        boxrad = 50.0;

        xmin = (long)(xcent[kk] - boxrad);
        xmax = (long)(xcent[kk] + boxrad);

        ymin = (long)(ycent[kk] - boxrad);
        ymax = (long)(ycent[kk] + boxrad);

        if(xmin < 0)
        {
            xmin = 0;
        }
        if(xmax > xsize - 1)
        {
            xmax = xsize - 1;
        }

        if(ymin < 0)
        {
            ymin = 0;
        }
        if(ymax > ysize - 1)
        {
            ymax = ysize - 1;
        }

        //printf(" %4ld %4ld    %4ld %4ld\n", xmin, xmax, ymin, ymax);
        //fflush(stdout);

        for(ii = xmin; ii < xmax; ii++)
            for(jj = ymin; jj < ymax; jj++)
            {
                v = dcimg[IDin].array.F[kk * xysize + jj * xsize + ii];
                xtot += v * ii;
                ytot += v * jj;
                tot += v;
            }
        xcent[kk] = xtot / tot;
        ycent[kk] = ytot / tot;

        printf("%6ld   %12lf   %12lf\n", kk, xcent[kk], ycent[kk]);
        fflush(stdout);
    }

    xsize1  = 128;
    ysize1  = 128;
    xysize1 = xsize1 * ysize1;

    create_3Dimage_ID("cropPSF", xsize1, ysize1, zsize, &ID1);

    for(kk = 0; kk < zsize; kk++)
    {
        for(ii1 = 0; ii1 < xsize1; ii1++)
        {
            for(jj1 = 0; jj1 < xsize1; jj1++)
            {
                ii = ii1 - xsize1 / 2 + (long)(xcent[kk] + 0.5);
                jj = jj1 - ysize1 / 2 + (long)(ycent[kk] + 0.5);
                if((ii > -1) && (ii < xsize) && (jj > -1) && (jj < ysize))
                {
                    dcimg[ID1].array.F[kk * xysize1 + jj1 * xsize1 + ii1] =
                        dcimg[IDin].array.F[kk * xysize + jj * xsize + ii];
                }
            }
        }
    }

    create_2Dimage_ID("translin", xsize1, ysize1, &IDt1);

    for(kk = 0; kk < zsize; kk++)
    {
        xtot = 0.0;
        ytot = 0.0;
        tot  = 0.0;

        xcent[kk] = 0.5 * xsize;
        ycent[kk] = 0.5 * ysize;

        xtot = 0.0;
        ytot = 0.0;
        tot  = 0.0;

        for(ii1 = 0; ii1 < xsize1; ii1++)
            for(jj1 = 0; jj1 < ysize1; jj1++)
            {
                v = dcimg[ID1].array.F[kk * xysize1 + jj1 * xsize1 + ii1];
                xtot += v * ii1;
                ytot += v * jj1;
                tot += v;
                dcimg[IDt1].array.F[jj1 * xsize1 + ii1] = v;
            }
        xcent[kk] = xtot / tot;
        ycent[kk] = ytot / tot;

        printf("%6ld   %12lf   %12lf\n", kk, xcent[kk], ycent[kk]);

        fft_image_translate("translin",
                            "translout",
                            xcent[kk] - 0.5 * xsize1,
                            ycent[kk] - 0.5 * ysize1);
        IDt2 = image_ID("translout", dcimg, dcnimg);

        for(ii = 0; ii < xysize1; ii++)
        {
            dcimg[ID1].array.F[kk * xysize1 + ii] =
                dcimg[IDt2].array.F[ii];
        }

        delete_image_ID("translout", DELETE_IMAGE_ERRMODE_WARNING);
    }

    free(xcent);
    free(ycent);

    save_fits("cropPSF", "cropPSF.fits");

    create_2Dimage_ID("corrmask", xsize1, ysize1, &ID);
    for(ii1 = 0; ii1 < xsize1; ii1++)
        for(jj1 = 0; jj1 < ysize1; jj1++)
        {
            x = 1.0 * ii1 - 0.5 * xsize1;
            y = 1.0 * jj1 - 0.5 * ysize1;
            r = sqrt(x * x + y * y);
            if((x < 0.0) && (r > 15.0))
            {
                dcimg[ID].array.F[jj1 * xsize1 + ii1] = 1.0f;
            }
            else
            {
                dcimg[ID].array.F[jj1 * xsize1 + ii1] = 0.0f;
            }
            if((fabs(y) > 30.0) || (x < -30) || (y < 0.0))
            {
                dcimg[ID].array.F[jj1 * xsize1 + ii1] = 0.0f;
            }
        }

    save_fits("corrmask", "corrmask.fits");

    IMG_REDUCE_correlMatrix("cropPSF", "corrmask", "cropPSF_corr");
    save_fits("cropPSF_corr", "cropPSF_corr.fits");

    ID   = image_ID("cropPSF_corr", dcimg, dcnimg);
    kk1  = 0;
    kk2  = 500;
    vmin = dcimg[ID].array.F[kk2 * zsize + kk1] * 2.0f;
    for(kk1 = 0; kk1 < zsize; kk1++)
        for(kk2 = kk1 + 100; kk2 < zsize; kk2++)
        {
            if(dcimg[ID].array.F[kk2 * zsize + kk1] < vmin)
            {
                vmin   = dcimg[ID].array.F[kk2 * zsize + kk1];
                kk1min = kk1;
                kk2min = kk2;
            }
        }

    printf("MOST SIMILAR PAIR : %ld %ld   [%lf]\n", kk1min, kk2min, vmin);

    create_2Dimage_ID("imp1", xsize1, ysize1, &ID1);
    create_2Dimage_ID("imp2", xsize1, ysize1, &ID2);
    create_2Dimage_ID("imdiff", xsize1, ysize1, &IDdiff);
    ID = image_ID("cropPSF", dcimg, dcnimg);

    list_image_ID();

    for(ii = 0; ii < xysize1; ii++)
    {
        dcimg[ID1].array.F[ii] =
            dcimg[ID].array.F[kk1min * xysize1 + ii];
        dcimg[ID2].array.F[ii] =
            dcimg[ID].array.F[kk2min * xysize1 + ii];
        dcimg[IDdiff].array.F[ii] =
            dcimg[ID].array.F[kk1min * xysize1 + ii] -
            dcimg[ID].array.F[kk2min * xysize1 + ii];
    }

    save_fits("imp1", "imp1.fits");
    save_fits("imp2", "imp2.fits");
    save_fits("imdiff", "impdiff.fits");

    return RETURN_SUCCESS;
}

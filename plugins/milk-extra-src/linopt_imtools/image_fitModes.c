#include <gsl/gsl_cblas.h>

#include "CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/savefits.h"

#include "image_construct.h"
#include "image_to_vec.h"
#include "mask_to_pixtable.h"

#include "compute_SVDpseudoInverse.h"
#include "linalgebra/magma_compute_SVDpseudoInverse.h"

static int fmInit = 0;


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imfitmodes",
    .cmdkey      = "imfitmodes",
    .description = "fit image as sum of modes"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   * inimname = NULL;
static char   * modesimname = NULL;
static char   * maskimname = NULL;
static double * SVDeps = NULL;
static char   * outcoeffimname = NULL;
static int    * reuse = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".inim", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".modes", &modesimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "modes image cube") \
    X(".mask", &maskimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "mask image") \
    X(".SVDeps", &SVDeps, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "SVD cutoff") \
    X(".outimcoeff", &outcoeffimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output coeff image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}

/** @brief Decompose image as linear sum
 *
 * if reuse = 1, do not recompute pixind, pixmul, respm, recm
 */
errno_t linopt_imtools_image_fitModes(const char *ID_name,
                                      const char *IDmodes_name,
                                      const char *IDmask_name,
                                      double      SVDeps,
                                      const char *IDcoeff_name,
                                      int         reuse,
                                      imageID    *outIDcoeff)
{
    DEBUG_TRACE_FSTART();

    imageID IDrecm;
    imageID IDmvec;
    imageID IDcoeff;

    //int use_magma = 0;

    if((reuse == 0) && (fmInit == 1))
    {
        delete_image_ID("_fm_pixind", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("_fm_pixmul", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("_fm_respm", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("_fm_recm", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("_fm_vtmat", DELETE_IMAGE_ERRMODE_WARNING);
    }

    if((reuse == 0) || (fmInit == 0))
    {
        FUNC_CHECK_RETURN(linopt_imtools_mask_to_pixtable(IDmask_name,
                          "_fm_pixind",
                          "_fm_pixmul",
                          NULL));

        FUNC_CHECK_RETURN(linopt_imtools_image_to_vec(IDmodes_name,
                          "_fm_pixind",
                          "_fm_pixmul",
                          "_fm_respm",
                          NULL));

#ifdef HAVE_MAGMA
        FUNC_CHECK_RETURN(
            LINALGEBRA_magma_compute_SVDpseudoInverse("_fm_respm",
                    "_fm_recm",
                    SVDeps,
                    10000,
                    "_fm_vtmat",
                    0,
                    1,
                    64,
                    0, // GPU device
                    NULL));

#else
        FUNC_CHECK_RETURN(linopt_compute_SVDpseudoInverse("_fm_respm",
                          "_fm_recm",
                          SVDeps,
                          10000,
                          "_fm_vtmat",
                          NULL));
#endif
    }

    FUNC_CHECK_RETURN(linopt_imtools_image_to_vec(ID_name,
                      "_fm_pixind",
                      "_fm_pixmul",
                      "_fm_measvec",
                      NULL));

    IDmvec     = image_ID("_fm_measvec", data.image, data.NB_MAX_IMAGE);
    IDrecm     = image_ID("_fm_recm", data.image, data.NB_MAX_IMAGE);
    uint32_t m = data.image[IDrecm].md[0].size[1];
    uint32_t n = data.image[IDrecm].md[0].size[0];
    // printf("m=%ld n=%ld\n", m, n);
    // m = number modes
    // n = number WFS elem

    FUNC_CHECK_RETURN(create_2Dimage_ID(IDcoeff_name, m, 1, &IDcoeff));

    //printf(" -> Entering cblas_sgemv \n");
    //fflush(stdout);
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                m,
                n,
                1.0,
                data.image[IDrecm].array.F,
                n,
                data.image[IDmvec].array.F,
                1,
                0.0,
                data.image[IDcoeff].array.F,
                1);
    //printf(" -> Exiting cblas_sgemv \n");
    //fflush(stdout);

    // for(ii=0;ii<m;ii++)
    //   printf("  coeff %03ld  =  %g\n", ii, data.image[IDcoeff].array.F[ii]);

    FUNC_CHECK_RETURN(
        delete_image_ID("_fm_measvec", DELETE_IMAGE_ERRMODE_WARNING));

    if(0)  // testing
    {
        printf("========  %s  %s  %s  %lf  %s  %d  ====\n",
               ID_name,
               IDmodes_name,
               IDmask_name,
               SVDeps,
               IDcoeff_name,
               reuse);
        list_image_ID();
        save_fits("_fm_respm", "fm_respm.fits");

        linopt_imtools_image_construct(IDmodes_name,
                                       IDcoeff_name,
                                       "testsol",
                                       NULL);

        save_fits("testsol", "testsol.fits");
        arith_image_sub(ID_name, "testsol", "fitres");
        save_fits("fitres", "fitres.fits");
        arith_image_mult("fitres", IDmask_name, "fitresm");
        save_fits("fitresm", "fitresm.fits");

        FUNC_RETURN_FAILURE("testing exit");
    }

    fmInit = 1;

    if(outIDcoeff != NULL)
    {
        *outIDcoeff = IDcoeff;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_image_fitModes(inimname,
                                  modesimname,
                                  maskimname,
                                  *SVDeps,
                                  outcoeffimname,
                                  *reuse,
                                  NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_linopt_imtools__image_fitModes()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif


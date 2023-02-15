#include <gsl/gsl_cblas.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/savefits.h"

#include "image_construct.h"
#include "image_to_vec.h"
#include "mask_to_pixtable.h"

#include "compute_SVDpseudoInverse.h"
#include "cudacomp/magma_compute_SVDpseudoInverse.h"

static int fmInit = 0;

// Local variables pointers
static char   *inimname;
static char   *modesimname;
static char   *maskimname;
static double *SVDeps;
static char   *outcoeffimname;
static int    *reuse;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".inim",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_IMG,
        ".modes",
        "modes image cube",
        "imcmode",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &modesimname,
        NULL
    },
    {
        CLIARG_IMG,
        ".mask",
        "mask image",
        "immask",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &maskimname,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".SVDeps",
        "SVD cutoff",
        "0.001",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &SVDeps,
        NULL
    },
    {
        CLIARG_STR,
        ".outimcoeff",
        "output coeff image",
        "immask",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outcoeffimname,
        NULL
    },
    {
        CLIARG_INT64,
        ".reuse",
        "reuse configuration flag",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &reuse,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "imfitmodes", "fit image as sum of modes", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
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
            CUDACOMP_magma_compute_SVDpseudoInverse("_fm_respm",
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

    IDmvec     = image_ID("_fm_measvec");
    IDrecm     = image_ID("_fm_recm");
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

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__image_fitModes()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

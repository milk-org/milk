#include "ImageStreamIO/ImageStruct.h"
#include "CLIcore.h"
#include "statistic/statistic.h" // ran1, gauss, gauss_trc

#include "COREMOD_memory/image_keyword_addL.h"
#include "COREMOD_memory/image_keyword_addS.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkrnd",
    .cmdkey      = "mkrnd",
    .description = "make random image"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static LOCVAR_OUTIMG2D outim;
static uint32_t       *distrib = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".distrib", &distrib, \
      FPTYPE_UINT32, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "distribution (0:uniform 1:gauss 2:trunc)")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FARG_OUTIM_NAME(outim),
    FARG_OUTIM_SHARED(outim),
    FARG_OUTIM_XSIZE(outim),
    FARG_OUTIM_YSIZE(outim),
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


/**
 * @brief Make random image
 *
 *
 * @param[out] img
 *      Output image
 *
 * @param[in] pdf
 *      Probability distribution function
 *
 * @return imageID
 */
static imageID make_image_random(
    IMGID *img,
    int pdf
)
{
    DEBUG_TRACE_FSTART();

    // 0: uniform
    // 1: gauss
    // 2: truncated gauss

    // Create image if needed
    imcreateIMGID(img);

    // openMP is slow when calling gsl random number generator : do not use openMP here
    if(pdf == 0)
    {
        for(uint64_t ii = 0; ii < img->md->nelement; ii++)
        {
            img->im->array.F[ii] = (float) ran1();
        }
    }
    if(pdf == 1)
    {
        for(uint64_t ii = 0; ii < img->md->nelement; ii++)
        {
            img->im->array.F[ii] = (float) gauss();
        }
    }
    if(pdf == 2)
    {
        for(uint64_t ii = 0; ii < img->md->nelement; ii++)
        {
            img->im->array.F[ii] = (float) gauss_trc();
        }
    }
    if(pdf == 3)  // test pattern
    {
        static uint64_t ii   = 0;
        img->im->array.F[ii] = 1.0 - img->im->array.F[ii];
        ii++;
        if(ii == img->md->nelement)
        {
            ii = 0;
        }
    }

    DEBUG_TRACE_FEXIT();
    return (img->ID);
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("make IMGID for %s", outim.name);
    IMGID img  = imgid_make_from_name_2D(outim.name, *outim.xsize, *outim.ysize);
    img.mdt->shared = *outim.shared;
    //img.NBkw   = *outim.NBkw;
    //img.CBsize = *outim.CBsize;

    printf("NBkw   = %d\n", img.mdt->NBkw);
    printf("CBsize = %d\n", img.mdt->CBsize);


    // Create image if needed
    imcreateIMGID(&img);


    list_image_ID();

/*
    image_keyword_addS(img, "MILKFUNC", "mkrandomim", "MILK function");
    image_keyword_addL(img,
                       "RNDPDF",
                       (long)(*distrib),
                       "random value distribution");
*/
    INSERT_STD_PROCINFO_COMPUTEFUNC_START


    make_image_random(&img, *distrib);

    DEBUG_TRACEPOINT("update output ID %ld", img.ID);
    processinfo_update_output_stream(processinfo, img.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&img);

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
CLIADDCMD_image_gen__mkrandomim()
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


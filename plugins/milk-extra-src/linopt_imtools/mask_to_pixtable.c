/**
 * @file mask_to_pixtable.c
 * @brief Mask to pixtable module
 */

#include "CLIcore.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mask2pixtable",
    .cmdkey      = "mask2pixtable",
    .description = "make pixel tables from mask"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * inimname = NULL;
static char * outpixiimname = NULL;
static char * outpixmimname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".inim", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".outpixi", &outpixiimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output index image") \
    X(".outpixm", &outpixmimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output mask image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


//   Maps image to array of pixel values using mask
// to decompose image into modes:
// STEP 1: create index and mult tables (linopt_imtools_mask_to_pixtable)
//

errno_t linopt_imtools_mask_to_pixtable(const char *IDmask_name,
                                        const char *IDpixindex_name,
                                        const char *IDpixmult_name,
                                        long       *outNBpix)
{
    DEBUG_TRACE_FSTART();

    long      NBpix;
    imageID   ID;
    long      size;
    float     eps = 1.0e-8;
    long      k;
    uint32_t *sizearray;
    imageID   IDpixindex, IDpixmult;

    ID = image_ID(IDmask_name, dcimg, dcnimg);

    size = dcimg[ID].md[0].nelement;

    NBpix = 0;
    for(long ii = 0; ii < size; ii++)
        if(dcimg[ID].array.F[ii] > eps)
        {
            NBpix++;
        }

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(sizearray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }
    sizearray[0] = NBpix;
    sizearray[1] = 1;

    FUNC_CHECK_RETURN(create_image_ID(IDpixindex_name,
                                      2,
                                      sizearray,
                                      _DATATYPE_INT64,
                                      0,
                                      0,
                                      0,
                                      &IDpixindex));

    FUNC_CHECK_RETURN(create_image_ID(IDpixmult_name,
                                      2,
                                      sizearray,
                                      _DATATYPE_FLOAT,
                                      0,
                                      0,
                                      0,
                                      &IDpixmult));
    free(sizearray);

    k = 0;
    for(long ii = 0; ii < size; ii++)
        if(dcimg[ID].array.F[ii] > eps)
        {
            dcimg[IDpixindex].array.SI64[k] = ii;
            dcimg[IDpixmult].array.F[k]     = dcimg[ID].array.F[ii];
            k++;
        }

    //  printf("%ld active pixels in mask %s\n", NBpix, IDmask_name);

    if(outNBpix != NULL)
    {
        *outNBpix = NBpix;
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_mask_to_pixtable(inimname,
                                    outpixiimname,
                                    outpixmimname,
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
CLIADDCMD_linopt_imtools__mask_to_pixtable()
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


/**
 * @file image_to_vec.c
 * @brief Image to vec module
 */

#include "CLIcore.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "im2vec",
    .cmdkey      = "im2vec",
    .description = "remap image to vector"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * inimname = NULL;
static char * inpixiname = NULL;
static char * inpixmultname = NULL;
static char * outvecname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".inim", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".inpixi", &inpixiname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input pixel index image") \
    X(".inpixmult", &inpixmultname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input pixel mult image") \
    X(".outvec", &outvecname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output vector image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


//
//
//
errno_t linopt_imtools_image_to_vec(const char *__restrict ID_name,
                                    const char *__restrict IDpixindex_name,
                                    const char *__restrict IDpixmult_name,
                                    const char *__restrict IDvec_name,
                                    imageID *outID)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s %s %s %s",
                     ID_name,
                     IDpixindex_name,
                     IDpixmult_name,
                     IDvec_name);

    imageID ID;
    imageID IDpixindex, IDpixmult;
    imageID IDvec;
    long    NBpix;
    long    naxisin;
    long    sizexy;
    uint8_t datatype;


    ID = image_ID(ID_name, dcimg, dcnimg);

    naxisin  = dcimg[ID].md[0].naxis;
    datatype = dcimg[ID].md[0].datatype;


    IDpixindex = image_ID(IDpixindex_name, dcimg, dcnimg);
    IDpixmult  = image_ID(IDpixmult_name, dcimg, dcnimg);
    NBpix      = dcimg[IDpixindex].md[0].nelement;


    if(naxisin < 3)
    {
        FUNC_CHECK_RETURN(create_2Dimage_ID(IDvec_name, NBpix, 1, &IDvec));
        for(long k = 0; k < NBpix; k++)
        {
            dcimg[IDvec].array.F[k] =
                dcimg[IDpixmult].array.F[k] *
                dcimg[ID].array.F[dcimg[IDpixindex].array.SI64[k]];
        }
    }
    else
    {
        sizexy = dcimg[ID].md[0].size[0] * dcimg[ID].md[0].size[1];
        if(datatype == _DATATYPE_FLOAT)
        {
            FUNC_CHECK_RETURN(create_2Dimage_ID(IDvec_name,
                                                NBpix,
                                                dcimg[ID].md[0].size[2],
                                                &IDvec));

            for(uint32_t kk = 0; kk < dcimg[ID].md[0].size[2]; kk++)
                for(long k = 0; k < NBpix; k++)
                {
                    dcimg[IDvec].array.F[kk * NBpix + k] =
                        dcimg[IDpixmult].array.F[k] *
                        dcimg[ID]
                        .array.F[kk * sizexy +
                                    dcimg[IDpixindex].array.SI64[k]];
                }
        }
        if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            FUNC_CHECK_RETURN(create_2Dimage_ID(IDvec_name,
                                                NBpix * 2,
                                                dcimg[ID].md[0].size[2],
                                                &IDvec));

            for(uint32_t kk = 0; kk < dcimg[ID].md[0].size[2]; kk++)
                for(long k = 0; k < NBpix; k++)
                {
                    dcimg[IDvec].array.F[kk * NBpix * 2 + 2 * k] =
                        dcimg[IDpixmult].array.F[k] *
                        dcimg[ID]
                        .array
                        .CF[kk * sizexy +
                               dcimg[IDpixindex].array.SI64[k]]
                        .re;
                    dcimg[IDvec].array.F[kk * NBpix * 2 + 2 * k + 1] =
                        dcimg[IDpixmult].array.F[k] *
                        dcimg[ID]
                        .array
                        .CF[kk * sizexy +
                               dcimg[IDpixindex].array.SI64[k]]
                        .im;
                }
        }
    }

    if(outID != NULL)
    {
        *outID = IDvec;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_image_to_vec(inimname,
                                inpixiname,
                                inpixmultname,
                                outvecname,
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
CLIADDCMD_linopt_imtools__image_to_vec()
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


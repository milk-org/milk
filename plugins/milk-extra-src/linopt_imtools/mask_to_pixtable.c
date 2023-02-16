#include "CommandLineInterface/CLIcore.h"

// Local variables pointers

// Local variables pointers
static char *inimname;
static char *outpixiimname;
static char *outpixmimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".inim",
        "input image",
        "",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outpixi",
        "output index image",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outpixiimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outpixm",
        "output mask image",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outpixmimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "mask2pixtable", "make pixel tables from mask", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

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

    ID = image_ID(IDmask_name);

    size = data.image[ID].md[0].nelement;

    NBpix = 0;
    for(long ii = 0; ii < size; ii++)
        if(data.image[ID].array.F[ii] > eps)
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
        if(data.image[ID].array.F[ii] > eps)
        {
            data.image[IDpixindex].array.SI64[k] = ii;
            data.image[IDpixmult].array.F[k]     = data.image[ID].array.F[ii];
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

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__mask_to_pixtable()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

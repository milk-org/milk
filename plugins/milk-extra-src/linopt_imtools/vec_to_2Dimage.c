#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *imvecname;
static char *inpixiname;
static char *inpixmultname;
static char *outimname;
static long *xsizein;
static long *ysizein;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".inim",
        "input vector",
        "imvec",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imvecname,
        NULL
    },
    {
        CLIARG_IMG,
        ".inpixi",
        "pixel index image",
        "pixi",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inpixiname,
        NULL
    },
    {
        CLIARG_IMG,
        ".inpixmult",
        "input pixel mult image",
        "pixmult",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inpixmultname,
        NULL
    },
    {
        CLIARG_STR,
        ".outim",
        "output 2D image",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_INT64,
        ".xsize",
        "X size",
        "512",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &xsizein,
        NULL
    },
    {
        CLIARG_INT64,
        ".ysize",
        "Y size",
        "512",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ysizein,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "vec2im", "remap vector to image", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

//
//
//
errno_t linopt_imtools_vec_to_2DImage(const char *IDvec_name,
                                      const char *IDpixindex_name,
                                      const char *IDpixmult_name,
                                      const char *ID_name,
                                      long        xsize,
                                      long        ysize,
                                      imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID ID;
    imageID IDvec;
    long    k;
    imageID IDpixindex, IDpixmult;
    long    NBpix;

    IDvec      = image_ID(IDvec_name);
    IDpixindex = image_ID(IDpixindex_name);
    IDpixmult  = image_ID(IDpixmult_name);
    NBpix      = data.image[IDpixindex].md[0].nelement;

    FUNC_CHECK_RETURN(create_2Dimage_ID(ID_name, xsize, ysize, &ID));

    for(k = 0; k < NBpix; k++)
    {
        data.image[ID].array.F[data.image[IDpixindex].array.SI64[k]] =
            data.image[IDvec].array.F[k] / data.image[IDpixmult].array.F[k];
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_vec_to_2DImage(imvecname,
                                  inpixiname,
                                  inpixmultname,
                                  outimname,
                                  *xsizein,
                                  *ysizein,
                                  NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__vec_to_2DImage()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

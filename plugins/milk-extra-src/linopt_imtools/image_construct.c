
#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *modesimname;
static char *invecname;
static char *outimname;

static CLICMDARGDEF farg[] = {{
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
        ".invec",
        "input vector",
        "imvec",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &invecname,
        NULL
    },
    {
        CLIARG_STR,
        ".outim",
        "output image",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata = {"imlinconstruct",
                                "construct image as linear sum of modes",
                                CLICMD_FIELDS_DEFAULTS
                               };

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

errno_t linopt_imtools_image_construct(const char *IDmodes_name,
                                       const char *IDcoeff_name,
                                       const char *ID_name,
                                       imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID ID;
    imageID IDmodes;
    imageID IDcoeff;
    uint8_t datatype;

    IDmodes  = image_ID(IDmodes_name);
    datatype = data.image[IDmodes].md[0].datatype;

    uint32_t xsize = data.image[IDmodes].md[0].size[0];
    uint32_t ysize = data.image[IDmodes].md[0].size[1];
    uint32_t zsize = data.image[IDmodes].md[0].size[2];

    uint64_t sizexy = xsize;
    sizexy *= ysize;

    if(datatype == _DATATYPE_FLOAT)
    {
        FUNC_CHECK_RETURN(create_2Dimage_ID(ID_name, xsize, ysize, &ID));
    }
    else
    {
        FUNC_CHECK_RETURN(create_2Dimage_ID_double(ID_name, xsize, ysize, &ID));
    }

    IDcoeff = image_ID(IDcoeff_name);

    if(datatype == _DATATYPE_FLOAT)
    {
        memset(data.image[ID].array.F,
               0,
               sizeof(float) * data.image[ID].md[0].nelement);
        for(uint32_t kk = 0; kk < zsize; kk++)
            for(uint64_t ii = 0; ii < sizexy; ii++)
            {
                data.image[ID].array.F[ii] +=
                    data.image[IDcoeff].array.F[kk] *
                    data.image[IDmodes].array.F[kk * sizexy + ii];
            }
    }
    else
    {
        memset(data.image[ID].array.D,
               0,
               sizeof(double) * data.image[ID].md[0].nelement);
        for(uint32_t kk = 0; kk < zsize; kk++)
            for(uint64_t ii = 0; ii < sizexy; ii++)
            {
                data.image[ID].array.D[ii] +=
                    data.image[IDcoeff].array.D[kk] *
                    data.image[IDmodes].array.D[kk * sizexy + ii];
            }
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

    linopt_imtools_image_construct(modesimname, invecname, outimname, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__image_construct()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

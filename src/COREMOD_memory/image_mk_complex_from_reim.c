#include <math.h>

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *inreimname;
static char *inimimname;
static char *outimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".imre_name",
        "real image",
        "imre",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inreimname,
        NULL
    },
    {
        CLIARG_IMG,
        ".imim_name",
        "imaginary image",
        "imim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimimname,
        NULL
    },
    {
        CLIARG_STR,
        ".out_name",
        "output complex image",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "ri2c", "real, imaginary -> complex", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

errno_t mk_complex_from_reim(const char *re_name,
                             const char *im_name,
                             const char *out_name,
                             int         sharedmem)
{
    DEBUG_TRACE_FSTART();

    imageID   IDre;
    imageID   IDim;
    imageID   IDout;
    uint32_t *naxes = NULL;
    int8_t    naxis;
    uint8_t   datatype_re;
    uint8_t   datatype_im;
    uint8_t   datatype_out;

    IDre = image_ID(re_name);
    IDim = image_ID(im_name);

    datatype_re = data.image[IDre].md[0].datatype;
    datatype_im = data.image[IDim].md[0].datatype;
    naxis       = data.image[IDre].md[0].naxis;

    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    for(int8_t i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[IDre].md[0].size[i];
    }
    uint64_t nelement = data.image[IDre].md[0].nelement;

    if((datatype_re == _DATATYPE_FLOAT) && (datatype_im == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_FLOAT;
        FUNC_CHECK_RETURN(create_image_ID(out_name,
                                          naxis,
                                          naxes,
                                          datatype_out,
                                          sharedmem,
                                          data.NBKEYWORD_DFT,
                                          0,
                                          &IDout));
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CF[ii].re = data.image[IDre].array.F[ii];
            data.image[IDout].array.CF[ii].im = data.image[IDim].array.F[ii];
        }
    }
    else if((datatype_re == _DATATYPE_FLOAT) &&
            (datatype_im == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        FUNC_CHECK_RETURN(create_image_ID(out_name,
                                          naxis,
                                          naxes,
                                          datatype_out,
                                          sharedmem,
                                          data.NBKEYWORD_DFT,
                                          0,
                                          &IDout));
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CD[ii].re = data.image[IDre].array.F[ii];
            data.image[IDout].array.CD[ii].im = data.image[IDim].array.D[ii];
        }
    }
    else if((datatype_re == _DATATYPE_DOUBLE) &&
            (datatype_im == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        FUNC_CHECK_RETURN(create_image_ID(out_name,
                                          naxis,
                                          naxes,
                                          datatype_out,
                                          sharedmem,
                                          data.NBKEYWORD_DFT,
                                          0,
                                          &IDout));
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CD[ii].re = data.image[IDre].array.D[ii];
            data.image[IDout].array.CD[ii].im = data.image[IDim].array.F[ii];
        }
    }
    else if((datatype_re == _DATATYPE_DOUBLE) &&
            (datatype_im == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        FUNC_CHECK_RETURN(create_image_ID(out_name,
                                          naxis,
                                          naxes,
                                          datatype_out,
                                          sharedmem,
                                          data.NBKEYWORD_DFT,
                                          0,
                                          &IDout));
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CD[ii].re = data.image[IDre].array.D[ii];
            data.image[IDout].array.CD[ii].im = data.image[IDim].array.D[ii];
        }
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        abort();
    }
    // Note: openMP doesn't help here

    free(naxes);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_complex_from_reim(inreimname, inimimname, outimname, 0);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD__mk_complex_from_reim()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

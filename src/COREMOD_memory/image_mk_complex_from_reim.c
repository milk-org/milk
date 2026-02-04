#include <math.h>

#include "CLIcore.h"

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

errno_t mk_complex_from_reim_IMGID(
    IMGID *imgre,
    IMGID *imgim,
    IMGID *imgout
)
{
    DEBUG_TRACE_FSTART();

    uint8_t   datatype_re;
    uint8_t   datatype_im;
    uint8_t   datatype_out;

    resolveIMGID(imgre, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    resolveIMGID(imgim, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    datatype_re = imgre->md[0].datatype;
    datatype_im = imgim->md[0].datatype;

    imgout->mdt->naxis = imgre->md[0].naxis;
    for(int8_t i = 0; i < imgout->mdt->naxis; i++)
    {
        imgout->mdt->size[i] = imgre->md[0].size[i];
    }
    uint64_t nelement = imgre->md[0].nelement;

    if((datatype_re == _DATATYPE_FLOAT) && (datatype_im == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_FLOAT;
        imgout->mdt->datatype = datatype_out;
        createimagefromIMGID(imgout);

        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.CF[ii].re = imgre->im->array.F[ii];
            imgout->im->array.CF[ii].im = imgim->im->array.F[ii];
        }
    }
    else if((datatype_re == _DATATYPE_FLOAT) &&
            (datatype_im == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        imgout->mdt->datatype = datatype_out;
        createimagefromIMGID(imgout);

        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.CD[ii].re = imgre->im->array.F[ii];
            imgout->im->array.CD[ii].im = imgim->im->array.D[ii];
        }
    }
    else if((datatype_re == _DATATYPE_DOUBLE) &&
            (datatype_im == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        imgout->mdt->datatype = datatype_out;
        createimagefromIMGID(imgout);

        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.CD[ii].re = imgre->im->array.D[ii];
            imgout->im->array.CD[ii].im = imgim->im->array.F[ii];
        }
    }
    else if((datatype_re == _DATATYPE_DOUBLE) &&
            (datatype_im == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        imgout->mdt->datatype = datatype_out;
        createimagefromIMGID(imgout);

        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.CD[ii].re = imgre->im->array.D[ii];
            imgout->im->array.CD[ii].im = imgim->im->array.D[ii];
        }
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        abort();
    }
    // Note: openMP doesn't help here

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_complex_from_reim(const char *re_name,
                             const char *im_name,
                             const char *out_name,
                             int         sharedmem)
{
    IMGID imgre = imgid_make_from_name(re_name);
    IMGID imgim = imgid_make_from_name(im_name);
    IMGID imgout = imgid_make_from_name(out_name);
    imgout.mdt->shared = sharedmem;

    errno_t ret = mk_complex_from_reim_IMGID(&imgre, &imgim, &imgout);
    imgid_free(&imgre);
    imgid_free(&imgim);
    imgid_free(&imgout);
    return ret;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgre = imgid_make_from_name(inreimname);
    IMGID imgim = imgid_make_from_name(inimimname);
    IMGID imgout = imgid_make_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_complex_from_reim_IMGID(&imgre, &imgim, &imgout);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgre);
    imgid_free(&imgim);
    imgid_free(&imgout);

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

#include <math.h>

#include "CLIcore.h"

// Local variables pointers
static char *inimname;
static char *outreimname;
static char *outimimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".imre_name",
        "input imaginary image",
        "imC",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".imim_name",
        "output real image",
        "outre",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outreimname,
        NULL
    },
    {
        CLIARG_STR,
        ".out_name",
        "output imaginary image",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "c2ap", "complex -> re, im", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

errno_t mk_reim_from_complex_IMGID(
    IMGID *imgin,
    IMGID *imgre,
    IMGID *imgim
)
{
    DEBUG_TRACE_FSTART();

    uint8_t  datatype;

    resolveIMGID(imgin, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    datatype = imgin->md[0].datatype;
    uint8_t naxis    = imgin->md[0].naxis;
    for(int i = 0; i < naxis; i++)
    {
        imgre->mdt->size[i] = imgin->md[0].size[i];
        imgim->mdt->size[i] = imgin->md[0].size[i];
    }
    imgre->mdt->naxis = naxis;
    imgim->mdt->naxis = naxis;

    uint64_t nelement = imgin->md[0].nelement;

    if(datatype == _DATATYPE_COMPLEX_FLOAT)  // single precision
    {
        imgre->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgre);

        imgim->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgim);

        imgre->md[0].write = 1;
        imgim->md[0].write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                imgre->im->array.F[ii] = imgin->im->array.CF[ii].re;
                imgim->im->array.F[ii] = imgin->im->array.CF[ii].im;
            }
#ifdef _OPENMP
        }
#endif
        if(imgre->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgre->ID, -1);
        }
        if(imgim->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgim->ID, -1);
        }
        imgre->md[0].cnt0++;
        imgim->md[0].cnt0++;
        imgre->md[0].write = 0;
        imgim->md[0].write = 0;
    }
    else if(datatype == _DATATYPE_COMPLEX_DOUBLE)  // double precision
    {
        imgre->mdt->datatype = _DATATYPE_DOUBLE;
        createimagefromIMGID(imgre);

        imgim->mdt->datatype = _DATATYPE_DOUBLE;
        createimagefromIMGID(imgim);

        imgre->md[0].write = 1;
        imgim->md[0].write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                imgre->im->array.D[ii] = imgin->im->array.CD[ii].re;
                imgim->im->array.D[ii] = imgin->im->array.CD[ii].im;
            }
#ifdef _OPENMP
        }
#endif
        if(imgre->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgre->ID, -1);
        }
        if(imgim->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgim->ID, -1);
        }
        imgre->md[0].cnt0++;
        imgim->md[0].cnt0++;
        imgre->md[0].write = 0;
        imgim->md[0].write = 0;
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        abort();
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_reim_from_complex(const char *in_name,
                             const char *re_name,
                             const char *im_name,
                             int         sharedmem)
{
    IMGID imgin = imgid_make_from_name(in_name);
    IMGID imgre = imgid_make_from_name(re_name);
    IMGID imgim = imgid_make_from_name(im_name);
    imgre.mdt->shared = sharedmem;
    imgim.mdt->shared = sharedmem;

    errno_t ret = mk_reim_from_complex_IMGID(&imgin, &imgre, &imgim);
    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);
    return ret;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(inimname);
    IMGID imgre = imgid_make_from_name(outreimname);
    IMGID imgim = imgid_make_from_name(outimimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_reim_from_complex_IMGID(&imgin, &imgre, &imgim);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD__mk_reim_from_complex()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

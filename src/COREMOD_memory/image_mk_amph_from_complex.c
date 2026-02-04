#include <math.h>

#include "CLIcore.h"

// Local variables pointers
static char *inimname;
static char *outampimname;
static char *outphaimname;

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
        "output amplitude image",
        "outamp",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outampimname,
        NULL
    },
    {
        CLIARG_STR,
        ".out_name",
        "output phase image",
        "outpha",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outphaimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "c2ap", "complex -> ampl, pha", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

errno_t mk_amph_from_complex_IMGID(
    IMGID *imgin,
    IMGID *imgamp,
    IMGID *imgpha
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(imgin, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    uint8_t datatype = imgin->md[0].datatype;
    uint8_t naxis    = imgin->md[0].naxis;

    for(uint8_t i = 0; i < naxis; i++)
    {
        imgamp->mdt->size[i] = imgin->md[0].size[i];
        imgpha->mdt->size[i] = imgin->md[0].size[i];
    }
    imgamp->mdt->naxis = naxis;
    imgpha->mdt->naxis = naxis;

    uint64_t nelement = imgin->md[0].nelement;

    if(datatype == _DATATYPE_COMPLEX_FLOAT)  // single precision
    {
        imgamp->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgamp);

        imgpha->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgpha);

        imgamp->md[0].write = 1;
        imgpha->md[0].write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                float amp_f =
                    (float) sqrt(imgin->im->array.CF[ii].re *
                                 imgin->im->array.CF[ii].re +
                                 imgin->im->array.CF[ii].im *
                                 imgin->im->array.CF[ii].im);
                float pha_f = (float) atan2(imgin->im->array.CF[ii].im,
                                            imgin->im->array.CF[ii].re);
                imgamp->im->array.F[ii] = amp_f;
                imgpha->im->array.F[ii] = pha_f;
            }
#ifdef _OPENMP
        }
#endif
        if(imgamp->md[0].shared == 1)
        {
            FUNC_CHECK_RETURN(COREMOD_MEMORY_image_set_sempost_byID(imgamp->ID, -1));
        }
        if(imgpha->md[0].shared == 1)
        {
            FUNC_CHECK_RETURN(COREMOD_MEMORY_image_set_sempost_byID(imgpha->ID, -1));
        }
        imgamp->md[0].cnt0++;
        imgpha->md[0].cnt0++;
        imgamp->md[0].write = 0;
        imgpha->md[0].write = 0;
    }
    else if(datatype == _DATATYPE_COMPLEX_DOUBLE)  // double precision
    {
        imgamp->mdt->datatype = _DATATYPE_DOUBLE;
        createimagefromIMGID(imgamp);

        imgpha->mdt->datatype = _DATATYPE_DOUBLE;
        createimagefromIMGID(imgpha);

        imgamp->md[0].write = 1;
        imgpha->md[0].write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                double amp_d = sqrt(imgin->im->array.CD[ii].re *
                                    imgin->im->array.CD[ii].re +
                                    imgin->im->array.CD[ii].im *
                                    imgin->im->array.CD[ii].im);
                double pha_d = atan2(imgin->im->array.CD[ii].im,
                                     imgin->im->array.CD[ii].re);
                imgamp->im->array.D[ii] = amp_d;
                imgpha->im->array.D[ii] = pha_d;
            }
#ifdef _OPENMP
        }
#endif
        if(imgamp->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgamp->ID, -1);
        }
        if(imgpha->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgpha->ID, -1);
        }
        imgamp->md[0].cnt0++;
        imgpha->md[0].cnt0++;
        imgamp->md[0].write = 0;
        imgpha->md[0].write = 0;
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        exit(0);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_amph_from_complex(const char *in_name,
                             const char *am_name,
                             const char *ph_name,
                             int         sharedmem)
{
    IMGID imgin = imgid_make_from_name(in_name);
    IMGID imgamp = imgid_make_from_name(am_name);
    IMGID imgpha = imgid_make_from_name(ph_name);
    imgamp.mdt->shared = sharedmem;
    imgpha.mdt->shared = sharedmem;

    errno_t ret = mk_amph_from_complex_IMGID(&imgin, &imgamp, &imgpha);
    imgid_free(&imgin);
    imgid_free(&imgamp);
    imgid_free(&imgpha);
    return ret;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(inimname);
    IMGID imgamp = imgid_make_from_name(outampimname);
    IMGID imgpha = imgid_make_from_name(outphaimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_amph_from_complex_IMGID(&imgin, &imgamp, &imgpha);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgin);
    imgid_free(&imgamp);
    imgid_free(&imgpha);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD__mk_amph_from_complex()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

/**
 * @file    image_mk_reim_from_complex.c
 * @brief   complex -> re, im
 *
 * Uses FPS V2 framework.
 */

#include <math.h>

#include "CLIcore.h"
#include "fps.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "c2ri",
    .cmdkey      = "c2ri",
    .description = "complex -> re, im"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *inimname    = NULL;
static char *outreimname = NULL;
static char *outimimname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imre_name", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input complex image") \
    X(".imim_name", &outreimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output real image") \
    X(".out_name", &outimimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output imaginary image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t mk_reim_from_complex_IMGID(
    IMGID *imgin,
    IMGID *imgre,
    IMGID *imgim
)
{
    DEBUG_TRACE_FSTART();

    uint8_t datatype;

    resolveIMGID(
        imgin, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    datatype = imgin->md[0].datatype;
    uint8_t naxis = imgin->md[0].naxis;
    for(int i = 0; i < naxis; i++)
    {
        imgre->mdt->size[i] =
            imgin->md[0].size[i];
        imgim->mdt->size[i] =
            imgin->md[0].size[i];
    }
    imgre->mdt->naxis = naxis;
    imgim->mdt->naxis = naxis;

    uint64_t nelement = imgin->md[0].nelement;

    if(datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        imgre->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgre);

        imgim->mdt->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgim);

        imgre->md[0].write = 1;
        imgim->md[0].write = 1;
#ifdef _OPENMP
        #pragma omp parallel \
            if (nelement > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint64_t ii = 0;
                 ii < nelement; ii++)
            {
                imgre->im->array.F[ii] =
                    imgin->im->array.CF[ii].re;
                imgim->im->array.F[ii] =
                    imgin->im->array.CF[ii].im;
            }
#ifdef _OPENMP
        }
#endif
        if(imgre->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(
                imgre->ID, -1);
        }
        if(imgim->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(
                imgim->ID, -1);
        }
        imgre->md[0].cnt0++;
        imgim->md[0].cnt0++;
        imgre->md[0].write = 0;
        imgim->md[0].write = 0;
    }
    else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        imgre->mdt->datatype =
            _DATATYPE_DOUBLE;
        createimagefromIMGID(imgre);

        imgim->mdt->datatype =
            _DATATYPE_DOUBLE;
        createimagefromIMGID(imgim);

        imgre->md[0].write = 1;
        imgim->md[0].write = 1;
#ifdef _OPENMP
        #pragma omp parallel \
            if (nelement > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint64_t ii = 0;
                 ii < nelement; ii++)
            {
                imgre->im->array.D[ii] =
                    imgin->im->array.CD[ii].re;
                imgim->im->array.D[ii] =
                    imgin->im->array.CD[ii].im;
            }
#ifdef _OPENMP
        }
#endif
        if(imgre->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(
                imgre->ID, -1);
        }
        if(imgim->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(
                imgim->ID, -1);
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

errno_t mk_reim_from_complex(
    const char *in_name,
    const char *re_name,
    const char *im_name,
    int         sharedmem)
{
    IMGID imgin =
        imgid_make_from_name(in_name);
    IMGID imgre =
        imgid_make_from_name(re_name);
    IMGID imgim =
        imgid_make_from_name(im_name);
    imgre.mdt->shared = sharedmem;
    imgim.mdt->shared = sharedmem;

    errno_t ret = mk_reim_from_complex_IMGID(
        &imgin, &imgre, &imgim);
    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);
    return ret;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
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


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgin =
        imgid_make_from_name(inimname);
    IMGID imgre =
        imgid_make_from_name(outreimname);
    IMGID imgim =
        imgid_make_from_name(outimimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_reim_from_complex_IMGID(
        &imgin, &imgre, &imgim);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);

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
CLIADDCMD_COREMOD__mk_reim_from_complex()
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

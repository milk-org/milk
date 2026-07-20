// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *inimname;
static char *outreimname;
static char *outimimname;

static CLICMDARGDEF farg[] = { { CLIARG_IMG, ".imre_name", "input imaginary image", "imC",
                                 CLIARG_VISIBLE_DEFAULT, (void **) &inimname, NULL },
                               { CLIARG_STR, ".imim_name", "output real image", "outre",
                                 CLIARG_VISIBLE_DEFAULT, (void **) &outreimname, NULL },
                               { CLIARG_STR, ".out_name", "output imaginary image", "outim",
                                 CLIARG_VISIBLE_DEFAULT, (void **) &outimimname, NULL } };

static CLICMDDATA CLIcmddata = { "c2ap", "complex -> re, im", CLICMD_FIELDS_DEFAULTS };

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

errno_t mk_reim_from_complex_IMGID(IMGID *imgin, IMGID *imgre, IMGID *imgim)
{
    DEBUG_TRACE_FSTART();

    uint8_t datatype;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype      = imgin->md[0].datatype;
    uint8_t naxis = imgin->md[0].naxis;
    for (int i = 0; i < naxis; i++)
    {
        imgre->size[i] = imgin->md[0].size[i];
        imgim->size[i] = imgin->md[0].size[i];
    }
    imgre->naxis = naxis;
    imgim->naxis = naxis;

    uint64_t nelement = imgin->md[0].nelement;

    if (datatype == _DATATYPE_COMPLEX_FLOAT) // single precision
    {
        imgre->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgre);

        imgim->datatype = _DATATYPE_FLOAT;
        createimagefromIMGID(imgim);

        imgre->md[0].write = 1;
        imgim->md[0].write = 1;
#ifdef _OPENMP
#    pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                imgre->im->array.F[ii] = imgin->im->array.CF[ii].re;
                imgim->im->array.F[ii] = imgin->im->array.CF[ii].im;
            }
#ifdef _OPENMP
        }
#endif
        if (imgre->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgre->ID, -1);
        }
        if (imgim->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgim->ID, -1);
        }
        imgre->md[0].cnt0++;
        imgim->md[0].cnt0++;
        imgre->md[0].write = 0;
        imgim->md[0].write = 0;
    }
    else if (datatype == _DATATYPE_COMPLEX_DOUBLE) // double precision
    {
        imgre->datatype = _DATATYPE_DOUBLE;
        createimagefromIMGID(imgre);

        imgim->datatype = _DATATYPE_DOUBLE;
        createimagefromIMGID(imgim);

        imgre->md[0].write = 1;
        imgim->md[0].write = 1;
#ifdef _OPENMP
#    pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
#    pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                imgre->im->array.D[ii] = imgin->im->array.CD[ii].re;
                imgim->im->array.D[ii] = imgin->im->array.CD[ii].im;
            }
#ifdef _OPENMP
        }
#endif
        if (imgre->md[0].shared == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgre->ID, -1);
        }
        if (imgim->md[0].shared == 1)
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
    IMGID imgin  = mkIMGID_from_name(in_name);
    IMGID imgre  = mkIMGID_from_name(re_name);
    IMGID imgim  = mkIMGID_from_name(im_name);
    imgre.shared = sharedmem;
    imgim.shared = sharedmem;

    return mk_reim_from_complex_IMGID(&imgin, &imgre, &imgim);
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    IMGID imgin = mkIMGID_from_name(inimname);
    IMGID imgre = mkIMGID_from_name(outreimname);
    IMGID imgim = mkIMGID_from_name(outimimname);

    mk_reim_from_complex_IMGID(&imgin, &imgre, &imgim);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

    // Register function in CLI
    errno_t CLIADDCMD_COREMOD__mk_reim_from_complex()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *inampimname;
static char *inphaimname;
static char *outimname;

static CLICMDARGDEF farg[] = {{CLIARG_IMG,
                               ".imamp_name",
                               "amplitude image",
                               "imamp",
                               CLIARG_VISIBLE_DEFAULT,
                               (void **) &inampimname,
                               NULL},
                              {CLIARG_IMG,
                               ".impha_name",
                               "phase image",
                               "impha",
                               CLIARG_VISIBLE_DEFAULT,
                               (void **) &inphaimname,
                               NULL},
                              {CLIARG_STR,
                               ".out_name",
                               "output complex image",
                               "outim",
                               CLIARG_VISIBLE_DEFAULT,
                               (void **) &outimname,
                               NULL}};

static CLICMDDATA CLIcmddata = {
    "ap2c", "amplitude, phase -> complex", CLICMD_FIELDS_DEFAULTS};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

errno_t mk_complex_from_amph(const char *am_name,
                             const char *ph_name,
                             const char *out_name,
                             int         sharedmem)
{
    DEBUG_TRACE_FSTART();

    imageID  IDam;
    imageID  IDph;
    imageID  IDout;
    uint32_t naxes[3];
    long     naxis;
    uint64_t nelement;
    long     i;
    uint8_t  datatype_am;
    uint8_t  datatype_ph;
    uint8_t  datatype_out;

    IDam        = image_ID(am_name);
    IDph        = image_ID(ph_name);
    datatype_am = data.image[IDam].md[0].datatype;
    datatype_ph = data.image[IDph].md[0].datatype;

    naxis = data.image[IDam].md[0].naxis;
    for (i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[IDam].md[0].size[i];
    }
    nelement = data.image[IDam].md[0].nelement;

    if ((datatype_am == _DATATYPE_FLOAT) && (datatype_ph == _DATATYPE_FLOAT))
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

        data.image[IDout].md[0].write = 1;
#ifdef _OPENMP
#pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
#pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CF[ii].re =
                    data.image[IDam].array.F[ii] *
                    ((float) cos(data.image[IDph].array.F[ii]));
                data.image[IDout].array.CF[ii].im =
                    data.image[IDam].array.F[ii] *
                    ((float) sin(data.image[IDph].array.F[ii]));
            }
#ifdef _OPENMP
        }
#endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }
    else if ((datatype_am == _DATATYPE_FLOAT) &&
             (datatype_ph == _DATATYPE_DOUBLE))
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
        data.image[IDout].md[0].write = 1;
#ifdef _OPENMP
#pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
#pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CD[ii].re =
                    data.image[IDam].array.F[ii] *
                    cos(data.image[IDph].array.D[ii]);
                data.image[IDout].array.CD[ii].im =
                    data.image[IDam].array.F[ii] *
                    sin(data.image[IDph].array.D[ii]);
            }
#ifdef _OPENMP
        }
#endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }
    else if ((datatype_am == _DATATYPE_DOUBLE) &&
             (datatype_ph == _DATATYPE_FLOAT))
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
        data.image[IDout].md[0].write = 1;
#ifdef _OPENMP
#pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
#pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CD[ii].re =
                    data.image[IDam].array.D[ii] *
                    cos(data.image[IDph].array.F[ii]);
                data.image[IDout].array.CD[ii].im =
                    data.image[IDam].array.D[ii] *
                    sin(data.image[IDph].array.F[ii]);
            }
#ifdef _OPENMP
        }
#endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }
    else if ((datatype_am == _DATATYPE_DOUBLE) &&
             (datatype_ph == _DATATYPE_DOUBLE))
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
        data.image[IDout].md[0].write = 1;
#ifdef _OPENMP
#pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
#pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CD[ii].re =
                    data.image[IDam].array.D[ii] *
                    cos(data.image[IDph].array.D[ii]);
                data.image[IDout].array.CD[ii].im =
                    data.image[IDam].array.D[ii] *
                    sin(data.image[IDph].array.D[ii]);
            }
#ifdef _OPENMP
        }
#endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        abort();
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_complex_from_amph(inampimname, inphaimname, outimname, 0);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

    // Register function in CLI
    errno_t
    CLIADDCMD_COREMOD__mk_complex_from_amph()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

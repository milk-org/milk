#include <math.h>

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *inimname;
static char *outampimname;
static char *outphaimname;

static CLICMDARGDEF farg[] = {
    {CLIARG_IMG, ".imre_name", "input imaginary image", "imC", CLIARG_VISIBLE_DEFAULT, (void **)&inimname, NULL},
    {CLIARG_STR, ".imim_name", "output amplitude image", "outamp", CLIARG_VISIBLE_DEFAULT, (void **)&outampimname,
     NULL},
    {CLIARG_STR, ".out_name", "output phase image", "outpha", CLIARG_VISIBLE_DEFAULT, (void **)&outphaimname, NULL}};

static CLICMDDATA CLIcmddata = {"c2ap", "complex -> ampl, pha", CLICMD_FIELDS_DEFAULTS};

// detailed help
static errno_t help_function() { return RETURN_SUCCESS; }

errno_t mk_amph_from_complex(const char *in_name, const char *am_name, const char *ph_name, int sharedmem)
{
    DEBUG_TRACE_FSTART();

    imageID IDam;
    imageID IDph;
    imageID IDin;
    uint32_t naxes[3];
    uint8_t datatype;

    IDin = image_ID(in_name);
    datatype = data.image[IDin].md[0].datatype;
    uint8_t naxis = data.image[IDin].md[0].naxis;

    for (uint8_t i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[IDin].md[0].size[i];
    }
    uint64_t nelement = data.image[IDin].md[0].nelement;

    if (datatype == _DATATYPE_COMPLEX_FLOAT) // single precision
    {
        FUNC_CHECK_RETURN(
            create_image_ID(am_name, naxis, naxes, _DATATYPE_FLOAT, sharedmem, data.NBKEYWORD_DFT, 0, &IDam));

        FUNC_CHECK_RETURN(
            create_image_ID(ph_name, naxis, naxes, _DATATYPE_FLOAT, sharedmem, data.NBKEYWORD_DFT, 0, &IDph));

        data.image[IDam].md[0].write = 1;
        data.image[IDph].md[0].write = 1;
#ifdef _OPENMP
#pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT) private(ii, amp_f, pha_f)
        {
#pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                float amp_f = (float)sqrt(data.image[IDin].array.CF[ii].re * data.image[IDin].array.CF[ii].re +
                                          data.image[IDin].array.CF[ii].im * data.image[IDin].array.CF[ii].im);
                float pha_f = (float)atan2(data.image[IDin].array.CF[ii].im, data.image[IDin].array.CF[ii].re);
                data.image[IDam].array.F[ii] = amp_f;
                data.image[IDph].array.F[ii] = pha_f;
            }
#ifdef _OPENMP
        }
#endif
        if (sharedmem == 1)
        {
            FUNC_CHECK_RETURN(COREMOD_MEMORY_image_set_sempost_byID(IDam, -1));

            FUNC_CHECK_RETURN(COREMOD_MEMORY_image_set_sempost_byID(IDph, -1));
        }
        data.image[IDam].md[0].cnt0++;
        data.image[IDph].md[0].cnt0++;
        data.image[IDam].md[0].write = 0;
        data.image[IDph].md[0].write = 0;
    }
    else if (datatype == _DATATYPE_COMPLEX_DOUBLE) // double precision
    {
        FUNC_CHECK_RETURN(
            create_image_ID(am_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem, data.NBKEYWORD_DFT, 0, &IDam));

        FUNC_CHECK_RETURN(
            create_image_ID(ph_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem, data.NBKEYWORD_DFT, 0, &IDph));

        data.image[IDam].md[0].write = 1;
        data.image[IDph].md[0].write = 1;
#ifdef _OPENMP
#pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT) private(ii, amp_d, pha_d)
        {
#pragma omp for
#endif
            for (uint64_t ii = 0; ii < nelement; ii++)
            {
                double amp_d = sqrt(data.image[IDin].array.CD[ii].re * data.image[IDin].array.CD[ii].re +
                                    data.image[IDin].array.CD[ii].im * data.image[IDin].array.CD[ii].im);
                double pha_d = atan2(data.image[IDin].array.CD[ii].im, data.image[IDin].array.CD[ii].re);
                data.image[IDam].array.D[ii] = amp_d;
                data.image[IDph].array.D[ii] = pha_d;
            }
#ifdef _OPENMP
        }
#endif
        if (sharedmem == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(IDam, -1);
            COREMOD_MEMORY_image_set_sempost_byID(IDph, -1);
        }
        data.image[IDam].md[0].cnt0++;
        data.image[IDph].md[0].cnt0++;
        data.image[IDam].md[0].write = 0;
        data.image[IDph].md[0].write = 0;
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        exit(0);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_amph_from_complex(inimname, outampimname, outphaimname, 0);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

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

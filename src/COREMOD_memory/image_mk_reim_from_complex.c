#include <math.h>

#include "CommandLineInterface/CLIcore.h"


// Local variables pointers
static char *inimname;
static char *outreimname;
static char *outimimname;





static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".imre_name", "input imaginary image", "imC",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname
    },
    {
        CLIARG_STR, ".imim_name", "output real image", "outre",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outreimname
    },
    {
        CLIARG_STR, ".out_name", "output imaginary image", "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimimname
    }
};

static CLICMDDATA CLIcmddata =
{
    "c2ap",
    "complex -> re, im",
    CLICMD_FIELDS_DEFAULTS
};


// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}






errno_t mk_reim_from_complex(
    const char *in_name,
    const char *re_name,
    const char *im_name,
    int         sharedmem
)
{
    DEBUG_TRACE_FSTART();

    imageID     IDre;
    imageID     IDim;
    imageID     IDin;
    uint32_t    naxes[3];
    long        naxis;
    uint64_t        nelement;
    long        i;
    uint8_t     datatype;

    IDin = image_ID(in_name);
    datatype = data.image[IDin].md[0].datatype;
    naxis = data.image[IDin].md[0].naxis;
    for(i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[IDin].md[0].size[i];
    }
    nelement = data.image[IDin].md[0].nelement;

    if(datatype == _DATATYPE_COMPLEX_FLOAT) // single precision
    {
        FUNC_CHECK_RETURN(
            create_image_ID(re_name, naxis, naxes, _DATATYPE_FLOAT, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDre)
        );

        FUNC_CHECK_RETURN(
            create_image_ID(im_name, naxis, naxes, _DATATYPE_FLOAT, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDim)
        );

        data.image[IDre].md[0].write = 1;
        data.image[IDim].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
# endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDre].array.F[ii] = data.image[IDin].array.CF[ii].re;
                data.image[IDim].array.F[ii] = data.image[IDin].array.CF[ii].im;
            }
# ifdef _OPENMP
        }
# endif
        if(sharedmem == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(IDre, -1);
            COREMOD_MEMORY_image_set_sempost_byID(IDim, -1);
        }
        data.image[IDre].md[0].cnt0++;
        data.image[IDim].md[0].cnt0++;
        data.image[IDre].md[0].write = 0;
        data.image[IDim].md[0].write = 0;
    }
    else if(datatype == _DATATYPE_COMPLEX_DOUBLE) // double precision
    {
        FUNC_CHECK_RETURN(
            create_image_ID(re_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDre)
        );

        FUNC_CHECK_RETURN(
            create_image_ID(im_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDim)
        );
        data.image[IDre].md[0].write = 1;
        data.image[IDim].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
# endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDre].array.D[ii] = data.image[IDin].array.CD[ii].re;
                data.image[IDim].array.D[ii] = data.image[IDin].array.CD[ii].im;
            }
# ifdef _OPENMP
        }
# endif
        if(sharedmem == 1)
        {
            COREMOD_MEMORY_image_set_sempost_byID(IDre, -1);
            COREMOD_MEMORY_image_set_sempost_byID(IDim, -1);
        }
        data.image[IDre].md[0].cnt0++;
        data.image[IDim].md[0].cnt0++;
        data.image[IDre].md[0].write = 0;
        data.image[IDim].md[0].write = 0;

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

    mk_reim_from_complex(
        inimname,
        outreimname,
        outimimname,
        0
    );

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


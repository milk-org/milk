#include <math.h>

#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *inampimname;
static char *inphaimname;
static char *outimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".imamp_name",
        "amplitude image",
        "imamp",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inampimname,
        NULL
    },
    {
        CLIARG_IMG,
        ".impha_name",
        "phase image",
        "impha",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inphaimname,
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
    "ap2c", "amplitude, phase -> complex", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}






errno_t mk_complexIMG_from_amphIMG(
    IMGID imginamp,
    IMGID imginpha,
    IMGID *imgoutC
)
{
    DEBUG_TRACE_FSTART();

    uint32_t naxes[3];


    uint8_t datatype_am = imginamp.md->datatype;
    uint8_t datatype_ph = imginpha.md->datatype;

    uint8_t naxisamp = imginamp.md->naxis;
    uint8_t naxispha = imginpha.md->naxis;
    uint64_t xysize = imginamp.md->size[0];
    imgoutC->size[0] = imginamp.md->size[0];
    imgoutC->size[1] = 1;

    uint8_t naxis = naxisamp;
    if(naxisamp > 1)
    {
        xysize *= imginamp.md->size[1];
        imgoutC->size[1] = imginamp.md->size[1];
    }
    if(naxispha > naxisamp)
    {
        naxis = naxispha;
    }



    uint32_t zsize = 1;
    uint32_t zsizeamp = 1;
    uint32_t zsizepha = 1;
    if(naxisamp > 2)
    {
        zsizeamp = imginamp.md->size[2];
    }
    if(naxispha > 2)
    {
        zsizepha = imginpha.md->size[2];
    }
    zsize = zsizeamp;
    if(zsizepha > zsizeamp)
    {
        zsize = zsizepha;
    }


    imgoutC->naxis = naxis;
    imgoutC->size[2] = zsize;



    //printf("xysize = %lu\n", xysize);

    if((datatype_am == _DATATYPE_FLOAT) && (datatype_ph == _DATATYPE_FLOAT))
    {
        imgoutC->datatype = _DATATYPE_COMPLEX_FLOAT;
        createimagefromIMGID(imgoutC);

        imgoutC->md->write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (xysize > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint32_t kk=0; kk<zsize; kk++)
            {
                uint32_t kkamp = kk;
                if(kkamp > zsizeamp-1)
                {
                    kkamp = zsizeamp-1;
                }

                uint32_t kkpha = kk;
                if(kkpha > zsizepha-1)
                {
                    kkpha = zsizepha-1;
                }

                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    imgoutC->im->array.CF[kk*xysize + ii].re =
                        imginamp.im->array.F[kkamp*xysize + ii] *
                        ((float) cos(imginpha.im->array.F[kkpha*xysize + ii]));

                    imgoutC->im->array.CF[kk*xysize +ii].im =
                        imginamp.im->array.F[kkamp*xysize + ii] *
                        ((float) sin(imginpha.im->array.F[kkpha*xysize + ii]));
                }
            }
#ifdef _OPENMP
        }
#endif
        imgoutC->md->cnt0++;
        imgoutC->md->write = 0;
    }
    else if((datatype_am == _DATATYPE_FLOAT) &&
            (datatype_ph == _DATATYPE_DOUBLE))
    {
        imgoutC->datatype = _DATATYPE_COMPLEX_DOUBLE;
        createimagefromIMGID(imgoutC);

        imgoutC->md->write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (xysize > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint32_t kk=0; kk<zsize; kk++)
            {
                uint32_t kkamp = kk;
                if(kkamp > zsizeamp-1)
                {
                    kkamp = zsizeamp-1;
                }

                uint32_t kkpha = kk;
                if(kkpha > zsizepha-1)
                {
                    kkpha = zsizepha-1;
                }


                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    imgoutC->im->array.CD[kk*xysize + ii].re =
                        imginamp.im->array.F[kkamp*xysize + ii] *
                        cos(imginpha.im->array.D[kkpha*xysize + ii]);

                    imgoutC->im->array.CD[kk*xysize + ii].im =
                        imginamp.im->array.F[kkamp*xysize + ii] *
                        sin(imginpha.im->array.D[kkpha*xysize + ii]);
                }
            }
#ifdef _OPENMP
        }
#endif
        imgoutC->md->cnt0++;
        imgoutC->md->write = 0;
    }
    else if((datatype_am == _DATATYPE_DOUBLE) &&
            (datatype_ph == _DATATYPE_FLOAT))
    {
        imgoutC->datatype = _DATATYPE_COMPLEX_DOUBLE;
        createimagefromIMGID(imgoutC);

        imgoutC->md->write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (xysize > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint32_t kk=0; kk<zsize; kk++)
            {
                uint32_t kkamp = kk;
                if(kkamp > zsizeamp-1)
                {
                    kkamp = zsizeamp-1;
                }

                uint32_t kkpha = kk;
                if(kkpha > zsizepha-1)
                {
                    kkpha = zsizepha-1;
                }


                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    imgoutC->im->array.CD[kk*xysize + ii].re =
                        imginamp.im->array.D[kkamp*xysize + ii] *
                        cos(imginpha.im->array.F[kkpha*xysize + ii]);

                    imgoutC->im->array.CD[kk*xysize + ii].im =
                        imginamp.im->array.D[kkamp*xysize + ii] *
                        sin(imginpha.im->array.F[kkpha*xysize + ii]);
                }
            }
#ifdef _OPENMP
        }
#endif
        imgoutC->md->cnt0++;
        imgoutC->md->write = 0;
    }
    else if((datatype_am == _DATATYPE_DOUBLE) &&
            (datatype_ph == _DATATYPE_DOUBLE))
    {
        imgoutC->datatype = _DATATYPE_COMPLEX_DOUBLE;
        createimagefromIMGID(imgoutC);

        imgoutC->md->write = 1;
#ifdef _OPENMP
        #pragma omp parallel if (xysize > OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
#endif
            for(uint32_t kk=0; kk<zsize; kk++)
            {
                uint32_t kkamp = kk;
                if(kkamp > zsizeamp-1)
                {
                    kkamp = zsizeamp-1;
                }

                uint32_t kkpha = kk;
                if(kkpha > zsizepha-1)
                {
                    kkpha = zsizepha-1;
                }


                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    imgoutC->im->array.CD[kk*xysize + ii].re =
                        imginamp.im->array.D[kkamp*xysize + ii] *
                        cos(imginpha.im->array.D[kkpha*xysize + ii]);

                    imgoutC->im->array.CD[kk*xysize + ii].im =
                        imginamp.im->array.D[kkamp*xysize + ii] *
                        sin(imginpha.im->array.D[kkpha*xysize + ii]);
                }
            }
#ifdef _OPENMP
        }
#endif
        imgoutC->md->cnt0++;
        imgoutC->md->write = 0;
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        abort();
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





errno_t mk_complex_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *out_name,
    int         sharedmem
)
{
    IMGID imgamp = mkIMGID_from_name(am_name);
    resolveIMGID(&imgamp, ERRMODE_ABORT);

    IMGID imgpha = mkIMGID_from_name(ph_name);
    resolveIMGID(&imgpha, ERRMODE_ABORT);

    IMGID imgoutC  = mkIMGID_from_name(out_name);
    imgoutC.shared = sharedmem;

    mk_complexIMG_from_amphIMG(imgamp, imgpha, &imgoutC);

    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgamp = mkIMGID_from_name(inampimname);
    resolveIMGID(&imgamp, ERRMODE_ABORT);

    IMGID imgpha = mkIMGID_from_name(inphaimname);
    resolveIMGID(&imgpha, ERRMODE_ABORT);

    IMGID imgoutC  = mkIMGID_from_name(outimname);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        //mk_complex_from_amph(inampimname, inphaimname, outimname, 0);

        mk_complexIMG_from_amphIMG(imgamp, imgpha, &imgoutC);

    }
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

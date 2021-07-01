/**
 * @file    image_complex.c
 * @brief   complex number conversion
 */


#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "create_image.h"
#include "stream_sem.h"
#include "delete_image.h"



# ifdef _OPENMP
# include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
# endif






// ==========================================
// Forward declaration(s)
// ==========================================


errno_t mk_complex_from_reim(
    const char *re_name,
    const char *im_name,
    const char *out_name,
    int         sharedmem
);

errno_t mk_complex_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *out_name,
    int         sharedmem
);

errno_t mk_reim_from_complex(
    const char *in_name,
    const char *re_name,
    const char *im_name,
    int         sharedmem
);

errno_t mk_amph_from_complex(
    const char *in_name,
    const char *am_name,
    const char *ph_name,
    int         sharedmem
);

errno_t mk_reim_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *re_out_name,
    const char *im_out_name,
    int         sharedmem
);

errno_t mk_amph_from_reim(
    const char *re_name,
    const char *im_name,
    const char *am_out_name,
    const char *ph_out_name,
    int         sharedmem
);




// ==========================================
// Command line interface wrapper function(s)
// ==========================================


static errno_t mk_complex_from_reim__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }
    if(data.cmdargtoken[2].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[2].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_complex_from_reim(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}



static errno_t mk_complex_from_amph__cli()
{
    if(data.cmdargtoken[1].type != 4)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }
    if(data.cmdargtoken[2].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[2].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_complex_from_amph(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}



static errno_t mk_reim_from_complex__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_reim_from_complex(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}



static errno_t mk_amph_from_complex__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    mk_amph_from_complex(
        data.cmdargtoken[1].val.string,
        data.cmdargtoken[2].val.string,
        data.cmdargtoken[3].val.string,
        0);

    return CLICMD_SUCCESS;
}






// ==========================================
// Register CLI command(s)
// ==========================================

errno_t image_complex_addCLIcmd()
{
    RegisterCLIcommand(
        "ri2c",
        __FILE__,
        mk_complex_from_reim__cli,
        "real, imaginary -> complex",
        "real imaginary complex",
        "ri2c imr imi imc",
        "int mk_complex_from_reim(const char *re_name, const char *im_name, const char *out_name)");

    RegisterCLIcommand(
        "ap2c",
        __FILE__,
        mk_complex_from_amph__cli,
        "ampl, pha -> complex",
        "ampl pha complex",
        "ap2c ima imp imc",
        "int mk_complex_from_amph(const char *re_name, const char *im_name, const char *out_name, int sharedmem)");

    RegisterCLIcommand(
        "c2ri",
        __FILE__,
        mk_reim_from_complex__cli,
        "complex -> real, imaginary",
        "complex real imaginary",
        "c2ri imc imr imi",
        "int mk_reim_from_complex(const char *re_name, const char *im_name, const char *out_name)");

    RegisterCLIcommand(
        "c2ap",
        __FILE__,
        mk_amph_from_complex__cli,
        "complex -> ampl, pha",
        "complex ampl pha",
        "c2ap imc ima imp",
        "int mk_amph_from_complex(const char *re_name, const char *im_name, const char *out_name, int sharedmem)");

    return RETURN_SUCCESS;
}















errno_t mk_complex_from_reim(
    const char *re_name,
    const char *im_name,
    const char *out_name,
    int         sharedmem
)
{
    imageID     IDre;
    imageID     IDim;
    imageID     IDout;
    uint32_t   *naxes = NULL;
    long        naxis;
    long        nelement;
    long        ii;
    long        i;
    uint8_t     datatype_re;
    uint8_t     datatype_im;
    uint8_t     datatype_out;

    IDre = image_ID(re_name);
    IDim = image_ID(im_name);

    datatype_re = data.image[IDre].md[0].datatype;
    datatype_im = data.image[IDim].md[0].datatype;
    naxis = data.image[IDre].md[0].naxis;

    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc error");
        exit(0);
    }

    for(i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[IDre].md[0].size[i];
    }
    nelement = data.image[IDre].md[0].nelement;


    if((datatype_re == _DATATYPE_FLOAT) && (datatype_im == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_FLOAT;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);
        for(ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CF[ii].re = data.image[IDre].array.F[ii];
            data.image[IDout].array.CF[ii].im = data.image[IDim].array.F[ii];
        }
    }
    else if((datatype_re == _DATATYPE_FLOAT) && (datatype_im == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);
        for(ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CD[ii].re = data.image[IDre].array.F[ii];
            data.image[IDout].array.CD[ii].im = data.image[IDim].array.D[ii];
        }
    }
    else if((datatype_re == _DATATYPE_DOUBLE) && (datatype_im == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);
        for(ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CD[ii].re = data.image[IDre].array.D[ii];
            data.image[IDout].array.CD[ii].im = data.image[IDim].array.F[ii];
        }
    }
    else if((datatype_re == _DATATYPE_DOUBLE) && (datatype_im == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);
        for(ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CD[ii].re = data.image[IDre].array.D[ii];
            data.image[IDout].array.CD[ii].im = data.image[IDim].array.D[ii];
        }
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        exit(0);
    }
    // Note: openMP doesn't help here

    free(naxes);

    return RETURN_SUCCESS;
}





errno_t mk_complex_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *out_name,
    int         sharedmem
)
{
    imageID    IDam;
    imageID    IDph;
    imageID    IDout;
    uint32_t   naxes[3];
    long       naxis;
    uint64_t   nelement;
    long       i;
    uint8_t    datatype_am;
    uint8_t    datatype_ph;
    uint8_t    datatype_out;

    IDam = image_ID(am_name);
    IDph = image_ID(ph_name);
    datatype_am = data.image[IDam].md[0].datatype;
    datatype_ph = data.image[IDph].md[0].datatype;

    naxis = data.image[IDam].md[0].naxis;
    for(i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[IDam].md[0].size[i];
    }
    nelement = data.image[IDam].md[0].nelement;

    if((datatype_am == _DATATYPE_FLOAT) && (datatype_ph == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_FLOAT;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);

        data.image[IDout].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
# endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CF[ii].re = data.image[IDam].array.F[ii] * ((float) cos(
                                                        data.image[IDph].array.F[ii]));
                data.image[IDout].array.CF[ii].im = data.image[IDam].array.F[ii] * ((float) sin(
                                                        data.image[IDph].array.F[ii]));
            }
# ifdef _OPENMP
        }
# endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;

    }
    else if((datatype_am == _DATATYPE_FLOAT) && (datatype_ph == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);
        data.image[IDout].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
# endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CD[ii].re = data.image[IDam].array.F[ii] * cos(
                                                        data.image[IDph].array.D[ii]);
                data.image[IDout].array.CD[ii].im = data.image[IDam].array.F[ii] * sin(
                                                        data.image[IDph].array.D[ii]);
            }
# ifdef _OPENMP
        }
# endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }
    else if((datatype_am == _DATATYPE_DOUBLE) && (datatype_ph == _DATATYPE_FLOAT))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);
        data.image[IDout].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
# endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CD[ii].re = data.image[IDam].array.D[ii] * cos(
                                                        data.image[IDph].array.F[ii]);
                data.image[IDout].array.CD[ii].im = data.image[IDam].array.D[ii] * sin(
                                                        data.image[IDph].array.F[ii]);
            }
# ifdef _OPENMP
        }
# endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;

    }
    else if((datatype_am == _DATATYPE_DOUBLE) && (datatype_ph == _DATATYPE_DOUBLE))
    {
        datatype_out = _DATATYPE_COMPLEX_DOUBLE;
        create_image_ID(out_name, naxis, naxes, datatype_out, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDout);
        data.image[IDout].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT)
        {
            #pragma omp for
# endif
            for(uint64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.CD[ii].re = data.image[IDam].array.D[ii] * cos(
                                                        data.image[IDph].array.D[ii]);
                data.image[IDout].array.CD[ii].im = data.image[IDam].array.D[ii] * sin(
                                                        data.image[IDph].array.D[ii]);
            }
# ifdef _OPENMP
        }
# endif
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        exit(0);
    }

    return RETURN_SUCCESS;
}




errno_t mk_reim_from_complex(
    const char *in_name,
    const char *re_name,
    const char *im_name,
    int         sharedmem
)
{
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
        create_image_ID(re_name, naxis, naxes, _DATATYPE_FLOAT, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDre);
        create_image_ID(im_name, naxis, naxes, _DATATYPE_FLOAT, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDim);

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
        create_image_ID(re_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDre);
        create_image_ID(im_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem,
                        data.NBKEYWORD_DFT, 0, &IDim);
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
        exit(0);
    }


    return RETURN_SUCCESS;
}




errno_t mk_amph_from_complex(
    const char *in_name,
    const char *am_name,
    const char *ph_name,
    int         sharedmem
)
{
    DEBUG_TRACE_FSTART();

    imageID    IDam;
    imageID    IDph;
    imageID    IDin;
    uint32_t   naxes[3];
    long       naxis;
    uint64_t       nelement;
    uint64_t   ii;
    long       i;
    float      amp_f;
    float      pha_f;
    double     amp_d;
    double     pha_d;
    uint8_t    datatype;

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
            create_image_ID(am_name, naxis, naxes,  _DATATYPE_FLOAT, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDam)
        );

        FUNC_CHECK_RETURN(
            create_image_ID(ph_name, naxis, naxes,  _DATATYPE_FLOAT, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDph)
        );

        data.image[IDam].md[0].write = 1;
        data.image[IDph].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT) private(ii,amp_f,pha_f)
        {
            #pragma omp for
# endif
            for(ii = 0; ii < nelement; ii++)
            {
                amp_f = (float) sqrt(data.image[IDin].array.CF[ii].re *
                                     data.image[IDin].array.CF[ii].re + data.image[IDin].array.CF[ii].im *
                                     data.image[IDin].array.CF[ii].im);
                pha_f = (float) atan2(data.image[IDin].array.CF[ii].im,
                                      data.image[IDin].array.CF[ii].re);
                data.image[IDam].array.F[ii] = amp_f;
                data.image[IDph].array.F[ii] = pha_f;
            }
# ifdef _OPENMP
        }
# endif
        if(sharedmem == 1)
        {
            FUNC_CHECK_RETURN(
                COREMOD_MEMORY_image_set_sempost_byID(IDam, -1)
            );

            FUNC_CHECK_RETURN(
                COREMOD_MEMORY_image_set_sempost_byID(IDph, -1)
            );
        }
        data.image[IDam].md[0].cnt0++;
        data.image[IDph].md[0].cnt0++;
        data.image[IDam].md[0].write = 0;
        data.image[IDph].md[0].write = 0;
    }
    else if(datatype == _DATATYPE_COMPLEX_DOUBLE) // double precision
    {
        FUNC_CHECK_RETURN(
            create_image_ID(am_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDam)
        );

        FUNC_CHECK_RETURN(
            create_image_ID(ph_name, naxis, naxes, _DATATYPE_DOUBLE, sharedmem,
                            data.NBKEYWORD_DFT, 0, &IDph)
        );

        data.image[IDam].md[0].write = 1;
        data.image[IDph].md[0].write = 1;
# ifdef _OPENMP
        #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT) private(ii,amp_d,pha_d)
        {
            #pragma omp for
# endif
            for(ii = 0; ii < nelement; ii++)
            {
                amp_d = sqrt(data.image[IDin].array.CD[ii].re * data.image[IDin].array.CD[ii].re
                             + data.image[IDin].array.CD[ii].im * data.image[IDin].array.CD[ii].im);
                pha_d = atan2(data.image[IDin].array.CD[ii].im,
                              data.image[IDin].array.CD[ii].re);
                data.image[IDam].array.D[ii] = amp_d;
                data.image[IDph].array.D[ii] = pha_d;
            }
# ifdef _OPENMP
        }
# endif
        if(sharedmem == 1)
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




errno_t mk_reim_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *re_out_name,
    const char *im_out_name,
    int         sharedmem
)
{
    mk_complex_from_amph(am_name, ph_name, "Ctmp", 0);
    mk_reim_from_complex("Ctmp", re_out_name, im_out_name, sharedmem);
    delete_image_ID("Ctmp", DELETE_IMAGE_ERRMODE_WARNING);

    return RETURN_SUCCESS;
}



errno_t mk_amph_from_reim(
    const char *re_name,
    const char *im_name,
    const char *am_out_name,
    const char *ph_out_name,
    int         sharedmem
)
{
    mk_complex_from_reim(re_name, im_name, "Ctmp", 0);
    mk_amph_from_complex("Ctmp", am_out_name, ph_out_name, sharedmem);
    delete_image_ID("Ctmp", DELETE_IMAGE_ERRMODE_WARNING);

    return RETURN_SUCCESS;
}





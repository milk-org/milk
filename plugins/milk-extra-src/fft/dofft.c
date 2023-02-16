/** @file dofft.c
 */

#include <fftw3.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "wisdom.h"

#define FFTWOPTMODE FFTW_ESTIMATE

// ==========================================
// Forward declaration(s)
// ==========================================

imageID do1dfft(const char *in_name, const char *out_name);

imageID do1drfft(const char *in_name, const char *out_name);

imageID do2dfft(const char *in_name, const char *out_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t fft_do1dfft_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) == 0)
    {
        do1dfft(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t fft_do1drfft_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) == 0)
    {
        do1drfft(data.cmdargtoken[1].val.string,
                 data.cmdargtoken[2].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t fft_do2dfft_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) == 0)
    {
        do2dfft(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t dofft_addCLIcmd()
{
    RegisterCLIcommand(
        "dofft",
        __FILE__,
        fft_do2dfft_cli,
        "perform FFT",
        "<input> <output>",
        "fofft in out",
        "int do2dfft(const char *in_name, const char *out_name)");

    RegisterCLIcommand(
        "do1Dfft",
        __FILE__,
        fft_do1dfft_cli,
        "perform 1D complex->complex FFT",
        "<input> <output>",
        "do1Dfft in out",
        "int do1dfft(const char *in_name, const char *out_name)");

    RegisterCLIcommand(
        "do1Drfft",
        __FILE__,
        fft_do1drfft_cli,
        "perform 1D real->complex FFT",
        "<input> <output>",
        "do1drfft in out",
        "int do1drfft(const char *in_name, const char *out_name)");

    return RETURN_SUCCESS;
}

int array_index(long size)
{
    int i;

    switch(size)
    {
        case 1:
            i = 0;
            break;
        case 2:
            i = 1;
            break;
        case 4:
            i = 2;
            break;
        case 8:
            i = 3;
            break;
        case 16:
            i = 4;
            break;
        case 32:
            i = 5;
            break;
        case 64:
            i = 6;
            break;
        case 128:
            i = 7;
            break;
        case 256:
            i = 8;
            break;
        case 512:
            i = 9;
            break;
        case 1024:
            i = 10;
            break;
        case 2048:
            i = 11;
            break;
        case 4096:
            i = 12;
            break;
        case 8192:
            i = 13;
            break;
        case 16384:
            i = 14;
            break;
        default:
            i = 100;
    }

    return (i);
}

/* 1d complex -> complex fft */
// supports single and double precisions
//
imageID FFT_do1dfft(const char *__restrict in_name,
                    const char *__restrict out_name,
                    int dir)
{
    int           *naxes;
    uint32_t      *naxesl;
    long           naxis;
    imageID        IDin, IDout;
    long           i;
    int            OK = 0;
    fftwf_plan     plan;
    fftw_plan      plan_double;
    long           jj;
    fftwf_complex *inptr, *outptr;
    fftw_complex  *inptr_double, *outptr_double;
    uint8_t        datatype;

    IDin  = image_ID(in_name);
    naxis = data.image[IDin].md[0].naxis;

    naxes = (int *) malloc(naxis * sizeof(int));
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    naxesl = (uint32_t *) malloc(naxis * sizeof(uint32_t));
    if(naxesl == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(i = 0; i < naxis; i++)
    {
        naxesl[i] = data.image[IDin].md[0].size[i];
        naxes[i]  = (int) data.image[IDin].md[0].size[i];
    }
    datatype = data.image[IDin].md[0].datatype;
    create_image_ID(out_name,
                    naxis,
                    naxesl,
                    datatype,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);

    if(naxis == 1)
    {
        if(array_index(naxes[0]) != 100)
        {
            OK = 1;
            if(datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                plan = fftwf_plan_dft_1d(
                           naxes[0],
                           (fftwf_complex *) data.image[IDin].array.CF,
                           (fftwf_complex *) data.image[IDout].array.CF,
                           dir,
                           FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double = fftw_plan_dft_1d(
                                  naxes[0],
                                  (fftw_complex *) data.image[IDin].array.CD,
                                  (fftw_complex *) data.image[IDout].array.CD,
                                  dir,
                                  FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
        else
        {
            OK = 1;
            if(datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                plan = fftwf_plan_dft_1d(
                           naxes[0],
                           (fftwf_complex *) data.image[IDin].array.CF,
                           (fftwf_complex *) data.image[IDout].array.CF,
                           dir,
                           FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double = fftw_plan_dft_1d(
                                  naxes[0],
                                  (fftw_complex *) data.image[IDin].array.CD,
                                  (fftw_complex *) data.image[IDout].array.CD,
                                  dir,
                                  FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
    }

    if(naxis == 2)
    {
        if((naxes[1] == 1) && (array_index(naxes[0]) != 100))
        {
            OK = 1;
            if(datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                inptr  = (fftwf_complex *) data.image[IDin].array.CF;
                outptr = (fftwf_complex *) data.image[IDout].array.CF;
                plan   = fftwf_plan_dft_1d(naxes[0],
                                           inptr,
                                           outptr,
                                           dir,
                                           FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                inptr_double  = (fftw_complex *) data.image[IDin].array.CD;
                outptr_double = (fftw_complex *) data.image[IDout].array.CD;
                plan_double   = fftw_plan_dft_1d(naxes[0],
                                                 inptr_double,
                                                 outptr_double,
                                                 dir,
                                                 FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
        else
        {
            OK = 1;
            if(datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                inptr =
                    (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[0]);
                if(inptr == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                outptr =
                    (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[0]);
                if(outptr == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                plan = fftwf_plan_dft_1d(naxes[0],
                                         inptr,
                                         outptr,
                                         dir,
                                         FFTWOPTMODE);

                for(jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr,
                           (char *) data.image[IDin].array.CF +
                           sizeof(fftwf_complex) * jj * naxes[0],
                           sizeof(fftwf_complex) * naxes[0]);
                    fftwf_execute(plan);
                    memcpy((char *) data.image[IDout].array.CF +
                           sizeof(complex_float) * jj * naxes[0],
                           outptr,
                           sizeof(fftwf_complex) * naxes[0]);
                }
                fftwf_destroy_plan(plan);
                free(inptr);
                free(outptr);
            }
            else
            {
                inptr_double =
                    (fftw_complex *) malloc(sizeof(fftw_complex) * naxes[0]);
                if(inptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                outptr_double =
                    (fftw_complex *) malloc(sizeof(fftw_complex) * naxes[0]);
                if(outptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                plan_double = fftw_plan_dft_1d(naxes[0],
                                               inptr_double,
                                               outptr_double,
                                               dir,
                                               FFTWOPTMODE);

                for(jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr_double,
                           (char *) data.image[IDin].array.CD +
                           sizeof(fftw_complex) * jj * naxes[0],
                           sizeof(fftw_complex) * naxes[0]);
                    fftw_execute(plan_double);
                    memcpy((char *) data.image[IDout].array.CD +
                           sizeof(complex_double) * jj * naxes[0],
                           outptr_double,
                           sizeof(fftw_complex) * naxes[0]);
                }
                fftw_destroy_plan(plan_double);
                free(inptr_double);
                free(outptr_double);
            }
        }
    }

    if(OK == 0)
    {
        printf("Error : image dimension not appropriate for FFT\n");
    }
    free(naxes);
    free(naxesl);

    return (IDout);
}

/* 1d real -> complex fft */
// supports single and double precision
imageID do1drfft(const char *__restrict in_name,
                 const char *__restrict out_name)
{
    int           *naxes;
    uint32_t      *naxesl;
    uint32_t      *naxesout;
    long           naxis;
    imageID        IDin;
    imageID        IDout;
    long           i;
    int            OK = 0;
    long           jj;
    fftwf_plan     plan;
    fftw_plan      plan_double;
    fftwf_complex *outptr;
    fftw_complex  *outptr_double;
    float         *inptr;
    double        *inptr_double;
    uint8_t        datatype;

    IDin  = image_ID(in_name);
    naxis = data.image[IDin].md[0].naxis;

    naxes = (int *) malloc(naxis * sizeof(int));
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    naxesl = (uint32_t *) malloc(naxis * sizeof(uint32_t));
    if(naxesl == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    naxesout = (uint32_t *) malloc(naxis * sizeof(uint32_t));
    if(naxesout == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    datatype = data.image[IDin].md[0].datatype;

    int fftaxis = 0;
    if(naxis == 3)
    {
        fftaxis = 2;
    }


    for(i = 0; i < naxis; i++)
    {
        naxesl[i]   = data.image[IDin].md[0].size[i];
        naxes[i]    = (int) data.image[IDin].md[0].size[i];
        naxesout[i] = data.image[IDin].md[0].size[i];
        if(i == fftaxis)
        {
            naxesout[i] = data.image[IDin].md[0].size[i] / 2 + 1;
        }
    }

    if(datatype == _DATATYPE_DOUBLE)
    {
        create_image_ID(out_name,
                        naxis,
                        naxesout,
                        _DATATYPE_COMPLEX_DOUBLE,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &IDout);
    }
    else
    {
        create_image_ID(out_name,
                        naxis,
                        naxesout,
                        _DATATYPE_COMPLEX_FLOAT,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &IDout);
    }

    if(naxis == 2)
    {
        if((naxes[1] == 1) && (array_index(naxes[0]) != 100))
        {
            OK = 1;
            if(datatype == _DATATYPE_FLOAT)
            {
                plan = fftwf_plan_dft_r2c_1d(
                           naxes[0],
                           data.image[IDin].array.F,
                           (fftwf_complex *) data.image[IDout].array.CF,
                           FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double = fftw_plan_dft_r2c_1d(
                                  naxes[0],
                                  data.image[IDin].array.D,
                                  (fftw_complex *) data.image[IDout].array.CD,
                                  FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
        else
        {
            OK = 1;
            if(datatype == _DATATYPE_FLOAT)
            {
                inptr = (float *) malloc(sizeof(float) * naxes[0]);
                if(inptr == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                outptr =
                    (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[0]);
                if(outptr == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                plan =
                    fftwf_plan_dft_r2c_1d(naxes[0], inptr, outptr, FFTWOPTMODE);

                for(jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr,
                           (char *) data.image[IDin].array.F +
                           sizeof(float) * jj * naxes[0],
                           sizeof(float) * naxes[0]);
                    fftwf_execute(plan);
                    memcpy((char *) data.image[IDout].array.CF +
                           sizeof(complex_float) * jj * naxesout[0],
                           outptr,
                           sizeof(fftwf_complex) * naxesout[0]);
                }
                fftwf_destroy_plan(plan);
                free(inptr);
                free(outptr);
            }
            else
            {
                inptr_double = (double *) malloc(sizeof(double) * naxes[0]);
                if(inptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                outptr_double =
                    (fftw_complex *) malloc(sizeof(fftw_complex) * naxes[0]);
                if(outptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                plan_double = fftw_plan_dft_r2c_1d(naxes[0],
                                                   inptr_double,
                                                   outptr_double,
                                                   FFTWOPTMODE);

                for(jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr_double,
                           (char *) data.image[IDin].array.D +
                           sizeof(double) * jj * naxes[0],
                           sizeof(double) * naxes[0]);
                    fftw_execute(plan_double);
                    memcpy((char *) data.image[IDout].array.CD +
                           sizeof(complex_double) * jj * naxesout[0],
                           outptr_double,
                           sizeof(fftw_complex) * naxesout[0]);
                }
                fftw_destroy_plan(plan_double);
                free(inptr_double);
                free(outptr_double);
            }
        }
    }
    if(naxis == 3)
    {
        // perform 1D FFT along last dimension
        OK = 1;

        if(datatype == _DATATYPE_FLOAT)
        {
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);
            uint64_t xysize = naxes[0];
            xysize *= naxes[1];

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for(uint32_t ii = 0; ii < xysize; ii++)
            {
                for(int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] = data.image[IDin].array.F[i * xysize + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0; i < naxesout[2]; i++)
                {
                    data.image[IDout].array.CF[i * xysize + ii].re =
                        outptr[i][0];
                    data.image[IDout].array.CF[i * xysize + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if(datatype == _DATATYPE_UINT16)
        {
            printf("UINT16 data type\n");
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);
            uint64_t xysize = naxes[0];
            xysize *= naxes[1];

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for(uint32_t ii = 0; ii < xysize; ii++)
            {
                for(int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] =
                        1.0 * data.image[IDin].array.UI16[i * xysize + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0; i < naxesout[2]; i++)
                {
                    data.image[IDout].array.CF[i * xysize + ii].re =
                        outptr[i][0];
                    data.image[IDout].array.CF[i * xysize + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if(datatype == _DATATYPE_UINT32)
        {
            printf("UINT32 data type\n");
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);
            uint64_t xysize = naxes[0];
            xysize *= naxes[1];

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for(uint32_t ii = 0; ii < xysize; ii++)
            {
                for(int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] =
                        1.0 * data.image[IDin].array.UI32[i * xysize + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0; i < naxesout[2]; i++)
                {
                    data.image[IDout].array.CF[i * xysize + ii].re =
                        outptr[i][0];
                    data.image[IDout].array.CF[i * xysize + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if(datatype == _DATATYPE_UINT64)
        {
            printf("UINT64 data type\n");
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);
            uint64_t xysize = naxes[0];
            xysize *= naxes[1];

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for(uint32_t ii = 0; ii < xysize; ii++)
            {
                for(int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] =
                        1.0 * data.image[IDin].array.UI64[i * xysize + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0; i < naxesout[2]; i++)
                {
                    data.image[IDout].array.CF[i * xysize + ii].re =
                        outptr[i][0];
                    data.image[IDout].array.CF[i * xysize + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }
    }




    if(OK == 0)
    {
        printf("Error : image dimension not appropriate for FFT\n");
    }
    free(naxes);
    free(naxesl);
    free(naxesout);

    return (IDout);
}

imageID do1dfft(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do1dfft(in_name, out_name, -1);

    return (IDout);
}

imageID do1dffti(const char *__restrict in_name,
                 const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do1dfft(in_name, out_name, 1);

    return (IDout);
}

/* 2d complex fft */
// supports single and double precisions
imageID FFT_do2dfft(const char *in_name, const char *out_name, int dir)
{
    int       *naxes;
    uint32_t  *naxesl;
    long       naxis;
    imageID    IDin;
    imageID    IDout;
    long       i;
    int        OK = 0;
    fftwf_plan plan;
    fftw_plan  plan_double;
    long       tmp1;

    char    ffttmpcpyname[STRINGMAXLEN_IMGNAME];
    uint8_t datatype;

    IDin  = image_ID(in_name);
    naxis = data.image[IDin].md[0].naxis;

    naxes = (int *) malloc(naxis * sizeof(int));
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    naxesl = (uint32_t *) malloc(naxis * sizeof(uint32_t));
    if(naxesl == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(i = 0; i < naxis; i++)
    {
        naxesl[i] = (long) data.image[IDin].md[0].size[i];
        naxes[i]  = (int) data.image[IDin].md[0].size[i];
    }

    datatype = data.image[IDin].md[0].datatype;
    create_image_ID(out_name,
                    naxis,
                    naxesl,
                    datatype,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);

    // need to swap first 2 axis for fftw
    if(naxis > 1)
    {
        tmp1     = naxes[0];
        naxes[0] = naxes[1];
        naxes[1] = tmp1;
    }

    if(naxis == 2)
    {
        OK = 1;

        if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            plan =
                fftwf_plan_dft_2d(naxes[0],
                                  naxes[1],
                                  (fftwf_complex *) data.image[IDin].array.CF,
                                  (fftwf_complex *) data.image[IDout].array.CF,
                                  dir,
                                  FFTWOPTMODE);
            if(plan == NULL)
            {
                //	  if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2dfft %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname,
                                "_ffttmpcpyname_%d",
                                (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_dft_2d(
                           naxes[0],
                           naxes[1],
                           (fftwf_complex *) data.image[IDin].array.CF,
                           (fftwf_complex *) data.image[IDout].array.CF,
                           dir,
                           FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        else
        {
            plan_double =
                fftw_plan_dft_2d(naxes[0],
                                 naxes[1],
                                 (fftw_complex *) data.image[IDin].array.CD,
                                 (fftw_complex *) data.image[IDout].array.CD,
                                 dir,
                                 FFTWOPTMODE);
            if(plan_double == NULL)
            {
                //	  if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2dfft %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname,
                                "_ffttmpcpyname_%d",
                                (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_dft_2d(
                                  naxes[0],
                                  naxes[1],
                                  (fftw_complex *) data.image[IDin].array.CD,
                                  (fftw_complex *) data.image[IDout].array.CD,
                                  dir,
                                  FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);
        }
    }

    if(naxis == 3)
    {
        OK = 1;
        if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            plan = fftwf_plan_many_dft(
                       2,
                       naxes,
                       naxes[2],
                       (fftwf_complex *) data.image[IDin].array.CF,
                       NULL,
                       1,
                       naxes[0] * naxes[1],
                       (fftwf_complex *) data.image[IDout].array.CF,
                       NULL,
                       1,
                       naxes[0] * naxes[1],
                       dir,
                       FFTWOPTMODE);
            if(plan == NULL)
            {
                //if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2dfft %d x %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0],
                        naxes[2]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname,
                                "_ffttmpcpyname_%d",
                                (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_many_dft(
                           2,
                           naxes,
                           naxes[2],
                           (fftwf_complex *) data.image[IDin].array.CF,
                           NULL,
                           1,
                           naxes[0] * naxes[1],
                           (fftwf_complex *) data.image[IDout].array.CF,
                           NULL,
                           1,
                           naxes[0] * naxes[1],
                           dir,
                           FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        else
        {
            plan_double =
                fftw_plan_many_dft(2,
                                   naxes,
                                   naxes[2],
                                   (fftw_complex *) data.image[IDin].array.CD,
                                   NULL,
                                   1,
                                   naxes[0] * naxes[1],
                                   (fftw_complex *) data.image[IDout].array.CD,
                                   NULL,
                                   1,
                                   naxes[0] * naxes[1],
                                   dir,
                                   FFTWOPTMODE);
            if(plan_double == NULL)
            {
                //if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2dfft %d x %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0],
                        naxes[2]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname,
                                "_ffttmpcpyname_%d",
                                (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_many_dft(
                                  2,
                                  naxes,
                                  naxes[2],
                                  (fftw_complex *) data.image[IDin].array.CD,
                                  NULL,
                                  1,
                                  naxes[0] * naxes[1],
                                  (fftw_complex *) data.image[IDout].array.CD,
                                  NULL,
                                  1,
                                  naxes[0] * naxes[1],
                                  dir,
                                  FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);
        }
    }

    if(OK == 0)
    {
        printf("Error : image dimension not appropriate for FFT\n");
    }

    free(naxes);

    return (IDout);
}

imageID do2dfft(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do2dfft(in_name, out_name, -1);

    return (IDout);
}

imageID do2dffti(const char *__restrict in_name,
                 const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do2dfft(in_name, out_name, 1);

    return (IDout);
}

/* real fft : real to complex */
// supports single and double precisions
imageID FFT_do2drfft(const char *__restrict in_name,
                     const char *__restrict out_name,
                     int dir)
{
    int      *naxes; // int format for fftw
    uint32_t *naxesl;
    uint32_t *naxestmp;

    long    naxis;
    imageID IDin;
    imageID IDout;
    imageID IDtmp;

    int        OK = 0;
    fftwf_plan plan;
    fftw_plan  plan_double;
    long       tmp1;

    uint8_t datatype;
    uint8_t datatypeout;

    IDin = image_ID(in_name);

    datatype = data.image[IDin].md[0].datatype;
    naxis    = data.image[IDin].md[0].naxis;

    naxes = (int *) malloc(naxis * sizeof(uint32_t));
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    naxesl = (uint32_t *) malloc(naxis * sizeof(uint32_t));
    if(naxesl == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    naxestmp = (uint32_t *) malloc(naxis * sizeof(uint32_t));
    if(naxestmp == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(int i = 0; i < naxis; i++)
    {
        naxes[i]    = (int) data.image[IDin].md[0].size[i];
        naxesl[i]   = (uint32_t) data.image[IDin].md[0].size[i];
        naxestmp[i] = data.image[IDin].md[0].size[i];
        if(i == 0)
        {
            naxestmp[i] = data.image[IDin].md[0].size[i] / 2 + 1;
        }
    }

    char ffttmpname[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(ffttmpname, "_ffttmp_%d", (int) getpid());

    if(datatype == _DATATYPE_FLOAT)
    {
        datatypeout = _DATATYPE_COMPLEX_FLOAT;
    }
    else
    {
        datatypeout = _DATATYPE_COMPLEX_DOUBLE;
    }

    create_image_ID(ffttmpname,
                    naxis,
                    naxestmp,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDtmp);

    create_image_ID(out_name,
                    naxis,
                    naxesl,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);

    if(naxis == 2)
    {
        OK = 1;

        if(datatype == _DATATYPE_FLOAT)
        {
            plan = fftwf_plan_dft_r2c_2d(
                       (int) naxes[1],
                       (int) naxes[0],
                       data.image[IDin].array.F,
                       (fftwf_complex *) data.image[IDtmp].array.CF,
                       FFTWOPTMODE);
            if(plan == NULL)
            {
                // if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2drfft %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_dft_r2c_2d(
                           naxes[1],
                           naxes[0],
                           data.image[IDin].array.F,
                           (fftwf_complex *) data.image[IDtmp].array.CF,
                           FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if(dir == -1)
            {
                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        data.image[IDout].array.CF[jj * naxes[0] + ii] =
                            data.image[IDtmp].array.CF[jj * naxestmp[0] + ii];
                    }

                for(uint32_t ii = 1; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                {
                    uint32_t jj = 0;
                    data.image[IDout]
                    .array.CF[jj * naxes[0] + (naxes[0] - ii)]
                    .re =
                        data.image[IDtmp].array.CF[jj * naxestmp[0] + ii].re;
                    data.image[IDout]
                    .array.CF[jj * naxes[0] + (naxes[0] - ii)]
                    .im =
                        -data.image[IDtmp].array.CF[jj * naxestmp[0] + ii].im;
                    for(uint32_t jj = 1; jj < (uint32_t) naxes[1]; jj++)
                    {
                        data.image[IDout]
                        .array.CF[jj * naxes[0] + (naxes[0] - ii)]
                        .re =
                            data.image[IDtmp]
                            .array.CF[(naxes[1] - jj) * naxestmp[0] + ii]
                            .re;
                        data.image[IDout]
                        .array.CF[jj * naxes[0] + (naxes[0] - ii)]
                        .im =
                            -data.image[IDtmp]
                            .array.CF[(naxes[1] - jj) * naxestmp[0] + ii]
                            .im;
                    }
                }
            }
        }
        else
        {
            plan_double = fftw_plan_dft_r2c_2d(
                              naxes[1],
                              naxes[0],
                              data.image[IDin].array.D,
                              (fftw_complex *) data.image[IDtmp].array.CD,
                              FFTWOPTMODE);
            if(plan_double == NULL)
            {
                // if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2drfft %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_dft_r2c_2d(
                                  naxes[1],
                                  naxes[0],
                                  data.image[IDin].array.D,
                                  (fftw_complex *) data.image[IDtmp].array.CD,
                                  FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);

            if(dir == -1)
            {
                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        data.image[IDout].array.CD[jj * naxes[0] + ii] =
                            data.image[IDtmp].array.CD[jj * naxestmp[0] + ii];
                    }

                for(uint32_t ii = 1; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                {
                    uint32_t jj = 0;
                    data.image[IDout]
                    .array.CD[jj * naxes[0] + (naxes[0] - ii)]
                    .re =
                        data.image[IDtmp].array.CD[jj * naxestmp[0] + ii].re;
                    data.image[IDout]
                    .array.CD[jj * naxes[0] + (naxes[0] - ii)]
                    .im =
                        -data.image[IDtmp].array.CD[jj * naxestmp[0] + ii].im;
                    for(uint32_t jj = 1; jj < (uint32_t) naxes[1]; jj++)
                    {
                        data.image[IDout]
                        .array.CD[jj * naxes[0] + (naxes[0] - ii)]
                        .re =
                            data.image[IDtmp]
                            .array.CD[(naxes[1] - jj) * naxestmp[0] + ii]
                            .re;
                        data.image[IDout]
                        .array.CD[jj * naxes[0] + (naxes[0] - ii)]
                        .im =
                            -data.image[IDtmp]
                            .array.CD[(naxes[1] - jj) * naxestmp[0] + ii]
                            .im;
                    }
                }
            }
        }
    }
    if(naxis == 3)
    {
        OK = 1;
        //idist = naxes[0]*naxes[1];

        // swapping first 2 axis
        tmp1     = naxes[0];
        naxes[0] = naxes[1];
        naxes[1] = tmp1;

        if(datatype == _DATATYPE_FLOAT)
        {
            plan = fftwf_plan_many_dft_r2c(
                       2,
                       naxes,
                       naxes[2],
                       data.image[IDin].array.F,
                       NULL,
                       1,
                       naxes[0] * naxes[1],
                       (fftwf_complex *) data.image[IDout].array.CF,
                       NULL,
                       1,
                       naxes[0] * naxes[1],
                       FFTWOPTMODE);
            if(plan == NULL)
            {
                //	  if ( Debug > 2) fprintf(stdout,"New FFT size [do2drfft %d x %d x %d]: optimizing ...",naxes[1],naxes[0],naxes[2]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_many_dft_r2c(
                           2,
                           naxes,
                           naxes[2],
                           data.image[IDin].array.F,
                           NULL,
                           1,
                           naxes[0] * naxes[1],
                           (fftwf_complex *) data.image[IDout].array.CF,
                           NULL,
                           1,
                           naxes[0] * naxes[1],
                           FFTWOPTMODE);

                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if(dir == -1)
            {
                // unswapping first 2 axis
                tmp1     = naxes[0];
                naxes[0] = naxes[1];
                naxes[1] = tmp1;

                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                        for(uint32_t kk = 0; kk < (uint32_t) naxes[2]; kk++)
                        {
                            data.image[IDout]
                            .array.CF[naxes[0] * naxes[1] * kk +
                                               jj * naxes[0] + ii] =
                                          data.image[IDtmp]
                                          .array.CF[naxestmp[0] * naxestmp[1] * kk +
                                                                jj * naxestmp[0] + ii];
                            if(ii != 0)
                            {
                                data.image[IDout]
                                .array.CF[naxes[0] * naxes[1] * kk +
                                                   jj * naxes[0] + (naxes[0] - ii)] =
                                              data.image[IDtmp]
                                              .array
                                              .CF[naxestmp[0] * naxestmp[1] * kk +
                                                              jj * naxestmp[0] + ii];
                            }
                        }
            }
        }
        else
        {
            plan_double = fftw_plan_many_dft_r2c(
                              2,
                              naxes,
                              naxes[2],
                              data.image[IDin].array.D,
                              NULL,
                              1,
                              naxes[0] * naxes[1],
                              (fftw_complex *) data.image[IDout].array.CD,
                              NULL,
                              1,
                              naxes[0] * naxes[1],
                              FFTWOPTMODE);
            if(plan == NULL)
            {
                //	  if ( Debug > 2) fprintf(stdout,"New FFT size [do2drfft %d x %d x %d]: optimizing ...",naxes[1],naxes[0],naxes[2]);
                //				fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_many_dft_r2c(
                                  2,
                                  naxes,
                                  naxes[2],
                                  data.image[IDin].array.D,
                                  NULL,
                                  1,
                                  naxes[0] * naxes[1],
                                  (fftw_complex *) data.image[IDout].array.CD,
                                  NULL,
                                  1,
                                  naxes[0] * naxes[1],
                                  FFTWOPTMODE);

                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }

            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);

            if(dir == -1)
            {
                // unswapping first 2 axis
                tmp1     = naxes[0];
                naxes[0] = naxes[1];
                naxes[1] = tmp1;

                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                        for(uint32_t kk = 0; kk < (uint32_t) naxes[2]; kk++)
                        {
                            data.image[IDout]
                            .array.CD[naxes[0] * naxes[1] * kk +
                                               jj * naxes[0] + ii] =
                                          data.image[IDtmp]
                                          .array.CD[naxestmp[0] * naxestmp[1] * kk +
                                                                jj * naxestmp[0] + ii];
                            if(ii != 0)
                            {
                                data.image[IDout]
                                .array.CD[naxes[0] * naxes[1] * kk +
                                                   jj * naxes[0] + (naxes[0] - ii)] =
                                              data.image[IDtmp]
                                              .array
                                              .CD[naxestmp[0] * naxestmp[1] * kk +
                                                              jj * naxestmp[0] + ii];
                            }
                        }
            }
        }
    }

    if(OK == 0)
    {
        printf("Error : image dimension not appropriate for FFT\n");
    }

    delete_image_ID(ffttmpname, DELETE_IMAGE_ERRMODE_WARNING);

    free(naxestmp);
    free(naxesl);
    free(naxes);

    return IDout;
}

imageID do2drfft(const char *in_name, const char *out_name)
{
    imageID IDout;

    IDout = FFT_do2drfft(in_name, out_name, -1);

    return IDout;
}

imageID do2drffti(const char *in_name, const char *out_name)
{
    imageID IDout;

    IDout = FFT_do2drfft(in_name, out_name, 1);

    return IDout;
}

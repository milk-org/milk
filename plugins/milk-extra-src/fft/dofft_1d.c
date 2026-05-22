/**
 * @file    dofft_1d.c
 * @brief   1D FFT operations (complex and real)
 *
 * Split from dofft.c for navigability.
 * Contains: array_index(), FFT_do1dfft(), do1drfft(),
 * do1dfft(), do1dffti().
 */

#include "dofft_internal.h"

int array_index(long size)
{
    int i;

    switch (size)
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
imageID FFT_do1dfft(const char *__restrict in_name, const char *__restrict out_name, int dir)
{
    int            OK = 0;
    fftwf_plan     plan;
    fftw_plan      plan_double;
    fftwf_complex *inptr, *outptr;
    fftw_complex  *inptr_double;
    fftw_complex  *outptr_double;

    IMGID imgin = imgid_make_from_name(in_name);
    if (resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg) != RETURN_SUCCESS)
    {
        return -1;
    }

    long    naxis    = imgin.md->naxis;
    uint8_t datatype = imgin.md->datatype;

    int *naxes = (int *) malloc(naxis * sizeof(int));
    if (naxes == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        return -1;
    }

    IMGID imgout      = imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for (long i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] = imgin.md->size[i];
        naxes[i]            = (int) imgin.md->size[i];
    }
    imgout.mdt->datatype = datatype;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    if (naxis == 1)
    {
        if (array_index(naxes[0]) != 100)
        {
            OK = 1;
            if (datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                plan = fftwf_plan_dft_1d(naxes[0], (fftwf_complex *) imgin.im->array.CF,
                                         (fftwf_complex *) imgout.im->array.CF, dir, FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double =
                    fftw_plan_dft_1d(naxes[0], (fftw_complex *) imgin.im->array.CD,
                                     (fftw_complex *) imgout.im->array.CD, dir, FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
        else
        {
            OK = 1;
            if (datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                plan = fftwf_plan_dft_1d(naxes[0], (fftwf_complex *) imgin.im->array.CF,
                                         (fftwf_complex *) imgout.im->array.CF, dir, FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double =
                    fftw_plan_dft_1d(naxes[0], (fftw_complex *) imgin.im->array.CD,
                                     (fftw_complex *) imgout.im->array.CD, dir, FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
    }

    if (naxis == 2)
    {
        if ((naxes[1] == 1) && (array_index(naxes[0]) != 100))
        {
            OK = 1;
            if (datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                inptr  = (fftwf_complex *) imgin.im->array.CF;
                outptr = (fftwf_complex *) imgout.im->array.CF;
                plan   = fftwf_plan_dft_1d(naxes[0], inptr, outptr, dir, FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                inptr_double  = (fftw_complex *) imgin.im->array.CD;
                outptr_double = (fftw_complex *) imgout.im->array.CD;
                plan_double =
                    fftw_plan_dft_1d(naxes[0], inptr_double, outptr_double, dir, FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
        else
        {
            OK = 1;
            if (datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                inptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[0]);
                if (inptr == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[0]);
                if (outptr == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                plan = fftwf_plan_dft_1d(naxes[0], inptr, outptr, dir, FFTWOPTMODE);

                for (long jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr,
                           (char *) imgin.im->array.CF + sizeof(fftwf_complex) * jj * naxes[0],
                           sizeof(fftwf_complex) * naxes[0]);
                    fftwf_execute(plan);
                    memcpy((char *) imgout.im->array.CF + sizeof(complex_float) * jj * naxes[0],
                           outptr, sizeof(fftwf_complex) * naxes[0]);
                }
                fftwf_destroy_plan(plan);
                free(inptr);
                free(outptr);
            }
            else
            {
                inptr_double = (fftw_complex *) malloc(sizeof(fftw_complex) * naxes[0]);
                if (inptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                outptr_double = (fftw_complex *) malloc(sizeof(fftw_complex) * naxes[0]);
                if (outptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                plan_double =
                    fftw_plan_dft_1d(naxes[0], inptr_double, outptr_double, dir, FFTWOPTMODE);

                for (long jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr_double,
                           (char *) imgin.im->array.CD + sizeof(fftw_complex) * jj * naxes[0],
                           sizeof(fftw_complex) * naxes[0]);
                    fftw_execute(plan_double);
                    memcpy((char *) imgout.im->array.CD + sizeof(complex_double) * jj * naxes[0],
                           outptr_double, sizeof(fftw_complex) * naxes[0]);
                }
                fftw_destroy_plan(plan_double);
                free(inptr_double);
                free(outptr_double);
            }
        }
    }

    if (OK == 0)
    {
        printf("Error : image dimension"
               " not appropriate"
               " for FFT\n");
    }
    free(naxes);

    return imgout.ID;
}

/* 1d real -> complex fft */
// supports single and double precision
imageID do1drfft(const char *__restrict in_name, const char *__restrict out_name)
{
    int            OK = 0;
    fftwf_plan     plan;
    fftw_plan      plan_double;
    fftwf_complex *outptr;
    fftw_complex  *outptr_double;
    float         *inptr;
    double        *inptr_double;

    IMGID imgin = imgid_make_from_name(in_name);
    if (resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg) != RETURN_SUCCESS)
    {
        return -1;
    }

    long    naxis    = imgin.md->naxis;
    uint8_t datatype = imgin.md->datatype;

    int *naxes = (int *) malloc(naxis * sizeof(int));
    if (naxes == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        return -1;
    }

    int fftaxis = 0;
    if (naxis == 3)
    {
        fftaxis = 2;
    }

    uint32_t naxesout[3];
    for (long i = 0; i < naxis; i++)
    {
        naxes[i]    = (int) imgin.md->size[i];
        naxesout[i] = imgin.md->size[i];
        if (i == fftaxis)
        {
            naxesout[i] = imgin.md->size[i] / 2 + 1;
        }
    }

    uint8_t outtype;
    if (datatype == _DATATYPE_DOUBLE)
    {
        outtype = _DATATYPE_COMPLEX_DOUBLE;
    }
    else
    {
        outtype = _DATATYPE_COMPLEX_FLOAT;
    }

    IMGID imgout      = imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for (long i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] = naxesout[i];
    }
    imgout.mdt->datatype = outtype;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    if (naxis == 2)
    {
        if ((naxes[1] == 1) && (array_index(naxes[0]) != 100))
        {
            OK = 1;
            if (datatype == _DATATYPE_FLOAT)
            {
                plan = fftwf_plan_dft_r2c_1d(naxes[0], imgin.im->array.F,
                                             (fftwf_complex *) imgout.im->array.CF, FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double = fftw_plan_dft_r2c_1d(
                    naxes[0], imgin.im->array.D, (fftw_complex *) imgout.im->array.CD, FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(plan_double);
            }
        }
        else
        {
            OK = 1;
            if (datatype == _DATATYPE_FLOAT)
            {
                inptr = (float *) malloc(sizeof(float) * naxes[0]);
                if (inptr == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[0]);
                if (outptr == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                plan = fftwf_plan_dft_r2c_1d(naxes[0], inptr, outptr, FFTWOPTMODE);

                for (long jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr,
                           (char *) imgin.im->array.F + sizeof(float) * jj * naxes[0],
                           sizeof(float) * naxes[0]);
                    fftwf_execute(plan);
                    memcpy((char *) imgout.im->array.CF + sizeof(complex_float) * jj * naxesout[0],
                           outptr, sizeof(fftwf_complex) * naxesout[0]);
                }
                fftwf_destroy_plan(plan);
                free(inptr);
                free(outptr);
            }
            else
            {
                inptr_double = (double *) malloc(sizeof(double) * naxes[0]);
                if (inptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                outptr_double = (fftw_complex *) malloc(sizeof(fftw_complex) * naxes[0]);
                if (outptr_double == NULL)
                {
                    PRINT_ERROR("malloc returns"
                                " NULL pointer");
                    return -1;
                }

                plan_double =
                    fftw_plan_dft_r2c_1d(naxes[0], inptr_double, outptr_double, FFTWOPTMODE);

                for (long jj = 0; jj < naxes[1]; jj++)
                {
                    memcpy((char *) inptr_double,
                           (char *) imgin.im->array.D + sizeof(double) * jj * naxes[0],
                           sizeof(double) * naxes[0]);
                    fftw_execute(plan_double);
                    memcpy((char *) imgout.im->array.CD + sizeof(complex_double) * jj * naxesout[0],
                           outptr_double, sizeof(fftw_complex) * naxesout[0]);
                }
                fftw_destroy_plan(plan_double);
                free(inptr_double);
                free(outptr_double);
            }
        }
    }
    if (naxis == 3)
    {
        /* 1D FFT along last dimension */
        OK              = 1;
        uint64_t xysize = naxes[0];
        xysize *= naxes[1];

        if (datatype == _DATATYPE_FLOAT)
        {
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                for (int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] = imgin.im->array.F[i * xysize + ii];
                }
                fftwf_execute(plan);
                for (uint32_t i = 0; i < naxesout[2]; i++)
                {
                    imgout.im->array.CF[i * xysize + ii].re = outptr[i][0];
                    imgout.im->array.CF[i * xysize + ii].im = outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if (datatype == _DATATYPE_UINT16)
        {
            printf("UINT16 data type\n");
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                for (int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] = 1.0 * imgin.im->array.UI16[i * xysize + ii];
                }
                fftwf_execute(plan);
                for (uint32_t i = 0; i < naxesout[2]; i++)
                {
                    imgout.im->array.CF[i * xysize + ii].re = outptr[i][0];
                    imgout.im->array.CF[i * xysize + ii].im = outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if (datatype == _DATATYPE_UINT32)
        {
            printf("UINT32 data type\n");
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                for (int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] = 1.0 * imgin.im->array.UI32[i * xysize + ii];
                }
                fftwf_execute(plan);
                for (uint32_t i = 0; i < naxesout[2]; i++)
                {
                    imgout.im->array.CF[i * xysize + ii].re = outptr[i][0];
                    imgout.im->array.CF[i * xysize + ii].im = outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if (datatype == _DATATYPE_UINT64)
        {
            printf("UINT64 data type\n");
            inptr  = (float *) malloc(sizeof(float) * naxes[2]);
            outptr = (fftwf_complex *) malloc(sizeof(fftwf_complex) * naxes[2]);

            plan = fftwf_plan_dft_r2c_1d(naxes[2], inptr, outptr, FFTWOPTMODE);
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                for (int i = 0; i < naxes[2]; i++)
                {
                    inptr[i] = 1.0 * imgin.im->array.UI64[i * xysize + ii];
                }
                fftwf_execute(plan);
                for (uint32_t i = 0; i < naxesout[2]; i++)
                {
                    imgout.im->array.CF[i * xysize + ii].re = outptr[i][0];
                    imgout.im->array.CF[i * xysize + ii].im = outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }
    }

    if (OK == 0)
    {
        printf("Error : image dimension"
               " not appropriate"
               " for FFT\n");
    }
    free(naxes);

    return imgout.ID;
}

imageID do1dfft(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do1dfft(in_name, out_name, -1);

    return (IDout);
}

imageID do1dffti(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do1dfft(in_name, out_name, 1);

    return (IDout);
}

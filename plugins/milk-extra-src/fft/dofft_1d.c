// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    dofft_1d.c
 * @brief   1D FFT operations (complex and real)
 *
 * Split from dofft.c for navigability.
 * Contains: array_index(), FFT_do1dfft(), do1drfft(),
 * do1dfft(), do1dffti().
 */

#include "dofft_internal.h"

#include "COREMOD_arith/image_elem_block.h"
#include "COREMOD_arith/image_spatial.h"

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
    if (resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg) == -1) /* the image's ID, or -1 */
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
    imgid_create_registered(&imgout);

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
/* 1d real -> complex fft along x of each row of an image, or along z of each pixel of a cube;
 * half spectra out (n/2+1 along that axis). Any real input type: float and double natively, the
 * others as float. */
imageID do1drfft(const char *__restrict in_name, const char *__restrict out_name)
{
    IMGID imgin = imgid_make_from_name(in_name);
    if (resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg) == -1) /* the image's ID, or -1 */
    {
        return -1;
    }
    long    naxis    = imgin.md->naxis;
    uint8_t datatype = imgin.md->datatype;
    int     is_dbl   = (datatype == _DATATYPE_DOUBLE);
    if ((naxis != 2 && naxis != 3) || !arith_is_real_dt(datatype))
    {
        printf("Error : image dimension or type not appropriate for FFT\n");
        return -1;
    }
    int      fftaxis = (naxis == 3) ? 2 : 0;
    uint64_t nelem   = imgin.md->nelement;
    int      n       = (int) imgin.md->size[fftaxis];
    int      howmany = (int) (nelem / (uint64_t) n);
    int      stride  = (naxis == 3) ? howmany : 1; /* between a line's samples, in and out */
    int      idist   = (naxis == 3) ? 1 : n;       /* between lines */
    int      odist   = (naxis == 3) ? 1 : n / 2 + 1;

    IMGID imgout      = imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for (long i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] = (i == fftaxis) ? (uint32_t) (n / 2 + 1) : imgin.md->size[i];
    }
    imgout.mdt->datatype = is_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    if (imgid_create_registered(&imgout) != RETURN_SUCCESS)
    {
        return -1;
    }

    int ok = 0;
    if (is_dbl)
    {
        fftw_plan plan = fftw_plan_many_dft_r2c(1, &n, howmany, imgin.im->array.D, NULL, stride,
                                                idist, (fftw_complex *) imgout.im->array.CD, NULL,
                                                stride, odist, FFTWOPTMODE);
        ok             = (plan != NULL);
        if (ok)
        {
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
    }
    else
    {
        float *in = imgin.im->array.F;
        if (datatype != _DATATYPE_FLOAT) /* the integer types, as float */
        {
            in = (float *) fftwf_malloc(sizeof(float) * nelem);
            for (uint64_t off = 0; in != NULL && off < nelem; off += ARITH_ELEM_BLK)
            {
                int len = (int) ((nelem - off < ARITH_ELEM_BLK) ? nelem - off : ARITH_ELEM_BLK);
                arith_load_block_f_inline(imgin.im, off, len, in + off);
            }
        }
        fftwf_plan plan = (in == NULL)
                              ? NULL
                              : fftwf_plan_many_dft_r2c(1, &n, howmany, in, NULL, stride, idist,
                                                        (fftwf_complex *) imgout.im->array.CF, NULL,
                                                        stride, odist, FFTWOPTMODE);
        ok              = (plan != NULL);
        if (ok)
        {
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        if (in != NULL && in != imgin.im->array.F)
        {
            fftwf_free(in);
        }
    }
    if (!ok)
    {
        printf("Error : FFTW could not plan the transform of %s\n", in_name);
        return -1;
    }
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

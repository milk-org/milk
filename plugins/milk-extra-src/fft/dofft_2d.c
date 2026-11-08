// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    dofft_2d.c
 * @brief   2D FFT operations (complex and real)
 *
 * Split from dofft.c for navigability.
 * Contains: FFT_do2dfft(), do2dfft(), do2dffti(),
 * FFT_do2drfft(), do2drfft(), do2drffti().
 */

#include "dofft_internal.h"

/* 2d complex fft */
// supports single and double precisions
imageID FFT_do2dfft(const char *in_name, const char *out_name, int dir)
{
    int        OK = 0;
    fftwf_plan plan;
    fftw_plan  plan_double;

    char ffttmpcpyname[STRINGMAXLEN_IMGNAME];

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

    /* swap first 2 axes for fftw */
    if (naxis > 1)
    {
        long tmp1 = naxes[0];
        naxes[0]  = naxes[1];
        naxes[1]  = tmp1;
    }

    if (naxis == 2)
    {
        OK = 1;

        if (datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            plan = fftwf_plan_dft_2d(naxes[0], naxes[1], (fftwf_complex *) imgin.im->array.CF,
                                     (fftwf_complex *) imgout.im->array.CF, dir, FFTWOPTMODE);
            if (plan == NULL)
            {
                fprintf(stdout,
                        "New FFT size"
                        " [do2dfft"
                        " %d x %d]:"
                        " optimizing ...",
                        naxes[1], naxes[0]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpyname_%d", (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_dft_2d(naxes[0], naxes[1], (fftwf_complex *) imgin.im->array.CF,
                                         (fftwf_complex *) imgout.im->array.CF, dir, FFTWOPTMODE);
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
            plan_double = fftw_plan_dft_2d(naxes[0], naxes[1], (fftw_complex *) imgin.im->array.CD,
                                           (fftw_complex *) imgout.im->array.CD, dir, FFTWOPTMODE);
            if (plan_double == NULL)
            {
                fprintf(stdout,
                        "New FFT size"
                        " [do2dfft"
                        " %d x %d]:"
                        " optimizing ...",
                        naxes[1], naxes[0]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpyname_%d", (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double =
                    fftw_plan_dft_2d(naxes[0], naxes[1], (fftw_complex *) imgin.im->array.CD,
                                     (fftw_complex *) imgout.im->array.CD, dir, FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);
        }
    }

    if (naxis == 3)
    {
        OK = 1;
        if (datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            plan =
                fftwf_plan_many_dft(2, naxes, naxes[2], (fftwf_complex *) imgin.im->array.CF, NULL,
                                    1, naxes[0] * naxes[1], (fftwf_complex *) imgout.im->array.CF,
                                    NULL, 1, naxes[0] * naxes[1], dir, FFTWOPTMODE);
            if (plan == NULL)
            {
                fprintf(stdout,
                        "New FFT size"
                        " [do2dfft"
                        " %d x %d x %d]:"
                        " optimizing ...",
                        naxes[1], naxes[0], naxes[2]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpyname_%d", (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_many_dft(2, naxes, naxes[2], (fftwf_complex *) imgin.im->array.CF,
                                           NULL, 1, naxes[0] * naxes[1],
                                           (fftwf_complex *) imgout.im->array.CF, NULL, 1,
                                           naxes[0] * naxes[1], dir, FFTWOPTMODE);
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
                fftw_plan_many_dft(2, naxes, naxes[2], (fftw_complex *) imgin.im->array.CD, NULL, 1,
                                   naxes[0] * naxes[1], (fftw_complex *) imgout.im->array.CD, NULL,
                                   1, naxes[0] * naxes[1], dir, FFTWOPTMODE);
            if (plan_double == NULL)
            {
                fprintf(stdout,
                        "New FFT size"
                        " [do2dfft"
                        " %d x %d x %d]:"
                        " optimizing ...",
                        naxes[1], naxes[0], naxes[2]);
                fflush(stdout);

                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpyname_%d", (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_many_dft(
                    2, naxes, naxes[2], (fftw_complex *) imgin.im->array.CD, NULL, 1,
                    naxes[0] * naxes[1], (fftw_complex *) imgout.im->array.CD, NULL, 1,
                    naxes[0] * naxes[1], dir, FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);
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

imageID do2dfft(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do2dfft(in_name, out_name, -1);

    return (IDout);
}

imageID do2dffti(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do2dfft(in_name, out_name, 1);

    return (IDout);
}

/* real fft : real to complex */
// supports single and double precisions
/* EXPAND_HALF() - fill the full spectra of @nframes real nx x ny frames in @out from the half spectra
 * FFTW's r2c wrote in @half ((nx/2+1) x ny each): a real frame's spectrum is Hermitian. The backward
 * transform (@dir +1) of a real frame is the conjugate of the forward one. */
#define EXPAND_HALF(NAME, CTYPE)                                                                    \
    static void NAME(CTYPE *out, const CTYPE *half, uint32_t nx, uint32_t ny, uint32_t nframes,     \
                     int dir)                                                                       \
    {                                                                                               \
        uint32_t nh = nx / 2 + 1;                                                                   \
        for (uint64_t k = 0; k < nframes; k++)                                                      \
        {                                                                                           \
            CTYPE       *o = out + k * nx * ny;                                                     \
            const CTYPE *h = half + k * nh * ny;                                                    \
            for (uint32_t jj = 0; jj < ny; jj++)                                                    \
            {                                                                                       \
                for (uint32_t ii = 0; ii < nx; ii++)                                                \
                {                                                                                   \
                    int   mirror = (ii >= nh);                                                      \
                    CTYPE v      = mirror ? h[((ny - jj) % ny) * nh + (nx - ii)] : h[jj * nh + ii]; \
                    if (mirror != (dir == 1))                                                       \
                    {                                                                               \
                        v.im = -v.im;                                                               \
                    }                                                                               \
                    o[jj * nx + ii] = v;                                                            \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
    }
EXPAND_HALF(expand_half_cf, complex_float)
EXPAND_HALF(expand_half_cd, complex_double)

/* 2d real -> complex fft of an image, or of each frame of a cube; full spectra out, forward
 * (@dir -1) or backward (+1), unnormalized like FFT_do2dfft(). Single and double precision. */
imageID FFT_do2drfft(const char *__restrict in_name, const char *__restrict out_name, int dir)
{
    IMGID imgin = imgid_make_from_name(in_name);
    if (resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg) == -1) /* the image's ID, or -1 */
    {
        return -1;
    }
    uint8_t datatype = imgin.md->datatype;
    long    naxis    = imgin.md->naxis;
    int     is_dbl   = (datatype == _DATATYPE_DOUBLE);
    if ((naxis != 2 && naxis != 3) || (datatype != _DATATYPE_FLOAT && !is_dbl))
    {
        printf("Error : image dimension or type not appropriate for FFT (2D or 3D, float or "
               "double)\n");
        return -1;
    }
    uint32_t nx = imgin.md->size[0];
    uint32_t ny = imgin.md->size[1];
    uint32_t nz = (naxis == 3) ? imgin.md->size[2] : 1;

    char ffttmpname[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(ffttmpname, "_ffttmp_%d", (int) getpid());

    IMGID imgtmp      = imgid_make_from_name(ffttmpname);
    IMGID imgout      = imgid_make_from_name(out_name);
    imgtmp.mdt->naxis = naxis;
    imgout.mdt->naxis = naxis;
    for (int i = 0; i < naxis; i++)
    {
        imgtmp.mdt->size[i] = (i == 0) ? nx / 2 + 1 : imgin.md->size[i];
        imgout.mdt->size[i] = imgin.md->size[i];
    }
    imgtmp.mdt->datatype = imgout.mdt->datatype =
        is_dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT;
    imgtmp.mdt->shared = imgout.mdt->shared = dcshareddft;
    imgtmp.mdt->NBkw = imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    if (imgid_create_registered(&imgtmp) != RETURN_SUCCESS ||
        imgid_create_registered(&imgout) != RETURN_SUCCESS)
    {
        delete_image_ID(ffttmpname, DELETE_IMAGE_ERRMODE_IGNORE);
        return -1;
    }

    int n[2]  = { (int) ny, (int) nx }; /* FFTW is row-major */
    int idist = (int) (nx * ny);
    int odist = (int) ((nx / 2 + 1) * ny);
    int ok    = 0;
    if (is_dbl)
    {
        fftw_plan plan = fftw_plan_many_dft_r2c(2, n, (int) nz, imgin.im->array.D, NULL, 1, idist,
                                                (fftw_complex *) imgtmp.im->array.CD, NULL, 1,
                                                odist, FFTWOPTMODE);
        ok             = (plan != NULL);
        if (ok)
        {
            fftw_execute(plan);
            fftw_destroy_plan(plan);
            expand_half_cd(imgout.im->array.CD, imgtmp.im->array.CD, nx, ny, nz, dir);
        }
    }
    else
    {
        fftwf_plan plan = fftwf_plan_many_dft_r2c(2, n, (int) nz, imgin.im->array.F, NULL, 1, idist,
                                                  (fftwf_complex *) imgtmp.im->array.CF, NULL, 1,
                                                  odist, FFTWOPTMODE);
        ok              = (plan != NULL);
        if (ok)
        {
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
            expand_half_cf(imgout.im->array.CF, imgtmp.im->array.CF, nx, ny, nz, dir);
        }
    }
    delete_image_ID(ffttmpname, DELETE_IMAGE_ERRMODE_WARNING);
    if (!ok)
    {
        printf("Error : FFTW could not plan the %u x %u x %u transform\n", nx, ny, nz);
        return -1;
    }
    return imgout.ID;
}

imageID do2drfft(const char *in_name, const char *out_name)
{
    imageID IDout;

    IDout = FFT_do2drfft(in_name, out_name, -1);

    return (IDout);
}

imageID do2drffti(const char *in_name, const char *out_name)
{
    imageID IDout;

    IDout = FFT_do2drfft(in_name, out_name, 1);

    return (IDout);
}

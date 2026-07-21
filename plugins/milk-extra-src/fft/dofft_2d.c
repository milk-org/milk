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
imageID FFT_do2drfft(const char *__restrict in_name, const char *__restrict out_name, int dir)
{
    int *naxes; /* int format for fftw */

    int        OK = 0;
    fftwf_plan plan;
    fftw_plan  plan_double;
    long       tmp1;

    uint8_t datatypeout;

    IMGID imgin = imgid_make_from_name(in_name);
    if (resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg) != RETURN_SUCCESS)
    {
        return -1;
    }

    uint8_t datatype = imgin.md->datatype;
    long    naxis    = imgin.md->naxis;

    naxes = (int *) malloc(naxis * sizeof(uint32_t));
    if (naxes == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        return -1;
    }

    uint32_t naxestmp[3];
    for (int i = 0; i < naxis; i++)
    {
        naxes[i]    = (int) imgin.md->size[i];
        naxestmp[i] = imgin.md->size[i];
        if (i == 0)
        {
            naxestmp[i] = imgin.md->size[i] / 2 + 1;
        }
    }

    char ffttmpname[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(ffttmpname, "_ffttmp_%d", (int) getpid());

    if (datatype == _DATATYPE_FLOAT)
    {
        datatypeout = _DATATYPE_COMPLEX_FLOAT;
    }
    else
    {
        datatypeout = _DATATYPE_COMPLEX_DOUBLE;
    }

    IMGID imgtmp      = imgid_make_from_name(ffttmpname);
    imgtmp.mdt->naxis = naxis;
    for (int i = 0; i < naxis; i++)
    {
        imgtmp.mdt->size[i] = naxestmp[i];
    }
    imgtmp.mdt->datatype = datatypeout;
    imgtmp.mdt->shared   = dcshareddft;
    imgtmp.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgtmp.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgtmp);

    IMGID imgout      = imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for (int i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] = imgin.md->size[i];
    }
    imgout.mdt->datatype = datatypeout;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    if (naxis == 2)
    {
        OK = 1;

        if (datatype == _DATATYPE_FLOAT)
        {
            plan = fftwf_plan_dft_r2c_2d((int) naxes[1], (int) naxes[0], imgin.im->array.F,
                                         (fftwf_complex *) imgtmp.im->array.CF, FFTWOPTMODE);
            if (plan == NULL)
            {
                // if ( Debug > 2)
                fprintf(stdout, "New FFT size [do2drfft %d x %d]: optimizing ...", naxes[1],
                        naxes[0]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_dft_r2c_2d(naxes[1], naxes[0], imgin.im->array.F,
                                             (fftwf_complex *) imgtmp.im->array.CF, FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if (dir == -1)
            {
                for (uint32_t ii = 0; ii < (uint32_t) (naxes[0] / 2 + 1); ii++)
                {
                    for (uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CF[jj * naxes[0] + ii] =
                            imgtmp.im->array.CF[jj * naxestmp[0] + ii];
                    }
                }

                for (uint32_t ii = 1; ii < (uint32_t) (naxes[0] / 2 + 1); ii++)
                {
                    uint32_t jj = 0;
                    imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)].re =
                        imgtmp.im->array.CF[jj * naxestmp[0] + ii].re;
                    imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)].im =
                        -imgtmp.im->array.CF[jj * naxestmp[0] + ii].im;
                    for (uint32_t jj = 1; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)].re =
                            imgtmp.im->array.CF[(naxes[1] - jj) * naxestmp[0] + ii].re;
                        imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)].im =
                            -imgtmp.im->array.CF[(naxes[1] - jj) * naxestmp[0] + ii].im;
                    }
                }
            }
        }
        else
        {
            plan_double = fftw_plan_dft_r2c_2d(naxes[1], naxes[0], imgin.im->array.D,
                                               (fftw_complex *) imgtmp.im->array.CD, FFTWOPTMODE);
            if (plan_double == NULL)
            {
                // if ( Debug > 2)
                fprintf(stdout, "New FFT size [do2drfft %d x %d]: optimizing ...", naxes[1],
                        naxes[0]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double =
                    fftw_plan_dft_r2c_2d(naxes[1], naxes[0], imgin.im->array.D,
                                         (fftw_complex *) imgtmp.im->array.CD, FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);

            if (dir == -1)
            {
                for (uint32_t ii = 0; ii < (uint32_t) (naxes[0] / 2 + 1); ii++)
                {
                    for (uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CD[jj * naxes[0] + ii] =
                            imgtmp.im->array.CD[jj * naxestmp[0] + ii];
                    }
                }

                for (uint32_t ii = 1; ii < (uint32_t) (naxes[0] / 2 + 1); ii++)
                {
                    uint32_t jj = 0;
                    imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)].re =
                        imgtmp.im->array.CD[jj * naxestmp[0] + ii].re;
                    imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)].im =
                        -imgtmp.im->array.CD[jj * naxestmp[0] + ii].im;
                    for (uint32_t jj = 1; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)].re =
                            imgtmp.im->array.CD[(naxes[1] - jj) * naxestmp[0] + ii].re;
                        imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)].im =
                            -imgtmp.im->array.CD[(naxes[1] - jj) * naxestmp[0] + ii].im;
                    }
                }
            }
        }
    }
    if (naxis == 3)
    {
        OK = 1;
        //idist = naxes[0]*naxes[1];

        // swapping first 2 axis
        tmp1     = naxes[0];
        naxes[0] = naxes[1];
        naxes[1] = tmp1;

        if (datatype == _DATATYPE_FLOAT)
        {
            plan = fftwf_plan_many_dft_r2c(
                2, naxes, naxes[2], imgin.im->array.F, NULL, 1, naxes[0] * naxes[1],
                (fftwf_complex *) imgout.im->array.CF, NULL, 1, naxes[0] * naxes[1], FFTWOPTMODE);
            if (plan == NULL)
            {
                //	  if ( Debug > 2) fprintf(stdout,"New FFT size [do2drfft %d x %d x %d]: optimizing ...",naxes[1],naxes[0],naxes[2]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_many_dft_r2c(2, naxes, naxes[2], imgin.im->array.F, NULL, 1,
                                               naxes[0] * naxes[1],
                                               (fftwf_complex *) imgout.im->array.CF, NULL, 1,
                                               naxes[0] * naxes[1], FFTWOPTMODE);

                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if (dir == -1)
            {
                // unswapping first 2 axis
                tmp1     = naxes[0];
                naxes[0] = naxes[1];
                naxes[1] = tmp1;

                for (uint32_t ii = 0; ii < (uint32_t) (naxes[0] / 2 + 1); ii++)
                {
                    for (uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        for (uint32_t kk = 0; kk < (uint32_t) naxes[2]; kk++)
                        {
                            imgout.im->array.CF[naxes[0] * naxes[1] * kk + jj * naxes[0] + ii] =
                                imgtmp.im->array
                                    .CF[naxestmp[0] * naxestmp[1] * kk + jj * naxestmp[0] + ii];
                            if (ii != 0)
                            {
                                imgout.im->array.CF[naxes[0] * naxes[1] * kk + jj * naxes[0] +
                                                    (naxes[0] - ii)] =
                                    imgtmp.im->array
                                        .CF[naxestmp[0] * naxestmp[1] * kk + jj * naxestmp[0] + ii];
                            }
                        }
                    }
                }
            }
        }
        else
        {
            plan_double = fftw_plan_many_dft_r2c(
                2, naxes, naxes[2], imgin.im->array.D, NULL, 1, naxes[0] * naxes[1],
                (fftw_complex *) imgout.im->array.CD, NULL, 1, naxes[0] * naxes[1], FFTWOPTMODE);
            if (plan == NULL)
            {
                //	  if ( Debug > 2) fprintf(stdout,"New FFT size [do2drfft %d x %d x %d]: optimizing ...",naxes[1],naxes[0],naxes[2]);
                //				fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_many_dft_r2c(2, naxes, naxes[2], imgin.im->array.D, NULL, 1,
                                                     naxes[0] * naxes[1],
                                                     (fftw_complex *) imgout.im->array.CD, NULL, 1,
                                                     naxes[0] * naxes[1], FFTWOPTMODE);

                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if (dir == -1)
            {
                // unswapping first 2 axis
                tmp1     = naxes[0];
                naxes[0] = naxes[1];
                naxes[1] = tmp1;

                for (uint32_t ii = 0; ii < (uint32_t) (naxes[0] / 2 + 1); ii++)
                {
                    for (uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        for (uint32_t kk = 0; kk < (uint32_t) naxes[2]; kk++)
                        {
                            imgout.im->array.CD[naxes[0] * naxes[1] * kk + jj * naxes[0] + ii] =
                                imgtmp.im->array
                                    .CD[naxestmp[0] * naxestmp[1] * kk + jj * naxestmp[0] + ii];
                            if (ii != 0)
                            {
                                imgout.im->array.CD[naxes[0] * naxes[1] * kk + jj * naxes[0] +
                                                    (naxes[0] - ii)] =
                                    imgtmp.im->array
                                        .CD[naxestmp[0] * naxestmp[1] * kk + jj * naxestmp[0] + ii];
                            }
                        }
                    }
                }
            }
        }
    }

    if (OK == 0)
    {
        printf("Error : image dimension not appropriate for FFT\n");
    }

    delete_image_ID(ffttmpname, DELETE_IMAGE_ERRMODE_WARNING);

    free(naxes);

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

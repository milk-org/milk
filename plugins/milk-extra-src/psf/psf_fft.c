#include "psf_internal.h"

imageID PSF_makeChromatPSF(const char *amp_name,
                           const char *pha_name,
                           float       coeff1,
                           float       coeff2,
                           long        NBstep,
                           float       ApoCoeff,
                           const char *out_name)
{
    imageID  IDin;
    imageID  IDout;
    uint32_t xsize, ysize;
    imageID  IDamp;
    imageID  IDpha;
    //  float lambdafact;
    long  step;
    float x, y, u, t;

    float coeff, mcoeff, tmp;
    float eps = 1.0e-5;

    IDamp = image_ID(amp_name, dcimg, dcnimg);
    IDpha = image_ID(pha_name, dcimg, dcnimg);

    xsize = dcimg[IDamp].md[0].size[0];
    ysize = dcimg[IDamp].md[0].size[1];

    if ((dcimg[IDpha].md[0].size[0] != xsize) || (dcimg[IDpha].md[0].size[0] != xsize))
    {
        printf("ERROR in makeChromatPSF: images %s and %s have different sizes\n", amp_name,
               pha_name);
        exit(0);
    }

    create_2Dimage_ID(out_name, xsize, ysize, &IDout);
    list_image_ID();

    for (step = 0; step < NBstep; step++)
    {
        fprintf(stdout, "\rMake chromatic PSF [%3ld]: %.2f %s completed", step,
                100.0 * step / NBstep, "%");
        fflush(stdout);
        coeff = coeff1 * pow(pow((coeff2 / coeff1), 1.0 / (NBstep - 1)),
                             step); // + (coeff2-coeff1)*(1.0*step/(NBstep-1));
        x     = (coeff - (coeff1 + coeff2) / 2.0) / ((coeff2 - coeff1) / 2.0);
        // x goes from -1 to 1
        if (ApoCoeff > eps)
        {
            mcoeff = pow((1.0 - pow((fabs(x) - (1.0 - ApoCoeff)) / ApoCoeff, 2.0)), 4.0);
        }
        else
        {
            mcoeff = 1.0;
        }

        if ((1.0 - x * x) < eps)
        {
            mcoeff = 0.0;
        }
        if (fabs(x) < ApoCoeff)
        {
            mcoeff = 1.0;
        }
        //      fprintf(stdout,"(%f %f %f %f %f)",coeff,coeff1,coeff2,x,mcoeff);

        arith_image_cstmult(pha_name, coeff, "phamult");
        mk_complex_from_amph(amp_name, "phamult", "tmpimc", 0);
        delete_image_ID("phamult", DELETE_IMAGE_ERRMODE_WARNING);
        permut("tmpimc");
        do2dfft("tmpimc", "tmpimc1");
        delete_image_ID("tmpimc", DELETE_IMAGE_ERRMODE_WARNING);
        permut("tmpimc1");
        mk_amph_from_complex("tmpimc1", "tmpamp", "tmppha", 0);
        delete_image_ID("tmpimc1", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("tmppha", DELETE_IMAGE_ERRMODE_WARNING);
        arith_image_cstpow("tmpamp", 2.0, "tmpint");
        delete_image_ID("tmpamp", DELETE_IMAGE_ERRMODE_WARNING);
        list_image_ID();
        IDin = image_ID("tmpint", dcimg, dcnimg);
        for (uint32_t ii = 0; ii < xsize; ii++)
        {
            for (uint32_t jj = 0; jj < ysize; jj++)
            {
                x      = (1.0 * (ii - xsize / 2) * coeff) + xsize / 2;
                y      = (1.0 * (jj - ysize / 2) * coeff) + ysize / 2;
                long i = (long) x;
                long j = (long) y;
                u      = x - i;
                t      = y - j;
                if ((i < xsize - 1) && (j < ysize - 1) && (i > -1) && (j > -1))
                {
                    tmp = (1.0 - u) * (1.0 - t) * dcimg[IDin].array.F[j * xsize + i];
                    tmp += (1.0 - u) * t * dcimg[IDin].array.F[(j + 1) * xsize + i];
                    tmp += u * (1.0 - t) * dcimg[IDin].array.F[j * xsize + i + 1];
                    tmp += u * t * dcimg[IDin].array.F[(j + 1) * xsize + i + 1];
                    dcimg[IDout].array.F[jj * xsize + ii] += mcoeff * tmp / coeff / coeff;
                }
            }
        }
        delete_image_ID("tmpint", DELETE_IMAGE_ERRMODE_WARNING);
    }

    printf("\n");

    return IDout;
}

imageID extract_psf_photcent(const char *ID_name, const char *out_name, long size)
{
    imageID  IDin;
    imageID  IDout;
    double   totx, toty, tot;
    uint32_t naxes[2];
    long     ii, jj, ii0, jj0, ii1, jj1;

    IDin     = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[IDin].md[0].size[0];
    naxes[1] = dcimg[IDin].md[0].size[1];

    totx = 0.0;
    toty = 0.0;
    tot  = 0.0;
    for (ii = 0; ii < naxes[0]; ii++)
    {
        for (jj = 0; jj < naxes[1]; jj++)
        {
            totx += dcimg[IDin].array.F[jj * naxes[0] + ii] * ii;
            toty += dcimg[IDin].array.F[jj * naxes[0] + ii] * jj;
            tot += dcimg[IDin].array.F[jj * naxes[0] + ii];
        }
    }
    totx /= tot;
    toty /= tot;

    printf("Photocenter = %lf %lf\n", totx, toty);

    create_2Dimage_ID(out_name, size, size, &IDout);
    ii0 = (long) totx;
    jj0 = (long) toty;

    for (ii1 = 0; ii1 < size; ii1++)
    {
        for (jj1 = 0; jj1 < size; jj1++)
        {
            ii = ii0 - size / 2 + ii1;
            jj = jj0 - size / 2 + jj1;
            if ((ii > -1) && (jj > -1) && (ii < naxes[0]) && (jj < naxes[1]))
            {
                dcimg[IDout].array.F[jj1 * size + ii1] = dcimg[IDin].array.F[jj * naxes[0] + ii];
            }
            else
            {
                dcimg[IDout].array.F[jj1 * size + ii1] = 0.0f;
            }
        }
    }

    return IDout;
}

float psf_measure_SR(const char *ID_name, float factor, float r1, float r2)
{
    imageID  ID;
    long     Csize = 128;
    long     Csize2;
    double  *xcenter;
    double  *ycenter;
    long     box_size;
    uint32_t naxes[2];
    double   tmp1;
    double   SR;
    long     ii, jj;
    double   peak;
    int      fzoomfactor = 2;
    double   background;
    double   max;

    double total, total1, total2;
    long   cnt1, cnt2;
    long   peakii, peakjj;
    double dist;

    peakii = 0;
    peakjj = 0;
    Csize2 = Csize * fzoomfactor;

    xcenter = (double *) malloc(sizeof(double));
    if (xcenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ycenter = (double *) malloc(sizeof(double));
    if (ycenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ID       = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    xcenter[0] = naxes[0] / 2;
    ycenter[0] = naxes[1] / 2;
    box_size   = naxes[0] / 3 - 1;

    /*remove_cosmics(ID_name,"tmpcen");*/
    copy_image_ID(ID_name, "tmpcen", 0);
    background = arith_image_percentile("tmpcen", 0.5);

    arith_image_trunc("tmpcen", arith_image_percentile("tmpcen", 0.99),
                      arith_image_percentile("tmpcen", 1.0), "tmpcen1");
    delete_image_ID("tmpcen", DELETE_IMAGE_ERRMODE_WARNING);

    center_PSF("tmpcen1", xcenter, ycenter, box_size);
    delete_image_ID("tmpcen1", DELETE_IMAGE_ERRMODE_WARNING);

    printf("center : %f %f\n", xcenter[0], ycenter[0]);

    if ((xcenter[0] < Csize / 2 + 1) || (xcenter[0] > naxes[0] - Csize / 2 - 1) ||
        (ycenter[0] < Csize / 2 + 1) || (ycenter[0] > naxes[1] - Csize / 2 - 1))
    {
        printf("PSF too close to edge of image - cannot measure SR\n");
        SR = -1;
    }
    else
    {
        arith_image_extract2D(ID_name, "tmpsr", Csize, Csize, ((long) xcenter[0]) - Csize / 2,
                              ((long) ycenter[0]) - Csize / 2);
        ID   = image_ID("tmpsr", dcimg, dcnimg);
        peak = 0.0;
        for (ii = Csize / 2 - 5; ii < Csize / 2 + 5; ii++)
        {
            for (jj = Csize / 2 - 5; jj < Csize / 2 + 5; jj++)
            {
                tmp1 = dcimg[ID].array.F[jj * Csize + ii];
                if (tmp1 > peak)
                {
                    peak   = tmp1;
                    peakii = ii;
                    peakjj = jj;
                }
            }
        }
        for (ii = 0; ii < Csize; ii++)
        {
            for (jj = 0; jj < Csize; jj++)
            {
                if (dcimg[ID].array.F[jj * Csize + ii] > peak * 1.001f)
                {
                    dcimg[ID].array.F[jj * Csize + ii] = 0.0f;
                }
            }
        }

        fftzoom("tmpsr", "tmpsrz", fzoomfactor);
        ID = image_ID("tmpsrz", dcimg, dcnimg);
        peakii *= fzoomfactor;
        peakjj *= fzoomfactor;
        total1 = 0.0;
        total2 = 0.0;
        cnt1   = 0;
        cnt2   = 0;
        for (ii = 0; ii < Csize2; ii++)
        {
            for (jj = 0; jj < Csize2; jj++)
            {
                dist = sqrt((peakii - ii) * (peakii - ii) + (peakjj - jj) * (peakjj - jj));
                if (dist < r2 * fzoomfactor)
                {
                    if (dist < r1 * fzoomfactor)
                    {
                        total1 += dcimg[ID].array.F[jj * Csize2 + ii];
                        cnt1++;
                    }
                    else
                    {
                        total2 += dcimg[ID].array.F[jj * Csize2 + ii];
                        cnt2++;
                    }
                }
            }
        }
        background = total2 / cnt2;
        total      = total1 - background * cnt1;
        max        = arith_image_max("tmpsrz");

        printf("background = %f\n", background);
        printf("max   = %f  (%f)\n", max, max * fzoomfactor * fzoomfactor);
        printf("total = %f (%f[%ld] %f[%ld])\n", total, total1, cnt1, total2, cnt2);

        printf("ratio = %f  \n", max / total * fzoomfactor);
        SR = max / total * fzoomfactor / factor;
        save_fl_fits("tmpsr", "tmpsr");
        save_fl_fits("tmpsrz", "tmpsrz");
        delete_image_ID("tmpsr", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("tmpsrz", DELETE_IMAGE_ERRMODE_WARNING);

        printf("SR = %f\n", SR);
    }

    free(xcenter);
    free(ycenter);

    return SR;
}

// simple lucky imaging
// input must be co-centered flux normalized cube
// algorithm will rank frames according to the total flux inside a radius r_pix

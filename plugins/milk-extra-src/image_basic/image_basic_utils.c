/**
 * @file image_basic_utils.c
 * @brief Utilities for basic image operations
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "image_basic.h"


int gauss_histo_image(const char *ID_name, const char *ID_out_name, float sigma, float center)
{
    imageID  ID, ID_out;
    uint32_t naxes[2];
    long     k, k1;
    float    x;
    long     N       = 100000;
    float   *histo   = NULL;
    float   *imp     = NULL;
    float   *impr    = NULL;
    float   *imprinv = NULL;

    ID       = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    histo = (float *) malloc(sizeof(float) * N);
    if (histo == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    imp = (float *) malloc(sizeof(float) * N);
    if (imp == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    impr = (float *) malloc(sizeof(float) * N);
    if (impr == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    imprinv = (float *) malloc(sizeof(float) * N);
    if (imprinv == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for (uint32_t ii = 0; ii < N; ii++)
    {
        histo[ii] = 0.0;
        imp[ii]   = 0.0;
    }

    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            k = (long) (dcimg[ID].array.F[jj * naxes[0] + ii] * N);
            if (k < 0)
            {
                k = 0;
            }
            if (k > N - 1)
            {
                k = N - 1;
            }
            histo[k]++;
        }
    }
    for (k = 0; k < N; k++)
    {
        histo[k] *= 1.0 / naxes[1] / naxes[0];
    }

    imp[0] = histo[0];
    for (k = 1; k < N; k++)
    {
        imp[k] = imp[k - 1] + histo[k];
    }
    for (k = 0; k < N; k++)
    {
        imp[k] /= imp[N - 1];
    }


    printf("SIGMA = %f, CENTER = %f\n", sigma, center);

    for (uint32_t ii = 0; ii < N; ii++)
    {
        x         = 2.0 * (1.0 * ii / N - center);
        histo[ii] = exp(-(x * x) / sigma / sigma);
        impr[ii]  = 0.0;
    }
    impr[0] = histo[0];
    for (k = 1; k < N; k++)
    {
        impr[k] = impr[k - 1] + histo[k];
    }
    for (k = 0; k < N; k++)
    {
        impr[k] /= impr[N - 1];
    }

    k = 0;
    for (k1 = 0; k1 < N; k1++)
    {
        x = 1.0 * k1 / N;
        while (impr[k] < x)
        {
            k++;
        }
        if (k > N - 1)
        {
            k = N - 1;
        }
        imprinv[k1] = 1.0 * k / N;
    }

    ID_out = create_2Dimage_ID(ID_out_name, naxes[0], naxes[1]);
    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            k1 = (long) (dcimg[ID].array.F[jj * naxes[0] + ii] * N);
            if (k1 < 0)
            {
                k1 = 0;
            }
            if (k1 > N - 1)
            {
                k1 = N - 1;
            }
            k = (long) (imp[k1] * N);
            if (k < 0)
            {
                k = 0;
            }
            if (k > N - 1)
            {
                k = N - 1;
            }
            dcimg[ID_out].array.F[jj * naxes[0] + ii] = imprinv[k];
        }
    }

    free(histo);
    free(imp);
    free(impr);
    free(imprinv);

    return (0);
}


// load all images matching strfilter + .fits
// return number of images loaded
// image name in buffer is same as file name without extension
long load_fitsimages(const char *strfilter)
{
    long  cnt = 0;
    char  fname[STRINGMAXLEN_FILENAME];
    char  fname1[STRINGMAXLEN_FILENAME];
    FILE *fp;

    EXECUTE_SYSTEM_COMMAND_NOCHECK("ls %s.fits > flist.tmp\n", strfilter);


    if ((fp = fopen("flist.tmp", "r")) == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("fopen() error");
        exit(0);
    }

    while (fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        strncpy(fname1, fname, STRINGMAXLEN_FILENAME);
        fname1[strlen(fname) - 5] = '\0';
        load_fits(fname, fname1, 1);
        printf("[%ld] Image %s loaded -> %s\n", cnt, fname, fname1);
        fflush(stdout);
        cnt++;
    }

    fclose(fp);

    EXECUTE_SYSTEM_COMMAND_NOCHECK("rm flist.tmp");

    printf("%ld images loaded\n", cnt);

    return (cnt);
}


// recenter cube frames such that the photocenter is on the central pixel
// images are recentered by integer number of pixels
imageID basic_cube_center(const char *ID_in_name, const char *ID_out_name)
{
    imageID IDin, IDout;
    long    xsize, ysize, ksize;
    long    ii, jj, kk, ii1, jj1;
    double  tot, totii, totjj;
    long    index0, index1, index;
    double  v;
    long   *tx = NULL;
    long   *ty = NULL;

    IDin  = image_ID(ID_in_name, dcimg, dcnimg);
    xsize = dcimg[IDin].md[0].size[0];
    ysize = dcimg[IDin].md[0].size[1];
    ksize = dcimg[IDin].md[0].size[2];

    tx = (long *) malloc(sizeof(long) * ksize);
    if (tx == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    ty = (long *) malloc(sizeof(long) * ksize);
    if (ty == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }


    IDout = create_3Dimage_ID(ID_out_name, xsize, ysize, ksize);

    for (kk = 0; kk < ksize; kk++)
    {
        tot    = 0.0;
        totii  = 0.0;
        totjj  = 0.0;
        index0 = kk * xsize * ysize;

        for (jj = 0; jj < ysize; jj++)
        {
            index1 = index0 + jj * xsize;
            for (ii = 0; ii < xsize; ii++)
            {
                index = index1 + ii;
                v     = dcimg[IDin].array.F[index];
                totii += v * ii;
                totjj += v * jj;
                tot += v;
            }
        }
        totii /= tot;
        totjj /= tot;
        tx[kk] = ((long) (totii + 0.5)) - xsize / 2;
        ty[kk] = ((long) (totjj + 0.5)) - ysize / 2;

        for (ii = 0; ii < xsize; ii++)
        {
            for (jj = 0; jj < ysize; jj++)
            {
                ii1 = ii + tx[kk];
                jj1 = jj + ty[kk];
                if ((ii1 > -1) && (ii1 < xsize) && (jj1 > -1) && (jj1 < ysize))
                {
                    dcimg[IDout].array.F[index0 + jj * xsize + ii] =
                        dcimg[IDin].array.F[index0 + jj1 * xsize + ii1];
                }
                else
                {
                    dcimg[IDout].array.F[index0 + jj * xsize + ii] = 0.0f;
                }
            }
        }
    }

    free(tx);
    free(ty);

    return IDout;
}


//
// average frames in a cube
// excludes point which are more than alpha x sigma
// writes an rms value frame as rmsim
//
imageID cube_average(const char *ID_in_name, const char *ID_out_name, float alpha)
{
    imageID IDin, IDout, IDrms;
    long    xsize, ysize, ksize;
    long    ii, kk;
    double *array = NULL;
    double  ave, ave1, rms;
    long    cnt;
    long    cnt1;

    IDin  = image_ID(ID_in_name, dcimg, dcnimg);
    xsize = dcimg[IDin].md[0].size[0];
    ysize = dcimg[IDin].md[0].size[1];
    ksize = dcimg[IDin].md[0].size[2];

    IDout = create_2Dimage_ID(ID_out_name, xsize, ysize);
    IDrms = create_2Dimage_ID("rmsim", xsize, ysize);

    array = (double *) malloc(sizeof(double) * ksize);
    if (array == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    cnt1 = 0;
    for (ii = 0; ii < xsize * ysize; ii++)
    {
        for (kk = 0; kk < ksize; kk++)
        {
            array[kk] = (double) dcimg[IDin].array.F[kk * xsize * ysize + ii];
        }

        ave = 0.0;
        for (kk = 0; kk < ksize; kk++)
        {
            ave += array[kk];
        }
        ave /= ksize;

        rms = 0.0;
        for (kk = 0; kk < ksize; kk++)
        {
            rms += (array[kk] - ave) * (array[kk] - ave);
        }
        rms = sqrt(rms / ksize);

        dcimg[IDrms].array.F[ii] = (float) rms;

        ave1 = 0.0;
        cnt  = 0;
        for (kk = 0; kk < ksize; kk++)
        {
            if (fabs(array[kk] - ave) < alpha * rms)
            {
                ave1 += array[kk];
                cnt++;
            }
        }
        if (cnt > 0.5)
        {
            dcimg[IDout].array.F[ii] = (float) (ave1 / cnt);
        }
        else
        {
            dcimg[IDout].array.F[ii] = (float) ave;
        }
        cnt1 += cnt;
    }

    free(array);

    printf("(alpha = %f) fraction of pixel values selected = %ld/%ld = %.20g\n", alpha, cnt1,
           xsize * ysize * ksize, (double) (1.0 * cnt1 / (xsize * ysize * ksize)));
    printf("RMS written into image rmsim\n");

    return (IDout);
}


// coadd all images matching strfilter + .fits
// return number of images added
long basic_addimagesfiles(const char *strfilter, const char *outname)
{
    long    cnt = 0;
    char    fname[STRINGMAXLEN_FILENAME];
    char    fname1[STRINGMAXLEN_FILENAME];
    FILE   *fp;
    imageID ID;
    int     init = 0; // becomes 1 when first image encountered

    EXECUTE_SYSTEM_COMMAND_NOCHECK("ls %s.fits > flist.tmp\n", strfilter);


    if ((fp = fopen("flist.tmp", "r")) == NULL)
    {
        PRINT_ERROR("fopen() error");
        exit(0);
    }
    while (fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        strncpy(fname1, fname, STRINGMAXLEN_FILENAME);

        fname1[strlen(fname) - 5] = '\0';
        ID                        = load_fits(fname, fname1, 1);
        printf("Image %s loaded -> %s\n", fname, fname1);
        if (init == 0)
        {
            init = 1;
            copy_image_ID(dcimg[ID].name, outname, 0);
        }
        else
        {
            arith_image_add_inplace(outname, dcimg[ID].name);
        }
        delete_image_ID(fname1);
        printf("Image %s added\n", dcimg[ID].name);
        cnt++;
    }

    fclose(fp);

    EXECUTE_SYSTEM_COMMAND_NOCHECK("rm flist.tmp");

    printf("%ld images coadded (stored in variable imcnt) -> %s\n", cnt, outname);
    create_variable_ID("imcnt", 1.0 * cnt);

    return (cnt);
}


// coadd all images matching strfilter + .fits
// return number of images added
long basic_aveimagesfiles(const char *strfilter, const char *outname)
{
    long cnt;

    cnt = basic_addimagesfiles(strfilter, outname);
    arith_image_cstmult_inplace(outname, 1.0 / cnt);

    return (cnt);
}


// add all images starting with prefix
// return number of images added
long basic_addimages(const char *prefix, const char *ID_out)
{
    long i;
    int  init = 0; // becomes 1 when first image encountered
    long cnt  = 0;

    for (i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            if (strncmp(prefix, dcimg[i].name, strlen(prefix)) == 0)
            {
                if (init == 0)
                {
                    init = 1;
                    copy_image_ID(dcimg[i].name, ID_out, 0);
                }
                else
                {
                    arith_image_add_inplace(ID_out, dcimg[i].name);
                }
                printf("Image %s added\n", dcimg[i].name);
                cnt++;
            }
        }
    }

    return (cnt);
}


// paste all images starting with prefix
long basic_pasteimages(const char *prefix, long NBcol, const char *IDout_name)
{
    long    i;
    long    cnt       = 0;
    long    row       = 0;
    long    col       = 0;
    long    colmax    = 0;
    long    xsizeout  = 0;
    long    ysizeout  = 0;
    long    xsize1max = 0;
    long    ysize1max = 0;
    long    xsize1, ysize1;
    long    iioffset, jjoffset;
    long    ii, jj, ii1, jj1;
    imageID IDout;

    for (i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            if (strncmp(prefix, dcimg[i].name, strlen(prefix)) == 0)
            {
                if (dcimg[i].md[0].size[0] > xsize1max)
                {
                    xsize1max = dcimg[i].md[0].size[0];
                }
                if (dcimg[i].md[0].size[1] > ysize1max)
                {
                    ysize1max = dcimg[i].md[0].size[1];
                }

                if (col == NBcol)
                {
                    col = 0;
                    row++;
                }
                if (col > colmax)
                {
                    colmax = col;
                }

                printf("Image %s[%ld] will be pasted at [%ld %ld]\n", dcimg[i].name, cnt, row, col);
                col++;
            }
        }
    }
    xsizeout = (colmax + 1) * xsize1max;
    ysizeout = (row + 1) * ysize1max;
    IDout    = create_2Dimage_ID(IDout_name, xsizeout, ysizeout);


    col = 0;
    row = 0;
    for (i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            if (strncmp(prefix, dcimg[i].name, strlen(prefix)) == 0)
            {
                if (col == NBcol)
                {
                    col = 0;
                    row++;
                }

                iioffset = col * xsize1max;
                jjoffset = row * ysize1max;

                xsize1 = dcimg[i].md[0].size[0];
                ysize1 = dcimg[i].md[0].size[1];
                for (ii = 0; ii < xsize1; ii++)
                {
                    for (jj = 0; jj < ysize1; jj++)
                    {
                        ii1 = ii + iioffset;
                        jj1 = jj + jjoffset;
                        dcimg[IDout].array.F[jj1 * xsizeout + ii1] =
                            dcimg[i].array.F[jj * xsize1 + ii];
                    }
                }

                printf("Image %s[%ld] pasted at [%ld %ld]\n", dcimg[i].name, cnt, row, col);
                col++;
            }
        }
    }

    return (cnt);
}


// average all images starting with prefix
// return number of images added
long basic_averageimages(const char *prefix, const char *ID_out)
{
    long cnt;

    cnt = basic_addimages(prefix, ID_out);
    arith_image_cstmult_inplace(ID_out, (float) (1.0 / cnt));

    return (cnt);
}

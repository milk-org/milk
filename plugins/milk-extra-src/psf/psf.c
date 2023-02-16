#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "image_basic/image_basic.h"
#include "image_filter/image_filter.h"
#include "image_gen/image_gen.h"

#include "fft/fft.h"

#include "psf/psf.h"

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "psf"

// Module short description
#define MODULE_DESCRIPTION "Point Spread Function analysis"

//extern struct DATA data;

double FWHM_MEASURED;

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(psf)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

errno_t PSF_sequence_measure_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 1) + CLI_checkarg(3, 3) == 0)
    {
        PSF_sequence_measure(data.cmdargtoken[1].val.string,
                             data.cmdargtoken[2].val.numf,
                             data.cmdargtoken[3].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t init_module_CLI()
{
    RegisterCLIcommand("psfseqmeas",
                       __FILE__,
                       PSF_sequence_measure_cli,
                       "measure PSF sequence",
                       "<input image cube> <estimated PSF size> <output file>",
                       "psfseqmeas imc 20.0 outimc.txt",
                       "int PSF_sequence_measure(const char *IDin_name, float "
                       "PSFsizeEst, const char *outfname)");

    return RETURN_SUCCESS;
}

// make a chromatic PSF, assuming an achromatic amplitude and OPD in the pupil
// the phase is secified for the wavelength lambda0
// lamda goes from lambda0*coeff1 to lambda0*coeff2
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

    IDamp = image_ID(amp_name);
    IDpha = image_ID(pha_name);

    xsize = data.image[IDamp].md[0].size[0];
    ysize = data.image[IDamp].md[0].size[1];

    if((data.image[IDpha].md[0].size[0] != xsize) ||
            (data.image[IDpha].md[0].size[0] != xsize))
    {
        printf(
            "ERROR in makeChromatPSF: images %s and %s have different sizes\n",
            amp_name,
            pha_name);
        exit(0);
    }

    create_2Dimage_ID(out_name, xsize, ysize, &IDout);
    list_image_ID();

    for(step = 0; step < NBstep; step++)
    {
        fprintf(stdout,
                "\rMake chromatic PSF [%3ld]: %.2f %s completed",
                step,
                100.0 * step / NBstep,
                "%");
        fflush(stdout);
        coeff = coeff1 * pow(pow((coeff2 / coeff1), 1.0 / (NBstep - 1)),
                             step); // + (coeff2-coeff1)*(1.0*step/(NBstep-1));
        x     = (coeff - (coeff1 + coeff2) / 2.0) / ((coeff2 - coeff1) / 2.0);
        // x goes from -1 to 1
        if(ApoCoeff > eps)
        {
            mcoeff =
                pow((1.0 - pow((fabs(x) - (1.0 - ApoCoeff)) / ApoCoeff, 2.0)),
                    4.0);
        }
        else
        {
            mcoeff = 1.0;
        }

        if((1.0 - x * x) < eps)
        {
            mcoeff = 0.0;
        }
        if(fabs(x) < ApoCoeff)
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
        IDin = image_ID("tmpint");
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                x      = (1.0 * (ii - xsize / 2) * coeff) + xsize / 2;
                y      = (1.0 * (jj - ysize / 2) * coeff) + ysize / 2;
                long i = (long) x;
                long j = (long) y;
                u      = x - i;
                t      = y - j;
                if((i < xsize - 1) && (j < ysize - 1) && (i > -1) && (j > -1))
                {
                    tmp = (1.0 - u) * (1.0 - t) *
                          data.image[IDin].array.F[j * xsize + i];
                    tmp += (1.0 - u) * t *
                           data.image[IDin].array.F[(j + 1) * xsize + i];
                    tmp += u * (1.0 - t) *
                           data.image[IDin].array.F[j * xsize + i + 1];
                    tmp += u * t *
                           data.image[IDin].array.F[(j + 1) * xsize + i + 1];
                    data.image[IDout].array.F[jj * xsize + ii] +=
                        mcoeff * tmp / coeff / coeff;
                }
            }
        delete_image_ID("tmpint", DELETE_IMAGE_ERRMODE_WARNING);
    }

    printf("\n");

    return IDout;
}

errno_t PSF_finddiskcent(const char *ID_name, float rad, float *result)
{
    // minimizes flux outside disk
    float    xcb, ycb;
    imageID  ID;
    imageID  IDd;
    uint32_t size;
    float    step;
    float    totin, totout;
    float    v, value, bvalue;
    float    xcstart, xcend, ycstart, ycend;
    long     NBiter = 20;

    ID   = image_ID(ID_name);
    size = data.image[ID].md[0].size[0];
    step = 0.1 * size;

    xcstart = 0.0;
    xcend   = 1.0 * size;
    ycstart = 0.0;
    ycend   = 1.0 * size;

    xcb = 0.0;
    ycb = 0.0;

    bvalue = arith_image_total(ID_name);
    for(long iter = 0; iter < NBiter; iter++)
    {
        fprintf(stderr,
                "iter %ld / %ld  (%f %f  %f %f   %f %f) %g\n",
                iter,
                NBiter,
                xcstart,
                xcend,
                ycstart,
                ycend,
                xcb,
                ycb,
                bvalue);
        for(float xc = xcstart; xc < xcend; xc += step)
            for(float yc = ycstart; yc < ycend; yc += step)
            {
                IDd = make_subpixdisk("tmpd1", size, size, xc, yc, rad);

                totin  = 0.0;
                totout = 0.0;
                for(uint64_t ii = 0; ii < size * size; ii++)
                {
                    v = data.image[ID].array.F[ii];
                    if(data.image[IDd].array.F[ii] > 0.5)
                    {
                        totin += v;
                    }
                    else
                    {
                        totout += v;
                    }
                }
                value = totout;
                if(value < bvalue)
                {
                    xcb    = xc;
                    ycb    = yc;
                    bvalue = value;
                }
                delete_image_ID("tmpd1", DELETE_IMAGE_ERRMODE_WARNING);
            }
        xcstart = 0.5 * (xcstart + xcb);
        xcend   = 0.5 * (xcend + xcb);
        ycstart = 0.5 * (ycstart + ycb);
        ycend   = 0.5 * (ycend + ycb);
        step *= 0.5;
    }

    printf("Disk center = %f x %f\n", xcb, ycb);
    result[0] = xcb;
    result[1] = ycb;

    return RETURN_SUCCESS;
}

errno_t PSF_finddiskcent_alone(const char *ID_name, float rad)
{
    float *result;

    result = (float *) malloc(sizeof(float) * 2);
    if(result == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    PSF_finddiskcent(ID_name, rad, result);
    free(result);

    return RETURN_SUCCESS;
}

errno_t PSF_measurePhotocenter(const char *ID_name)
{
    imageID  ID;
    uint32_t naxes[2];
    float    iitot, jjtot, tot;
    float    v;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    iitot = 0.0;
    jjtot = 0.0;
    tot   = 0.0;
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            v = data.image[ID].array.F[jj * naxes[1] + ii];
            tot += v;
            iitot += v * ii;
            jjtot += v * jj;
        }

    printf("photocenter = %.2f %.2f\n", iitot / tot, jjtot / tot);
    data.FLOATARRAY[0] = iitot / tot;
    data.FLOATARRAY[1] = jjtot / tot;

    return RETURN_SUCCESS;
}

float measure_enc_NRJ(const char *ID_name,
                      float       xcenter,
                      float       ycenter,
                      float       fraction)
{
    imageID  ID;
    uint32_t naxes[2];
    float   *total;
    float    distance;
    long     index;
    float    sum;
    float    sum_all;
    long     arraysize;
    float    value;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    arraysize = (long)(sqrt(2) * naxes[0]);

    total = (float *) malloc(sizeof(float) * arraysize);
    if(total == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(uint32_t ii = 0; ii < arraysize; ii++)
    {
        total[ii] = 0.0;
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            index    = (long) distance;
            if(index < arraysize)
            {
                total[index] += data.image[ID].array.F[jj * naxes[0] + ii];
            }
        }

    sum_all = 0.0;
    for(uint32_t ii = 0; ii < arraysize; ii++)
    {
        sum_all += total[ii];
    }

    sum = 0.0;
    sum_all *= fraction;

    {
        uint64_t ii = 0;
        while(sum < sum_all)
        {
            sum += total[ii];
            ii++;
        }

        /*  printf("%ld %f %f\n",ii,total[ii-1],sum_all);*/
        value = 1.0 * (ii - 2) +
                (sum_all - (sum - total[ii - 1])) / (total[ii - 1]);
    }
    printf("Enc. NRJ = %f pix\n", value);
    free(total);

    return (value);
}

errno_t measure_enc_NRJ1(const char *ID_name,
                         float       xcenter,
                         float       ycenter,
                         const char *filename)
{
    imageID  ID;
    uint32_t naxes[2];
    float   *total;
    float    distance;
    long     index;
    float    sum_all;
    uint32_t arraysize;
    FILE    *fp;
    float   *ENCNRJ;

    printf("Center is %f %f\n", xcenter, ycenter);

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    arraysize = (uint32_t)(sqrt(2) * naxes[0]);

    total = (float *) malloc(sizeof(float) * arraysize);
    if(total == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ENCNRJ = (float *) malloc(sizeof(float) * arraysize);
    if(ENCNRJ == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(index = 0; index < arraysize; index++)
    {
        ENCNRJ[index] = 0.0;
        total[index]  = 0.0;
    }
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            index    = (long) distance;
            if(index < arraysize)
            {
                total[index] += data.image[ID].array.F[jj * naxes[0] + ii];
            }
        }

    if((fp = fopen(filename, "w")) == NULL)
    {
        printf("ERROR: cannot create file \"%s\"\n", filename);
        fflush(stdout);
        exit(0);
    }

    sum_all = 0.0;
    for(uint32_t ii = 0; ii < arraysize; ii++)
    {
        ENCNRJ[ii] = sum_all;
        sum_all += total[ii];
        fprintf(fp, "%u %f\n", ii, ENCNRJ[ii]);
    }
    fclose(fp);

    free(total);
    free(ENCNRJ);

    return RETURN_SUCCESS;
}

/* measures the FWHM of a "perfect" PSF */
float measure_FWHM(
    const char *ID_name, float xcenter, float ycenter, float step, long nb_step)
{
    imageID  ID;
    uint32_t naxes[2];
    // long nelements;
    float  distance;
    float *dist;
    float *mean;
    float *rms;
    long  *counts;
    long   i;
    float  FWHM;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    //nelements = naxes[0] * naxes[1];

    dist = (float *) malloc(nb_step * sizeof(float));
    if(dist == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    mean = (float *) malloc(nb_step * sizeof(float));
    if(mean == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    rms = (float *) malloc(nb_step * sizeof(float));
    if(rms == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    counts = (long *) malloc(nb_step * sizeof(long));
    if(counts == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(i = 0; i < nb_step; i++)
    {
        dist[i]   = 0;
        mean[i]   = 0;
        rms[i]    = 0;
        counts[i] = 0;
    }

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            distance = sqrt((1.0 * ii - xcenter) * (1.0 * ii - xcenter) +
                            (1.0 * jj - ycenter) * (1.0 * jj - ycenter));
            i        = (long) distance / step;
            if(i < nb_step)
            {
                dist[i] += distance;
                mean[i] += data.image[ID].array.F[jj * naxes[0] + ii];
                rms[i] += data.image[ID].array.F[jj * naxes[0] + ii] *
                          data.image[ID].array.F[jj * naxes[0] + ii];
                counts[i] += 1;
            }
        }

    for(i = 0; i < nb_step; i++)
    {
        dist[i] /= counts[i];
        mean[i] /= counts[i];
        rms[i] = sqrt(rms[i] - 1.0 * counts[i] * mean[i] * mean[i]) /
                 sqrt(counts[i]);
    }

    FWHM = 0.0;
    for(i = 0; i < nb_step; i++)
    {
        if((mean[i + 1] < mean[0] / 2) && (mean[i] > mean[0] / 2))
        {
            FWHM = 2.0 * dist[i] + (dist[i + 1] - dist[i]) *
                   (mean[i] - mean[0] / 2) /
                   (mean[i] - mean[i + 1]);
        }
    }

    free(counts);
    free(dist);
    free(mean);
    free(rms);

    return FWHM;
}

/* finds a PSF center with no a priori position information */
errno_t
center_PSF(const char *ID_name, double *xcenter, double *ycenter, long box_size)
{
    imageID ID;
    long
    n3; /* effective box size. =box_size if the star is not at the edge of the image field */
    double   back_cont;
    double   centerx, centery;
    double   ocenterx, ocentery;
    double   total_fl;
    uint32_t naxes[2];
    int      nbiter = 10;
    long     iistart, iiend, jjstart, jjend;

    n3 = box_size;

    /* for better performance, the background continuum needs to be computed for each image */
    back_cont = arith_image_median(ID_name);
    /* first approximation given by barycenter after median of image */
    median_filter(ID_name, "PSFctmp", 1);

    ID       = image_ID("PSFctmp");
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    centerx  = (double) naxes[0] / 2;
    centery  = (double) naxes[1] / 2;
    ocenterx = centerx;
    ocentery = centery;

    for(int k = 0; k < nbiter; k++)
    {
        n3 = (long)(1.0 * naxes[0] / 2 /
                    (1.0 + (0.1 * naxes[0] / 2 * k / (4 * nbiter))));

        iistart = (long)(0.5 + ocenterx - n3);
        if(iistart < 0)
        {
            iistart = 0;
        }
        iiend = (long)(0.5 + ocenterx + n3);
        if(iiend > naxes[0] - 1)
        {
            iiend = naxes[0] - 1;
        }

        jjstart = (long)(0.5 + ocentery - n3);
        if(jjstart < 0)
        {
            jjstart = 0;
        }
        jjend = (long)(0.5 + ocentery + n3);
        if(jjend > naxes[1] - 1)
        {
            jjend = naxes[1] - 1;
        }

        //      printf("effective box size is %ld - center is %f %f\n",n3,ocenterx,ocentery);
        // fflush(stdout);
        centerx  = 0.0;
        centery  = 0.0;
        total_fl = 0.0;
        for(uint32_t jj = jjstart; jj < jjend; jj++)
            for(uint32_t ii = iistart; ii < iiend; ii++)
            {
                if(data.image[ID].array.F[jj * naxes[0] + ii] > back_cont)
                {
                    centerx += 1.0 * ii *
                               (data.image[ID].array.F[jj * naxes[0] + ii] -
                                1.0 * back_cont);
                    centery += 1.0 * jj *
                               (data.image[ID].array.F[jj * naxes[0] + ii] -
                                1.0 * back_cont);
                    total_fl += data.image[ID].array.F[jj * naxes[0] + ii] -
                                1.0 * back_cont;
                }
            }
        centerx /= total_fl;
        centery /= total_fl;

        /*      printf("step %d\n",k);
            printf("image %s: center is %f %f for %ld by %ld pixels. Total_fl is %f\n",data.image[ID].name,centerx,centery,naxes[0],naxes[1],total_fl);
        */
        ocenterx = centerx;
        ocentery = centery;
    }

    delete_image_ID("PSFctmp", DELETE_IMAGE_ERRMODE_WARNING);

    xcenter[0] = centerx;
    ycenter[0] = centery;

    return RETURN_SUCCESS;
}

/* finds a PSF center with no a priori position information */
errno_t fast_center_PSF(const char *ID_name,
                        double     *xcenter,
                        double     *ycenter,
                        long        box_size)
{
    imageID ID;
    long
    n3; /* effective box size. =box_size if the star is not at the edge of the image field */
    double   centerx, centery;
    double   ocenterx, ocentery;
    double   total_fl;
    uint32_t naxes[2];
    int      nbiter = 6;

    long iimin, iimax, jjmin, jjmax;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    centerx  = (double) naxes[0] / 2;
    centery  = (double) naxes[1] / 2;
    ocenterx = centerx;
    ocentery = centery;

    for(int k = 0; k < nbiter; k++)
    {
        n3 = (long)(1.0 * naxes[0] / 2 /
                    (1.0 + (0.1 * naxes[0] / 2 * k / (4 * nbiter))));
        if(((long)(0.5 + ocenterx) - n3) < 0)
        {
            n3 = (long)(0.5 + ocenterx);
        }
        if(((long)(0.5 + ocenterx) + n3 + 1) > naxes[0])
        {
            n3 = naxes[0] - ((long)(0.5 + ocenterx) + 1);
        }
        if(((long)(0.5 + ocentery) - n3) < 0)
        {
            n3 = (long)(0.5 + ocentery);
        }
        if(((long)(0.5 + ocentery) + n3 + 1) > naxes[1])
        {
            n3 = naxes[1] - ((long)(0.5 + ocentery) + 1);
        }
        n3 -= 1;

        if(n3 < box_size)
        {
            n3 = box_size;
        }

        centerx  = 0.0;
        centery  = 0.0;
        total_fl = 0.0;

        iimin = ((long)(0.5 + ocenterx) - n3);
        if(iimin < 0)
        {
            iimin = 0.0;
        }
        iimax = ((long)(0.5 + ocenterx) + n3 + 1);
        if(iimax > naxes[0] - 1)
        {
            iimax = naxes[0] - 1;
        }

        jjmin = ((long)(0.5 + ocentery) - n3);
        if(jjmin < 0)
        {
            jjmin = 0.0;
        }
        jjmax = ((long)(0.5 + ocentery) + n3 + 1);
        if(jjmax > naxes[1] - 1)
        {
            jjmax = naxes[1] - 1;
        }

        for(uint32_t jj = (uint32_t) jjmin; jj < (uint32_t) jjmax; jj++)
            for(uint32_t ii = (uint32_t) iimin; ii < (uint32_t) iimax; ii++)
            {
                centerx +=
                    1.0 * ii * (data.image[ID].array.F[jj * naxes[0] + ii]);
                centery +=
                    1.0 * jj * (data.image[ID].array.F[jj * naxes[0] + ii]);
                total_fl += data.image[ID].array.F[jj * naxes[0] + ii];
            }

        //        printf("effective box size is %ld (%ld) - center is %f %f   [ %3ld %3ld   %3ld %3ld]   ", n3, box_size, ocenterx, ocentery, iimin, iimax, jjmin, jjmax);
        //       fflush(stdout);

        //    printf("total_fl is %f\n",total_fl);
        centerx /= total_fl;
        centery /= total_fl;

        //printf("step %d\n",k);
        //printf("image %s: center is %f %f for %ld by %ld pixels. Total_fl is %f\n",data.image[ID].name,centerx,centery,naxes[0],naxes[1],total_fl);

        ocenterx = centerx;
        ocentery = centery;
    }

    xcenter[0] = centerx;
    ycenter[0] = centery;

    return RETURN_SUCCESS;
}

errno_t center_PSF_alone(const char *ID_name)
{
    imageID  ID;
    double  *xcenter;
    double  *ycenter;
    long     box_size;
    uint32_t naxes[2];

    xcenter = (double *) malloc(sizeof(double));
    if(xcenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ycenter = (double *) malloc(sizeof(double));
    if(ycenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    xcenter[0] = naxes[0] / 2;
    ycenter[0] = naxes[1] / 2;
    box_size   = naxes[0] / 3 - 1;

    /*remove_cosmics(ID_name,"tmpcen");*/
    copy_image_ID(ID_name, "tmpcen", 0);

    arith_image_trunc("tmpcen",
                      arith_image_percentile("tmpcen", 0.99),
                      arith_image_percentile("tmpcen", 1.0),
                      "tmpcen1");
    delete_image_ID("tmpcen", DELETE_IMAGE_ERRMODE_WARNING);

    center_PSF("tmpcen1", xcenter, ycenter, box_size);
    delete_image_ID("tmpcen1", DELETE_IMAGE_ERRMODE_WARNING);

    printf("center : %f %f\n", xcenter[0], ycenter[0]);

    create_variable_ID("xc", xcenter[0]);
    create_variable_ID("yc", ycenter[0]);

    free(xcenter);
    free(ycenter);

    return RETURN_SUCCESS;
}

/* this simple routine finds the center of a PSF by barycenter technique */
errno_t center_star(const char *ID_in_name, double *x_star, double *y_star)
{
    imageID  ID_in;
    uint32_t naxes[2];
    long     n1, n2, n3;
    /* n2,n3 are the pixel coordinate, n3 is the pixel radius of the sampling box used.*/
    double sum, coeff;
    double xsum, ysum;
    int    max_nb_iter = 500;
    int    i, found;
    double limit;

    limit = 1.0 / 10000000000.0;

    ID_in    = image_ID(ID_in_name);
    naxes[0] = data.image[ID_in].md[0].size[0];
    naxes[1] = data.image[ID_in].md[0].size[1];
    n1       = (long) x_star[0];
    n2       = (long) y_star[0];
    n3       = 20;

    i     = 0;
    found = 0;
    while((i < max_nb_iter) && (found == 0))
    {
        xsum = 0;
        ysum = 0;
        sum  = 0;
        for(uint32_t jj = (n2 - n3); jj < (n2 + n3); jj++)
            for(uint32_t ii = (n1 - n3); ii < (n1 + n3); ii++)
            {
                coeff = (ii - x_star[0]) * (ii - x_star[0]) +
                        (jj - y_star[0]) * (jj - y_star[0]);
                coeff = coeff / n3 / n3;
                coeff = exp(-coeff * 50 * (1.0 * i / max_nb_iter));
                sum =
                    sum + coeff * data.image[ID_in].array.F[jj * naxes[0] + ii];
                xsum = xsum +
                       coeff * (data.image[ID_in].array.F[jj * naxes[0] + ii]) *
                       ii;
                ysum = ysum +
                       coeff * (data.image[ID_in].array.F[jj * naxes[0] + ii]) *
                       jj;
            }
        xsum = xsum / sum;
        ysum = ysum / sum;
        if(((x_star[0] - xsum) * (x_star[0] - xsum) +
                (y_star[0] - ysum) * (y_star[0] - ysum)) <
                (limit * limit * n3 * n3))
        {
            found = 1;
        }
        x_star[0] = xsum;
        y_star[0] = ysum;
        n1        = (long) xsum;
        n2        = (long) ysum;
        /*    printf(" i=%d x=%e  y=%e  sum=%20.18e \n",i,xsum,ysum,sum);*/
        i++;
    }

    x_star[0] = xsum;
    y_star[0] = ysum;
    printf("%f %f\n", xsum, ysum);

    return RETURN_SUCCESS;
}

float get_sigma(const char *ID_name, float x, float y, const char *options)
{
    imageID  ID;
    uint32_t naxes[2];
    float    C, distsq;
    int      n3 = 30;
    /* , nb_iter=40; */
    /* float sum,count; */
    long   n1, n2;
    int    str_pos;
    char   boxsize[50];
    float *x1;
    float *y1;
    float *sig;
    long   nbpixel;
    long   pixelnb;
    float  SATURATION = 50000.0000;
    float  best_A, best_err, best_sigma, err, A, sigma, sigmasq;
    float  dist[100];
    float  value[100];
    int    count[100];
    float  peak, FWHM_m;
    int    getmfwhm;
    int    backg;

    printf("get_sigma .... ");
    fflush(stdout);
    if(strstr(options, "-box ") != NULL)
    {
        str_pos = strstr(options, "-box ") - options;
        str_pos = str_pos + strlen("-box ");
        int i   = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            boxsize[i] = options[i + str_pos];
            i++;
        }
        boxsize[i] = '\0';
        n3         = atoi(boxsize);
        printf("box radius is %d pixels \n", n3);
    }

    getmfwhm = 0;
    if(strstr(options, "-mfwhm ") != NULL)
    {
        getmfwhm = 1;
    }

    backg = 0;
    if(strstr(options, "-backg ") != NULL)
    {
        backg = 1;
    }

    if(strstr(options, "-sat ") != NULL)
    {
        str_pos = strstr(options, "-sat ") - options;
        str_pos = str_pos + strlen("-sat ");
        int i   = 0;
        while((options[i + str_pos] != ' ') &&
                (options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
        {
            boxsize[i] = options[i + str_pos];
            i++;
        }
        boxsize[i] = '\0';
        SATURATION = atof(boxsize);
        printf("saturation level is %f\n", SATURATION);
    }

    n1       = (long) x;
    n2       = (long) y;
    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    C = 0.0;
    A = data.image[ID].array.F[n2 * naxes[0] + n1 + 1];
    printf("A initial is %f\n", A);
    /* f = Aexp(-a*a*x*x)+C */

    nbpixel = 0;
    for(int jj = (n2 - n3); jj < (n2 + n3); jj++)
        for(int ii = (n1 - n3); ii < (n1 + n3); ii++)
            if((ii > 0) && (ii < (int) naxes[0]) && (jj > 0) &&
                    (jj < (int) naxes[1]))
            {
                nbpixel += 1;
            }

    x1 = (float *) malloc(nbpixel * sizeof(float));
    if(x1 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    y1 = (float *) malloc(nbpixel * sizeof(float));
    if(y1 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sig = (float *) malloc(nbpixel * sizeof(float));
    if(sig == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    printf("background is ");
    fflush(stdout);
    if(backg == 1)
    {
        C = 0.0;
    }
    else
    {
        C = arith_image_percentile(ID_name, 0.5);
    }
    printf("%f\n", C);
    fflush(stdout);
    pixelnb = 0;

    for(long jj = (n2 - n3); jj < (n2 + n3); jj++)
        for(long ii = (n1 - n3); ii < (n1 + n3); ii++)
            if((ii > 0) && (ii < (int) naxes[0]) && (jj > 0) &&
                    (jj < (int) naxes[1]))
            {
                distsq = (ii - x) * (ii - x) + (jj - y) * (jj - y);
                if(data.image[ID].array.F[jj * naxes[0] + ii] < SATURATION)
                {
                    x1[pixelnb]  = distsq;
                    y1[pixelnb]  = data.image[ID].array.F[jj * naxes[0] + ii];
                    sig[pixelnb] = 1.0;
                    pixelnb++;
                }
            }

    /* do the radial average */
    for(int i = 0; i < 100; i++)
    {
        count[i] = 0.0;
        dist[i]  = 0.0;
        value[i] = 0.0;
    }

    for(int i = 0; i < pixelnb; i++)
    {
        count[(int)(sqrt(x1[i]))]++;
        dist[(int)(sqrt(x1[i]))] += sqrt(x1[i]);
        value[(int)(sqrt(x1[i]))] += y1[i];
    }

    for(int i = 0; i < 100; i++)
    {
        dist[i] /= count[i];
        value[i] /= count[i];
    }

    sigma   = 10.0;
    err     = 0.0;
    sigmasq = sigma * sigma;
    for(int i = 0; i < 100; i++)
        if(count[i] > 0)
        {
            err =
                err +
                pow((value[i] - C - A * exp(-dist[i] * dist[i] / sigmasq)), 2) *
                count[i];
        }
    best_err = err;

    best_sigma = 10.0;
    best_A     = 1000.0;
    for(sigma = 2.0; sigma < 50.0; sigma = sigma * 1.01)
        for(A = best_A * 0.1; A < best_A * 10.0; A = A * 1.01)
        {
            err     = 0.0;
            sigmasq = sigma * sigma;
            for(int i = 0; i < 100; i++)
                if(count[i] > 0)
                {
                    err = err + pow((value[i] - C -
                                     A * exp(-dist[i] * dist[i] / sigmasq)),
                                    2) *
                          0.00001 * count[i];
                }
            if(err < best_err)
            {
                best_err   = err;
                best_A     = A;
                best_sigma = sigma;
            }
        }

    peak = value[0] - C;
    {
        int i = 0;
        while((value[i] - C) > (peak / 2.0))
        {
            i++;
        }
        FWHM_m = 2.0 * dist[i - 1] + 2.0 * (dist[i] - dist[i - 1]) *
                 (value[i - 1] - C - peak / 2.0) /
                 (value[i - 1] - value[i]);
    }

    printf("PSF center is %f x %f\n", x, y);
    printf("A = %f\n", best_A);
    printf("sigma = %f\n", best_sigma);
    printf("gaussian FWHM = %f\n", 2.0 * best_sigma * sqrt(log(2)));
    printf("measured peak = %f (background subtracted)\n", peak);
    printf("measured FWHM = %f\n", FWHM_m);

    free(sig);
    free(x1);
    free(y1);

    if(getmfwhm == 1)
    {
        sigma = FWHM_m;
    }
    else
    {
        sigma = best_sigma;
    }

    return sigma;
}

float get_sigma_alone(const char *ID_name)
{
    double  *xcenter;
    double  *ycenter;
    double   sigma = 0.0;
    imageID  ID;
    uint32_t naxes[2];
    long     box_size;
    /*  char lstring[1000];*/
    int FAST = 0;
    int FWHM = 0;

    xcenter = (double *) malloc(sizeof(double));
    if(xcenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ycenter = (double *) malloc(sizeof(double));
    if(ycenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    xcenter[0] = naxes[0] / 2;
    ycenter[0] = naxes[1] / 2;
    box_size   = xcenter[0] / 2 - 1;

    /*  remove_cosmics(ID_name,"tmpcen");*/
    copy_image_ID(ID_name, "tmpcen", 0);

    if(FAST == 0)
    {
        arith_image_trunc("tmpcen",
                          arith_image_percentile("tmpcen", 0.9),
                          arith_image_percentile("tmpcen", 1.0),
                          "tmpcen1");
        delete_image_ID("tmpcen", DELETE_IMAGE_ERRMODE_WARNING);
        center_PSF("tmpcen1", xcenter, ycenter, box_size);
        delete_image_ID("tmpcen1", DELETE_IMAGE_ERRMODE_WARNING);

        sigma = get_sigma(ID_name, xcenter[0], ycenter[0], "");
    }
    else
    {
        fast_center_PSF("tmpcen", xcenter, ycenter, box_size);
        center_star("tmpcen", xcenter, ycenter);
        printf("peak = %f\n",
               data.image[ID].array.F[((long) ycenter[0]) * naxes[0] +
                                      ((long) xcenter[0])]);
        if(FWHM == 1)
        {
            sigma = get_sigma(ID_name, xcenter[0], ycenter[0], "");
        }
    }

    free(xcenter);
    free(ycenter);

    return (sigma);
}

errno_t extract_psf(const char *ID_name, const char *out_name, long size)
{
    imageID  ID;
    double  *xcenter;
    double  *ycenter;
    long     box_size;
    uint32_t naxes[2];

    xcenter = (double *) malloc(sizeof(double));
    if(xcenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ycenter = (double *) malloc(sizeof(double));
    if(ycenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    xcenter[0] = naxes[0] / 2;
    ycenter[0] = naxes[1] / 2;
    box_size   = naxes[0] / 2 - 1;
    /*remove_cosmics(ID_name,"tmpcen");*/
    copy_image_ID(ID_name, "tmpcen", 0);
    arith_image_trunc("tmpcen",
                      arith_image_percentile("tmpcen", 0.99),
                      arith_image_percentile("tmpcen", 1.0),
                      "tmpcen1");
    delete_image_ID("tmpcen", DELETE_IMAGE_ERRMODE_WARNING);
    center_PSF("tmpcen1", xcenter, ycenter, box_size);

    printf("PSF center = %f %f   extracting window size %ld\n",
           xcenter[0],
           ycenter[0],
           size);
    delete_image_ID("tmpcen1", DELETE_IMAGE_ERRMODE_WARNING);
    /*  arith_image_extract2D(ID_name,out_name,size,size,((long) (xcenter[0]+0.5))-(size/2),((long) (ycenter[0]+0.5))-(size/2));*/

    arith_image_extract2D(ID_name,
                          "tmpf",
                          size,
                          size,
                          ((long)(xcenter[0] + 0.5)) - (size / 2),
                          ((long)(ycenter[0] + 0.5)) - (size / 2));

    fft_image_translate("tmpf",
                        out_name,
                        xcenter[0] - ((long)(xcenter[0] + 0.5)),
                        ycenter[0] - ((long)(ycenter[0] + 0.5)));
    //arith_image_translate("tmpf", out_name,xcenter[0]-((long) (xcenter[0]+0.5)), ycenter[0]-((long) (ycenter[0]+0.5)));

    delete_image_ID("tmpf", DELETE_IMAGE_ERRMODE_WARNING);
    free(xcenter);
    free(ycenter);

    return RETURN_SUCCESS;
}

imageID
extract_psf_photcent(const char *ID_name, const char *out_name, long size)
{
    imageID  IDin;
    imageID  IDout;
    double   totx, toty, tot;
    uint32_t naxes[2];
    long     ii, jj, ii0, jj0, ii1, jj1;

    IDin     = image_ID(ID_name);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];

    totx = 0.0;
    toty = 0.0;
    tot  = 0.0;
    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
        {
            totx += data.image[IDin].array.F[jj * naxes[0] + ii] * ii;
            toty += data.image[IDin].array.F[jj * naxes[0] + ii] * jj;
            tot += data.image[IDin].array.F[jj * naxes[0] + ii];
        }
    totx /= tot;
    toty /= tot;

    printf("Photocenter = %lf %lf\n", totx, toty);

    create_2Dimage_ID(out_name, size, size, &IDout);
    ii0 = (long) totx;
    jj0 = (long) toty;

    for(ii1 = 0; ii1 < size; ii1++)
        for(jj1 = 0; jj1 < size; jj1++)
        {
            ii = ii0 - size / 2 + ii1;
            jj = jj0 - size / 2 + jj1;
            if((ii > -1) && (jj > -1) && (ii < naxes[0]) && (jj < naxes[1]))
            {
                data.image[IDout].array.F[jj1 * size + ii1] =
                    data.image[IDin].array.F[jj * naxes[0] + ii];
            }
            else
            {
                data.image[IDout].array.F[jj1 * size + ii1] = 0.0;
            }
        }

    return IDout;
}

errno_t
psf_variance(const char *ID_out_m, const char *ID_out_v, const char *options)
{
    int      Nb_files;
    int      file_nb;
    long     ii, jj, i, j;
    int      str_pos;
    int     *IDn;
    char     file_name[50];
    uint32_t naxes[2];
    float    mean, tmp, rms;
    imageID  IDoutm, IDoutv;

    Nb_files = 1;
    i        = 0;
    str_pos  = 0;

    printf("option is :%s\n", options);
    while((options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
    {
        if(options[i + str_pos] == ' ')
        {
            Nb_files += 1;
        }
        i++;
    }

    printf("%d files\n", Nb_files);
    IDn = (int *) malloc(Nb_files * sizeof(int));
    if(IDn == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    j       = 0;
    i       = 0;
    file_nb = 0;
    while(file_nb < Nb_files)
    {
        if((options[i + str_pos] == ' ') || (options[i + str_pos] == '\0') ||
                (options[i + str_pos] == '\n'))
        {
            file_name[j] = '\0';
            IDn[file_nb] = image_ID(file_name);
            printf("%d %s \n", IDn[file_nb], file_name);
            file_nb += 1;
            j = 0;
        }
        else
        {
            file_name[j] = options[i + str_pos];
            j++;
        }
        i++;
    }
    naxes[0] = data.image[IDn[0]].md[0].size[0];
    naxes[1] = data.image[IDn[0]].md[0].size[1];

    create_2Dimage_ID(ID_out_m, naxes[0], naxes[1], &IDoutm);
    create_2Dimage_ID(ID_out_v, naxes[0], naxes[1], &IDoutv);
    /*  printf("%d %d - starting computations\n",naxes[0],naxes[1]);*/
    fflush(stdout);
    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            mean = 0.0;
            for(file_nb = 0; file_nb < Nb_files; file_nb++)
            {
                mean += data.image[IDn[file_nb]].array.F[jj * naxes[0] + ii];
            }
            mean /= Nb_files;
            data.image[IDoutm].array.F[jj * naxes[0] + ii] = mean;
            rms                                            = 0.0;
            for(file_nb = 0; file_nb < Nb_files; file_nb++)
            {
                tmp = (mean -
                       data.image[IDn[file_nb]].array.F[jj * naxes[0] + ii]);
                rms += tmp * tmp;
            }
            rms = sqrt(rms / Nb_files);
            data.image[IDoutv].array.F[jj * naxes[0] + ii] = rms;
        }

    free(IDn);

    return RETURN_SUCCESS;
}

imageID combine_2psf(const char *ID_name,
                     const char *ID_name1,
                     const char *ID_name2,
                     float       radius,
                     float       index)
{
    imageID  ID1;
    imageID  ID2;
    imageID  IDout;
    uint32_t naxes[2];
    float    dist;

    ID1      = image_ID(ID_name1);
    ID2      = image_ID(ID_name2);
    naxes[0] = data.image[ID1].md[0].size[0];
    naxes[1] = data.image[ID1].md[0].size[1];
    create_2Dimage_ID(ID_name, naxes[0], naxes[1], &IDout);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dist = sqrt((ii - naxes[0] / 2) * (ii - naxes[0] / 2) +
                        (jj - naxes[1] / 2) * (jj - naxes[1] / 2));
            data.image[IDout].array.F[jj * naxes[0] + ii] =
                exp(-pow(dist / radius, index)) *
                data.image[ID1].array.F[jj * naxes[0] + ii] +
                (1.0 - exp(-pow(dist / radius, index))) *
                data.image[ID2].array.F[jj * naxes[0] + ii];
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
    if(xcenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ycenter = (double *) malloc(sizeof(double));
    if(ycenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    xcenter[0] = naxes[0] / 2;
    ycenter[0] = naxes[1] / 2;
    box_size   = naxes[0] / 3 - 1;

    /*remove_cosmics(ID_name,"tmpcen");*/
    copy_image_ID(ID_name, "tmpcen", 0);
    background = arith_image_percentile("tmpcen", 0.5);

    arith_image_trunc("tmpcen",
                      arith_image_percentile("tmpcen", 0.99),
                      arith_image_percentile("tmpcen", 1.0),
                      "tmpcen1");
    delete_image_ID("tmpcen", DELETE_IMAGE_ERRMODE_WARNING);

    center_PSF("tmpcen1", xcenter, ycenter, box_size);
    delete_image_ID("tmpcen1", DELETE_IMAGE_ERRMODE_WARNING);

    printf("center : %f %f\n", xcenter[0], ycenter[0]);

    if((xcenter[0] < Csize / 2 + 1) ||
            (xcenter[0] > naxes[0] - Csize / 2 - 1) ||
            (ycenter[0] < Csize / 2 + 1) || (ycenter[0] > naxes[1] - Csize / 2 - 1))
    {
        printf("PSF too close to edge of image - cannot measure SR\n");
        SR = -1;
    }
    else
    {
        arith_image_extract2D(ID_name,
                              "tmpsr",
                              Csize,
                              Csize,
                              ((long) xcenter[0]) - Csize / 2,
                              ((long) ycenter[0]) - Csize / 2);
        ID   = image_ID("tmpsr");
        peak = 0.0;
        for(ii = Csize / 2 - 5; ii < Csize / 2 + 5; ii++)
            for(jj = Csize / 2 - 5; jj < Csize / 2 + 5; jj++)
            {
                tmp1 = data.image[ID].array.F[jj * Csize + ii];
                if(tmp1 > peak)
                {
                    peak   = tmp1;
                    peakii = ii;
                    peakjj = jj;
                }
            }
        for(ii = 0; ii < Csize; ii++)
            for(jj = 0; jj < Csize; jj++)
                if(data.image[ID].array.F[jj * Csize + ii] > peak * 1.001)
                {
                    data.image[ID].array.F[jj * Csize + ii] = 0.0;
                }

        fftzoom("tmpsr", "tmpsrz", fzoomfactor);
        ID = image_ID("tmpsrz");
        peakii *= fzoomfactor;
        peakjj *= fzoomfactor;
        total1 = 0.0;
        total2 = 0.0;
        cnt1   = 0;
        cnt2   = 0;
        for(ii = 0; ii < Csize2; ii++)
            for(jj = 0; jj < Csize2; jj++)
            {
                dist = sqrt((peakii - ii) * (peakii - ii) +
                            (peakjj - jj) * (peakjj - jj));
                if(dist < r2 * fzoomfactor)
                {
                    if(dist < r1 * fzoomfactor)
                    {
                        total1 += data.image[ID].array.F[jj * Csize2 + ii];
                        cnt1++;
                    }
                    else
                    {
                        total2 += data.image[ID].array.F[jj * Csize2 + ii];
                        cnt2++;
                    }
                }
            }
        background = total2 / cnt2;
        total      = total1 - background * cnt1;
        max        = arith_image_max("tmpsrz");

        printf("background = %f\n", background);
        printf("max   = %f  (%f)\n", max, max * fzoomfactor * fzoomfactor);
        printf("total = %f (%f[%ld] %f[%ld])\n",
               total,
               total1,
               cnt1,
               total2,
               cnt2);

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
imageID
PSF_coaddbest(const char *IDcin_name, const char *IDout_name, float r_pix)
{
    imageID  IDcin, IDout;
    imageID  IDmask;
    uint32_t xsize, ysize, ksize;
    long     ii, kk, kk1;
    double  *flux_array;
    long    *imgindex;

    IDcin = image_ID(IDcin_name);
    xsize = data.image[IDcin].md[0].size[0];
    ysize = data.image[IDcin].md[0].size[1];
    ksize = data.image[IDcin].md[0].size[2];

    //  printf("\"%s\" %ld SIZE = %ld %ld\n",IDcin_name, IDcin, xsize,ysize);

    flux_array = (double *) malloc(sizeof(double) * ksize);
    if(flux_array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imgindex = (long *) malloc(sizeof(long) * ksize);
    if(imgindex == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IDmask =
        make_subpixdisk("tmpMask", xsize, ysize, xsize / 2, ysize / 2, r_pix);

    for(kk = 0; kk < ksize; kk++)
    {
        imgindex[kk]   = kk;
        flux_array[kk] = 0.0;
        for(ii = 0; ii < xsize * ysize; ii++)
        {
            flux_array[kk] -=
                data.image[IDcin].array.F[kk * xsize * ysize + ii] *
                data.image[IDmask].array.F[ii];
        }
    }

    delete_image_ID("tmpMask", DELETE_IMAGE_ERRMODE_WARNING);

    quick_sort2l(flux_array, imgindex, ksize);

    create_3Dimage_ID(IDout_name, xsize, ysize, ksize, &IDout);

    for(kk = 0; kk < ksize; kk++)
    {
        kk1 = imgindex[kk];
        for(ii = 0; ii < xsize * ysize; ii++)
        {
            data.image[IDout].array.F[kk * xsize * ysize + ii] =
                data.image[IDcin].array.F[kk1 * xsize * ysize + ii];
        }
        if(kk > 0)
            for(ii = 0; ii < xsize * ysize; ii++)
            {
                data.image[IDout].array.F[kk * xsize * ysize + ii] +=
                    data.image[IDout].array.F[(kk - 1) * xsize * ysize + ii];
            }
    }

    free(imgindex);
    free(flux_array);

    return IDout;
}

//
// if timing file exists, use it for output
// PSFsizeEst: estimated size of PSF (sigma)
//
errno_t PSF_sequence_measure(const char *IDin_name,
                             float       PSFsizeEst,
                             const char *outfname)
{
    imageID     IDin;
    uint32_t    xsize, ysize, xysize, zsize;
    FILE       *fpout;
    imageID     IDtmp;
    double     *xcenter;
    double     *ycenter;
    long        boxsize;
    const char *ptr;
    long        kk;

    boxsize = (long)(2.0 * PSFsizeEst);
    printf("box size : %f -> %ld\n", PSFsizeEst, boxsize);

    xcenter = (double *) malloc(sizeof(double));
    if(xcenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ycenter = (double *) malloc(sizeof(double));
    if(ycenter == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IDin   = image_ID(IDin_name);
    xsize  = data.image[IDin].md[0].size[0];
    ysize  = data.image[IDin].md[0].size[1];
    xysize = xsize * ysize;
    if(data.image[IDin].md[0].naxis == 3)
    {
        zsize = data.image[IDin].md[0].size[2];
    }
    else
    {
        zsize = 1;
    }

    create_2Dimage_ID("_tmppsfim", xsize, ysize, &IDtmp);

    fpout = fopen(outfname, "w");
    for(kk = 0; kk < zsize; kk++)
    {
        ptr = (char *) data.image[IDin].array.F;
        ptr += sizeof(float) * xysize * kk;
        memcpy((void *) data.image[IDtmp].array.F,
               (void *) ptr,
               sizeof(float) * xysize);
        fast_center_PSF("_tmppsfim", xcenter, ycenter, boxsize);
        printf("%5ld   CENTER = %f %f\n", kk, xcenter[0], ycenter[0]);
        fprintf(fpout, "%ld %20f %20f\n", kk, xcenter[0], ycenter[0]);

        //	sprintf(fname, "_tmppsfim_%04ld.fits", kk);
        //	save_fits("_tmppsfim", fname);
    }
    fclose(fpout);

    free(xcenter);
    free(ycenter);

    return RETURN_SUCCESS;
}

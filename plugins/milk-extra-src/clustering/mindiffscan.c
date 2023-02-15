#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_tools/COREMOD_tools.h"

static char     *farg_inimname;
static char     *farg_outdname;
static uint32_t *farg_kNNsize;

// List of arguments to function
//
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in_name",
        "input image cube",
        "imc1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &farg_inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outdname",
        "output directory name",
        "outd",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &farg_outdname,
        NULL
    },
    {
        CLIARG_INT64,
        ".kNNsize",
        "number of samples in cluster",
        "20",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &farg_kNNsize,
        NULL
    }
};

// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "mindiffscan",                       // keyword to call function in CLI
    "scan image cube for similar pairs", // description of what the function does
    CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("find nearest neighbors\n");

    return RETURN_SUCCESS;
}

static errno_t
imcube_mindiffscan(IMGID img, const char *__restrict outdname, uint32_t kNNsize)
{
    // entering function, updating trace accordingly
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s", outdname);

    resolveIMGID(&img, ERRMODE_ABORT);

    uint32_t xsize = img.md->size[0];
    uint32_t ysize = img.md->size[1];
    uint32_t zsize = img.md->size[2];

    uint64_t xysize = xsize;
    xysize *= ysize;

    if(zsize == 0)
    {
        // if 2D image, assume ysize is number of samples
        xysize = xsize;
        zsize  = ysize;
    }

    printf("image size %u %u %u\n", xsize, ysize, zsize);

    // FLUX MINIMIZATION MODE: ACTIVE PIXELS
    long    fluxpixcnt = 0;
    imageID IDmaskflux = image_ID("maskfluxim");
    long   *fluxpix    = NULL;
    if(IDmaskflux != -1)
    {
        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            if(data.image[IDmaskflux].array.F[ii] > 0.5)
            {
                fluxpixcnt++;
            }
        }
        fluxpix = (long *) malloc(sizeof(long) * fluxpixcnt);

        fluxpixcnt = 0;
        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            if(data.image[IDmaskflux].array.F[ii] > 0.5)
            {
                fluxpix[fluxpixcnt] = ii;
                fluxpixcnt++;
            }
        }
    }

    // looking for selection mask image
    imageID IDmask = image_ID("maskim");
    if(IDmask == -1)
    {
        printf("Creating default mask image %ld pixel\n", xysize);
        create_2Dimage_ID("maskim", xsize, ysize, &IDmask);

        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            data.image[IDmask].array.F[ii] = 1.0;
        }
    }
    else
    {
        printf("Mask image loaded\n");
    }

    // build pixmap to load input images in vectors
    float maskeps = 0.01; // threshold below which pixels are ignored
    long  pixcnt  = 0;
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        if(data.image[IDmask].array.F[ii] > maskeps)
        {
            pixcnt++;
        }
    }
    long npix = pixcnt;
    DEBUG_TRACEPOINT("npix = %ld", npix);

    long *pixmap = (long *) malloc(sizeof(long) * npix);
    if(pixmap == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }
    double *pixgain = (double *) malloc(sizeof(double) * npix);
    if(pixgain == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    long inpixindex = 0;
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        if(data.image[IDmask].array.F[ii] > maskeps)
        {
            pixmap[inpixindex]  = ii;
            pixgain[inpixindex] = data.image[IDmask].array.F[ii];
            inpixindex++;
        }
    }

    // prepare input compact image

    imageID IDc; // compact image
    create_2Dimage_ID("mindiffscan_imc", npix, zsize, &IDc);
    for(long zi = 0; zi < zsize; zi++)
    {
        for(long ii = 0; ii < npix; ii++)
        {
            data.image[IDc].array.F[zi * npix + ii] =
                pixgain[ii] * img.im->array.F[zi * xysize + pixmap[ii]];
        }
    }

    save_fl_fits("mindiffscan_imc", "mindiffscan_imc.fits");

    // looking for distmat image
    imageID IDdmat = image_ID("distmat");
    if(IDdmat == -1)
    {
        printf("Computing distmat");

        create_2Dimage_ID("distmat", zsize, zsize / 2, &IDdmat);

        printf("\n\n");
        long  diffcnt  = 0;
        float fracdone = 0.0; // fraction completed

        float deltasave       = 0.02; // save every frac
        float fracdonesavelim = deltasave;
        for(long zi0 = 0; zi0 < zsize; zi0++)
        {
            for(long zi1 = zi0 + 1; zi1 < zsize; zi1++)
            {
                long double dist2 = 0.0;
                for(long ii = 0; ii < npix; ii++)
                {
                    float v0 = data.image[IDc].array.F[zi0 * npix + ii];
                    float v1 = data.image[IDc].array.F[zi1 * npix + ii];
                    float dv = v0 - v1;
                    dist2 += dv * dv;
                }

                long zi0p = zi0;
                long zi1p = zi1;

                if(zi0p >= zsize / 2)
                {
                    zi0p = zsize - zi0p - 1;
                    zi1p = zsize - zi1p - 1;
                }

                //data.image[IDdmat].array.F[zi0p*zsize + zi1p] = (float) dist2;
                data.image[IDdmat].array.F[zi0p * zsize + zi1p] = (float) dist2;

                diffcnt++;
            }
            fracdone = 1.0 * diffcnt / (zsize * (zsize - 1) / 2);
            printf("diffcnt = %8ld  %5.3f %% done   \r",
                   diffcnt,
                   100.0 * fracdone);

            if(fracdone > fracdonesavelim)
            {
                printf("\nsaving to filesystem\n");
                save_fl_fits("distmat", "distmat.fits");
                fracdonesavelim += deltasave;
            }
        }
        printf("\n\n");

        save_fl_fits("distmat", "distmat.fits");
    }
    else
    {
        printf("distmat image loaded\n");
    }
    free(pixmap);
    free(pixgain);

    // identify k-NN for each entry
    long kN = kNNsize;

    long   kNdistbesti   = -1;
    double kNdistbestval = 0.0;

    double *distarray = (double *) malloc(sizeof(double) * zsize);
    long   *iarray    = (long *) malloc(sizeof(long) * zsize);

    double *distarray_zbest = (double *) malloc(sizeof(double) * kN);
    long   *iarray_zbest    = (long *) malloc(sizeof(long) * kN);

    char fnamelog[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fnamelog, "bkNN.%ld.log", kN);
    FILE *fp = fopen(fnamelog, "w");

    long zibest = 0;
    for(long zi0 = 0; zi0 < zsize; zi0++)
    {
        unsigned long cnt = 0;

        for(long zi1 = 0; zi1 < zsize; zi1++)
        {
            double zdist = 1.0 * (zi0 - zi1);
            if(fabs(zdist) > 0)  // 0.1*zsize)
            {
                long zi0p = zi0;
                long zi1p = zi1;

                if(zi0p > zi1p)
                {
                    long ztmp = zi0p;
                    zi0p      = zi1p;
                    zi1p      = ztmp;
                }

                if(zi0p >= zsize / 2)
                {
                    zi0p = zsize - zi0p - 1;
                    zi1p = zsize - zi1p - 1;
                }

                distarray[cnt] =
                    data.image[IDdmat].array.F[zi0p * zsize + zi1p];
                iarray[cnt] = zi1;
                cnt++;
            }
        }
        quick_sort2l(distarray, iarray, cnt);
        double kNdistval = distarray[kN];

        // avoid duplicates
        long offsetkN = 0;
        while(kNdistval < 1.0e-8)
        {
            offsetkN++;
            kNdistval = distarray[kN];
        }

        fprintf(fp, "%5ld  %20g", zi0, kNdistval);
        long double fluxtot = 0.0;
        if(fluxpixcnt > 0)
        {
            for(long k = 0; k < kN; k++)
            {
                for(long ii = 0; ii < fluxpixcnt; ii++)
                {
                    fluxtot += data.image[img.ID]
                               .array.F[xysize * iarray[k] + fluxpix[ii]];
                }
            }
            kNdistval = 1.0 * log10(kNdistval) + 0.8 * log10(fluxtot);
            fprintf(fp, "   %20g    %20g", (double) fluxtot, kNdistval);
        }
        else
        {
            fluxtot = 1.0;
        }

        fprintf(fp, "\n");

        int bestupdate = 0;
        if(kNdistbesti == -1)
        {
            bestupdate    = 1;
            kNdistbesti   = zi0;
            kNdistbestval = kNdistval;
            printf("INIT: frame %5ld   %ld-NN dist = %g\n",
                   zi0,
                   kN,
                   kNdistbestval);
        }
        else
        {
            if(kNdistval < kNdistbestval)
            {
                bestupdate    = 1;
                kNdistbesti   = zi0;
                kNdistbestval = kNdistval;

                printf("BEST: frame %5ld   %ld-NN dist = %g\n",
                       zi0,
                       kN,
                       kNdistbestval);
            }
        }
        if(bestupdate == 1)
        {
            zibest = zi0;
            printf("NN list for frame %ld:\n", zibest);
            for(long k = 0; k < kN; k++)
            {
                distarray_zbest[k] = distarray[k];
                iarray_zbest[k]    = iarray[k];
                printf("    %ld    %3ld  %5ld  %g\n",
                       zibest,
                       k,
                       iarray[k],
                       distarray[k]);
            }
            printf("\n");
        }
    }
    fclose(fp);

    // Write output to data cube
    char fnamebNN[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fnamebNN, "bkNN.%ld.cluster.log", kN);
    FILE *fpb = fopen(fnamebNN, "w");

    imageID IDbc; // best cluster
    create_3Dimage_ID("bkNNcube", xsize, ysize, kN, &IDbc);
    for(long k = 0; k < kN; k++)
    {
        printf("    %3ld selecting frame %5ld   [%u %u %ld] offset %ld\n",
               k,
               iarray_zbest[k],
               xsize,
               ysize,
               k,
               k * xysize);

        fprintf(fpb,
                "%ld   %ld     %g\n",
                k,
                iarray_zbest[k],
                distarray_zbest[k]);
        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            data.image[IDbc].array.F[k * xysize + ii] =
                data.image[img.ID].array.F[xysize * iarray_zbest[k] + ii];
        }
    }
    fclose(fpb);
    list_image_ID();

    free(distarray);
    free(iarray);

    free(distarray_zbest);
    free(iarray_zbest);

    if(fluxpix != NULL)
    {
        free(fluxpix);
    }

    // normal successful return from function :
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// Wrapper function, used by all CLI calls
// Defines how local variables are fed to computation code
// Always local to this translation unit
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    imcube_mindiffscan(mkIMGID_from_name(farg_inimname),
                       farg_outdname,
                       *farg_kNNsize);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

/** @brief Register CLI command
*
* Adds function to list of CLI commands.
* Called by main module initialization function init_module_CLI().
*/
errno_t
CLIADDCMD_clustering__imcube_mindiffscan()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

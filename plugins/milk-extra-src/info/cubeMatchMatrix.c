/**
 * @file cubeMatchMatrix.c
 * @brief Compute sqsum diffs between cube slices
 */

#include <math.h>

#include "CLIcore.h"
#include "fps.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

// Forward declaration
imageID info_cubeMatchMatrix(
    const char *IDin_name,
    const char *IDout_name);

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "incube";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outim";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "cubeslmatch",
    .cmdkey      = "cubeslmatch",
    .description =
        "compute sqsum diffs between slices",
    .description_long =
        "Compute a matrix of sum-of-squared differences between all pairs of slices in an image cube. Used for similarity analysis."
};

#define FPS_PARAMS(X) \
    X(".in_name", p_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image cube") \
    X(".out_name", p_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};
static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static MILK_HOT errno_t compute_function()
{
    info_cubeMatchMatrix(p_in, p_out);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_info__cubeMatchMatrix()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

imageID info_cubeMatchMatrix(const char *IDin_name, const char *IDout_name)
{
    imageID  IDout;
    imageID  IDin;
    uint32_t xsize, ysize, zsize;
    uint64_t xysize;

    long        kk1, kk2;
    long double totv;
    long double v;
    double      v1, v2;

    FILE *fpout;

    uint32_t ksize;
    double  *array_matchV;
    long    *array_matchii;
    long    *array_matchjj;

    imageID ID0;
    long    k;
    // float        zfrac = 0.01;
    imageID IDrmsim;

    long kdiffmin = 995;
    long kdiffmax = 1005;
    long kmax     = 10;

    IDin   = image_ID(IDin_name, dcimg, dcnimg);
    xsize  = dcimg[IDin].md[0].size[0];
    ysize  = dcimg[IDin].md[0].size[1];
    zsize  = dcimg[IDin].md[0].size[2];
    xysize = xsize * ysize;

    IDout = image_ID(IDout_name, dcimg, dcnimg);

    if(IDout == -1)
    {
        create_2Dimage_ID(IDout_name, zsize, zsize, &IDout);

        fpout = fopen("outtest.txt", "w");
        fclose(fpout);

        printf("Computing differences - cube size is %u %u   %lu\n",
               zsize,
               zsize,
               xysize);
        printf("\n\n");
        for(kk1 = 0; kk1 < zsize; kk1++)
        {
            printf("%4ld / %4u    \n", kk1, zsize);
            fflush(stdout);
            for(kk2 = kk1 + 1; kk2 < zsize; kk2++)
            {
                totv = 0.0;
                for(unsigned long ii = 0; ii < xysize; ii++)
                {
                    v1 = (double) dcimg[IDin].array.F[kk1 * xysize + ii];
                    v2 = (double) dcimg[IDin].array.F[kk2 * xysize + ii];
                    v  = v1 - v2;
                    totv += v * v;
                    //						printf("   %5ld
                    //%20f
                    //-> %g\n", ii, (double) v, (double) totv);
                }
                printf("    %4ld %4ld   %g\n", kk1, kk2, (double) totv);

                fpout = fopen("outtest.txt", "a");
                fprintf(fpout,
                        "%5ld  %20f  %5ld %5ld\n",
                        kk2 - kk1,
                        (double) totv,
                        kk1,
                        kk2);
                fclose(fpout);

                dcimg[IDout].array.F[kk2 * zsize + kk1] = (float) totv;
            }

            save_fits(IDout_name, "testout.fits");
        }
        printf("\n");
    }
    else
    {
        zsize = dcimg[IDout].md[0].size[0];
    }

    ksize         = (zsize - 1) * (zsize) / 2;
    array_matchV  = (double *) malloc(sizeof(double) * ksize);
    array_matchii = (long *) malloc(sizeof(long) * ksize);
    array_matchjj = (long *) malloc(sizeof(long) * ksize);

    list_image_ID();
    printf("Reading %u pixels from ID = %ld\n",
           (zsize - 1) * (zsize) / 2,
           IDout);

    unsigned long ii = 0;

    for(kk1 = 0; kk1 < zsize; kk1++)
        for(kk2 = kk1 + 1; kk2 < zsize; kk2++)
        {
            if(ii > (unsigned long)(ksize - 1))
            {
                printf("ERROR: %ld %ld  %ld / %u\n", kk1, kk2, ii, ksize);
                exit(0);
            }
            if(((double) dcimg[IDout].array.F[kk2 * zsize + kk1] > 1.0f) &&
                    (kk2 - kk1 > kdiffmin) && (kk2 - kk1 < kdiffmax))
            {
                array_matchV[ii] =
                    (double) dcimg[IDout].array.F[kk2 * zsize + kk1];
                array_matchii[ii] = kk1;
                array_matchjj[ii] = kk2;
                ii++;
            }
        }
    ksize = ii;

    fpout = fopen("outtest.unsorted.txt", "w");
    for(ii = 0; ii < ksize; ii++)
    {
        fprintf(fpout,
                "%5ld  %5ld  %+5ld   %g\n",
                array_matchii[ii],
                array_matchjj[ii],
                array_matchjj[ii] - array_matchii[ii],
                array_matchV[ii]);
    }
    fclose(fpout);

    quick_sort3ll_double(array_matchV, array_matchii, array_matchjj, ksize);

    fpout = fopen("outtest.sorted.txt", "w");
    for(ii = 0; ii < ksize; ii++)
    {
        fprintf(fpout,
                "%5ld  %5ld  %+5ld   %g\n",
                array_matchii[ii],
                array_matchjj[ii],
                array_matchjj[ii] - array_matchii[ii],
                array_matchV[ii]);
    }
    fclose(fpout);

    ID0    = image_ID("imcfull", dcimg, dcnimg);
    xsize  = dcimg[ID0].md[0].size[0];
    ysize  = dcimg[ID0].md[0].size[1];
    xysize = xsize * ysize;

    if(ID0 != -1)
    {
        printf("PROCESSING IMAGE  %ld pixels\n", xysize);

        create_2Dimage_ID("imRMS", xsize, ysize, &IDrmsim);
        // kmax = (long) (zfrac*ksize);
        printf("KEEPING %ld out of %u pairs\n", kmax, ksize);

        for(k = 0; k < kmax; k++)
        {
            kk1 = array_matchii[k];
            kk2 = array_matchjj[k];
            for(unsigned long ii = 0; ii < xysize; ii++)
            {
                v1 = dcimg[ID0].array.F[kk1 * xysize + ii];
                v2 = dcimg[ID0].array.F[kk2 * xysize + ii];
                v  = v1 - v2;
                dcimg[IDrmsim].array.F[ii] += v * v;
            }
        }
        for(unsigned long ii = 0; ii < xysize; ii++)
        {
            dcimg[IDrmsim].array.F[ii] =
                sqrt(dcimg[IDrmsim].array.F[ii] / kmax);
        }
        save_fits("imRMS", "imRMS.fits");
    }

    free(array_matchV);
    free(array_matchii);
    free(array_matchjj);

    return (IDout);
}

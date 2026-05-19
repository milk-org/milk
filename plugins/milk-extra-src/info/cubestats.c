/**
 * @file cubestats.c
 * @brief Image cube stats
 */

#include <math.h>

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID info_cubestats(
    const char *ID_name,
    const char *IDmask_name,
    const char *outfname);

static char p_cube[FUNCTION_PARAMETER_STRMAXLEN]
    = "imc";
static char p_mask[FUNCTION_PARAMETER_STRMAXLEN]
    = "immask";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "imc_stats.txt";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "cubestats",
    .cmdkey      = "cubestats",
    .description = "image cube stats",
    .description_long =
        "Compute per-slice statistics (mean, rms, min, max) of a 3D image cube. Outputs a summary table."
};

#define FPS_PARAMS(X) \
    X(".in_cube", p_cube, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input 3D image") \
    X(".in_mask", p_mask, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "mask image") \
    X(".out_fname", p_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output file")

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
    info_cubestats(p_cube, p_mask, p_out);
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
CLIADDCMD_info__cubestats()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

// mask pixel values are 0 or 1
// prints:
//		index
//		min
//		max
//		total
//		average
//		tot power
//		RMS
imageID info_cubestats(const char *ID_name,
                       const char *IDmask_name,
                       const char *outfname)
{
    imageID  ID, IDm;
    float    min, max, tot, tot2;
    uint64_t xysize;
    FILE    *fp;
    int      init = 0;
    float    mtot;
    float    val;

    int    COMPUTE_CORR = 1;
    long   kcmax        = 100;
    double valn1, valn2, v1, v2, valxp, vcorr;
    long   k1, k2, kc;

    ID = image_ID(ID_name, dcimg, dcnimg);
    if(dcimg[ID].md[0].naxis != 3)
    {
        printf("ERROR: info_cubestats requires 3D image\n");
        exit(0);
    }

    IDm = image_ID(IDmask_name, dcimg, dcnimg);

    xysize = dcimg[ID].md[0].size[0] * dcimg[ID].md[0].size[1];

    mtot = 0.0;
    for(unsigned long ii = 0; ii < xysize; ii++)
    {
        mtot += dcimg[IDm].array.F[ii];
    }

    fp = fopen(outfname, "w");
    for(unsigned long kk = 0; kk < dcimg[ID].md[0].size[2]; kk++)
    {
        init = 0;
        tot  = 0.0;
        tot2 = 0.0;
        for(unsigned long ii = 0; ii < xysize; ii++)
        {
            if(dcimg[IDm].array.F[ii] > 0.5f)
            {
                val = dcimg[ID].array.F[kk * xysize + ii];
                if(init == 0)
                {
                    init = 1;
                    min  = val;
                    max  = val;
                }
                if(val > max)
                {
                    max = val;
                }
                if(val < min)
                {
                    min = val;
                }
                tot += val;
                tot2 += val * val;
            }
        }
        fprintf(fp,
                "%5ld  %20f  %20f  %20f  %20f  %20f  %20f\n",
                kk,
                min,
                max,
                tot,
                tot / mtot,
                tot2,
                sqrt((tot2 - tot * tot / mtot) / mtot));
    }
    fclose(fp);

    if(COMPUTE_CORR == 1)
    {
        fp = fopen("corr.txt", "w");
        for(kc = 1; kc < kcmax; kc++)
        {
            vcorr = 0.0;
            for(unsigned long kk = 0;
                    kk < (unsigned long)(dcimg[ID].md[0].size[2] - kc);
                    kk++)
            {
                k1    = kk;
                k2    = kk + kc;
                valn1 = 0.0;
                valn2 = 0.0;
                valxp = 0.0;
                for(unsigned long ii = 0; ii < xysize; ii++)
                {
                    if(dcimg[IDm].array.F[ii] > 0.5f)
                    {
                        v1 = dcimg[ID].array.F[k1 * xysize + ii];
                        v2 = dcimg[ID].array.F[k2 * xysize + ii];
                        valn1 += v1 * v1;
                        valn2 += v2 * v2;
                        valxp += v1 * v2;
                    }
                }
                vcorr += valxp / sqrt(valn1 * valn2);
            }
            vcorr /= dcimg[ID].md[0].size[2] - kc;
            fprintf(fp, "%3ld   %g\n", kc, vcorr);
        }
        fclose(fp);
    }

    return (ID);
}

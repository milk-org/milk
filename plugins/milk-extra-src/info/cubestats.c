/** @file cubestats.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"


// ==========================================
// Forward declaration(s)
// ==========================================

imageID info_cubestats(const char *ID_name,
                       const char *IDmask_name,
                       const char *outfname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t info_cubestats_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_IMG) +
            CLI_checkarg(3, CLIARG_STR_NOT_IMG) ==
            0)
    {
        info_cubestats(data.cmdargtoken[1].val.string,
                       data.cmdargtoken[2].val.string,
                       data.cmdargtoken[3].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t cubestats_addCLIcmd()
{
    RegisterCLIcommand("cubestats",
                       __FILE__,
                       info_cubestats_cli,
                       "image cube stats",
                       "<3Dimage> <mask> <output file>",
                       "cubestats imc immask imc_stats.txt",
                       "long info_cubestats(const char *ID_name, const char "
                       "*IDmask_name, const char *outfname)");

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

    ID = image_ID(ID_name);
    if(data.image[ID].md[0].naxis != 3)
    {
        printf("ERROR: info_cubestats requires 3D image\n");
        exit(0);
    }

    IDm = image_ID(IDmask_name);

    xysize = data.image[ID].md[0].size[0] * data.image[ID].md[0].size[1];

    mtot = 0.0;
    for(unsigned long ii = 0; ii < xysize; ii++)
    {
        mtot += data.image[IDm].array.F[ii];
    }

    fp = fopen(outfname, "w");
    for(unsigned long kk = 0; kk < data.image[ID].md[0].size[2]; kk++)
    {
        init = 0;
        tot  = 0.0;
        tot2 = 0.0;
        for(unsigned long ii = 0; ii < xysize; ii++)
        {
            if(data.image[IDm].array.F[ii] > 0.5)
            {
                val = data.image[ID].array.F[kk * xysize + ii];
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
                    kk < (unsigned long)(data.image[ID].md[0].size[2] - kc);
                    kk++)
            {
                k1    = kk;
                k2    = kk + kc;
                valn1 = 0.0;
                valn2 = 0.0;
                valxp = 0.0;
                for(unsigned long ii = 0; ii < xysize; ii++)
                {
                    if(data.image[IDm].array.F[ii] > 0.5)
                    {
                        v1 = data.image[ID].array.F[k1 * xysize + ii];
                        v2 = data.image[ID].array.F[k2 * xysize + ii];
                        valn1 += v1 * v1;
                        valn2 += v2 * v2;
                        valxp += v1 * v2;
                    }
                }
                vcorr += valxp / sqrt(valn1 * valn2);
            }
            vcorr /= data.image[ID].md[0].size[2] - kc;
            fprintf(fp, "%3ld   %g\n", kc, vcorr);
        }
        fclose(fp);
    }

    return (ID);
}

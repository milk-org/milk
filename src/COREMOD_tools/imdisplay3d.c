/**
 * @file imdisplay3d.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

static FILE *fpgnuplot;

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t COREMOD_TOOLS_imgdisplay3D(const char *IDname, long step);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t COREMOD_TOOLS_imgdisplay3D_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_LONG) == 0)
    {
        COREMOD_TOOLS_imgdisplay3D(data.cmdargtoken[1].val.string,
                                   data.cmdargtoken[2].val.numl);

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

errno_t imdisplay3d_addCLIcmd()
{
    RegisterCLIcommand(
        "dispim3d",
        __FILE__,
        COREMOD_TOOLS_imgdisplay3D_cli,
        "display 2D image as 3D surface using gnuplot",
        "<imname> <step>",
        "dispim3d im1 5",
        "int COREMOD_TOOLS_imgdisplay3D(const char *IDname, long step)");

    return RETURN_SUCCESS;
}

// displays 2D image in 3D using gnuplot
//
errno_t COREMOD_TOOLS_imgdisplay3D(const char *IDname, long step)
{
    imageID ID;
    long    xsize, ysize;
    long    ii, jj;
    char    cmd[512];
    FILE   *fp;

    ID    = image_ID(IDname);
    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];

    snprintf(cmd, 512, "gnuplot");

    if((fpgnuplot = popen(cmd, "w")) == NULL)
    {
        fprintf(stderr, "could not connect to gnuplot\n");
        return -1;
    }

    printf("image: %s [%ld x %ld], step = %ld\n", IDname, xsize, ysize, step);

    fprintf(fpgnuplot, "set pm3d\n");
    fprintf(fpgnuplot, "set hidden3d\n");
    fprintf(fpgnuplot, "set palette\n");
    //fprintf(gnuplot, "set xrange [0:%li]\n", image.md[0].size[0]);
    //fprintf(gnuplot, "set yrange [0:1e-5]\n");
    //fprintf(gnuplot, "set xlabel \"Mode #\"\n");
    //fprintf(gnuplot, "set ylabel \"Mode RMS\"\n");
    fflush(fpgnuplot);

    fp = fopen("pts.dat", "w");
    fprintf(fpgnuplot, "splot \"-\" w d notitle\n");
    for(ii = 0; ii < xsize; ii += step)
    {
        for(jj = 0; jj < xsize; jj += step)
        {
            fprintf(fpgnuplot,
                    "%ld %ld %f\n",
                    ii,
                    jj,
                    data.image[ID].array.F[jj * xsize + ii]);
            fprintf(fp,
                    "%ld %ld %f\n",
                    ii,
                    jj,
                    data.image[ID].array.F[jj * xsize + ii]);
        }
        fprintf(fpgnuplot, "\n");
        fprintf(fp, "\n");
    }
    fprintf(fpgnuplot, "e\n");
    fflush(fpgnuplot);
    fclose(fp);

    return RETURN_SUCCESS;
}

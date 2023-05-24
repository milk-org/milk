/**
 * @file    linARfilterPred.c
 * @brief   linear auto-regressive predictive filter
 *
 * Implements Empirical Orthogonal Functions
 *
 *
 */

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "larpf"

// Module short description
#define MODULE_DESCRIPTION "Linear auto-regressive predictive filters"

#include <assert.h>
#include <ctype.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <fitsio.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <time.h>

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "info/info.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"

#include "linARfilterPred/linARfilterPred.h"

#include "build_linPF.h"
#include "applyPF.h"



#ifdef HAVE_CUDA
#include "cudacomp/cudacomp.h"
#endif

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(linARfilterPred)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

errno_t LINARFILTERPRED_LoadASCIIfiles_cli()
{
    if(CLI_checkarg(1, 1) + CLI_checkarg(2, 1) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 2) + CLI_checkarg(5, 5) ==
            0)
    {
        LINARFILTERPRED_LoadASCIIfiles(data.cmdargtoken[1].val.numf,
                                       data.cmdargtoken[2].val.numf,
                                       data.cmdargtoken[3].val.numl,
                                       data.cmdargtoken[4].val.numl,
                                       data.cmdargtoken[5].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t LINARFILTERPRED_SelectBlock_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 3) ==
            0)
    {
        LINARFILTERPRED_SelectBlock(data.cmdargtoken[1].val.string,
                                    data.cmdargtoken[2].val.string,
                                    data.cmdargtoken[3].val.numl,
                                    data.cmdargtoken[4].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t linARfilterPred_repeat_shift_X_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 2) + CLI_checkarg(3, 3) == 0)
    {
        linARfilterPred_repeat_shift_X(data.cmdargtoken[1].val.string,
                                       data.cmdargtoken[2].val.numl,
                                       data.cmdargtoken[3].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t LINARFILTERPRED_Build_LinPredictor_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 2) + CLI_checkarg(3, 1) +
            CLI_checkarg(4, 1) + CLI_checkarg(5, 1) + CLI_checkarg(6, 3) +
            CLI_checkarg(7, 2) + CLI_checkarg(8, 1) + CLI_checkarg(9, 2) ==
            0)
    {
        LINARFILTERPRED_Build_LinPredictor(data.cmdargtoken[1].val.string,
                                           data.cmdargtoken[2].val.numl,
                                           data.cmdargtoken[3].val.numf,
                                           data.cmdargtoken[4].val.numf,
                                           data.cmdargtoken[5].val.numf,
                                           data.cmdargtoken[6].val.string,
                                           1,
                                           data.cmdargtoken[7].val.numl,
                                           data.cmdargtoken[8].val.numf,
                                           data.cmdargtoken[9].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t LINARFILTERPRED_Apply_LinPredictor_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 1) +
            CLI_checkarg(4, 3) ==
            0)
    {
        LINARFILTERPRED_Apply_LinPredictor(data.cmdargtoken[1].val.string,
                                           data.cmdargtoken[2].val.string,
                                           data.cmdargtoken[3].val.numf,
                                           data.cmdargtoken[4].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t LINARFILTERPRED_ScanGain_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 1) + CLI_checkarg(3, 1) == 0)
    {
        LINARFILTERPRED_ScanGain(data.cmdargtoken[1].val.string,
                                 data.cmdargtoken[2].val.numf,
                                 data.cmdargtoken[3].val.numf);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t LINARFILTERPRED_PF_updatePFmatrix_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 5) + CLI_checkarg(3, 1) == 0)
    {
        LINARFILTERPRED_PF_updatePFmatrix(data.cmdargtoken[1].val.string,
                                          data.cmdargtoken[2].val.string,
                                          data.cmdargtoken[3].val.numf);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t LINARFILTERPRED_PF_RealTimeApply_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 2) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 4) + CLI_checkarg(5, 2) + CLI_checkarg(6, 5) +
            CLI_checkarg(7, 2) + CLI_checkarg(8, 2) + CLI_checkarg(9, 2) +
            CLI_checkarg(10, 2) + CLI_checkarg(11, 1) + CLI_checkarg(12, 2) ==
            0)
    {
        LINARFILTERPRED_PF_RealTimeApply(data.cmdargtoken[1].val.string,
                                         data.cmdargtoken[2].val.numl,
                                         data.cmdargtoken[3].val.numl,
                                         data.cmdargtoken[4].val.string,
                                         data.cmdargtoken[5].val.numl,
                                         data.cmdargtoken[6].val.string,
                                         data.cmdargtoken[7].val.numl,
                                         data.cmdargtoken[8].val.numl,
                                         data.cmdargtoken[9].val.numl,
                                         data.cmdargtoken[10].val.numl,
                                         data.cmdargtoken[11].val.numf,
                                         data.cmdargtoken[12].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t init_module_CLI()
{
    RegisterCLIcommand(
        "pfloadascii",
        __FILE__,
        LINARFILTERPRED_LoadASCIIfiles_cli,
        "load ascii files to PF input",
        "<tstart> <dt> <NBpt> <NBfr> <output>",
        "pfloadascii 200.0 0.001 10000 4 pfin",
        "long LINARFILTERPRED_LoadASCIIfiles(double tstart, double dt, long "
        "NBpt, long NBfr, const char *IDoutname)");

    RegisterCLIcommand(
        "mselblock",
        __FILE__,
        LINARFILTERPRED_SelectBlock_cli,
        "select modes belonging to a block",
        "<input mode values> <block map> <selected block> <output>",
        "mselblock modevals blockmap 23 blk23modevals",
        "long LINARFILTERPRED_SelectBlock(const char *IDin_name, const char "
        "*IDblknb_name, long blkNB, "
        "const char *IDout_name)");

    RegisterCLIcommand("imrepshiftx",
                       __FILE__,
                       linARfilterPred_repeat_shift_X_cli,
                       "repeat and shift image, extend along X axis",
                       "<input image> <NBstep> <output image>",
                       "imrepshiftx imin 5 imout",
                       "long linARfilterPred_repeat_shift_X(const char "
                       "*IDin_name, long NBstep, const char *IDout_name)");

    RegisterCLIcommand(
        "mkARpfilt",
        __FILE__,
        LINARFILTERPRED_Build_LinPredictor_cli,
        "Make linear auto-regressive filter",
        "<input data> <PForder> <PFlag> <SVDeps> <regularization param> "
        "<output filters> <LOOPmode> <LOOPgain> "
        "<testmode>",
        "mkARpfilt indata 5 2.4 0.0001 0.0 outPF 0 0.1 1",
        "int LINARFILTERPRED_Build_LinPredictor(const char *IDin_name, long "
        "PForder, float PFlag, double SVDeps, "
        "double RegLambda, const char *IDoutPF, int outMode, int LOOPmode, "
        "float LOOPgain, int testmode)");

    /*  strcpy(data.cmd[data.NBcmd].key,"applyPfiltRT");
      strcpy(data.cmd[data.NBcmd].module,__FILE__);
      data.cmd[data.NBcmd].fp = LINARFILTERPRED_Apply_LinPredictor_RT_cli;
      strcpy(data.cmd[data.NBcmd].info,"Apply real-time linear predictive filter");
      strcpy(data.cmd[data.NBcmd].syntax,"<input data> <predictor filter> <output>");
      strcpy(data.cmd[data.NBcmd].example,"applyPfiltRT indata Pfilt outPF");
      strcpy(data.cmd[data.NBcmd].Ccall,"long LINARFILTERPRED_Apply_LinPredictor_RT(const char *IDfilt_name, const char *IDin_name, const char *IDout_name)");
      data.NBcmd++;
    */

    RegisterCLIcommand("applyARpfilt",
                       __FILE__,
                       LINARFILTERPRED_Apply_LinPredictor_cli,
                       "Apply linear auto-regressive filter",
                       "<input data> <predictor> <PFlag> <prediction>",
                       "applyARpfilt indata Pfilt 2.4 outPF",
                       "long LINARFILTERPRED_Apply_LinPredictor(const char "
                       "*IDfilt_name, const char *IDin_name, float "
                       "PFlag, const char *IDout_name)");

    RegisterCLIcommand(
        "mscangain",
        __FILE__,
        LINARFILTERPRED_ScanGain_cli,
        "scan gain",
        "<input mode values> <multiplicative factor (leak)> <latency [frame]>",
        "mscangain olwfsmeas 0.98 2.65",
        "LINARFILTERPRED_ScanGain(char* IDin_name, float multfact, float "
        "framelag)");

    RegisterCLIcommand("linARPFMupdate",
                       __FILE__,
                       LINARFILTERPRED_PF_updatePFmatrix_cli,
                       "update predictive filter matrix",
                       "<input 3D predictor> <output 2D matrix> <update coeff>",
                       "linARPFMupdate outPF PFMat 0.1",
                       "long LINARFILTERPRED_PF_updatePFmatrix(const char "
                       "*IDPF_name, const char *IDPFM_name, float alpha)");

    RegisterCLIcommand(
        "linARapplyRT",
        __FILE__,
        LINARFILTERPRED_PF_RealTimeApply_cli,
        "Real-time apply predictive filter",
        "<input open loop coeffs stream> <offset index> <trigger semaphore "
        "index> <2D predictive "
        "matrix> <filter order> <output stream> <nbGPU> <loop> <NBiter> "
        "<savemode> <timelag> <PFindex>",
        "linARapplyRT modevalOL 0 2 PFmat 5 outPFmodeval 0 0 0 0 1.8 0",
        "long LINARFILTERPRED_PF_RealTimeApply(const char *IDmodevalOL_name, "
        "long IndexOffset, int "
        "semtrig, const char *IDPFM_name, long NBPFstep, const char "
        "*IDPFout_name, int nbGPU, long "
        "loop, long NBiter, int SAVEMODE, float tlag, long PFindex)");




    CLIADDCMD_LinARfilterPred__build_linPF();
    CLIADDCMD_LinARfilterPred__applyPF();

    // add atexit functions here

    return RETURN_SUCCESS;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 1. INITIALIZATION                                                                               */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 2. I/O TOOLS                                                                                    */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

int NBwords(const char sentence[])
{
    int counted = 0; // result

    // state:
    const char *it     = sentence;
    int         inword = 0;

    do
        switch(*it)
        {
            case '\0':
            case ' ':
            case '\t':
            case '\n':
            case '\r':
                if(inword)
                {
                    inword = 0;
                    counted++;
                }
                break;
            default:
                inword = 1;
        }
    while(*it++);

    return counted;
}

/**
 * @brief load ascii file(s) into image cube
 *
 *  resamples sequence(s) of data points
 * INPUT FILES HAVE TO BE NAMED seq000.dat, seq001.dat etc...
 *
 * file starts at tstart, sampling = dt
 * NBpt per file
 * NBfr files
*/

long LINARFILTERPRED_LoadASCIIfiles(
    double tstart, double dt, long NBpt, long NBfr, const char *IDoutname)
{
    FILE       *fp;
    long        NBfiles;
    double      runtime;
    char        fname[200];
    struct stat fstat;
    int         fOK;
    long        NBvarin[200];
    long        fcnt;
    FILE       *fparray[200];
    long        kk;
    size_t      linesiz = 0;
    char       *linebuf = 0;
    //ssize_t linelen=0;
    //int     ret;
    long    vcnt;
    double  ftime0[200];
    double  var0[200][200];
    double  ftime1[200];
    double  var1[200][200];
    double  varC[200][200];
    float   alpha;
    long    nbvar;
    long    fr;
    char    imoutname[200];
    FILE   *fpout;
    imageID IDout[200];
    //int     HPfilt = 1; // high pass filter
    float HPgain = 0.005;

    long ii;
    long kkpt, kkfr;

    runtime = tstart;

    fOK     = 1;
    NBfiles = 0;
    nbvar   = 0;
    while(fOK == 1)
    {
        sprintf(fname, "seq%03ld.dat", NBfiles);
        if(stat(fname, &fstat) == 0)
        {
            printf("Found file %s\n", fname);
            fflush(stdout);
            fp = fopen(fname, "r");
            //linelen =
            if(getline(&linebuf, &linesiz, fp) == -1)
            {
                PRINT_ERROR("getline error");
            }
            fclose(fp);
            NBvarin[NBfiles] = NBwords(linebuf) - 1;
            free(linebuf);
            linebuf = NULL;
            printf("   NB variables = %ld\n", NBvarin[NBfiles]);
            nbvar += NBvarin[NBfiles];
            NBfiles++;
        }
        else
        {
            printf("No more files\n");
            fflush(stdout);
            fOK = 0;
        }
    }
    printf("NBfiles = %ld\n", NBfiles);

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        sprintf(fname, "seq%03ld.dat", fcnt);
        printf("   %03ld  OPENING FILE %s\n", fcnt, fname);
        fflush(stdout);
        fparray[fcnt] = fopen(fname, "r");
    }

    kk      = 0; // time
    runtime = tstart;

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        if(fscanf(fparray[fcnt], "%lf", &ftime0[fcnt]) != 1)
        {
            PRINT_ERROR("fscanf error");
        }

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            if(fscanf(fparray[fcnt], "%lf", &var0[fcnt][vcnt]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
        }
        if(fscanf(fparray[fcnt], "\n") != 0)
        {
            PRINT_ERROR("fscanf error");
        }

        if(fscanf(fparray[fcnt], "%lf", &ftime1[fcnt]) != 1)
        {
            PRINT_ERROR("fscanf error");
        }

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            if(fscanf(fparray[fcnt], "%lf", &var1[fcnt][vcnt]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
        }
        if(fscanf(fparray[fcnt], "\n") != 0)
        {
            PRINT_ERROR("fscanf error");
        }

        printf("FILE %ld :  \n", fcnt);
        printf(" time :    %20f  %20f\n", ftime0[fcnt], ftime1[fcnt]);
        fflush(stdout);

        for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
        {
            printf("    variable %3ld   :   %20f  %20f\n",
                   vcnt,
                   var0[fcnt][vcnt],
                   var1[fcnt][vcnt]);
            varC[fcnt][vcnt] = var0[fcnt][vcnt];
        }
        printf("\n");
    }

    for(fr = 0; fr < NBfr; fr++)
    {
        sprintf(imoutname, "%s_%03ld", IDoutname, fr);
        create_3Dimage_ID(imoutname, nbvar, 1, NBpt, &(IDout[fr]));
    }

    fpout = fopen("out.txt", "w");

    kk   = 0;
    kkpt = 0;
    kkfr = 0;
    while(kkfr < NBfr)
    {
        fprintf(fpout, "%20f", runtime);

        ii = 0;
        for(fcnt = 0; fcnt < NBfiles; fcnt++)
        {
            while(ftime1[fcnt] < runtime)
            {
                ftime0[fcnt] = ftime1[fcnt];
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    var0[fcnt][vcnt] = var1[fcnt][vcnt];
                }

                if(fscanf(fparray[fcnt], "%lf", &ftime1[fcnt]) != 1)
                {
                    PRINT_ERROR("fscanf error");
                }
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    if(fscanf(fparray[fcnt], "%lf", &var1[fcnt][vcnt]) != 1)
                    {
                        PRINT_ERROR("fscanf error");
                    }
                }
                if(fscanf(fparray[fcnt], "\n") != 0)
                {
                    PRINT_ERROR("fscanf error");
                }
            }
            if(kk == 0)
                for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
                {
                    varC[fcnt][vcnt] = var0[fcnt][vcnt];
                }

            alpha = (runtime - ftime0[fcnt]) / (ftime1[fcnt] - ftime0[fcnt]);
            for(vcnt = 0; vcnt < NBvarin[fcnt]; vcnt++)
            {
                fprintf(fpout,
                        " %20f",
                        (1.0 - alpha) * var0[fcnt][vcnt] +
                        alpha * var1[fcnt][vcnt] - varC[fcnt][vcnt]);
                varC[fcnt][vcnt] = (1.0 - HPgain) * varC[fcnt][vcnt] +
                                   HPgain * ((1.0 - alpha) * var0[fcnt][vcnt] +
                                             alpha * var1[fcnt][vcnt]);

                data.image[IDout[kkfr]].array.F[kkpt * nbvar + ii] =
                    (1.0 - alpha) * var0[fcnt][vcnt] +
                    alpha * var1[fcnt][vcnt] - varC[fcnt][vcnt];
                ii++;
            }
        }

        fprintf(fpout, "\n");

        kk++;
        kkpt++;
        runtime += dt;
        if(kkpt == NBpt)
        {
            kkpt = 0;
            kkfr++;
        }
    }

    fclose(fpout);

    for(fcnt = 0; fcnt < NBfiles; fcnt++)
    {
        fclose(fparray[fcnt]);
    }

    return (NBfiles);
}

// select block on first dimension
imageID LINARFILTERPRED_SelectBlock(const char *IDin_name,
                                    const char *IDblknb_name,
                                    long        blkNB,
                                    const char *IDout_name)
{
    imageID IDin;
    imageID IDblknb;
    uint8_t naxis;

    long          m;
    long          NBmodes1;
    uint32_t     *sizearray;
    uint32_t      xsize, ysize, zsize;
    unsigned long cnt;
    imageID       IDout;
    //char imname[200];
    long mmax;

    printf("Selecting block %ld ...\n", blkNB);
    fflush(stdout);

    IDin    = image_ID(IDin_name);
    IDblknb = image_ID(IDblknb_name);
    naxis   = data.image[IDin].md[0].naxis;
    mmax    = data.image[IDblknb].md[0].size[0];

    if(data.image[IDin].md[0].size[0] != data.image[IDblknb].md[0].size[0])
    {
        printf(
            "WARNING: block index file and telemetry have different sizes\n");
        fflush(stdout);
        mmax = data.image[IDin].md[0].size[0];
        if(data.image[IDblknb].md[0].size[0] < mmax)
        {
            mmax = data.image[IDblknb].md[0].size[0];
        }
    }

    NBmodes1 = 0;
    for(m = 0; m < mmax; m++)
    {
        if(data.image[IDblknb].array.UI16[m] == blkNB)
        {
            NBmodes1++;
        }
    }

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(uint8_t axis = 0; axis < naxis; axis++)
    {
        sizearray[axis] = data.image[IDin].md[0].size[axis];
    }
    sizearray[0] = NBmodes1;

    create_image_ID(IDout_name,
                    naxis,
                    sizearray,
                    _DATATYPE_FLOAT,
                    0,
                    0,
                    0,
                    &IDout);

    xsize = data.image[IDin].md[0].size[0];
    if(naxis > 1)
    {
        ysize = data.image[IDin].md[0].size[1];
    }
    else
    {
        ysize = 1;
    }
    if(naxis > 2)
    {
        zsize = data.image[IDin].md[0].size[2];
    }
    else
    {
        zsize = 1;
    }

    cnt = 0;

    for(uint32_t jj = 0; jj < ysize; jj++)
        for(uint32_t kk = 0; kk < zsize; kk++)
            for(uint32_t ii = 0; ii < mmax; ii++)
                if(data.image[IDblknb].array.UI16[ii] == blkNB)
                {
                    //printf("%ld / %ld   cnt = %8ld / %ld\n", ii, xsize, cnt, NBmodes1*ysize*zsize);
                    //fflush(stdout);
                    data.image[IDout].array.F[cnt] =
                        data.image[IDin]
                        .array.F[kk * xsize * ysize + jj * ysize + ii];
                    cnt++;
                }

    free(sizearray);

    return (IDout);
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 3. BUILD PREDICTIVE FILTER                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Expand 2D image/matrix in X direction by repeat and shift
 *
 */
imageID linARfilterPred_repeat_shift_X(const char *IDin_name,
                                       long        NBstep,
                                       const char *IDout_name)
{
    imageID  IDin;
    uint32_t xsize, ysize;

    imageID  IDout;
    uint32_t xsizeout, ysizeout;

    uint32_t *imsizeout;

    IDin     = image_ID(IDin_name);
    xsize    = data.image[IDin].md[0].size[0];
    ysize    = data.image[IDin].md[0].size[1];
    xsizeout = xsize * NBstep;
    ysizeout = ysize - NBstep;

    imsizeout = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsizeout == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imsizeout[0] = xsizeout;
    imsizeout[1] = ysizeout;
    create_image_ID(IDout_name, 2, imsizeout, _DATATYPE_FLOAT, 1, 0, 0, &IDout);
    free(imsizeout);

    long step;
    for(step = 0; step < NBstep; step++)
    {
        for(uint32_t ii = 0; ii < xsize; ii++)
        {
            for(uint32_t jjout = 0; jjout < ysize - NBstep; jjout++)
            {
                data.image[IDout]
                .array.F[jjout * xsizeout + step * xsize + ii] =
                    data.image[IDin]
                    .array.F[(jjout + NBstep - step - 1) * xsize + ii];
            }
        }
    }

    return IDout;
}

/** ## Purpose
 *
 * Build predictive filter from real-time AO telemetry
 *
 *
 * ## Masking
 *
 *  Optional input and output pixel masks select active input & output
 *
 *
 * ## Loop mode
 *
 * If LOOPmode = 1, operate in a loop, and re-run filter computation everytime IDin_name changes
 *
 *
 * ## Input parameters: dynamic mode
 *
 * if <IFoutPF_name>_PFparam image exist, read parameters from it: PFlag, SVDeps, RegLambda, LOOPgain
 * create it in shared memory by default
 *
 *
 * @return If testmode=2, write 3D output filter
 * @return output filter image indentifier
 *
   */

imageID LINARFILTERPRED_Build_LinPredictor(const char *IDin_name,
        long        PForder,
        float       PFlag,
        double      SVDeps,
        double      RegLambda,
        const char *IDoutPF_name,
        __attribute__((unused)) int outMode,
        int                         LOOPmode,
        float                       LOOPgain,
        int                         testmode)
{
    /// ---
    /// # Code Description

    imageID IDin;
    imageID IDmatA;
    //imageID IDout;
    imageID IDinmask;
    imageID IDoutmask;
    long    nbspl; // Number of samples
    long    NBpixin, NBpixout;
    long    NBmvec, NBmvec1;
    long    mvecsize;
    long    xsize, ysize;
    long   *pixarray_x;
    long   *pixarray_y;
    long   *pixarray_xy;

    long *outpixarray_x;
    long *outpixarray_y;
    long *outpixarray_xy;

    double *ave_inarray;
    int     REG = 0; // 1 if regularization
    long    m, pix, k0, dt;
    int     Save = 0;
    long    xysize;
    long    IDmatC;
    //int use_magma = 1;                         // use MAGMA library if available
    //int magmacomp = 0;

    //imageID IDfiltC;
    // float *valfarray;
    float alpha;
    long  PFpix;
    //char filtname[200];
    //char filtfname[200];
    //imageID ID_Pfilt;
    float   val, val0;
    long    ind1;
    imageID IDoutPF2D;    // averaged with previous filters
    imageID IDoutPF2Draw; // individual filter
    char    IDoutPF_name_raw[200];
    //  long IDoutPF3D;
    //  char IDoutPF_name3D[500];

    long NB_SVD_Modes;

    int DC_MODE = 0; // 1 if average value of each mode is removed

    long      NBiter, iter;
    long      semtrig = 2;
    uint32_t *imsizearray;

    //char fname[200];

    //time_t t;
    //struct tm *uttime;
    //struct timespec timenow;

    struct timespec t0;
    struct timespec t1;
    struct timespec t2;
    struct timespec tdiff;
    double          tdiffv01; // waiting time
    double          tdiffv12; // computing time

    imageID IDPFparam; // parameters in shared memory (optional)
    char    imname[200];
    int     ExternalPFparam;

    float PFlag_run;
    float SVDeps_run;
    float RegLambda_run;
    float LOOPgain_run;
    float gain;

    uint32_t *imsize;
    long      IDincp;
    long      inNBelem;

    list_variable_ID();

    int  PSINV_MODE = 0;
    long IDv;
    if((IDv = variable_ID("_SVD_PSINV")) != -1)
    {
        PSINV_MODE = (int)(data.variable[IDv].value.f + 0.1);
        printf("PSINV_MODE = %d\n", PSINV_MODE);
    }

    float PSINV_s = 1.0e-6;
    if((IDv = variable_ID("_SVD_s")) != -1)
    {
        PSINV_s = data.variable[IDv].value.f;
        printf("PSINV_s = %f\n", PSINV_s);
    }

    float PSINV_tol = 1.0;
    if((IDv = variable_ID("_SVD_tol")) != -1)
    {
        PSINV_tol = data.variable[IDv].value.f;
        printf("PSINV_tol = %f\n", PSINV_tol);
    }

    /// ## Reading Parameters from Image

    /// If image named <IDoutPF_name>_PFparam exists, the predictive filter
    /// parameters are read from it instead of the function arguments. \n
    /// This mode is particularly useful in LOOP mode if the user needs
    /// to change the parameters between LOOP iterations.\n

    sprintf(imname, "%s_PFparam", IDoutPF_name);
    imsize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsize == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }
    imsize[0] = 4;
    imsize[1] = 1;
    create_image_ID(imname, 2, imsize, _DATATYPE_FLOAT, 1, 0, 0, &IDPFparam);
    free(imsize);

    if((IDPFparam = image_ID(imname)) != -1)
    {
        ExternalPFparam                  = 1;
        data.image[IDPFparam].array.F[0] = PFlag;
        data.image[IDPFparam].array.F[1] = SVDeps;
        data.image[IDPFparam].array.F[2] = RegLambda;
        data.image[IDPFparam].array.F[3] = LOOPgain;
    }
    else
    {
        ExternalPFparam = 0;
    }

    LOOPgain_run = LOOPgain;
    if(LOOPmode == 0)
    {
        LOOPgain_run = 1.0;
        NBiter       = 1;
    }
    else
    {
        NBiter = 100000000;
    }

    //sprintf(IDoutPF_name3D, "%s_3D", IDoutPF_name);

    /// ## Selecting input values

    /// The goal of this function is to build a linear link between
    /// input and output variables. \n
    /// Input variables values are provided by the input telemetry image
    /// which is first read to measure dimensions, and allocate memory.\n
    /// Note that an optional variable selection step allows only a
    /// subset of the telemetry variables to be considered.

    /// ### Read input telemetry image IDin_name to measure xsize, ysize and number of samples
    IDin = image_ID(IDin_name);

    switch(data.image[IDin].md[0].naxis)
    {

        case 2:
            /// If 2D image:
            /// - xysize <- size[0] is number of variables
            /// - nbspl <- size[1] is number of samples
            nbspl = data.image[IDin].md[0].size[1];
            xsize = data.image[IDin].md[0].size[0];
            ysize = 1;
            // copy of image to avoid input change during computation
            create_2Dimage_ID("PFin_cp",
                              data.image[IDin].md[0].size[0],
                              data.image[IDin].md[0].size[1],
                              &IDincp);
            inNBelem =
                data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1];
            break;

        case 3:
            /// If 3D image
            /// - xysize <- size[0] * size[1] is number of variables
            /// - nbspl <- size[2] is number of samples
            nbspl = data.image[IDin].md[0].size[2];
            xsize = data.image[IDin].md[0].size[0];
            ysize = data.image[IDin].md[0].size[1];
            create_3Dimage_ID("PFin_copy",
                              data.image[IDin].md[0].size[0],
                              data.image[IDin].md[0].size[1],
                              data.image[IDin].md[0].size[2],
                              &IDincp);

            inNBelem = data.image[IDin].md[0].size[0] *
                       data.image[IDin].md[0].size[1] *
                       data.image[IDin].md[0].size[2];
            break;

        default:
            printf("Invalid image size\n");
            break;
    }
    xysize = xsize * ysize;
    printf("xysize = %ld\n", xysize);

    /// Once input telemetry size measured, arrays are created:
    /// - pixarray_x  : x coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_y  : y coordinate of each variable (useful to keep track of spatial coordinates)
    /// - pixarray_xy : combined index (avoids re-computing index frequently)
    /// - ave_inarray : time averaged value, useful because the predictive filter often needs average to be zero, so we will remove it

    pixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    pixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    pixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(pixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ave_inarray = (double *) malloc(sizeof(double) * xsize * ysize);
    if(ave_inarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    /// ### Select input variables from mask (optional)
    /// If image "inmask" exists, use it to select which variables are active.
    /// Otherwise, all variables are active\n
    /// The number of active input variables is stored in NBpixin.

    IDinmask = image_ID("inmask");
    if(IDinmask == -1)
    {
        NBpixin = 0; //xsize*ysize;

        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                pixarray_x[NBpixin]  = ii;
                pixarray_y[NBpixin]  = jj;
                pixarray_xy[NBpixin] = jj * xsize + ii;
                NBpixin++;
            }
    }
    else
    {
        NBpixin = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(data.image[IDinmask].array.F[jj * xsize + ii] > 0.5)
                {
                    pixarray_x[NBpixin]  = ii;
                    pixarray_y[NBpixin]  = jj;
                    pixarray_xy[NBpixin] = jj * xsize + ii;
                    NBpixin++;
                }
    }
    printf("NBpixin = %ld\n", NBpixin);

    /// ## Selecting Output Variables
    /// By default, the output variables are the same as the input variables,
    /// so the prediction is performed on the same variables as the input.\n
    ///
    /// With inmask and outmask, input AND output variables can be
    /// selected amond the telemetry.

    /// Arrays are created:
    /// - outpixarray_x  : x coordinate of each output variable (useful to keep track of spatial coordinates)
    /// - outpixarray_y  : y coordinate of each output variable (useful to keep track of spatial coordinates)
    /// - outpixarray_xy : combined output index (avoids re-computing index frequently)

    outpixarray_x = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_x == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outpixarray_y = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_y == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outpixarray_xy = (long *) malloc(sizeof(long) * xsize * ysize);
    if(outpixarray_xy == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    IDoutmask = image_ID("outmask");
    if(IDoutmask == -1)
    {
        NBpixout = 0; //xsize*ysize;

        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
            {
                outpixarray_x[NBpixout]  = ii;
                outpixarray_y[NBpixout]  = jj;
                outpixarray_xy[NBpixout] = jj * xsize + ii;
                NBpixout++;
            }
    }
    else
    {
        NBpixout = 0;
        for(uint32_t ii = 0; ii < xsize; ii++)
            for(uint32_t jj = 0; jj < ysize; jj++)
                if(data.image[IDoutmask].array.F[jj * xsize + ii] > 0.5)
                {
                    outpixarray_x[NBpixout]  = ii;
                    outpixarray_y[NBpixout]  = jj;
                    outpixarray_xy[NBpixout] = jj * xsize + ii;
                    NBpixout++;
                }
    }

    /// ## Reading PFlag from image (optional)
    /// PFlag_run needs to be read before entering the loop as some
    /// array sizes depend on its value.
    if(ExternalPFparam == 1)
    {
        PFlag_run = data.image[IDPFparam].array.F[0];
    }
    else
    {
        PFlag_run = PFlag;
    }

    /// ## Build Empty Data Matrix
    ///
    /// Note: column / row description follows FITS file viewing conventions.\n
    /// The data matrix is build from the telemetry. Each column (= time sample) of the
    /// data matrix consists of consecutives columns (= time sample) of the input telemetry.\n
    ///
    /// Variable naming:
    /// - NBmvec is the number of telemetry vectors (each corresponding to a different time) in the data matrix.
    /// - mvecsize is the size of each vector, equal to NBpixin times PForder
    ///
    /// Data matrix is stored as image of size NBmvec x mvecsize, to be fed to routine compute_SVDpseudoInverse in linopt_imtools (CPU mode) or in cudacomp (GPU mode)\n
    ///
    NBmvec =
        nbspl - PForder -
        (int)(PFlag_run) -
        2; // could put "-1", but "-2" allows user to change PFlag_run by up to 1 frame without reading out of array
    mvecsize =
        NBpixin *
        PForder; // size of each sample vector for AR filter, excluding regularization

    /// Regularization can be added to penalize strong coefficients in the predictive filter.
    /// It is optionally implemented by adding extra columns at the end of the data matrix.\n
    if(REG == 0)  // no regularization
    {
        printf("NBmvec   = %ld  -> %ld \n", NBmvec, NBmvec);
        NBmvec1 = NBmvec;
        create_2Dimage_ID("PFmatD", NBmvec, mvecsize, &IDmatA);
    }
    else // with regularization
    {
        printf("NBmvec   = %ld  -> %ld \n", NBmvec, NBmvec + mvecsize);
        NBmvec1 = NBmvec + mvecsize;
        create_2Dimage_ID("PFmatD", NBmvec + mvecsize, mvecsize, &IDmatA);
    }

    IDmatA = image_ID("PFmatD");

    /// Data matrix conventions :
    /// - each column (ii = cst) is a measurement
    /// - m index is measurement
    /// - dt*NBpixin+pix index is pixel

    printf("mvecsize = %ld  (%ld x %ld)\n", mvecsize, PForder, NBpixin);
    printf("NBpixin = %ld\n", NBpixin);
    printf("NBpixout = %ld\n", NBpixout);
    printf("NBmvec1 = %ld\n", NBmvec1);
    printf("PForder = %ld\n", PForder);

    printf("xysize = %ld\n", xysize);
    printf("IDin = %ld\n\n", IDin);
    list_image_ID();

    /// ## Predictive Filter Computation
    ///
    /// In LOOP mode, LOOP STARTS HERE \n
    ///

    if(LOOPmode == 1)
    {
        COREMOD_MEMORY_image_set_semflush(IDin_name, semtrig);
    }

    for(iter = 0; iter < NBiter; iter++)
    {

        /// ### Prepare data matrix PFmatD

        /// *STEP: Read parameters from external image (optional)*\n
        if(ExternalPFparam == 1)
        {
            PFlag_run     = data.image[IDPFparam].array.F[0];
            SVDeps_run    = data.image[IDPFparam].array.F[1];
            RegLambda_run = data.image[IDPFparam].array.F[2];
            LOOPgain_run  = data.image[IDPFparam].array.F[3];
        }
        else
        {
            PFlag_run     = PFlag;
            SVDeps_run    = SVDeps;
            RegLambda_run = RegLambda;
            LOOPgain_run  = LOOPgain;
        }

        printf(
            "=========== LOOP ITERATION %6ld ======= [ExternalPFparam = %d ]\n",
            iter,
            ExternalPFparam);
        printf(" parameters read from %s\n", data.image[IDPFparam].md[0].name);
        printf("  PFlag     = %20f      ", PFlag_run);
        printf("  SVDeps    = %20f\n", SVDeps_run);
        printf("  RegLambda = %20f      ", RegLambda_run);
        printf("  LOOPgain  = %20f\n", LOOPgain_run);
        printf("\n");

        gain = 1.0 / (iter + 1);
        if(gain < LOOPgain_run)
        {
            gain = LOOPgain_run;
        }

        /// *STEP: In loop mode, wait for input data to arrive*

        printf("WAITING FOR INPUT DATA ...... \n");
        clock_gettime(CLOCK_MILK, &t0);
        if(LOOPmode == 1)
        {
            ImageStreamIO_semwait(data.image+IDin, semtrig);
        }

        /// *STEP: Copy IDin to IDincp*
        ///
        /// Necessary as input may be continuously changing between consecutive loop iterations.
        ///
        IDincp = image_ID("PFin_copy");
        memcpy(data.image[IDincp].array.F,
               data.image[IDin].array.F,
               sizeof(float) * inNBelem);

        //save_fits("PFin_copy", "test_PFin_copy.fits");
        //save_fits(IDin_name, "test_PFin.fits");

        clock_gettime(CLOCK_MILK, &t1);

        /// *STEP: if DC_MODE==1, compute average value from each variable*
        if(DC_MODE == 1)  // remove average
        {
            for(pix = 0; pix < NBpixin; pix++)
            {
                ave_inarray[pix] = 0.0;
                for(m = 0; m < nbspl; m++)
                {
                    ave_inarray[pix] +=
                        data.image[IDincp]
                        .array.F[m * xysize + pixarray_xy[pix]];
                }
                ave_inarray[pix] /= nbspl;
            }
        }
        else
        {
            for(pix = 0; pix < NBpixin; pix++)
            {
                ave_inarray[pix] = 0.0;
            }
        }

        ///
        /// *STEP: Fill up data matrix PFmatD from input telemetry*
        ///
        for(m = 0; m < NBmvec1; m++)
        {
            k0 = m + PForder - 1; // dt=0 index
            for(pix = 0; pix < NBpixin; pix++)
                for(dt = 0; dt < PForder; dt++)
                {
                    data.image[IDmatA]
                    .array.F[(NBpixin * dt + pix) * NBmvec1 + m] =
                        data.image[IDincp]
                        .array.F[(k0 - dt) * xysize + pixarray_xy[pix]] -
                        ave_inarray[pix];
                }
        }

        if(LOOPmode == 0)
        {
            free(ave_inarray); // No need to hold on to array
        }

        ///
        /// *STEP: Write regularization coefficients (optional)*
        ///
        if(REG == 1)
        {
            for(m = 0; m < mvecsize; m++)
            {
                //m1 = NBmvec + m;
                data.image[IDmatA].array.F[(m) *NBmvec1 + (NBmvec + m)] =
                    RegLambda_run;
            }
        }

        if(Save == 1)
        {
            save_fits("PFmatD", "PFmatD.fits");
        }
        //list_image_ID();

        /// ### Compute pseudo-inverse of PFmatD
        ///
        /// *STEP: Compute Pseudo-Inverse of PFmatD*
        ///
        printf("Assembling pseudoinverse\n");
        fflush(stdout);

        // Assemble future measured data matrix
        imageID IDfm;
        create_2Dimage_ID("PFfmdat", NBmvec, NBpixout, &IDfm);

        alpha = PFlag_run - ((long) PFlag_run);
        for(PFpix = 0; PFpix < NBpixout; PFpix++)
            for(m = 0; m < NBmvec; m++)
            {
                k0 = m + PForder - 1;
                k0 += (long) PFlag_run;

                data.image[IDfm].array.F[PFpix * NBmvec + m] =
                    (1.0 - alpha) *
                    data.image[IDincp]
                    .array.F[(k0) * xysize + outpixarray_xy[PFpix]] +
                    alpha *
                    data.image[IDincp]
                    .array.F[(k0 + 1) * xysize + outpixarray_xy[PFpix]];
            }
        save_fits("PFfmdat", "PFfmdat.fits");

        /// If using MAGMA, call function CUDACOMP_magma_compute_SVDpseudoInverse()\n
        /// Otherwise, call function linopt_compute_SVDpseudoInverse()\n

        NB_SVD_Modes = 10000;

#ifdef HAVE_MAGMA
        printf("Using magma ...\n");
        CUDACOMP_magma_compute_SVDpseudoInverse("PFmatD",
                                                "PFmatC",
                                                SVDeps_run,
                                                NB_SVD_Modes,
                                                "PF_VTmat",
                                                LOOPmode,
                                                testmode,
                                                64,
                                                0, // GPU device
                                                NULL);
#else
        printf("Not using magma ...\n");
        linopt_compute_SVDpseudoInverse("PFmatD",
                                        "PFmatC",
                                        SVDeps_run,
                                        NB_SVD_Modes,
                                        "PF_VTmat",
                                        NULL);
#endif

        /// Result (pseudoinverse) is stored in image PFmatC\n
        printf("Done assembling pseudoinverse\n");
        fflush(stdout);

        if(Save == 1)
        {
            save_fits("PF_VTmat", "PF_VTmat.fits");
            save_fits("PFmatC", "PFmatC.fits");
        }
        IDmatC = image_ID("PFmatC");

        ///
        /// ### Assemble Predictive Filter
        ///
        printf("Compute filters\n");
        fflush(stdout);

        if(system("mkdir -p pixfilters") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }

        // 3D FILTER MATRIX - contains all pixels
        // axis 0 [ii] : input mode
        // axis 1 [jj] : reconstructed mode
        // axis 2 [kk] : time step

        // 2D Filter - contains only used input and output
        // axis 0 [ii1] : input mode x time step
        // axis 1 [jj1] : output mode

        if(LOOPmode == 0)
        {
            create_2Dimage_ID(IDoutPF_name,
                              NBpixin * PForder,
                              NBpixout,
                              &IDoutPF2D);
        }

        else
        {
            if(iter == 0)  // create 2D and 3D filters as shared memory
            {
                imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
                if(imsizearray == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                imsizearray[0] = NBpixin * PForder;
                imsizearray[1] = NBpixout;
                sprintf(IDoutPF_name_raw, "%s_raw", IDoutPF_name);

                create_image_ID(IDoutPF_name,
                                2,
                                imsizearray,
                                _DATATYPE_FLOAT,
                                1,
                                1,
                                0,
                                &IDoutPF2D);
                create_image_ID(IDoutPF_name_raw,
                                2,
                                imsizearray,
                                _DATATYPE_FLOAT,
                                1,
                                1,
                                0,
                                &IDoutPF2Draw);
                free(imsizearray);
                COREMOD_MEMORY_image_set_semflush(IDoutPF_name, -1);
                COREMOD_MEMORY_image_set_semflush(IDoutPF_name_raw, -1);
            }
            else
            {
                IDoutPF2D = image_ID(IDoutPF_name);
            }
        }

        IDoutmask = image_ID("outmask");

        printf("===========================================================\n");
        printf("ASSEMBLING OUTPUT\n");
        printf("  NBpixout = %ld\n", NBpixout);
        printf("  NBmvec   = %ld\n", NBmvec);
        printf("  NBmvec1  = %ld\n", NBmvec1);
        printf("  NBpixin  = %ld\n", NBpixin);
        printf("  PForder  = %ld\n", PForder);
        printf("===========================================================\n");

        long IDoutPF2Dn = image_ID("psinvPFmat");
        if(IDoutPF2Dn == -1)
        {
            printf("------------------- CPU computing PF matrix\n");

            create_2Dimage_ID("psinvPFmat",
                              NBpixin * PForder,
                              NBpixout,
                              &IDoutPF2Dn);
            for(
                PFpix = 0; PFpix < NBpixout;
                PFpix++) // PFpix is the pixel for which the filter is created (axis 1 in cube, jj)
            {

                // loop on input values
                for(pix = 0; pix < NBpixin; pix++)
                {
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val  = 0.0;
                        ind1 = (NBpixin * dt + pix) * NBmvec1;
                        for(m = 0; m < NBmvec; m++)
                        {
                            val += data.image[IDmatC].array.F[ind1 + m] *
                                   data.image[IDfm].array.F[PFpix * NBmvec + m];
                        }

                        data.image[IDoutPF2Dn]
                        .array.F[PFpix * (PForder * NBpixin) +
                                       dt * NBpixin + pix] = val;
                    }
                }
            }
        }
        else
        {
            printf("------------------- Using GPU-computed PF matrix\n");
        }
        delete_image_ID("PFfmdat", DELETE_IMAGE_ERRMODE_WARNING);

        if(LOOPmode == 1)
        {
            data.image[IDoutPF2Draw].md[0].write = 1;
            memcpy(data.image[IDoutPF2Draw].array.F,
                   data.image[IDoutPF2Dn].array.F,
                   sizeof(float) * NBpixout * NBpixin * PForder);
            COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2Draw, -1);
            data.image[IDoutPF2Draw].md[0].cnt0++;
            data.image[IDoutPF2Draw].md[0].write = 0;
        }

        // Mix current PF with last one
        data.image[IDoutPF2D].md[0].write = 1;
        if(LOOPmode == 0)
        {
            memcpy(data.image[IDoutPF2D].array.F,
                   data.image[IDoutPF2Dn].array.F,
                   sizeof(float) * NBpixout * NBpixin * PForder);
            save_fits(IDoutPF_name, "_outPF.fits");
        }
        else
        {
            printf("Mixing PF matrix with gain = %f ....", gain);
            fflush(stdout);
            for(PFpix = 0; PFpix < NBpixout; PFpix++)
                for(pix = 0; pix < NBpixin; pix++)
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val0 = data.image[IDoutPF2D]
                               .array.F[PFpix * (PForder * NBpixin) +
                                              dt * NBpixin + pix]; // Previous
                        val = data.image[IDoutPF2Dn]
                              .array.F[PFpix * (PForder * NBpixin) +
                                             dt * NBpixin + pix]; // New
                        data.image[IDoutPF2D]
                        .array.F[PFpix * (PForder * NBpixin) +
                                       dt * NBpixin + pix] =
                                     (1.0 - gain) * val0 + gain * val;
                    }
            printf(" done\n");
            fflush(stdout);
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDoutPF2D, -1);
        data.image[IDoutPF2D].md[0].cnt0++;
        data.image[IDoutPF2D].md[0].write = 0;

        if(testmode == 2)
        {
            printf("Prepare 3D output \n");

            imageID IDoutPF3D;
            create_3Dimage_ID("outPF3D",
                              NBpixin,
                              NBpixout,
                              PForder,
                              &IDoutPF3D);

            for(pix = 0; pix < NBpixin; pix++)
                for(PFpix = 0; PFpix < NBpixout; PFpix++)
                    for(dt = 0; dt < PForder; dt++)
                    {
                        val = data.image[IDoutPF2D]
                              .array.F[PFpix * (PForder * NBpixin) +
                                             dt * NBpixin + pix];
                        data.image[IDoutPF3D].array.F[NBpixout * NBpixin * dt +
                                                      NBpixin * PFpix + pix] =
                                                          val;
                    }
            save_fits("outPF3D", "_outPF3D.fits");
        }

        printf("DONE\n");
        fflush(stdout);
        clock_gettime(CLOCK_MILK, &t2);

        tdiff    = timespec_diff(t0, t1);
        tdiffv01 = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff    = timespec_diff(t1, t2);
        tdiffv12 = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        printf("Computing time = %5.3f s / %5.3f s -> fraction = %8.6f\n",
               tdiffv12,
               tdiffv01 + tdiffv12,
               tdiffv12 / (tdiffv01 + tdiffv12));
    }
    ///
    /// In LOOP mode, LOOP ENDS HERE \n
    ///

    // free(valfarray);

    free(pixarray_x);
    free(pixarray_y);
    free(pixarray_xy);

    free(outpixarray_x);
    free(outpixarray_y);
    free(outpixarray_xy);

    ///
    /// ---
    ///

    return IDoutPF2D;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 4. APPLY PREDICTIVE FILTER                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

//
// real-time apply predictive filter
//
// filter can be smaller than input telemetry but needs to include contiguous pixels at the beginning of the input telemetry
//
imageID LINARFILTERPRED_Apply_LinPredictor_RT(const char *IDfilt_name,
        const char *IDin_name,
        const char *IDout_name)
{
    imageID   IDout;
    imageID   IDin;
    imageID   IDfilt;
    long      PForder;
    long      NBpix_in;
    long      NBpix_out;
    uint32_t *imsizearray;
    int       semtrig = 7;

    float *inarray;
    float *outarray;

    //    long ii; // input index
    //    long jj; // output index
    //    long kk; // time step index

    IDfilt = image_ID(IDfilt_name);
    IDin   = image_ID(IDin_name);

    PForder   = data.image[IDfilt].md[0].size[2];
    NBpix_in  = data.image[IDfilt].md[0].size[0];
    NBpix_out = data.image[IDfilt].md[0].size[1];

    list_image_ID();

    if(data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] !=
            NBpix_in)
    {
        printf(
            "ERROR: lin predictor engine: filter input size does not match "
            "input telemetry\n");
        exit(0);
    }

    printf("Create prediction output %s\n", IDout_name);
    fflush(stdout);
    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(imsizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imsizearray[0] = NBpix_out;
    imsizearray[1] = 1;
    create_image_ID(IDout_name,
                    2,
                    imsizearray,
                    _DATATYPE_FLOAT,
                    1,
                    1,
                    0,
                    &IDout);
    free(imsizearray);
    COREMOD_MEMORY_image_set_semflush(IDout_name, -1);
    printf("Done\n");
    fflush(stdout);

    inarray = (float *) malloc(sizeof(float) * NBpix_in * PForder);
    if(inarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outarray = (float *) malloc(sizeof(float) * NBpix_out);
    if(outarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    while(ImageStreamIO_semtrywait(data.image+IDin, semtrig) == 0)
    {
    }
    while(1)
    {
        // initialize output array to zero
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            outarray[jj] = 0.0;
        }

        // shift input buffer entries back one time step
        for(uint32_t kk = PForder - 1; kk > 0; kk--)
            for(uint32_t ii = 0; ii < NBpix_in; ii++)
            {
                inarray[kk * NBpix_in + ii] = inarray[(kk - 1) * NBpix_in + ii];
            }

        // multiply input by prediction matrix .. except for measurement yet to come
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
            for(uint32_t ii = 0; ii < NBpix_in; ii++)
                for(uint32_t kk = 1; kk < PForder; kk++)
                {
                    outarray[jj] +=
                        data.image[IDfilt].array.F[kk * NBpix_in * NBpix_out +
                                                   jj * NBpix_in + ii] *
                        inarray[kk * NBpix_in + ii];
                }

        ImageStreamIO_semwait(data.image+IDin, semtrig);

        // write new input in inarray vector
        for(uint32_t ii = 0; ii < NBpix_in; ii++)
        {
            inarray[ii] = data.image[IDin].array.F[ii];
        }

        // multiply input by prediction matrix
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
            for(uint32_t ii = 0; ii < NBpix_in; ii++)
            {
                outarray[jj] += data.image[IDfilt].array.F[jj * NBpix_in + ii] *
                                inarray[ii];
            }

        data.image[IDout].md[0].write = 1;
        for(uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            data.image[IDout].array.F[jj] = outarray[jj];
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }

    free(inarray);
    free(outarray);

    return IDout;
}

//
//
// out : prediction
//
// ADDITIONAL OUTPUTS:
// outf : time-shifted measurement
//

imageID LINARFILTERPRED_Apply_LinPredictor(const char *IDfilt_name,
        const char *IDin_name,
        float       PFlag,
        const char *IDout_name)
{
    imageID  IDout;
    imageID  IDin;
    imageID  IDfilt;
    uint32_t xsize;
    uint32_t ysize;
    uint64_t xysize;

    long  nbspl;
    long  PForder;
    long  step;
    long  kk;
    float alpha;
    long  PFlagl;
    float valp, valf;

    imageID IDoutf;

    IDin   = image_ID(IDin_name);
    IDfilt = image_ID(IDfilt_name);

    switch(data.image[IDin].md[0].naxis)
    {

        case 2:
            nbspl = data.image[IDin].md[0].size[1];
            xsize = data.image[IDin].md[0].size[0];
            ysize = 1;
            create_2Dimage_ID(IDout_name, xsize, nbspl, &IDout);
            create_2Dimage_ID("outf", xsize, nbspl, &IDoutf);
            break;

        case 3:
            nbspl = data.image[IDin].md[0].size[2];
            xsize = data.image[IDin].md[0].size[0];
            ysize = data.image[IDin].md[0].size[1];
            create_3Dimage_ID(IDout_name, xsize, ysize, nbspl, &IDout);
            create_3Dimage_ID("outf", xsize, ysize, nbspl, &IDoutf);
            break;

        default:
            printf("Invalid image size\n");
            break;
    }
    xysize = xsize * ysize;

    PForder = data.image[IDfilt].md[0].size[2];

    if((data.image[IDfilt].md[0].size[0] != xysize) ||
            (data.image[IDfilt].md[0].size[1] != xysize))
    {
        printf("ERROR: filter \"%s\" size is incorrect\n", IDfilt_name);
        exit(0);
    }

    alpha  = PFlag - ((long) PFlag);
    PFlagl = (long) PFlag;

    for(kk = PForder; kk < nbspl; kk++)  // time step
    {
        for(uint32_t iip = 0; iip < xysize; iip++)  // predicted variable
        {
            valp = 0.0; // prediction
            for(step = 0; step < PForder; step++)
            {
                for(uint32_t ii = 0; ii < xsize * ysize;
                        ii++) // input variable
                {
                    valp += data.image[IDfilt].array.F[xysize * xysize * step +
                                                       iip * xysize + ii] *
                            data.image[IDin].array.F[(kk - step) * xysize + ii];
                }
            }
            data.image[IDout].array.F[kk * xysize + iip] = valp;

            valf = 0.0;
            if(kk + PFlag + 1 < nbspl)
            {
                valf =
                    (1.0 - alpha) *
                    data.image[IDin].array.F[(kk + PFlagl) * xysize + iip] +
                    alpha * data.image[IDin]
                    .array.F[(kk + PFlagl + 1) * xysize + iip];
            }
            data.image[IDoutf].array.F[kk * xysize + iip] = valf;
        }
    }

    return IDout;
}

//
// IDPF_name and IDPFM_name should be pre-loaded
//
imageID LINARFILTERPRED_PF_updatePFmatrix(const char *IDPF_name,
        const char *IDPFM_name,
        float       alpha)
{
    imageID IDPF;
    imageID IDPFM;
    long    inmode, NBmode, outmode, NBmode2;
    long    tstep, NBtstep;

    uint32_t *sizearray;
    uint8_t   naxis;

    // IDPF should be square
    IDPF    = image_ID(IDPF_name);
    NBmode  = data.image[IDPF].md[0].size[0];
    NBmode2 = NBmode * NBmode;
    assert(data.image[IDPF].md[0].size[0] == data.image[IDPF].md[0].size[1]);
    NBtstep = data.image[IDPF].md[0].size[2];

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = NBmode * NBtstep;
    sizearray[1] = NBmode;
    naxis        = 2;

    IDPFM = image_ID(IDPFM_name);

    if(IDPFM == -1)
    {
        printf("Creating shared mem image %s  [ %ld  x  %ld ]\n",
               IDPFM_name,
               (long) sizearray[0],
               (long) sizearray[1]);
        fflush(stdout);
        create_image_ID(IDPFM_name,
                        naxis,
                        sizearray,
                        _DATATYPE_FLOAT,
                        1,
                        0,
                        0,
                        &IDPFM);
    }
    free(sizearray);

    data.image[IDPFM].md[0].write = 1;
    for(outmode = 0; outmode < NBmode; outmode++)
    {
        for(tstep = 0; tstep < NBtstep; tstep++)
            for(inmode = 0; inmode < NBmode; inmode++)
                data.image[IDPFM].array.F[outmode * (NBmode * NBtstep) +
                                          tstep * NBmode + inmode] =
                                              (1.0 - alpha) *
                                              data.image[IDPFM].array.F[outmode * (NBmode * NBtstep) +
                                                      tstep * NBmode + inmode] +
                                              alpha * data.image[IDPF].array.F[tstep * NBmode2 +
                                                      outmode * NBmode + inmode];
    }
    COREMOD_MEMORY_image_set_sempost_byID(IDPFM, -1);
    data.image[IDPFM].md[0].write = 0;
    data.image[IDPFM].md[0].cnt0++;

    return IDPFM;
}

//
// IDmodevalIN_name : open loop modal coefficients
// IndexOffset      : predicted mode start at this input index
// semtrig          : semaphore trigger index in input input
// IDPFM_name       : predictive filter matrix
// IDPFout_name     : prediction
//
//  NBiter: run for fixed number of iteration
//  SAVEMODE:   0 no file output
//  			1	write txt and FITS output
//				2	write FITS telemetry with prediction: replace output measurements with predictions
//
//	tlag is only used if SAVEMODE = 2
//  used outmask to identify outputs
//
imageID LINARFILTERPRED_PF_RealTimeApply(const char *IDmodevalIN_name,
        long        IndexOffset,
        int         semtrig,
        const char *IDPFM_name,
        long        NBPFstep,
        const char *IDPFout_name,
        int         nbGPU,
        long        loop,
        long        NBiter,
        int         SAVEMODE,
        float       tlag,
        long        PFindex)
{
    imageID IDmodevalIN;
    long    NBmodeIN, NBmodeIN0, NBmodeOUT, mode;
    imageID IDPFM;

    imageID   IDINbuff;
    long      tstep;
    uint32_t *sizearray;
    uint8_t   naxis;

    imageID IDPFout;

    int *GPUsetPF;
    char GPUsetfname[200];
    int  gpuindex;

#ifdef HAVE_CUDA
    int status;
    int GPUstatus[100];
    int GPUMATMULTCONFindex = 2;
#endif

    FILE *fp;

    //time_t t;
    //struct tm *uttime;
    struct timespec timenow;
    double          timesec, timesec0;
    long            IDsave;

    FILE *fpout;
    long  iter;
    long  kk;

    imageID IDinmask;
    long   *inmaskindex;
    long    NBinmaskpix;

    long  tlag0;
    float tlagalpha = 0.0;

    imageID IDoutmask;
    long   *outmaskindex;
    long    NBoutmaskpix;
    long    kk0, kk1;
    float   val, val0, val1;
    long    ii0, ii1;

    long IDmasterout;
    char imname[200];

    IDmodevalIN = image_ID(IDmodevalIN_name);
    NBmodeIN0   = data.image[IDmodevalIN].md[0].size[0];

    IDPFM     = image_ID(IDPFM_name);
    NBmodeOUT = data.image[IDPFM].md[0].size[1];

    sprintf(imname, "aol%ld_modevalPF", loop);
    IDmasterout = image_ID(imname);

    IDinmask = image_ID("inmask");
    if(IDinmask != -1)
    {
        NBinmaskpix = 0;
        for(uint32_t ii = 0; ii < data.image[IDinmask].md[0].size[0]; ii++)
            if(data.image[IDinmask].array.F[ii] > 0.5)
            {
                NBinmaskpix++;
            }

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);
        if(inmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBinmaskpix = 0;
        for(uint32_t ii = 0; ii < data.image[IDinmask].md[0].size[0]; ii++)
            if(data.image[IDinmask].array.F[ii] > 0.5)
            {
                inmaskindex[NBinmaskpix] = ii;
                NBinmaskpix++;
            }
        //printf("Number of active input modes  = %ld\n", NBinmaskpix);
    }
    else
    {
        NBinmaskpix = NBmodeIN0;
        printf("no input mask -> assuming NBinmaskpix = %ld\n", NBinmaskpix);
        create_2Dimage_ID("inmask", NBinmaskpix, 1, &IDinmask);
        for(uint32_t ii = 0; ii < data.image[IDinmask].md[0].size[0]; ii++)
        {
            data.image[IDinmask].array.F[ii] = 1.0;
        }

        inmaskindex = (long *) malloc(sizeof(long) * NBinmaskpix);
        if(inmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for(uint32_t ii = 0; ii < data.image[IDinmask].md[0].size[0]; ii++)
        {
            inmaskindex[NBinmaskpix] = ii;
        }
    }
    NBmodeIN = NBinmaskpix;

    NBPFstep = data.image[IDPFM].md[0].size[0] / NBmodeIN;

    printf("Number of input modes         = %ld\n", NBmodeIN0);
    printf("Number of active input modes  = %ld\n", NBmodeIN);
    printf("Number of output modes        = %ld\n", NBmodeOUT);
    printf("Number of time steps          = %ld\n", NBPFstep);
    if(IDmasterout != -1)
    {
        printf("Writing result in master output stream %s  (%ld)\n",
               imname,
               IDmasterout);
    }

    if((SAVEMODE > 0) || (IDmasterout != -1))
    {
        IDoutmask = image_ID("outmask");
        if(IDoutmask == -1)
        {
            printf("ERROR: outmask image required\n");
            exit(0);
        }
        NBoutmaskpix = 0;
        for(uint32_t ii = 0; ii < data.image[IDoutmask].md[0].size[0]; ii++)
            if(data.image[IDoutmask].array.F[ii] > 0.5)
            {
                NBoutmaskpix++;
            }

        outmaskindex = (long *) malloc(sizeof(long) * NBoutmaskpix);
        if(outmaskindex == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        NBoutmaskpix = 0;
        for(uint32_t ii = 0; ii < data.image[IDoutmask].md[0].size[0]; ii++)
            if(data.image[IDoutmask].array.F[ii] > 0.5)
            {
                outmaskindex[NBoutmaskpix] = ii;
                NBoutmaskpix++;
            }
        if(NBoutmaskpix != NBmodeOUT)
        {
            printf("ERROR: NBoutmaskpix (%ld)   !=   NBmodeOUT (%ld)\n",
                   NBoutmaskpix,
                   NBmodeOUT);
            list_image_ID();
            exit(0);
        }
    }

    create_2Dimage_ID("INbuffer", NBmodeIN, NBPFstep, &IDINbuff);

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = NBmodeOUT;
    sizearray[1] = 1;
    naxis        = 2;
    IDPFout      = image_ID(IDPFout_name);

    if(IDPFout == -1)
    {
        create_image_ID(IDPFout_name,
                        naxis,
                        sizearray,
                        _DATATYPE_FLOAT,
                        1,
                        0,
                        0,
                        &IDPFout);
    }
    free(sizearray);

    if(nbGPU > 0)
    {
        GPUsetPF = (int *) malloc(sizeof(int) * nbGPU);
        if(GPUsetPF == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for(gpuindex = 0; gpuindex < nbGPU; gpuindex++)
        {
            sprintf(GPUsetfname,
                    "./conf/param_PFb%ldGPU%ddevice.txt",
                    PFindex,
                    gpuindex);
            fp = fopen(GPUsetfname, "r");
            if(fp == NULL)
            {
                printf("ERROR: file %s not found\n", GPUsetfname);
                exit(0);
            }
            if(fscanf(fp, "%d", &GPUsetPF[gpuindex]) != 1)
            {
                PRINT_ERROR("fscanf error");
            }
            fclose(fp);
        }
        printf("USING %d GPUs: ", nbGPU);
        for(gpuindex = 0; gpuindex < nbGPU; gpuindex++)
        {
            printf(" %d", GPUsetPF[gpuindex]);
        }
        printf("\n\n");
    }
    else
    {
        printf("Using CPU\n");
    }

    iter = 0;
    if(SAVEMODE > 0)
        if(NBiter > 50000)
        {
            NBiter = 50000;
        }

    if(SAVEMODE == 1)
    {
        create_2Dimage_ID("testPFsave",
                          1 + NBmodeIN0 + NBmodeOUT,
                          NBiter,
                          &IDsave);
    }
    if(SAVEMODE == 2)
    {
        create_3Dimage_ID("testPFTout", NBmodeIN0, 1, NBiter, &IDsave);
    }

    //	t = time(NULL);
    //    uttime = gmtime(&t);
    //	clock_gettime(CLOCK_MILK, &timenow);
    //	timesec0 = 3600.0*uttime->tm_hour  + 60.0*uttime->tm_min + 1.0*(timenow.tv_sec % 60) + 1.0e-9*timenow.tv_nsec;

    printf("Running on semaphore trigger %d of image %s\n",
           semtrig,
           data.image[IDmodevalIN].md[0].name);

    while(iter != NBiter)
    {
        //	printf("iter %5ld / %5ld", iter, NBiter);
        //	fflush(stdout);

        ImageStreamIO_semwait(data.image+IDmodevalIN, semtrig);
        //	printf("\n");
        //	fflush(stdout);

        // fill in buffer
        for(mode = 0; mode < NBmodeIN; mode++)
        {
            data.image[IDINbuff].array.F[mode] =
                data.image[IDmodevalIN]
                .array.F[IndexOffset + inmaskindex[mode]];
        }

        //
        // Main matrix multiplication is done here
        // input vector contains recent history of mode coefficients
        // output vector contains the predicted mode coefficients
        //
        if(nbGPU > 0)  // if using GPU
        {

#ifdef HAVE_CUDA
            if(iter == 0)
            {
                printf("INITIALIZE GPU(s)\n\n");
                fflush(stdout);

                GPU_loop_MultMat_setup(GPUMATMULTCONFindex,
                                       IDPFM_name,
                                       "INbuffer",
                                       IDPFout_name,
                                       nbGPU,
                                       GPUsetPF,
                                       0,
                                       1,
                                       1,
                                       loop);

                printf("INITIALIZATION DONE\n\n");
                fflush(stdout);
            }
            GPU_loop_MultMat_execute(GPUMATMULTCONFindex,
                                     &status,
                                     &GPUstatus[100],
                                     1.0,
                                     0.0,
                                     0,
                                     0);
#endif
        }
        else // if using CPU
        {
            // compute output : matrix vector mult with a CPU-based loop
            data.image[IDPFout].md[0].write = 1;
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                data.image[IDPFout].array.F[mode] = 0.0;
                for(uint32_t ii = 0; ii < NBmodeIN * NBPFstep; ii++)
                {
                    data.image[IDPFout].array.F[mode] +=
                        data.image[IDINbuff].array.F[ii] *
                        data.image[IDPFM]
                        .array
                        .F[mode * data.image[IDPFM].md[0].size[0] + ii];
                }
            }
            COREMOD_MEMORY_image_set_sempost_byID(IDPFout, -1);
            data.image[IDPFout].md[0].write = 0;
            data.image[IDPFout].md[0].cnt0++;
        }

        if(iter == 0)
        {
            /// measure time
            //t = time(NULL);
            //uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, &timenow);
            timesec0 = 1.0 * timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

            // fprintf(fp, "%02d:%02d:%02ld.%09ld ", uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
        }

        if(SAVEMODE == 1)
        {
            //		printf("	Saving step (mode = 1) ...");
            //		fflush(stdout);

            //t = time(NULL);
            //uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, &timenow);
            timesec = 1.0 * timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

            kk = 0;
            data.image[IDsave].array.F[iter * (1 + NBmodeIN0 + NBmodeOUT)] =
                (float)(timesec - timesec0);
            //printf(" [%f] ", data.image[IDsave].array.F[iter*(1+NBmodeIN0+NBmodeOUT)]);
            kk++;
            for(mode = 0; mode < NBmodeIN0; mode++)
            {
                data.image[IDsave]
                .array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk] =
                    data.image[IDmodevalIN].array.F[IndexOffset + mode];
                kk++;
            }
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                data.image[IDsave]
                .array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk] =
                    data.image[IDPFout].array.F[mode];
                kk++;
            }
            //	printf(" done\n");
            //	fflush(stdout);
        }
        if(SAVEMODE == 2)
        {
            //	printf("	Saving step (mode = 2) ...");
            //	fflush(stdout);

            for(mode = 0; mode < NBmodeIN0; mode++)
            {
                data.image[IDsave].array.F[iter * NBmodeIN0 + mode] =
                    data.image[IDmodevalIN].array.F[IndexOffset + mode];
            }
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                data.image[IDsave]
                .array.F[iter * NBmodeIN0 + outmaskindex[mode]] =
                    data.image[IDPFout].array.F[mode];
            }
            //	printf(" done\n");
            //	fflush(stdout);
        }

        if(IDmasterout != -1)
        {
            data.image[IDmasterout].md[0].write = 1;
            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                data.image[IDmasterout].array.F[outmaskindex[mode]] =
                    data.image[IDPFout].array.F[mode];
            }
            COREMOD_MEMORY_image_set_sempost_byID(IDmasterout, -1);
            data.image[IDmasterout].md[0].write = 0;
            data.image[IDmasterout].md[0].cnt0++;
        }

        iter++;

        if(iter != NBiter)
        {
            // do this now to save time when semaphore is posted
            for(tstep = NBPFstep - 1; tstep > 0; tstep--)
            {
                // tstep-1 -> tstep
                for(mode = 0; mode < NBmodeIN; mode++)
                {
                    data.image[IDINbuff].array.F[NBmodeIN * tstep + mode] =
                        data.image[IDINbuff]
                        .array.F[NBmodeIN * (tstep - 1) + mode];
                }
            }
        }
    }
    printf("LOOP done\n");
    fflush(stdout);

    // output ASCII file
    if(SAVEMODE == 1)
    {
        printf("SAVING DATA [1] ...");
        fflush(stdout);

        printf("IDsave = %ld     %ld  %ld\n",
               IDsave,
               1 + NBmodeIN0 + NBmodeOUT,
               NBmodeOUT);
        list_image_ID();

        //	for(mode=0;mode<NBmodeOUT;mode++)
        //	printf("output %4ld -> %5ld\n", outmaskindex[mode]);

        fpout = fopen("testPFsave.dat", "w");
        for(iter = 0; iter < NBiter; iter++)
        {
            fprintf(fpout, "%5ld ", iter);
            for(kk = 0; kk < (1 + NBmodeIN0 + NBmodeOUT); kk++)
            {
                fprintf(fpout,
                        "%10f ",
                        data.image[IDsave]
                        .array.F[iter * (1 + NBmodeIN0 + NBmodeOUT) + kk]);
            }

            tlag0     = (long) tlag;
            tlagalpha = tlag - tlag0;

            ii0 = iter - (tlag0 + 1);
            ii1 = iter - (tlag0);

            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                if(ii0 > -1)
                {
                    val0 = data.image[IDsave]
                           .array.F[ii0 * (1 + NBmodeIN0 + NBmodeOUT) + 1 +
                                        NBmodeIN0 + mode];
                    val1 = data.image[IDsave]
                           .array.F[ii1 * (1 + NBmodeIN0 + NBmodeOUT) + 1 +
                                        NBmodeIN0 + mode];
                }
                val = tlagalpha * val0 + (1.0 - tlagalpha) * val1;
                fprintf(fpout, "%10f ", val);
            }
            fprintf(fpout, "\n");
        }
        fclose(fpout);

        printf(" done\n");
        fflush(stdout);
    }

    free(inmaskindex);

    if(SAVEMODE == 2)  // time shift predicted output into FITS output
    {
        tlag0     = (long) tlag;
        tlagalpha = tlag - tlag0;
        for(kk = NBiter - 1; kk > tlag0; kk--)
        {
            kk0 = kk - (tlag0 + 1);
            kk1 = kk - (tlag0);

            for(mode = 0; mode < NBmodeOUT; mode++)
            {
                val0 = data.image[IDmodevalIN]
                       .array.F[kk0 * NBmodeIN0 + outmaskindex[mode]];
                val1 = data.image[IDmodevalIN]
                       .array.F[kk1 * NBmodeIN0 + outmaskindex[mode]];
                val = tlagalpha * val0 + (1.0 - tlagalpha) * val1;

                data.image[IDsave]
                .array.F[kk * NBmodeIN0 + outmaskindex[mode]] = val;
            }
        }

        save_fits("testPFTout", "testPFTout.fits");
    }

    if(SAVEMODE > 0)
    {
        free(outmaskindex);
    }

    return IDPFout;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 5. MISC TOOLS, DIAGNOSTICS                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

//
// IDin_name is a 2 or 3D image, open-loop disturbance
// last axis is time (step)
// this optimization asssumes no correlation in noise
//
float LINARFILTERPRED_ScanGain(char *IDin_name, float multfact, float framelag)
{
    float   gain;
    float   gainmax = 1.1;
    float   optgainblock;
    float   residualblock;
    float   residualblock0;
    float   gainstep = 0.01;
    imageID IDin;

    long nbstep;
    long step, step0, step1;

    long  framelag0;
    long  framelag1;
    float alpha;

    float *actval_array; // actuator value
    float  actval;

    long nbvar;
    long axis, naxis;

    double *errval;
    double  errvaltot;
    long    cnt;

    FILE *fp;
    char  fname[200];
    float mval;
    long  ii;
    float tmpv;

    int   TEST       = 0;
    float TESTperiod = 20.0;

    // results
    float *optgain;
    float *optres;
    float *res0;
    int    optinit = 0;

    if(framelag < 1.00000001)
    {
        printf("ERROR: framelag should be be > 1\n");
        exit(0);
    }

    IDin  = image_ID(IDin_name);
    naxis = data.image[IDin].md[0].naxis;

    nbvar = 1;
    for(axis = 0; axis < naxis - 1; axis++)
    {
        nbvar *= data.image[IDin].md[0].size[axis];
    }

    errval = (double *) malloc(sizeof(double) * nbvar);
    if(errval == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nbstep = data.image[IDin].md[0].size[naxis - 1];

    framelag0 = (long) framelag;
    framelag1 = framelag0 + 1;
    alpha     = framelag - framelag0;

    printf("alpha = %f    nbvar = %ld\n", alpha, nbvar);

    list_image_ID();
    if(TEST == 1)
    {
        for(ii = 0; ii < nbvar; ii++)
            for(step = 0; step < nbstep; step++)
            {
                data.image[IDin].array.F[step * nbvar + ii] =
                    1.0 * sin(2.0 * M_PI * step / TESTperiod);
            }
    }

    actval_array = (float *) malloc(sizeof(float) * nbstep);
    if(actval_array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    optgain = (float *) malloc(sizeof(float) * nbvar);
    if(optgain == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    optres = (float *) malloc(sizeof(float) * nbvar);
    if(optres == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    res0 = (float *) malloc(sizeof(float) * nbvar);
    if(res0 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sprintf(fname, "gainscan.txt");

    gain          = 0.2;
    ii            = 0;
    fp            = fopen(fname, "w");
    residualblock = 1.0e20;
    optgainblock  = 0.0;
    for(gain = 0; gain < gainmax; gain += gainstep)
    {
        fprintf(fp, "%5.3f", gain);

        errvaltot = 0.0;
        for(ii = 0; ii < nbvar; ii++)
        {
            errval[ii] = 0.0;
            cnt        = 0.0;
            for(step = 0; step < framelag1 + 2; step++)
            {
                actval_array[step] = 0.0;
            }
            for(step = framelag1; step < nbstep; step++)
            {
                step0 = step - framelag0;
                step1 = step - framelag1;

                actval = (1.0 - alpha) * actval_array[step0] +
                         alpha * actval_array[step1];
                mval = ((1.0 - alpha) *
                        data.image[IDin].array.F[step0 * nbvar + ii] +
                        alpha * data.image[IDin].array.F[step1 * nbvar + ii]) -
                       actval;
                actval_array[step] =
                    multfact * (actval_array[step - 1] + gain * mval);
                tmpv = data.image[IDin].array.F[step * nbvar + ii] -
                       actval_array[step];
                errval[ii] += tmpv * tmpv;
                cnt++;
            }
            errval[ii] = sqrt(errval[ii] / cnt);
            fprintf(fp, " %10f", errval[ii]);
            errvaltot += errval[ii] * errval[ii];

            if(optinit == 0)
            {
                optgain[ii] = gain;
                optres[ii]  = errval[ii];
                res0[ii]    = errval[ii];
            }
            else
            {
                if(errval[ii] < optres[ii])
                {
                    optres[ii]  = errval[ii];
                    optgain[ii] = gain;
                }
            }
        }

        if(optinit == 0)
        {
            residualblock0 = errvaltot;
        }

        optinit = 1;
        fprintf(fp, "%10f\n", errvaltot);

        if(errvaltot < residualblock)
        {
            residualblock = errvaltot;
            optgainblock  = gain;
        }
    }
    fclose(fp);

    free(actval_array);
    free(errval);

    for(ii = 0; ii < nbvar; ii++)
    {
        printf(
            "MODE %4ld    optimal gain = %5.2f     residual = %.6f -> %.6f \n",
            ii,
            optgain[ii],
            res0[ii],
            optres[ii]);
    }

    printf("\noptimal block gain = %f     residual = %.6f -> %.6f\n\n",
           optgainblock,
           sqrt(residualblock0),
           sqrt(residualblock));
    printf("RMS per mode = %f -> %f\n",
           sqrt(residualblock0 / nbvar),
           sqrt(residualblock / nbvar));

    free(optgain);
    free(optres);
    free(res0);

    return (optgainblock);
}

#include <math.h>

// log all debug trace points to file
#define DEBUGLOG

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_tools/COREMOD_tools.h"

// Local variables pointers
static char   *outimname;

static uint32_t *sizexout;
static uint32_t *sizeyout;


static uint64_t *centered;
long fpi_centered;

static float *xcent;
static float *ycent;

static float *CPAmaxval;

static float *deltaCPAval;

static float *radiusval;

static float *radiusfactorlimval;

static float *fpowerlaw;

static float *fpowerlaw_minf;

static float *fpowerlaw_maxf;

static uint32_t   *writefileval;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR,
        ".out_name",
        "output image",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_UINT32,
        ".sizex",
        "sizex",
        "512",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &sizexout,
        NULL
    },
    {
        CLIARG_UINT32,
        ".sizey",
        "sizey",
        "512",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &sizeyout,
        NULL
    },
    {
        CLIARG_ONOFF,
        ".align.centered",
        "on if centered",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &centered,
        &fpi_centered
    },
    {
        CLIARG_FLOAT32,
        ".align.xcenter",
        "x axis center",
        "200",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &xcent,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".align.ycenter",
        "y axis center",
        "200",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &ycent,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".CPAmax",
        "maximum cycle per aperture",
        "8.0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &CPAmaxval,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".deltaCPA",
        "CPA interval",
        "0.8",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &deltaCPAval,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".radius",
        "disk radius",
        "160.0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &radiusval,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".radfactlim",
        "radius factor limit",
        "1.5",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &radiusfactorlimval,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".fpowerlaw",
        "frequency power law (amp x f^a)",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &fpowerlaw,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".fpowerlaw_minf",
        "frequency power law max freq",
        "1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &fpowerlaw_minf,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".fpowerlaw_maxf",
        "frequency power law max freq",
        "100.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &fpowerlaw_maxf,
        NULL
    },
    {
        CLIARG_UINT32,
        ".writefile",
        "write file flag",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &writefileval,
        NULL
    }
};




// Optional custom configuration setup.
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {

    }

    return RETURN_SUCCESS;
}

// Optional custom configuration checks.
// Runs at every configuration check loop iteration
//
static errno_t customCONFcheck()
{

    if(data.fpsptr != NULL)
    {

    }

    return RETURN_SUCCESS;
}







static CLICMDDATA CLIcmddata =
{
    "mkFouriermodes", "make basis of Fourier Modes", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




errno_t linopt_imtools_makeCPAmodes(
    const char *ID_name,
    uint32_t        sizex,
    uint32_t        sizey,
    float       xcenter,
    float       ycenter,
    float       CPAmax,
    float       deltaCPA,
    float       radius,
    float       radfactlim,
    float       fpowerlaw,
    float       fpowerlaw_minf,
    float       fpowerlaw_maxf,
    uint32_t         writeMfile,
    long       *outNBmax
)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s", ID_name);

    imageID ID;
    imageID IDx, IDy, IDr;
    float   CPAx, CPAy;
    float   x, y, r;
    long    ii, jj;
    long    k, k1;
    long    NBmax;
    float  *CPAxarray;
    float  *CPAyarray;
    float  *CPArarray;
    long    size2;
    long    NBfrequ;
    float   eps;
    FILE   *fp;

    long IDfreq;

    eps = 0.1 * deltaCPA;
    printf("size       = %u %u\n", sizex, sizey);
    printf("CPAmax     = %f\n", CPAmax);
    printf("deltaCPA   = %f\n", deltaCPA);
    printf("radius     = %f\n", radius);
    printf("radfactlim = %f\n", radfactlim);


    size2 = sizex * sizey;
    FUNC_CHECK_RETURN(create_2Dimage_ID("cpa_tmpx", sizex, sizey, &IDx));

    FUNC_CHECK_RETURN(create_2Dimage_ID("cpa_tmpy", sizex, sizey, &IDy));

    FUNC_CHECK_RETURN(create_2Dimage_ID("cpa_tmpr", sizex, sizey, &IDr));

    printf("precomputing x, y, r\n");
    fflush(stdout);

    for(ii = 0; ii < sizex; ii++)
    {
        x = (1.0 * ii - xcenter + 0.5) / radius;
        for(jj = 0; jj < sizey; jj++)
        {
            y = (1.0 * jj - ycenter + 0.5) / radius;
            r = sqrt(x * x + y * y);
            data.image[IDx].array.F[jj * sizex + ii] = x;
            data.image[IDy].array.F[jj * sizex + ii] = y;
            data.image[IDr].array.F[jj * sizex + ii] = r;
        }
    }

    printf("CPA: max = %f   delta = %f\n", CPAmax, deltaCPA);
    fflush(stdout);
    NBfrequ = 0;
    for(CPAx = 0; CPAx < CPAmax; CPAx += deltaCPA)
        for(CPAy = -CPAmax; CPAy < CPAmax; CPAy += deltaCPA)
        {
            NBfrequ++;
        }

    DEBUG_TRACEPOINT("NBfrequ = %ld", NBfrequ);

    CPAxarray = (float *) malloc(sizeof(float) * NBfrequ);
    if(CPAxarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    CPAyarray = (float *) malloc(sizeof(float) * NBfrequ);
    if(CPAyarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    CPArarray = (float *) malloc(sizeof(float) * NBfrequ);
    if(CPArarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    NBfrequ = 0;
    //ydist = 2.0*deltaCPA;
    //y0 = 0.0;
    for(CPAx = 0; CPAx < CPAmax; CPAx += deltaCPA)
    {
        for(CPAy = 0; CPAy < CPAmax; CPAy += deltaCPA)
        {
            CPAxarray[NBfrequ] = CPAx;
            CPAyarray[NBfrequ] = CPAy;
            CPArarray[NBfrequ] = sqrt(CPAx * CPAx + CPAy * CPAy);
            NBfrequ++;
        }
        if(CPAx > eps)
        {
            for(CPAy = -deltaCPA; CPAy > -CPAmax; CPAy -= deltaCPA)
            {
                CPAxarray[NBfrequ] = CPAx;
                CPAyarray[NBfrequ] = CPAy;
                CPArarray[NBfrequ] = sqrt(CPAx * CPAx + CPAy * CPAy);
                NBfrequ++;
            }
        }
    }

    //  for(k1=0;k1<NBfrequ;k1++)
    //printf("%ld %f %f %f\n", k1, CPAxarray[k1], CPAyarray[k1], CPArarray[k1]);

    //  printf("sorting\n");
    // fflush(stdout);

    quick_sort3_float(CPArarray, CPAxarray, CPAyarray, NBfrequ);

    NBmax = NBfrequ * 2;
    FUNC_CHECK_RETURN(create_3Dimage_ID(ID_name, sizex, sizey, NBmax - 1, &ID));


    if(writeMfile == 1)
    {
        fp = fopen("ModesExpr_CPA.txt", "w");
        fprintf(fp, "# size       = %u %u\n", sizex, sizey);
        fprintf(fp, "# CPAmax     = %f\n", CPAmax);
        fprintf(fp, "# deltaCPA   = %f\n", deltaCPA);
        fprintf(fp, "# radius     = %f\n", radius);
        fprintf(fp, "# radfactlim = %f\n", radfactlim);
        fprintf(fp, "# \n");
        fprintf(fp, "# Unit for x and y = radius [pixel]\n");
        fprintf(fp, "# \n");
        fprintf(fp, "%4ld %10.5f %10.5f    1.0\n", (long) 0, 0.0, 0.0);
        k1 = 1;
        k  = 2;
        while(k < NBmax)
        {
            CPAx = CPAxarray[k1];
            CPAy = CPAyarray[k1];
            float frequency = sqrt(CPAx * CPAx + CPAy * CPAy);


            float fampl = 1.0;
            if(frequency < fpowerlaw_minf)
            {
                fampl = 1.0;
            }
            else if(frequency > fpowerlaw_maxf)
            {
                fampl = pow(fpowerlaw_maxf / fpowerlaw_minf,  fpowerlaw);
            }
            else
            {
                float f1 = frequency / fpowerlaw_minf;
                fampl = pow(f1, fpowerlaw);
            }


            if(CPAy < 0)
            {
                fprintf(fp,
                        "%4ld   %8.3f -> %8.3f  %10.5f %10.5f    cos(M_PI*(x*%.5f-y*%.5f))\n",
                        k - 1,
                        frequency, fampl,
                        CPAx,
                        CPAy,
                        CPAx,
                        -CPAy);
                fprintf(fp,
                        "%4ld   %8.3f -> %8.3f  %10.5f %10.5f    sin(M_PI*(x*%.5f-y*%.5f))\n",
                        k,
                        frequency, fampl,
                        CPAx,
                        CPAy,
                        CPAx,
                        -CPAy);
            }
            else
            {
                fprintf(fp,
                        "%4ld   %8.3f -> %8.3f  %10.5f %10.5f    cos(M_PI*(x*%.5f+y*%.5f))\n",
                        k - 1,
                        frequency, fampl,
                        CPAx,
                        CPAy,
                        CPAx,
                        CPAy);
                fprintf(fp,
                        "%4ld   %8.3f -> %8.3f  %10.5f %10.5f    sin(M_PI*(x*%.5f+y*%.5f))\n",
                        k,
                        frequency, fampl,
                        CPAx,
                        CPAy,
                        CPAx,
                        CPAy);
            }
            k += 2;
            k1++;
        }

        fclose(fp);
    }

    FUNC_CHECK_RETURN(
        delete_image_ID("cpamodesfreq", DELETE_IMAGE_ERRMODE_IGNORE));

    DEBUG_TRACEPOINT("Create cpamodesfreq");

    FUNC_CHECK_RETURN(create_2Dimage_ID("cpamodesfreq", NBmax - 1, 1, &IDfreq));

    DEBUG_TRACEPOINT("IDfreq %ld", IDfreq);
    DEBUG_TRACEPOINT("IDx %ld", IDx);
    DEBUG_TRACEPOINT("IDy %ld", IDy);
    DEBUG_TRACEPOINT("IDr %ld", IDr);
    DEBUG_TRACEPOINT("ID %ld", ID);
    DEBUG_TRACEPOINT("size2 %ld", size2);
    list_image_ID();

    // mode 0 (piston)
    data.image[IDfreq].array.F[0] = 0.0;
    for(ii = 0; ii < size2; ii++)
    {
        x = data.image[IDx].array.F[ii];
        y = data.image[IDy].array.F[ii];
        r = data.image[IDr].array.F[ii];
        if(r < radfactlim)
        {
            data.image[ID].array.F[ii] = 1.0;
        }
    }

    k1 = 1;
    k  = 2;
    while(k < NBmax)
    {
        DEBUG_TRACEPOINT("k = %ld / %ld   k1 = %ld / %ld",
                         k,
                         NBmax,
                         k1,
                         NBfrequ);

        CPAx = CPAxarray[k1];
        CPAy = CPAyarray[k1];
        DEBUG_TRACEPOINT("    %ld %f %f", k1, CPAx, CPAy);

        float frequency = sqrt(CPAx * CPAx + CPAy * CPAy);

        float fampl = 1.0;
        if(frequency < fpowerlaw_minf)
        {
            fampl = 1.0;
        }
        else if(frequency > fpowerlaw_maxf)
        {
            fampl = pow(fpowerlaw_maxf / fpowerlaw_minf,  fpowerlaw);
        }
        else
        {
            float f1 = frequency / fpowerlaw_minf;
            fampl = pow(f1, fpowerlaw);
        }

        for(ii = 0; ii < size2; ii++)
        {
            x                                 = data.image[IDx].array.F[ii];
            y                                 = data.image[IDy].array.F[ii];
            r                                 = data.image[IDr].array.F[ii];
            data.image[IDfreq].array.F[k - 1] = frequency;
            data.image[IDfreq].array.F[k]     = frequency;
            if(r < radfactlim)
            {
                data.image[ID].array.F[(k - 1) * size2 + ii] =
                    fampl * cos(M_PI * (x * CPAx + y * CPAy));
                data.image[ID].array.F[k * size2 + ii] =
                    fampl * sin(M_PI * (x * CPAx + y * CPAy));
            }
        }
        k += 2;
        k1++;
    }

    DEBUG_TRACEPOINT("free memory");

    free(CPAxarray);
    free(CPAyarray);
    free(CPArarray);

    DEBUG_TRACEPOINT("delete tmp files");

    FUNC_CHECK_RETURN(
        delete_image_ID("cpa_tmpx", DELETE_IMAGE_ERRMODE_WARNING));

    FUNC_CHECK_RETURN(
        delete_image_ID("cpa_tmpy", DELETE_IMAGE_ERRMODE_WARNING));

    FUNC_CHECK_RETURN(
        delete_image_ID("cpa_tmpr", DELETE_IMAGE_ERRMODE_WARNING));


    if(outNBmax != NULL)
    {
        *outNBmax = NBmax;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    printf("outimname                %s\n", outimname);
    printf("sizeout                  %u %u\n", *sizexout, *sizeyout);
    printf("CPAmaxval                %f\n", *CPAmaxval);
    printf("deltaCPAval              %f\n", *deltaCPAval);
    printf("radiusval                %f\n", *radiusval);
    printf("radiusfactorlimval       %f\n", *radiusfactorlimval);
    printf("writefileval             %u\n", *writefileval);


    float x0 = 0.0;
    float y0 = 0.0;

    printf("centered flag  :   %lu\n", *centered);
    if(*centered == 1)
    {
        printf("CENTERED      ");
        x0 = 0.5 * *sizexout;
        y0 = 0.5 * *sizeyout;
    }
    else
    {
        printf("NOT CENTERED  ");
        x0 = *xcent;
        y0 = *ycent;
    }
    printf(" %8.3f x %8.3f\n", x0, y0);


    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {


        linopt_imtools_makeCPAmodes(outimname,
                                    *sizexout,
                                    *sizeyout,
                                    x0,
                                    y0,
                                    *CPAmaxval,
                                    *deltaCPAval,
                                    *radiusval,
                                    *radiusfactorlimval,
                                    *fpowerlaw,
                                    *fpowerlaw_minf,
                                    *fpowerlaw_maxf,
                                    *writefileval,
                                    NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__makeCPAmodes()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

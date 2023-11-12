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

// Radial CPA
static float *rCPAminval;
static float *rCPAmaxval;

// sampling xy CPA
static float *CPAmaxval;

static float *deltaCPAval;

static float *radiusval;

static float *radiusfactorlimval;

static float *fpowerlaw;

static float *fpowerlaw_minf;

static float *fpowerlaw_maxf;

static uint32_t   *writefileval;

static char *maskim;

// extrapolation factor
// extrapolate out to extrfactor radian
static float *extrfactor;

// extrapolation offset
static float *extroffset;



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
        ".rCPAmin",
        "minimum radial cycle per aperture",
        "-1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &rCPAminval,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".rCPAmax",
        "maximum radial cycle per aperture",
        "1008.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &rCPAmaxval,
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
        "frequency power law min freq",
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
    },
    {
        CLIARG_IMG,
        ".maskim",
        "optional mask for extrapolation",
        "mask",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &maskim,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".extrfactor",
        "extrapolation factor [radian]",
        "1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &extrfactor,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".extroffset",
        "extrapolation offset [pix]",
        "0.5",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &extroffset,
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
    IMGID *imgoutm,
    uint32_t        sizex,
    uint32_t        sizey,
    float       xcenter,
    float       ycenter,
    float       rCPAmin,
    float       rCPAmax,
    float       CPAmax,
    float       deltaCPA,
    float       radius,
    float       radfactlim,
    float       fpowerlaw,
    float       fpowerlaw_minf,
    float       fpowerlaw_maxf,
    uint32_t    writeMfile,
    long       *outNBmax,
    IMGID       imgmask,
    float       extrfactor,
    float       extroffset
)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s", ID_name);

    float  *CPAxarray;
    float  *CPAyarray;
    float  *CPArarray;
    long    NBfrequ;
    float   eps;
    FILE   *fp;

    long IDfreq;

    eps = 0.1 * deltaCPA;
    printf("size       = %u %u\n", sizex, sizey);
    printf("rCPAmin    = %f\n", rCPAmin);
    printf("rCPAmax    = %f\n", rCPAmax);
    printf("CPAmax     = %f\n", CPAmax);
    printf("deltaCPA   = %f\n", deltaCPA);
    printf("radius     = %f\n", radius);
    printf("radfactlim = %f\n", radfactlim);


    long sizexy = sizex * sizey;


    IMGID imgx = mkIMGID_from_name("cpa_tmpx");
    imgx.naxis = 2;
    imgx.datatype = _DATATYPE_FLOAT;
    imgx.size[0] = sizex;
    imgx.size[1] = sizey;
    createimagefromIMGID(&imgx);

    IMGID imgy = mkIMGID_from_name("cpa_tmpy");
    imgy.naxis = 2;
    imgy.datatype = _DATATYPE_FLOAT;
    imgy.size[0] = sizex;
    imgy.size[1] = sizey;
    createimagefromIMGID(&imgy);

    IMGID imgr = mkIMGID_from_name("cpa_tmpr");
    imgr.naxis = 2;
    imgr.datatype = _DATATYPE_FLOAT;
    imgr.size[0] = sizex;
    imgr.size[1] = sizey;
    createimagefromIMGID(&imgr);

    list_image_ID();


    printf("precomputing x, y, r\n");
    fflush(stdout);

    for(uint32_t ii = 0; ii < sizex; ii++)
    {
        float x = (1.0 * ii - xcenter ) / radius;
        for(uint32_t jj = 0; jj < sizey; jj++)
        {
            float y = (1.0 * jj - ycenter ) / radius;
            float r = sqrt(x * x + y * y);
            imgx.im->array.F[jj * sizex + ii] = x;
            imgy.im->array.F[jj * sizex + ii] = y;
            imgr.im->array.F[jj * sizex + ii] = r;
        }
    }


    // If mask exists, compute distance to mask for extrapolation
    //
    int MASKext = 0; // toggles to 1 if applying mask for extrapolation
    resolveIMGID(&imgmask, ERRMODE_WARN);

    IMGID imgpixdist = mkIMGID_from_name("pixdist");

    if( imgmask.ID != -1)
    {
        MASKext = 1;
        printf("processing mask\n");
        fflush(stdout);

        imgpixdist.naxis = 2;
        imgpixdist.datatype = _DATATYPE_FLOAT;
        imgpixdist.size[0] = sizex;
        imgpixdist.size[1] = sizey;
        createimagefromIMGID(&imgpixdist);


        // store mask pixel

        // count number of active pixel in mask
        long NBmaskpix = 0;
        for(uint32_t ii = 0; ii < sizexy; ii++)
        {
            if( imgmask.im->array.F[ii] > 0.5)
            {
                NBmaskpix ++;
            }
        }

        printf("mask has %ld active pixels\n", NBmaskpix);
        fflush(stdout);

        float * maskx = (float*) malloc(sizeof(float)*NBmaskpix);
        float * masky = (float*) malloc(sizeof(float)*NBmaskpix);

        {
            long mpix = 0;
            for(uint32_t ii = 0; ii < sizexy; ii++)
            {
                if( imgmask.im->array.F[ii] > 0.5)
                {
                    maskx[mpix] = imgx.im->array.F[ii];
                    masky[mpix] = imgy.im->array.F[ii];
                    mpix ++;
                }
            }
        }


        printf("maskx, masky : done\n");
        fflush(stdout);


        for(uint32_t ii0 = 0; ii0 < sizex*sizey; ii0++)
        {
            if( imgmask.im->array.F[ii0] > 0.5 )
            {
                imgpixdist.im->array.F[ii0] = 0.0;
            }
            else
            {
                // initialize to absurdly large value
                imgpixdist.im->array.F[ii0] = sizex + sizey;

                float x0 = imgx.im->array.F[ii0];
                float y0 = imgy.im->array.F[ii0];
                for(uint32_t mpix = 0; mpix < NBmaskpix; mpix++)
                {
                    float dx = x0 - maskx[mpix];
                    float dy = y0 - masky[mpix];
                    float dr2 = dx*dx + dy*dy;
                    float dr = sqrt(dr2);

                    if( dr < imgpixdist.im->array.F[ii0] )
                    {
                        imgpixdist.im->array.F[ii0] = dr;
                    }
                }
            }
        }

        free(maskx);
        free(masky);
    }




    printf("CPA: max = %f   delta = %f\n", CPAmax, deltaCPA);
    fflush(stdout);
    NBfrequ = 0;
    for(float CPAx = 0; CPAx < CPAmax; CPAx += deltaCPA)
        for(float CPAy = -CPAmax; CPAy < CPAmax; CPAy += deltaCPA)
        {
            float CPAr = sqrt(CPAx*CPAx + CPAy*CPAy);
            if(CPAr>0.001) // excluding piston from array
            {
                if( (CPAr > rCPAmin) && (CPAr < rCPAmax))
                {
                    NBfrequ++;
                }
            }
        }
    printf("%ld spatial frequencies\n", NBfrequ);

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
    for(float CPAx = 0; CPAx < CPAmax; CPAx += deltaCPA)
    {
        for(float CPAy = 0; CPAy < CPAmax; CPAy += deltaCPA)
        {
            float CPAr = sqrt(CPAx*CPAx + CPAy*CPAy);
            if(CPAr>0.001) // excluding piston from array
            {
                CPAxarray[NBfrequ] = CPAx;
                CPAyarray[NBfrequ] = CPAy;
                CPArarray[NBfrequ] = CPAr;
                if( (CPAr > rCPAmin) && (CPAr < rCPAmax))
                {
                    NBfrequ++;
                }
            }
        }

        if(CPAx > eps)
        {
            for(float CPAy = -deltaCPA; CPAy > -CPAmax; CPAy -= deltaCPA)
            {
                float CPAr = sqrt(CPAx*CPAx + CPAy*CPAy);
                CPAxarray[NBfrequ] = CPAx;
                CPAyarray[NBfrequ] = CPAy;
                CPArarray[NBfrequ] = CPAr;
                if( (CPAr > rCPAmin) && (CPAr < rCPAmax))
                {
                    NBfrequ++;
                }
            }
        }
    }
    printf("%ld spatial frequencies\n", NBfrequ);

//  for(k1=0;k1<NBfrequ;k1++)
//printf("%ld %f %f %f\n", k1, CPAxarray[k1], CPAyarray[k1], CPArarray[k1]);

//  printf("sorting\n");
// fflush(stdout);

    quick_sort3_float(CPArarray, CPAxarray, CPAyarray, NBfrequ);

// 2 modes (sin, cos) per frequency
    long NBmax = NBfrequ * 2;
    if ( rCPAmin < 0.0 )
    {
        // piston term included
        NBmax = NBfrequ * 2 + 1;
    }

    printf("%ld modes\n", NBmax);


    imgoutm->naxis = 3;
    imgoutm->datatype = _DATATYPE_FLOAT;
    imgoutm->size[0] = sizex;
    imgoutm->size[1] = sizey;
    imgoutm->size[2] = NBmax;
    createimagefromIMGID(imgoutm);

//    imageID ID;
//    FUNC_CHECK_RETURN(create_3Dimage_ID(ID_name, sizex, sizey, NBmax, &ID));


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
        long k1 = 1;
        long k  = 2;
        while(k < NBmax)
        {
            float CPAx = CPAxarray[k1];
            float CPAy = CPAyarray[k1];
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

    FUNC_CHECK_RETURN(create_2Dimage_ID("cpamodesfreq", NBmax, 1, &IDfreq));

    DEBUG_TRACEPOINT("IDfreq %ld", IDfreq);
    DEBUG_TRACEPOINT("IDx %ld", IDx);
    DEBUG_TRACEPOINT("IDy %ld", IDy);
    DEBUG_TRACEPOINT("IDr %ld", IDr);
    DEBUG_TRACEPOINT("ID %ld", ID);
    DEBUG_TRACEPOINT("size2 %ld", size2);
    list_image_ID();


// CPA array index
    long k1 = 0;

// cube slice index
    long k  = 0;

    if ( rCPAmin <= 0.0 )
    {
        // mode 0 (piston) included

        data.image[IDfreq].array.F[0] = 0.0;
        for(uint32_t ii = 0; ii < sizexy; ii++)
        {
            float r = imgr.im->array.F[ii];
            if(r < radfactlim)
            {
                imgoutm->im->array.F[ii] = 1.0;
            }
        }
        k ++;
    }


    while(k < NBmax)
    {
        DEBUG_TRACEPOINT("k = %ld / %ld   k1 = %ld / %ld",
                         k,
                         NBmax,
                         k1,
                         NBfrequ);

        float CPAx = CPAxarray[k1];
        float CPAy = CPAyarray[k1];
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

        for(uint32_t ii = 0; ii < sizexy; ii++)
        {
            float x                           = imgx.im->array.F[ii];
            float y                           = imgy.im->array.F[ii];
            float r                           = imgr.im->array.F[ii];
            data.image[IDfreq].array.F[k] = frequency;
            data.image[IDfreq].array.F[k+1]     = frequency;
            if(r < radfactlim)
            {
                // attenuation factor for extrapolation
                float afact = 1.0;
                if( MASKext == 1 )
                {
                    float pdist = imgpixdist.im->array.F[ii];
                    float afact0 = 1.0 + extroffset - pdist * sqrt(CPAx*CPAx + CPAy*CPAy) * M_PI / extrfactor;
                    if(afact0 > 1.0)
                    {
                        afact0 = 1.0;
                    }
                    if(afact0 < 0.0)
                    {
                        afact0 = 0.0;
                    }
                    //afact0 = 0.5;
                    afact = 0.5 * (cos( (1.0-afact0)*M_PI) + 1.0);
                }

                imgoutm->im->array.F[(k) * sizexy + ii] =
                    fampl * afact * cos(M_PI * (x * CPAx + y * CPAy));

                imgoutm->im->array.F[(k+1) * sizexy + ii] =
                    fampl * afact * sin(M_PI * (x * CPAx + y * CPAy));
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
    printf("rCPAminval               %f\n", *rCPAminval);
    printf("rCPAmaxval               %f\n", *rCPAmaxval);
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


    // optional mask
    //
    IMGID imgmask = mkIMGID_from_name(maskim);
    resolveIMGID(&imgmask, ERRMODE_WARN);

    IMGID imgoutm = mkIMGID_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {


        linopt_imtools_makeCPAmodes(&imgoutm,
                                    *sizexout,
                                    *sizeyout,
                                    x0,
                                    y0,
                                    *rCPAminval,
                                    *rCPAmaxval,
                                    *CPAmaxval,
                                    *deltaCPAval,
                                    *radiusval,
                                    *radiusfactorlimval,
                                    *fpowerlaw,
                                    *fpowerlaw_minf,
                                    *fpowerlaw_maxf,
                                    *writefileval,
                                    NULL,
                                    imgmask,
                                    *extrfactor,
                                    *extroffset);

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

/**
 * @file makeCPAmodes.c
 * @brief log all debug trace points to file
 */

#include <math.h>

// log all debug trace points to file
#define DEBUGLOG

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_tools/COREMOD_tools.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkFouriermodes",
    .cmdkey      = "mkFouriermodes",
    .description = "make basis of Fourier Modes",
    .description_long =
        "Generate a basis of Fourier (CPA) modes within a pupil aperture. Creates sinusoidal modes at specified spatial frequencies."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   * outimname = NULL;
static uint32_t * sizexout = NULL;
static uint32_t * sizeyout = NULL;
static uint64_t * centered = NULL;
static float * xcent = NULL;
static float * ycent = NULL;
static float * rCPAminval = NULL;
static float * rCPAmaxval = NULL;
static float * CPAmaxval = NULL;
static float * deltaCPAval = NULL;
static float * radiusval = NULL;
static float * radiusfactorlimval = NULL;
static float * fpowerlaw = NULL;
static float * fpowerlaw_minf = NULL;
static float * fpowerlaw_maxf = NULL;
static uint32_t   * writefileval = NULL;
static char * maskim = NULL;
static float * extrfactor = NULL;
static float * extroffset = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".out_name", &outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".sizex", &sizexout, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sizex") \
    X(".sizey", &sizeyout, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sizey") \
    X(".CPAmax", &CPAmaxval, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "maximum cycle per aperture") \
    X(".deltaCPA", &deltaCPAval, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "CPA interval") \
    X(".radius", &radiusval, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "disk radius") \
    X(".radfactlim", &radiusfactorlimval, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "radius factor limit") \
    X(".writefile", &writefileval, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "write file flag")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

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


    long    NBfrequ;
    float   eps __attribute__((unused));
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


    IMGID imgx = imgid_make_from_name("cpa_tmpx");
    imgx.mdt->naxis = 2;
    imgx.mdt->datatype = _DATATYPE_FLOAT;
    imgx.mdt->size[0] = sizex;
    imgx.mdt->size[1] = sizey;
    createimagefromIMGID(&imgx);

    IMGID imgy = imgid_make_from_name("cpa_tmpy");
    imgy.mdt->naxis = 2;
    imgy.mdt->datatype = _DATATYPE_FLOAT;
    imgy.mdt->size[0] = sizex;
    imgy.mdt->size[1] = sizey;
    createimagefromIMGID(&imgy);

    IMGID imgr = imgid_make_from_name("cpa_tmpr");
    imgr.mdt->naxis = 2;
    imgr.mdt->datatype = _DATATYPE_FLOAT;
    imgr.mdt->size[0] = sizex;
    imgr.mdt->size[1] = sizey;
    createimagefromIMGID(&imgr);

    list_image_ID();


    printf("precomputing x, y, r\n");
    fflush(stdout);

    for(uint32_t ii = 0; ii < sizex; ii++)
    {
        float x = (1.0 * ii - xcenter) / radius;
        for(uint32_t jj = 0; jj < sizey; jj++)
        {
            float y = (1.0 * jj - ycenter) / radius;
            float r = sqrtf(x * x + y * y);
            imgx.im->array.F[jj * sizex + ii] = x;
            imgy.im->array.F[jj * sizex + ii] = y;
            imgr.im->array.F[jj * sizex + ii] = r;
        }
    }


    // If mask exists, compute distance to mask for extrapolation
    //
    int MASKext = 0; // toggles to 1 if applying mask for extrapolation
    resolveIMGID(&imgmask, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imgpixdist = imgid_make_from_name("pixdist");

    if(imgmask.ID != -1)
    {
        MASKext = 1;
        printf("processing mask\n");
        fflush(stdout);

        imgpixdist.mdt->naxis = 2;
        imgpixdist.mdt->datatype = _DATATYPE_FLOAT;
        imgpixdist.mdt->size[0] = sizex;
        imgpixdist.mdt->size[1] = sizey;
        createimagefromIMGID(&imgpixdist);


        // store mask pixel

        // count number of active pixel in mask
        long NBmaskpix = 0;
        for(uint32_t ii = 0; ii < sizexy; ii++)
        {
            if(imgmask.im->array.F[ii] > 0.5)
            {
                NBmaskpix ++;
            }
        }

        printf("mask has %ld active pixels\n", NBmaskpix);
        fflush(stdout);

        float *maskx = (float *) malloc(sizeof(float) * NBmaskpix);
        float *masky = (float *) malloc(sizeof(float) * NBmaskpix);

        {
            long mpix = 0;
            for(uint32_t ii = 0; ii < sizexy; ii++)
            {
                if(imgmask.im->array.F[ii] > 0.5)
                {
                    maskx[mpix] = imgx.im->array.F[ii];
                    masky[mpix] = imgy.im->array.F[ii];
                    mpix ++;
                }
            }
        }


        printf("maskx, masky : done\n");
        fflush(stdout);


        for(uint32_t ii0 = 0; ii0 < sizex * sizey; ii0++)
        {
            if(imgmask.im->array.F[ii0] > 0.5)
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
                    float dr2 = dx * dx + dy * dy;
                    float dr = sqrtf(dr2);

                    if(dr < imgpixdist.im->array.F[ii0])
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

    {
        int initCPAx = 0;
        for(float CPAx = 0; CPAx < CPAmax; CPAx += deltaCPA)
        {
            int initCPAy = 0;
            for(float CPAy = 0.0; CPAy < CPAmax; CPAy += deltaCPA)
            {
                float CPAr = sqrtf(CPAx * CPAx + CPAy * CPAy);
                if(CPAr > 0.001) // excluding piston from array
                {
                    if((CPAr > rCPAmin) && (CPAr < rCPAmax))
                    {
                        //printf("%5ld  CORE  : %f %f\n", NBfrequ, CPAx, CPAy);
                        NBfrequ++;


                        if(initCPAx == 1)   // not on the x=0 line
                        {
                            if(initCPAy == 1)   // not on the y=0 line
                            {
                                NBfrequ++;
                            }
                        }

                    }
                }
                initCPAy = 1;
            }
            initCPAx = 1; // no longer on x=0 line
        }
    }
    printf("%ld spatial frequencies\n", NBfrequ);


    DEBUG_TRACEPOINT("NBfrequ = %ld", NBfrequ);

    float *CPAxarray = (float *) malloc(sizeof(float) * NBfrequ);
    if(CPAxarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    float *CPAyarray = (float *) malloc(sizeof(float) * NBfrequ);
    if(CPAyarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    float *CPArarray = (float *) malloc(sizeof(float) * NBfrequ);
    if(CPArarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }


    NBfrequ = 0;
    {
        int initCPAx = 0;
        for(float CPAx = 0; CPAx < CPAmax; CPAx += deltaCPA)
        {
            int initCPAy = 0;
            for(float CPAy = 0.0; CPAy < CPAmax; CPAy += deltaCPA)
            {
                float CPAr = sqrtf(CPAx * CPAx + CPAy * CPAy);
                if(CPAr > 0.001) // excluding piston from array
                {
                    if((CPAr > rCPAmin) && (CPAr < rCPAmax))
                    {
                        //printf("%5ld  CORE  : %+f %+f   %6.3f\n", NBfrequ, CPAx, CPAy, CPAr);
                        CPAxarray[NBfrequ] = CPAx;
                        CPAyarray[NBfrequ] = CPAy;
                        CPArarray[NBfrequ] = CPAr;
                        NBfrequ ++;
                        if(initCPAx == 1)   // not on the x=0 line
                        {
                            if(initCPAy == 1)   // not on the y=0 line
                            {
                                CPAxarray[NBfrequ] = CPAx;
                                CPAyarray[NBfrequ] = -CPAy;
                                CPArarray[NBfrequ] = CPAr;
                                //printf("%5ld  EXTRA : %+f %+f   %6.3f\n", NBfrequ, CPAx, -CPAy, CPAr);
                                NBfrequ ++;
                            }
                        }
                    }
                }
                initCPAy = 1;
            }
            initCPAx = 1; // no longer on x=0 line
        }
    }
    printf("%ld spatial frequencies\n", NBfrequ);


    quick_sort3_float(CPArarray, CPAxarray, CPAyarray, NBfrequ);

    // 2 modes (sin, cos) per frequency
    long NBmax = NBfrequ * 2;
    /*if ( rCPAmin < 0.0 )
    {
        // piston term included
        NBmax = NBfrequ * 2 + 1;
    }*/

    printf("%ld modes\n", NBmax);


    imgoutm->mdt->naxis = 3;
    imgoutm->mdt->datatype = _DATATYPE_FLOAT;
    imgoutm->mdt->size[0] = sizex;
    imgoutm->mdt->size[1] = sizey;
    imgoutm->mdt->size[2] = NBmax;
    createimagefromIMGID(imgoutm);


    if(writeMfile == 1)
    {
        printf("Writing ouput file ModesExpr_CPA.txt\n");
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
            float frequency = sqrtf(CPAx * CPAx + CPAy * CPAy);


            float fampl = 1.0;
            if(frequency < fpowerlaw_minf)
            {
                fampl = 1.0;
            }
            else if(frequency > fpowerlaw_maxf)
            {
                fampl = powf(fpowerlaw_maxf / fpowerlaw_minf,
                             fpowerlaw);
            }
            else
            {
                float f1 = frequency / fpowerlaw_minf;
                fampl = powf(f1, fpowerlaw);
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
    list_image_ID();


    // CPA array index
    long k1 = 0;

    // cube slice index
    long k  = 0;

    /*    if ( rCPAmin <= 0.0 )
        {
            // mode 0 (piston) included

            dcimg[IDfreq].array.F[0] = 0.0f;
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
    */

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

        float frequency = sqrtf(CPAx * CPAx + CPAy * CPAy);

        float fampl = 1.0;
        if(frequency < fpowerlaw_minf)
        {
            fampl = 1.0;
        }
        else if(frequency > fpowerlaw_maxf)
        {
            fampl = powf(fpowerlaw_maxf / fpowerlaw_minf,
                         fpowerlaw);
        }
        else
        {
            float f1 = frequency / fpowerlaw_minf;
            fampl = powf(f1, fpowerlaw);
        }

        for(uint32_t ii = 0; ii < sizexy; ii++)
        {
            float x                           = imgx.im->array.F[ii];
            float y                           = imgy.im->array.F[ii];
            float r                           = imgr.im->array.F[ii];

            dcimg[IDfreq].array.F[k]    = frequency;
            dcimg[IDfreq].array.F[k + 1]  = frequency;
            if(r < radfactlim)
            {
                // attenuation factor for extrapolation
                float afact = 1.0;
                if(MASKext == 1)
                {
                    float pdist = imgpixdist.im->array.F[ii];
                    float afact0 = 1.0 + extroffset - pdist * sqrtf(CPAx * CPAx + CPAy * CPAy) *
                                   M_PI / extrfactor;
                    if(afact0 > 1.0)
                    {
                        afact0 = 1.0;
                    }
                    if(afact0 < 0.0)
                    {
                        afact0 = 0.0;
                    }
                    //afact0 = 0.5;
                    afact = 0.5 * (cos((1.0 - afact0) * M_PI) + 1.0);
                }

                imgoutm->im->array.F[(k) * sizexy + ii] =
                    fampl * afact * cos(M_PI * (x * CPAx + y * CPAy));

                imgoutm->im->array.F[(k + 1) * sizexy + ii] =
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

    imgid_free(&imgx);
    imgid_free(&imgy);
    imgid_free(&imgr);
    imgid_free(&imgpixdist);

    if(outNBmax != NULL)
    {
        *outNBmax = NBmax;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
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
    IMGID imgmask = imgid_make_from_name(maskim);
    resolveIMGID(&imgmask, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imgoutm = imgid_make_from_name(outimname);

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

    imgid_free(&imgmask);
    imgid_free(&imgoutm);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_linopt_imtools__makeCPAmodes()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif


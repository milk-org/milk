/**
 * @file image_unfold.c
 * @brief Image unfold module
 */

#include "ImageStreamIO/ImageStruct.h"

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "unfold",
    .cmdkey      = "unfold",
    .description = "image unfold, merge axis A into axis B",
    .description_long =
        "Reshape an image by merging one axis into another. For example, unfold a 3D cube (x, y, z) into a 2D image by merging z into y, producing dimensions (x, y*z). The total pixel count is preserved."
};

// input image names
static char inimname[
    FUNCTION_PARAMETER_STRMAXLEN];
static char outimname[
    FUNCTION_PARAMETER_STRMAXLEN];
static uint32_t axisA   = 0;
static uint32_t axisB   = 0;
static uint32_t colsize = 1;


#define FPS_PARAMS(X) \
    X(".inim", inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input image") \
    X(".outim", outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output image") \
    X(".axisA", &axisA, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "axis to merged") \
    X(".axisB", &axisB, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "merge into this axis") \
    X(".colsize", &colsize, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "column size")

errno_t image_unfold(
    IMGID   inimg,
    IMGID   *outimg,
    uint8_t axisA,
    uint8_t axisB,
    int     colsize)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);

    resolveIMGID(outimg, ERRMODE_NULL, dcimg, dcnimg);
    if (inimg.ID == -1) {
        return RETURN_FAILURE;
    }
    if( outimg->ID == -1)
    {
        imgid_copy(&inimg, outimg);
    }

    // output image size
    outimg->mdt->naxis = inimg.md->naxis - 1;

    // remove missing axis
    {
        uint8_t axout = 0;
        for( uint8_t axin=0; axin<inimg.md->naxis; axin++)
        {
            if( axin != axisA )
            {
                outimg->mdt->size[axout] = inimg.md->size[axin];
                axout ++;
            }
        }
    }

    // destination axis to grow
    uint8_t axis0 = 0;
    if( axisA > axisB )
    {
        axis0 = axisB;
    }
    else
    {
        axis0 = axisB-1;
    }

    // overflow destination axis to grow
    uint8_t axis1 = 0;
    uint8_t axisC = 0; // in input image
    if( (axis0 == 0 ) && (outimg->mdt->naxis >1) )
    {
        axis1 = 1;
        axisC = 1;
    }

    int mdimsize = 0;
    if( axis0 == axis1 )
    {
        outimg->mdt->size[axis0] *= inimg.md->size[axisA];
    }
    else
    {
        int mdim0 = 0;  // multiplicative on axis0
        int mdim1 = 1;  // multiplicative on axis1

        for( uint32_t ii=0; ii<inimg.md->size[axisA]; ii++)
        {
            mdim0 ++;
            if(mdim0 == colsize)
            {
                mdim0 = 0;
                mdim1++;
            }
        }
        if(mdim1 > 0)
        {
            mdim0 = colsize;
        }

        outimg->mdt->size[axis0] *= mdim0;
        outimg->mdt->size[axis1] *= mdim1;

        mdimsize = inimg.md->size[axisC] * outimg->mdt->size[axis0];
    }

    createimagefromIMGID(outimg);

    // copy data to ouput

    // destination pix coord
    uint32_t ii = 0;
    uint32_t jj = 0;

    uint64_t pixi = 0;
    uint64_t pixo = 0;
    for( uint32_t pixi2=0; pixi2 < inimg.md->size[2]; pixi2++)
    {
        for( uint32_t pixi1=0; pixi1 < inimg.md->size[1]; pixi1++)
        {
            for( uint32_t pixi0=0; pixi0 < inimg.md->size[0]; pixi0++)
            {
                pixo = jj;
                pixo *= outimg->md->size[0];

                pixo += ii % outimg->md->size[0];
                pixo += mdimsize * ( ii / outimg->md->size[0] );

                outimg->im->array.F[pixo] = inimg.im->array.F[pixi];
                pixi ++;

                ii++;
            }
            if (( axisA == 1) && ( axisB == 0))
            {
                // do nothing
            }
            else
            {
                jj ++;
                ii -= inimg.md->size[0];
            }
        }

        if (( axisA == 2) && ( axisB == 1))
        {
            // do nothing
        }
        else
        {
            ii += inimg.md->size[0];
            jj -= inimg.md->size[1];
        }

    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

FPS_V2_SECTION5(FPS_PARAMS)


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);

    IMGID outimg = imgid_make_from_name(outimname);
    if (inimg.ID == -1) {
        return RETURN_FAILURE;
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        image_unfold(
            inimg,
            &outimg,
            axisA,
            axisB,
            colsize
        );

        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__image_unfold()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif

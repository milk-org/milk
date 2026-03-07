#include "ImageStreamIO/ImageStruct.h"
#include <math.h>

#include "CLIcore.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "unfold",
    .cmdkey      = "unfold",
    .description = "image unfold, merge axis A into axis B"
};

// input image names
static char *inimname;
static char *outimname;
static uint32_t *axisA;
static uint32_t *axisB;
static uint32_t *colsize;


#define FPS_PARAMS(X) \
    X(".inim", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input image") \
    X(".outim", &outimname, \
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
    IMGID inimg,
    IMGID *outimg,
    uint8_t axisA,
    uint8_t axisB,
    int colsize
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    resolveIMGID(outimg, ERRMODE_NULL, data.image, data.NB_MAX_IMAGE);
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

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    IMGID outimg = imgid_make_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        image_unfold(
            inimg,
            &outimg,
            *axisA,
            *axisB,
            *colsize
        );

        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
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

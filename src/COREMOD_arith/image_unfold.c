#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"



// input image names
static char *inimname;
static char *maskimname;

static char *outimname;


static uint32_t *axisA;
static long      fpi_axisA = -1;

static uint32_t *axisB;
static long      fpi_axisB = -1;

static uint32_t *colsize;
static long      fpi_colsize = -1;

static char *auxin;

static uint64_t *modeRMS;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inim",
        "input image",
        "im0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outim",
        "output image",
        "imout",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_UINT32,
        ".axisA",
        "axis to merged",
        "2",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &axisA,
        &fpi_axisA
    },
    {
        CLIARG_UINT32,
        ".axisB",
        "merge into this axis",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &axisB,
        &fpi_axisB
    },
    {
        CLIARG_UINT32,
        ".colsize",
        "column size",
        "10",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &colsize,
        &fpi_colsize
    }
};




static CLICMDDATA CLIcmddata =
{
    "unfold",
    "image unfold, merge axis A into axis B",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    printf("Unfold image: redude number of axis\n");
    printf("Example, 3D image input [X,Y,Z]\n");
    printf("merge axis 2 into axis 0 -> size [XZ, Y]\n");
    printf("if colsize specified, array size is [X*colsize, Y*n]\n");
    printf("with n sufficiently large to include all pixels\n");

    return RETURN_SUCCESS;
}




errno_t image_unfold(
    IMGID inimg,
    IMGID *outimg,
    uint8_t axisA,
    uint8_t axisB,
    int colsize
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg, ERRMODE_ABORT);


    resolveIMGID(outimg, ERRMODE_NULL);
    if( outimg->ID == -1)
    {
        copyIMGID(&inimg, outimg);
    }

    // output image size
    outimg->naxis = inimg.md->naxis - 1;


    // remove missing axis
    {
        uint8_t axout = 0;
        for( uint8_t axin=0; axin<inimg.md->naxis; axin++)
        {
            if( axin != axisA )
            {
                outimg->size[axout] = inimg.md->size[axin];
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
    if( (axis0 == 0 ) && (outimg->naxis >1) )
    {
        axis1 = 1;
        axisC = 1;
    }


    int mdimsize = 0;
    if( axis0 == axis1 )
    {
        outimg->size[axis0] *= inimg.md->size[axisA];
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

        outimg->size[axis0] *= mdim0;
        outimg->size[axis1] *= mdim1;

        mdimsize = inimg.md->size[axisC] * outimg->size[axis0];
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






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);

    IMGID outimg = mkIMGID_from_name(outimname);

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

        processinfo_update_output_stream(processinfo, outimg.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__image_unfold()
{
    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

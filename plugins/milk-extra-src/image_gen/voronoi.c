#include <math.h>

#include "CommandLineInterface/CLIcore.h"


// input points positions, ASCII file
static char *inpos;

static LOCVAR_OUTIMG2D outim;

static uint32_t *xsize;
static long      fpi_xsize = -1;

static uint32_t *ysize;
static long      fpi_ysize = -1;

static float   *radius;
static long      fpi_radius = -1;

static float   *gapsize;
static long      fpi_gapsize = -1;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inpos",
        "points positions, filename",
        "pts.dat",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inpos,
        NULL
    },
    FARG_OUTIM2D(outim),
    {
        CLIARG_FLOAT32,
        ".radius",
        "radius",
        "0.1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &radius,
        &fpi_radius
    },
    {
        CLIARG_FLOAT32,
        ".gapsize",
        "gap size",
        "0.1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &gapsize,
        &fpi_gapsize
    }
};


static CLICMDDATA CLIcmddata =
{
    "voronoi",
    "make Voronoi map from points file",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}





/**
 * Create Voronoi map
 *
 * imgptspos holds ponts posistions
 * size coords x nbpt
 * for 100 points, 2D, this will be 2*100
 *
 *
 * Each following line is a point, with following format:
 * index x y
 *
 * (x,y) coordinates in range [0:1]
 *
 */
imageID
image_gen_make_voronoi_map(
    IMGID *imgpos,
    IMGID *imgout,
    float radius, // maximum radius of each Voronoi zone
    float maxsep  // gap between Voronoi zones
)
{
    // resolve imgpos
    resolveIMGID(imgpos, ERRMODE_ABORT);

    // Create output image if needed
    imcreateIMGID(imgout);


    uint32_t xsize = imgout->md->size[0];
    uint32_t ysize = imgout->md->size[1];
    uint64_t xysize = xsize * ysize;
    uint32_t NBpt = imgpos->md->size[1];


    //printf("%u points\n", NBpt);

    int64_t * __restrict nearest_index;
    float   * __restrict nearest_distance2;
    int64_t * __restrict nextnearest_index;
    float   * __restrict nextnearest_distance2;
    int     * __restrict gapim;

    nearest_index = (int64_t *) malloc(sizeof(int64_t) * xysize);
    if(nearest_index == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nearest_distance2 = (float *) malloc(sizeof(float) * xysize);
    if(nearest_distance2 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nextnearest_index = (int64_t *) malloc(sizeof(int64_t) * xysize);
    if(nextnearest_index == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nextnearest_distance2 = (float *) malloc(sizeof(float) * xysize);
    if(nextnearest_distance2 == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    gapim = (int *) malloc(sizeof(int) * xysize);
    if(gapim == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    // initialize arrays
    float bigval = 1.0e20;
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        nearest_index[ii]                = -1;
        nearest_distance2[ii]            = bigval;
        nextnearest_index[ii]            = -1;
        nextnearest_distance2[ii]        = bigval;
        imgout->im->array.SI32[ii]       = -1;
    }

    for(uint32_t ii = 0; ii < xsize; ii++)
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            int   pindex = jj * xsize + ii;
            float x      = 2.0 * ii / xsize - 1.0;
            float y      = 2.0 * jj / ysize - 1.0;

            for(int pt = 0; pt < NBpt; pt++)
            {
                float dx = x - imgpos->im->array.F[ 2*pt ];
                float dy = y - imgpos->im->array.F[ 2*pt + 1 ];

                float dist2 = dx * dx + dy * dy;

                if(dist2 < nearest_distance2[pindex])
                {
                    nextnearest_index[pindex]    = nearest_index[pindex];
                    nextnearest_distance2[pindex] = nearest_distance2[pindex];

                    nearest_index[pindex]    = pt;
                    nearest_distance2[pindex] = dist2;
                }
                else if(dist2 < nextnearest_distance2[pindex])
                {
                    nextnearest_index[pindex]    = pt;
                    nextnearest_distance2[pindex] = dist2;
                }
            }
            if((nearest_distance2[pindex] < radius*radius))
            {
                imgout->im->array.SI32[pindex] = nearest_index[pindex];
            }
        }


    // add gap
    int gapsizepix = (int)(maxsep * xsize);
    // int gapsizepix2 = (int) (maxsep*xsize/sqrt(2.0));

    for(uint32_t ii = 0; ii < xsize; ii++)
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            gapim[jj * xsize + ii] = 0;
        }

    for(uint32_t ii = gapsizepix; ii < xsize - gapsizepix; ii++)
        for(uint32_t jj = gapsizepix; jj < ysize - gapsizepix; jj++)
        {
            int pindex0  = jj * xsize + ii;
            int pindex0p = jj * xsize + ii + gapsizepix;
            int pindex0m = jj * xsize + ii - gapsizepix;
            int pindexp0 = (jj + gapsizepix) * xsize + ii;
            int pindexm0 = (jj - gapsizepix) * xsize + ii;
            int pindexpp = (jj + gapsizepix) * xsize + ii + gapsizepix;
            int pindexpm = (jj + gapsizepix) * xsize + ii - gapsizepix;
            int pindexmp = (jj - gapsizepix) * xsize + ii + gapsizepix;
            int pindexmm = (jj - gapsizepix) * xsize + ii - gapsizepix;

            int32_t pv0p = imgout->im->array.SI32[pindex0p];
            int32_t pv0m = imgout->im->array.SI32[pindex0m];
            int32_t pvp0 = imgout->im->array.SI32[pindexp0];
            int32_t pvm0 = imgout->im->array.SI32[pindexm0];
            int32_t pvpp = imgout->im->array.SI32[pindexpp];
            int32_t pvpm = imgout->im->array.SI32[pindexpm];
            int32_t pvmp = imgout->im->array.SI32[pindexmp];
            int32_t pvmm = imgout->im->array.SI32[pindexmm];

            gapim[pindex0] = 1;

            if((pv0p != pv0m) || (pvp0 != pvm0) || (pvpp != pvmm) ||
                    (pvpm != pvmp))
            {
                gapim[pindex0] = 0;
            }
        }

    for(uint32_t ii = 0; ii < xsize; ii++)
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            int pindex = jj * xsize + ii;
            if(gapim[pindex] == 0)
            {
                imgout->im->array.SI32[pindex] = -1;
            }
        }


    free(nearest_index);
    free(nearest_distance2);
    free(nextnearest_index);
    free(nextnearest_distance2);

    free(gapim);



    return (imgout->ID);
}







static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgpos = mkIMGID_from_name(inpos);
    resolveIMGID(&imgpos, ERRMODE_ABORT);

    // link/create output image/stream
    FARG_OUTIM2DCREATE(outim, imgout, _DATATYPE_INT32);


    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    {

        image_gen_make_voronoi_map(
            &imgpos,
            &imgout,
            *radius, // maximum radius of each Voronoi zone
            *gapsize  // gap between Voronoi zones
        );



        processinfo_update_output_stream(processinfo, imgout.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_image_gen__voronoi()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}




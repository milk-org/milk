#include <math.h>

#include "CommandLineInterface/CLIcore.h"


// input points positions, ASCII file
static char *inptspos_fname;

static char *outimname;

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
        CLIARG_STR,
        ".ptpos_fname",
        "points positions, filename",
        "pts.dat",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inptspos_fname,
        NULL
    },
    {
        CLIARG_IMG,
        ".out_name",
        "output image",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_UINT32,
        ".xsize",
        "output x size",
        "100",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &xsize,
        &fpi_xsize
    },
    {
        CLIARG_UINT32,
        ".ysize",
        "output y size",
        "100",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &ysize,
        &fpi_ysize
    },
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
 * filename is an ASCII file defining points
 *
 * First line is number of point
 *
 * Each following line is a point, with following format:
 * index x y
 *
 * (x,y) coordinates in range [0:1]
 *
 */
imageID
image_gen_make_voronoi_map(
    const char *filename,
    const char *IDout_name,
    uint32_t    xsize,
    uint32_t    ysize,
    float radius, // maximum radius of each Voronoi zone
    float maxsep  // gap between Voronoi zones
)
{


    // Read input ASCII file
    //
    long  NBpt;
    uint32_t * __restrict vpt_index;
    float * __restrict vpt_x;
    float * __restrict vpt_y;
    {
        FILE *fp;
        fp = fopen(filename, "r");
        if(fp == NULL)
        {
            printf("file %s not found\n", filename);
            return 1;
        }

        {
            int fscanfcnt = fscanf(fp, "%ld", &NBpt);
            if(fscanfcnt == EOF)
            {
                if(ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr,
                            "Error: fscanf reached end of file, no matching "
                            "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if(fscanfcnt != 2)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i input "
                        "items, 2 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }
        }


        printf("Loading %ld points\n", NBpt);

        vpt_index = (uint32_t *) malloc(sizeof(uint32_t) * NBpt);
        if(vpt_index == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        vpt_x = (float *) malloc(sizeof(float) * NBpt);
        if(vpt_x == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        vpt_y = (float *) malloc(sizeof(float) * NBpt);
        if(vpt_y == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for(int pt = 0; pt < NBpt; pt++)
        {
            int fscanfcnt =
                fscanf(fp, "%u %f %f\n", &vpt_index[pt], &vpt_x[pt], &vpt_y[pt]);
            if(fscanfcnt == EOF)
            {
                if(ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr,
                            "Error: fscanf reached end of file, no matching "
                            "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if(fscanfcnt != 3)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i input "
                        "items, 3 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }
        }

        fclose(fp);
    }



    // Create output image
    imageID   IDout;
    {
        uint32_t *sizearray;
        uint8_t naxis = 2;
        sizearray = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
        if(sizearray == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        sizearray[0] = xsize;
        sizearray[1] = ysize;
        create_image_ID(IDout_name,
                        naxis,
                        sizearray,
                        _DATATYPE_INT32,
                        0,
                        0,
                        0,
                        &IDout);
        free(sizearray);
    }





    int64_t * __restrict nearest_index;
    float   * __restrict nearest_distance;
    int64_t * __restrict nextnearest_index;
    float   * __restrict nextnearest_distance;
    int     * __restrict gapim;

    nearest_index = (int64_t *) malloc(sizeof(int64_t) * xsize * ysize);
    if(nearest_index == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nearest_distance = (float *) malloc(sizeof(float) * xsize * ysize);
    if(nearest_distance == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nextnearest_index = (int64_t *) malloc(sizeof(int64_t) * xsize * ysize);
    if(nextnearest_index == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    nextnearest_distance = (float *) malloc(sizeof(float) * xsize * ysize);
    if(nextnearest_distance == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    gapim = (int *) malloc(sizeof(int) * xsize * ysize);
    if(gapim == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    // initialize arrays
    float bigval = 1.0e20;
    for(uint64_t ii = 0; ii < xsize * ysize; ii++)
    {
        nearest_index[ii]                = -1;
        nearest_distance[ii]             = bigval;
        nextnearest_index[ii]            = -1;
        nextnearest_distance[ii]         = bigval;
        data.image[IDout].array.SI32[ii] = -1;
    }

    for(uint32_t ii = 0; ii < xsize; ii++)
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            int   pindex = jj * xsize + ii;
            float x      = 1.0 * ii / xsize;
            float y      = 1.0 * jj / ysize;

            for(int pt = 0; pt < NBpt; pt++)
            {
                float dx = x - vpt_x[pt];
                float dy = y - vpt_y[pt];

                float dist = sqrt(dx * dx + dy * dy);

                if(dist < nearest_distance[pindex])
                {
                    nextnearest_index[pindex]    = nearest_index[pindex];
                    nextnearest_distance[pindex] = nearest_distance[pindex];

                    nearest_index[pindex]    = pt;
                    nearest_distance[pindex] = dist;
                }
                else if(dist < nextnearest_distance[pindex])
                {
                    nextnearest_index[pindex]    = pt;
                    nextnearest_distance[pindex] = dist;
                }
            }
            if((nearest_distance[pindex] < radius))
            {
                data.image[IDout].array.SI32[pindex] = nearest_index[pindex];
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

            int32_t pv0p = data.image[IDout].array.SI32[pindex0p];
            int32_t pv0m = data.image[IDout].array.SI32[pindex0m];
            int32_t pvp0 = data.image[IDout].array.SI32[pindexp0];
            int32_t pvm0 = data.image[IDout].array.SI32[pindexm0];
            int32_t pvpp = data.image[IDout].array.SI32[pindexpp];
            int32_t pvpm = data.image[IDout].array.SI32[pindexpm];
            int32_t pvmp = data.image[IDout].array.SI32[pindexmp];
            int32_t pvmm = data.image[IDout].array.SI32[pindexmm];

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
                data.image[IDout].array.SI32[pindex] = -1;
            }
        }

    free(vpt_index);
    free(vpt_x);
    free(vpt_y);

    free(nearest_index);
    free(nearest_distance);
    free(nextnearest_index);
    free(nextnearest_distance);

    free(gapim);

    return (IDout);
}







static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    {


        image_gen_make_voronoi_map(
            inptspos_fname,
            outimname,
            *xsize,
            *ysize,
            *radius, // maximum radius of each Voronoi zone
            *gapsize  // gap between Voronoi zones
        );


        // stream is updated here, and not in the function called above, so that multiple
        // the above function can be chained with others
        //processinfo_update_output_stream(processinfo, outimg.ID);

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




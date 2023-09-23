#include "CommandLineInterface/CLIcore.h"


// input image
//
static char *insname;


// unmapping array to 1D
// same size as input, values are pix index pointing to output
// datatype = int32
// negative value are not mapped
//
static char *mapsname;


// output (remapped) image
//
static LOCVAR_OUTIMG2D outim;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".insname",
        "input image name",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &insname,
        NULL
    },
    {
        CLIARG_IMG,
        ".map",
        "mapping image name",
        "mapim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &mapsname,
        NULL
    },
    FARG_OUTIM_NAME(outim),
    FARG_OUTIM_SHARED(outim)
};





static CLICMDDATA CLIcmddata =
{
    "pixunmap", "pixel unmapping of image to 1D", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("Unmap input image to ouput 1D array by pixel loopup\n");

    return RETURN_SUCCESS;
}







static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


    // connect to input
    //
    IMGID imgin = mkIMGID_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_ABORT);
    int64_t insize = imgin.md->size[0]*imgin.md->size[1];


    IMGID imgmap = mkIMGID_from_name(mapsname);
    resolveIMGID(&imgmap, ERRMODE_ABORT);

    // read map size
    // Note: currently assumes 2D ... to be updated
    //
    uint32_t xsize = imgmap.md->size[0];
    uint32_t ysize = imgmap.md->size[1];
    uint64_t xysize = (uint64_t) xsize;
    xysize *= ysize;

    // read output 1D array size from max value of mapping file
    int x1Dsize = 0;
    for(uint64_t ii=0; ii<xysize; ii++)
    {
        int pixi = imgmap.im->array.SI32[ii];
        if( pixi > x1Dsize )
        {
            x1Dsize = pixi;
        }
    }
    x1Dsize ++;

    printf("output 1D size = %d\n", x1Dsize);
    fflush(stdout);




    // link/create output image/stream
    uint8_t outdatatype;
    switch ( imgin.md->datatype )
    {
    case (_DATATYPE_DOUBLE) :
        outdatatype = _DATATYPE_DOUBLE;
        break;
    case (_DATATYPE_INT64) :
        outdatatype = _DATATYPE_DOUBLE;
        break;
    case (_DATATYPE_UINT64) :
        outdatatype = _DATATYPE_DOUBLE;
        break;
    default :
        outdatatype = _DATATYPE_FLOAT;
    }



    IMGID imgout = mkIMGID_from_name(outim.name);
    imgout.shared = *outim.shared;
    if(*outim.shared == 1)
    {
        imgout = stream_connect_create_2D(outim.name, x1Dsize, 1, outdatatype);
    }
    else
    {
        imgout.naxis = 2;
        imgout.size[0] = x1Dsize;
        imgout.size[1] = 1;
        imgout.datatype = outdatatype;
        createimagefromIMGID(&imgout);
    }
    imcreateIMGID(&imgout);




    // build mapping table
    //
    uint64_t nbpix = 0;
    for(uint64_t ii = 0; ii < xsize*ysize; ii++)
    {
        int64_t pixindex = imgmap.im->array.SI32[ii];
        if ( ( pixindex > -1 )
                && ( pixindex < insize) )
        {
            nbpix ++;
        }
    }

    printf("mapping table has %lu elements\n", nbpix);

    uint64_t * __restrict map_2Dpixindex = (uint64_t*) malloc(sizeof(uint64_t) * nbpix);
    uint64_t * __restrict map_1Dpixindex  = (uint64_t*) malloc(sizeof(uint64_t) * nbpix);

    uint64_t * __restrict map_pixcnt      = (uint64_t*) malloc(sizeof(uint64_t) * x1Dsize);
    for(int zone=0; zone<x1Dsize; zone++)
    {
        map_pixcnt[zone] = 0;
    }

    nbpix = 0;
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        int64_t pixindex = imgmap.im->array.SI32[ii];
        if ( ( pixindex > -1 )
                && ( pixindex < x1Dsize) )
        {
            map_2Dpixindex[nbpix] = ii;
            map_1Dpixindex[nbpix] = pixindex;

            map_pixcnt[pixindex] ++;
            nbpix ++;
        }
    }

    // avoid division by zero
    for(int zone=0; zone<x1Dsize; zone++)
    {
        if(map_pixcnt[zone] == 0)
        {
            map_pixcnt[zone] = 1;
        }
    }




    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        switch ( imgin.md->datatype)
        {
        case _DATATYPE_FLOAT:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.F[map_1Dpixindex[pixi]] += imgin.im->array.F[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.F[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_DOUBLE:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.D[map_1Dpixindex[pixi]] += imgin.im->array.D[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.D[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_INT8:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.F[map_1Dpixindex[pixi]] += imgin.im->array.SI8[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.F[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_UINT8:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.F[map_1Dpixindex[pixi]] += imgin.im->array.UI8[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.F[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_INT16:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.F[map_1Dpixindex[pixi]] += imgin.im->array.SI16[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.F[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_UINT16:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.F[map_1Dpixindex[pixi]] += imgin.im->array.UI16[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.F[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_INT32:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.F[map_1Dpixindex[pixi]] += imgin.im->array.SI32[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.F[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_UINT32:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.F[map_1Dpixindex[pixi]] += imgin.im->array.UI32[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.F[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_INT64:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.D[map_1Dpixindex[pixi]] += imgin.im->array.SI64[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.D[ii] /= map_pixcnt[ii];
            }
            break;

        case _DATATYPE_UINT64:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.D[map_1Dpixindex[pixi]] += imgin.im->array.UI64[map_2Dpixindex[pixi]];
            }
            for(uint32_t ii=0; ii<x1Dsize; ii++)
            {
                imgout.im->array.D[ii] /= map_pixcnt[ii];
            }
            break;
        }



        processinfo_update_output_stream(processinfo, imgout.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(map_2Dpixindex);
    free(map_1Dpixindex);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__pixunmap()
{

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

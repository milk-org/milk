#include "CommandLineInterface/CLIcore.h"


// input image
//
static char *insname;


// mapping array
// same size as output, values are pix index pointing to input
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
    "pixremap", "pixel remapping of image", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("Remap input image to ouput image by pixel lookup\n");

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

    // link/create output image/stream

    IMGID imgout = mkIMGID_from_name(outim.name);
    imgout.shared = *outim.shared;
    if(*outim.shared == 1)
    {
        imgout = stream_connect_create_2D(outim.name, xsize, ysize, imgin.md->datatype);
    }
    else
    {
        imgout.naxis = 2;
        imgout.size[0] = xsize;
        imgout.size[1] = ysize;
        imgout.datatype = imgin.md->datatype;
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

    uint64_t * __restrict map_outpixindex = (uint64_t*) malloc(sizeof(uint64_t) * nbpix);
    uint64_t * __restrict map_inpixindex  = (uint64_t*) malloc(sizeof(uint64_t) * nbpix);


    nbpix = 0;
    for(uint64_t ii = 0; ii < xsize*ysize; ii++)
    {
        int64_t pixindex = imgmap.im->array.SI32[ii];
        if ( ( pixindex > -1 )
                && ( pixindex < insize) )
        {
            map_outpixindex[nbpix] = ii;
            map_inpixindex[nbpix] = pixindex;
            nbpix ++;
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
                imgout.im->array.F[map_outpixindex[pixi]] = imgin.im->array.F[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_DOUBLE:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.D[map_outpixindex[pixi]] = imgin.im->array.D[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_INT8:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.SI8[map_outpixindex[pixi]] = imgin.im->array.SI8[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_UINT8:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.UI8[map_outpixindex[pixi]] = imgin.im->array.UI8[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_INT16:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.SI16[map_outpixindex[pixi]] = imgin.im->array.SI16[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_UINT16:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.UI16[map_outpixindex[pixi]] = imgin.im->array.UI16[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_INT32:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.SI32[map_outpixindex[pixi]] = imgin.im->array.SI32[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_UINT32:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.UI32[map_outpixindex[pixi]] = imgin.im->array.UI32[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_INT64:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.SI64[map_outpixindex[pixi]] = imgin.im->array.SI64[map_inpixindex[pixi]];
            }
            break;

        case _DATATYPE_UINT64:
            for(uint64_t pixi=0; pixi<nbpix; pixi++)
            {
                imgout.im->array.UI64[map_outpixindex[pixi]] = imgin.im->array.UI64[map_inpixindex[pixi]];
            }
            break;
        }



        processinfo_update_output_stream(processinfo, imgout.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(map_outpixindex);
    free(map_inpixindex);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__pixremap()
{

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

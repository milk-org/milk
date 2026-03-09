#include "ImageStreamIO/ImageStruct.h"
#include "CLIcore.h"
#include "fps.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "pixunmap",
    .cmdkey      = "pixunmap",
    .description = "pixel unmapping of image to 1D"
};

// input image
static char *insname;

// unmapping array to 1D
static char *mapsname;

// output image
static char *outimname;
static int32_t *outshared;

#define FPS_PARAMS(X) \
    X(".insname", &insname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input image name") \
    X(".map", &mapsname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "mapping image name") \
    X(".out_name", &outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output image name") \
    X(".out_shared", &outshared, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, "output shared (1) or not (0)")


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

    // connect to input
    //
    IMGID imgin = imgid_make_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);
    int64_t insize = imgin.md->size[0]*imgin.md->size[1];

    IMGID imgmap = imgid_make_from_name(mapsname);
    resolveIMGID(&imgmap, ERRMODE_ABORT, dcimg, dcnimg);

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

    IMGID imgout = imgid_make_from_name(outimname);
    imgout.mdt->shared = *outshared;
    if(*outshared == 1)
    {
        imgid_free(&imgout);
        imgout = stream_connect_create_2D(outimname, x1Dsize, 1, outdatatype);
    }
    else
    {
        imgout.mdt->naxis = 2;
        imgout.mdt->size[0] = x1Dsize;
        imgout.mdt->size[1] = 1;
        imgout.mdt->datatype = outdatatype;
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

        processinfo_update_output_stream(processinfo, imgout.im, NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(map_2Dpixindex);
    free(map_1Dpixindex);

    DEBUG_TRACE_FEXIT();
    imgid_free(&imgin);
    imgid_free(&imgmap);
    imgid_free(&imgout);
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
CLIADDCMD_COREMODE_arith__pixunmap()
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

#include "ImageStreamIO/ImageStruct.h"
#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "pixremap",
    .cmdkey      = "pixremap",
    .description = "pixel remapping of image"
};

// input image
static char *insname;

// mapping array
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

    // link/create output image/stream
    // output size is same as input size (default)

    IMGID imgout = imgid_make_from_name(outimname);
    imgout.mdt->shared = *outshared;
    if(*outshared == 1)
    {
        imgid_free(&imgout);
        imgout = stream_connect_create_2D(outimname, xsize, ysize, imgin.md->datatype);
    }
    else
    {
        imgout.mdt->naxis = 2;
        imgout.mdt->size[0] = xsize;
        imgout.mdt->size[1] = ysize;
        imgout.mdt->datatype = imgin.md->datatype;
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

        processinfo_update_output_stream(processinfo, imgout.im, NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(map_outpixindex);
    free(map_inpixindex);
    imgid_free(&imgin);
    imgid_free(&imgmap);
    imgid_free(&imgout);

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
CLIADDCMD_COREMODE_arith__pixremap()
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

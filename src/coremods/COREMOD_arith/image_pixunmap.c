/**
 * @file image_pixunmap.c
 * @brief Image pixunmap module
 */

#include "ImageStreamIO/ImageStruct.h"
#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "pixunmap",
    .cmdkey      = "pixunmap",
    .description = "pixel unmapping of image to 1D",
    .description_long =
        "Reverse-map a 2D image to a 1D array using an index map. Each pixel in the input image is placed at the position specified by the unmap table. Inverse operation of pixremap."
};

// input image
static char insname[FUNCTION_PARAMETER_STRMAXLEN];

// unmapping array to 1D
static char mapsname[FUNCTION_PARAMETER_STRMAXLEN];

// output image
static char outimname[FUNCTION_PARAMETER_STRMAXLEN];
static int32_t outshared = 0;

#define FPS_PARAMS(X) \
    X(".insname", insname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input image name") \
    X(".map", mapsname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "mapping image name") \
    X(".out_name", outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output image name") \
    X(".out_shared", &outshared, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, "output shared (1) or not (0)")


FPS_V2_SECTION5(FPS_PARAMS)


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    // connect to input
    //
    IMGID imgin = imgid_make_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    int64_t insize = imgin.md->size[0]*imgin.md->size[1];
    if (imgin.ID == -1) {
        return RETURN_FAILURE;
    }

    IMGID imgmap = imgid_make_from_name(mapsname);
    resolveIMGID(&imgmap, ERRMODE_WARN, dcimg, dcnimg);

    // read map size
    // Note: currently assumes 2D ... to be updated
    //
    uint32_t xsize = imgmap.md->size[0];
    if (imgmap.ID == -1) {
        return RETURN_FAILURE;
    }
    uint32_t ysize = imgmap.md->size[1];
    uint64_t xysize = (uint64_t) xsize;
    xysize *= ysize;

    // read output 1D array size from max value of mapping file
    int x1Dsize = 0;
        MILK_IVDEP
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
    case (_DATATYPE_DOUBLE) : outdatatype = _DATATYPE_DOUBLE;
        break;
    case (_DATATYPE_INT64) : outdatatype = _DATATYPE_DOUBLE;
        break;
    case (_DATATYPE_UINT64) : outdatatype = _DATATYPE_DOUBLE;
        break;
    default : outdatatype = _DATATYPE_FLOAT;
    }

    IMGID imgout = imgid_make_from_name(outimname);
    imgout.mdt->shared = outshared;
    if(outshared == 1)
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
        MILK_IVDEP
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
        MILK_IVDEP
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

/*
 * Accumulate + average mapped pixels.
 * OACC = output accessor (F or D)
 * IACC = input accessor
 */
#define UNMAP_CASE_(DT, IACC, CT, OACC)             \
    case DT:                                         \
        for(uint64_t pixi=0; pixi<nbpix; pixi++)     \
        {                                            \
            imgout.im->array.OACC[                   \
                map_1Dpixindex[pixi]] +=              \
                imgin.im->array.IACC[                \
                    map_2Dpixindex[pixi]];           \
        }                                            \
        for(uint32_t ii=0; ii<x1Dsize; ii++)         \
        {                                            \
            imgout.im->array.OACC[ii] /=             \
                map_pixcnt[ii];                      \
        }                                            \
        break;

        switch ( imgin.md->datatype)
        {
        /* types that accumulate into float */
        UNMAP_CASE_(_DATATYPE_FLOAT,  F,    float,    F)
        UNMAP_CASE_(_DATATYPE_INT8,   SI8,  int8_t,   F)
        UNMAP_CASE_(_DATATYPE_UINT8,  UI8,  uint8_t,  F)
        UNMAP_CASE_(_DATATYPE_INT16,  SI16, int16_t,  F)
        UNMAP_CASE_(_DATATYPE_UINT16, UI16, uint16_t, F)
        UNMAP_CASE_(_DATATYPE_INT32,  SI32, int32_t,  F)
        UNMAP_CASE_(_DATATYPE_UINT32, UI32, uint32_t, F)
        /* types that accumulate into double */
        UNMAP_CASE_(_DATATYPE_DOUBLE, D,    double,   D)
        UNMAP_CASE_(_DATATYPE_INT64,  SI64, int64_t,  D)
        UNMAP_CASE_(_DATATYPE_UINT64, UI64, uint64_t, D)
        default: PRINT_ERROR("unsupported datatype");
            break;
        }
#undef UNMAP_CASE_

        processinfo_update_output_stream(processinfo, imgout.im, NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END  free(map_2Dpixindex);
    free(map_1Dpixindex);

    DEBUG_TRACE_FEXIT();
    imgid_free(&imgin);
    imgid_free(&imgmap);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}


#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__pixunmap()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC  return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif

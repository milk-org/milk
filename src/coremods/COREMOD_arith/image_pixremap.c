/**
 * @file image_pixremap.c
 * @brief Image pixremap module
 */

#include "ImageStreamIO/ImageStruct.h"
#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "pixremap",
    .cmdkey           = "pixremap",
    .description      = "pixel remapping of image",
    .description_long = "Remap pixels from one image to another using an index map. Each output "
                        "pixel is assigned the value of the input pixel at the index specified by "
                        "the mapping array. Supports arbitrary geometric transformations."
};

// input image
static char insname[FUNCTION_PARAMETER_STRMAXLEN];

// mapping array
static char mapsname[FUNCTION_PARAMETER_STRMAXLEN];

// output image
static char    outimname[FUNCTION_PARAMETER_STRMAXLEN];
static int32_t outshared = 0;

#define FPS_PARAMS(X)                                                                      \
    X(".insname", insname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image name") \
    X(".map", mapsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "mapping image name")  \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image name") \
    X(".out_shared", &outshared, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                    \
      "output shared (1) or not (0)")


FPS_V2_SECTION5(FPS_PARAMS)


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    // connect to input
    //
    IMGID imgin = imgid_make_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    int64_t insize = imgin.md->size[0] * imgin.md->size[1];
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgmap = imgid_make_from_name(mapsname);
    resolveIMGID(&imgmap, ERRMODE_WARN, dcimg, dcnimg);

    // read map size
    // Note: currently assumes 2D ... to be updated
    //
    uint32_t xsize = imgmap.md->size[0];
    if (imgmap.ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint32_t ysize = imgmap.md->size[1];

    // link/create output image/stream
    // output size is same as input size (default)

    IMGID imgout       = imgid_make_from_name(outimname);
    imgout.mdt->shared = outshared;
    if (outshared == 1)
    {
        imgid_free(&imgout);
        imgout = stream_connect_create_2D(outimname, xsize, ysize, imgin.md->datatype);
    }
    else
    {
        imgout.mdt->naxis    = 2;
        imgout.mdt->size[0]  = xsize;
        imgout.mdt->size[1]  = ysize;
        imgout.mdt->datatype = imgin.md->datatype;
        createimagefromIMGID(&imgout);
    }
    imcreateIMGID(&imgout);

    // build mapping table
    //
    uint64_t nbpix = 0;
    for (uint64_t ii = 0; ii < (uint64_t) xsize * ysize; ii++)
    {
        int64_t pixindex = imgmap.im->array.SI32[ii];
        if ((pixindex > -1) && (pixindex < insize))
        {
            nbpix++;
        }
    }

    printf("mapping table has %lu elements\n", nbpix);

    uint64_t *MILK_RESTRICT map_outpixindex = (uint64_t *) malloc(sizeof(uint64_t) * nbpix);
    uint64_t *MILK_RESTRICT map_inpixindex  = (uint64_t *) malloc(sizeof(uint64_t) * nbpix);

    nbpix = 0;
    for (uint64_t ii = 0; ii < (uint64_t) xsize * ysize; ii++)
    {
        int64_t pixindex = imgmap.im->array.SI32[ii];
        if ((pixindex > -1) && (pixindex < insize))
        {
            map_outpixindex[nbpix] = ii;
            map_inpixindex[nbpix]  = pixindex;
            nbpix++;
        }
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
#define REMAP_CASE_(DT, ACC, CT)                           \
    case DT:                                               \
        for (uint64_t pixi = 0; pixi < nbpix; pixi++)      \
        {                                                  \
            imgout.im->array.ACC[map_outpixindex[pixi]] =  \
                imgin.im->array.ACC[map_inpixindex[pixi]]; \
        }                                                  \
        break;

        switch (imgin.md->datatype)
        {
            FOREACH_REAL_DATATYPE(REMAP_CASE_) default : PRINT_ERROR("unsupported datatype");
            break;
        }
#undef REMAP_CASE_

        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END free(map_outpixindex);
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
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

// Register function in CLI
errno_t CLIADDCMD_COREMODE_arith__pixremap()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif


#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif

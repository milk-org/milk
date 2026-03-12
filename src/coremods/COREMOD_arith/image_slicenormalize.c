/**
 * @file image_slicenormalize.c
 * @brief Image slicenormalize module
 */

#include "ImageStreamIO/ImageStruct.h"
#include <math.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkdata/pixel_dispatch.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "slicenorm",
    .cmdkey      = "normalizeslice",
    .description = "image normalize over mask by slice"
};

// input image names
static char inimname[
    FUNCTION_PARAMETER_STRMAXLEN];
static char maskimname[
    FUNCTION_PARAMETER_STRMAXLEN];
static char outimname[
    FUNCTION_PARAMETER_STRMAXLEN];
static uint32_t sliceaxis = 0;
static char auxin[
    FUNCTION_PARAMETER_STRMAXLEN];

// changed from uint64_t* to int32_t for V2
static int32_t modeRMS = 0;

#define FPS_PARAMS(X) \
    X(".in0name", inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input image 0") \
    X(".maskim", maskimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "input image mask") \
    X(".outname", outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output image") \
    X(".axis", &sliceaxis, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "norm axis") \
    X(".auxin", auxin, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "auxillary input image, in-place update") \
    X(".RMS", &modeRMS, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, "output RMS=1 over mask")

errno_t image_slicenormalize(
    IMGID inimg,
    IMGID maskimg,
    IMGID *outimg,
    uint8_t sliceaxis,
    IMGID imgaux,
    int modeRMS
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&maskimg, ERRMODE_ABORT, dcimg, dcnimg);

    resolveIMGID(&imgaux, ERRMODE_NULL, dcimg, dcnimg);

    resolveIMGID(outimg, ERRMODE_NULL, dcimg, dcnimg);
    if(outimg->ID == -1)
    {
        imgid_copy(&inimg, outimg);
    }

    outimg->mdt->datatype = _DATATYPE_FLOAT;

    createimagefromIMGID(outimg);

    // input image
    //
    uint32_t sizescan[3];
    sizescan[0] = inimg.md->size[0];
    sizescan[1] = inimg.md->size[1];
    sizescan[2] = inimg.md->size[2];
    if(inimg.md->naxis < 3)
    {
        sizescan[2] = 1;
    }
    if(inimg.md->naxis < 2)
    {
        sizescan[1] = 1;
    }

    // aux input image
    //
    uint32_t auxsizescan[3];
    if(imgaux.ID != -1)
    {
        auxsizescan[0] = imgaux.md->size[0];
        auxsizescan[1] = imgaux.md->size[1];
        auxsizescan[2] = imgaux.md->size[2];
        if(imgaux.md->naxis < 3)
        {
            auxsizescan[2] = 1;
        }
        if(imgaux.md->naxis < 2)
        {
            auxsizescan[1] = 1;
        }
    }

    // mask image
    //
    uint32_t sizescanm[3];
    sizescanm[0] = sizescan[0];
    sizescanm[1] = sizescan[1];
    sizescanm[2] = sizescan[2];
    sizescanm[sliceaxis] = 1;

    uint32_t sizemmask[3];
    sizemmask[0] = 1;
    sizemmask[1] = 1;
    sizemmask[2] = 1;
    sizemmask[sliceaxis] = 0;

    double *__restrict normarray = (double *) malloc(sizeof(
                                       double) * sizescan[sliceaxis]);
    for(uint32_t ii = 0; ii < inimg.md->size[sliceaxis]; ii++)
    {
        normarray[ii] = 0.0;
    }

    double *__restrict avarray = (double *) malloc(sizeof(double) *
                                 sizescan[sliceaxis]);
    for(uint32_t ii = 0; ii < inimg.md->size[sliceaxis]; ii++)
    {
        avarray[ii] = 0.0;
    }

    double *__restrict maskcntarray = (double *) malloc(sizeof(
                                          double) * sizescan[sliceaxis]);
    for(uint32_t ii = 0; ii < inimg.md->size[sliceaxis]; ii++)
    {
        maskcntarray[ii] = 0.0;
    }

    // input image
    uint32_t pixcoord[3];

    for(uint32_t ii = 0; ii < sizescan[0]; ii++)
    {
        pixcoord[0] = ii;
        uint32_t iim = ii * sizemmask[0];

        for(uint32_t jj = 0; jj < sizescan[1]; jj++)
        {
            pixcoord[1] = jj;
            uint32_t jjm = jj * sizemmask[1];

            for(uint32_t kk = 0; kk < sizescan[2]; kk++)
            {
                pixcoord[2] = kk;
                uint32_t kkm = kk * sizemmask[2];

                uint64_t pixi = kk * sizescan[1] * sizescan[0];
                pixi += jj * sizescan[0];
                pixi += ii;

                uint64_t pixim = kkm * sizescanm[1] * sizescanm[0];
                pixim += jjm * sizescanm[0];
                pixim += iim;

                double valm; // masked value

#define MASKMUL_(DT, ACC, CT)             \
    case DT:                               \
        valm = maskimg.im->array.F[pixim]  \
            * inimg.im->array.ACC[pixi];   \
        break;

                switch(inimg.md->datatype)
                {
                    FOREACH_REAL_DATATYPE(MASKMUL_)
                default:
                    valm = 0.0;
                    PRINT_ERROR(
                        "unsupported datatype");
                    break;
                }
#undef MASKMUL_
                normarray[pixcoord[sliceaxis]] += valm * valm;
                avarray[pixcoord[sliceaxis]] += valm;
                maskcntarray[pixcoord[sliceaxis]] += maskimg.im->array.F[pixim];
            }
        }
    }

    for(uint32_t ii = 0; ii < sizescan[sliceaxis]; ii++)
    {
        avarray[ii] /= maskcntarray[ii];

        normarray[ii] /= maskcntarray[ii];
        // REMOVED FROM DEF BEHAVIOR: no mean sub.
        // normarray[ii] -= avarray[ii]*avarray[ii];
        if(normarray[ii] > 0.0)
        {
            normarray[ii] = sqrt(normarray[ii]);
        }
        // printf("slice %3u : cnt=%lf  av=%lf  std=%lf\n", ii, maskcntarray[ii], avarray[ii], normarray[ii]);

        if(modeRMS == 0)
        {
            normarray[ii] *= sqrt(maskcntarray[ii]);
        }
    }

    // process input image
    //
    for(uint32_t ii = 0; ii < sizescan[0]; ii++)
    {
        pixcoord[0] = ii;
        for(uint32_t jj = 0; jj < sizescan[1]; jj++)
        {
            pixcoord[1] = jj;
            for(uint32_t kk = 0; kk < sizescan[2]; kk++)
            {
                pixcoord[2] = kk;

                uint64_t pixi = kk * sizescan[1] * sizescan[0];
                pixi += jj * sizescan[0];
                pixi += ii;

#define NORMDIV_(DT, ACC, CT)                       \
    case DT:                                         \
        outimg->im->array.F[pixi] =                  \
            (1.0 * inimg.im->array.ACC[pixi]) /      \
            normarray[pixcoord[sliceaxis]];           \
        break;

                switch(inimg.md->datatype)
                {
                    FOREACH_REAL_DATATYPE(NORMDIV_)
                default:
                    PRINT_ERROR(
                        "unsupported datatype");
                    break;
                }
#undef NORMDIV_
            }
        }
    }

    if(imgaux.ID != -1)
    {
        // process aux input image
        // FLOAT only
        // scaling only, no offset
        //
        for(uint32_t ii = 0; ii < auxsizescan[0]; ii++)
        {
            pixcoord[0] = ii;
            for(uint32_t jj = 0; jj < auxsizescan[1]; jj++)
            {
                pixcoord[1] = jj;
                for(uint32_t kk = 0; kk < auxsizescan[2]; kk++)
                {
                    pixcoord[2] = kk;

                    uint64_t pixi = kk * auxsizescan[1] * auxsizescan[0];
                    pixi += jj * auxsizescan[0];
                    pixi += ii;

                    imgaux.im->array.F[pixi] = imgaux.im->array.F[pixi] /
                                               normarray[pixcoord[sliceaxis]];
                }
            }
        }
    }

    free(normarray);
    free(avarray);
    free(maskcntarray);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


FPS_V2_SECTION5(FPS_PARAMS)


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID maskimg = imgid_make_from_name(maskimname);
    resolveIMGID(&maskimg, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID imgaux = imgid_make_from_name(auxin);
    resolveIMGID(&imgaux, ERRMODE_WARN, dcimg, dcnimg);

    IMGID outimg = imgid_make_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        image_slicenormalize(
            inimg,
            maskimg,
            &outimg,
            sliceaxis,
            imgaux,
            modeRMS
        );

        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&inimg);
    imgid_free(&maskimg);
    imgid_free(&imgaux);
    imgid_free(&outimg);

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
CLIADDCMD_COREMOD_arith__image_slicenormalize()
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

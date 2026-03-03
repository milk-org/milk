#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "image_norm.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *norm_inimname  = NULL;
char     *norm_outimname = NULL;
uint32_t *norm_sliceaxis = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "normslice",
    .cmdkey      = "normslice",
    .description = "image norm by slice"
};

static uint64_t processinfo_change_cnt_local;

errno_t image_slicenorm_IMGID(
    IMGID *inimg, IMGID *outimg,
    uint8_t sliceaxis
)
{
#ifndef FPS_STANDALONE
    resolveIMGID(
        inimg, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    resolveIMGID(
        outimg, ERRMODE_NULL,
        data.image, data.NB_MAX_IMAGE);
#endif
    if (outimg->ID == -1) {
        imgid_copy(inimg, outimg);
    }
    for (uint8_t ax = 0;
         ax < inimg->md->naxis; ax++)
    {
        if (ax != sliceaxis) {
            outimg->mdt->size[ax] = 1;
        }
    }
    outimg->mdt->datatype = _DATATYPE_FLOAT;
#ifndef FPS_STANDALONE
    createimagefromIMGID(outimg);
#else
    outimg->im =
        (IMAGE *) malloc(sizeof(IMAGE));
    strncpy(outimg->name,
            norm_outimname, 79);
    ImageStreamIO_createIm_gpu(
        outimg->im, outimg->name,
        outimg->mdt->naxis,
        outimg->mdt->size,
        outimg->mdt->datatype,
        -1, 1, 10, 0, 0, 0);
    outimg->md = outimg->im->md;
#endif
    uint32_t sizes[3] = {
        inimg->md->size[0],
        inimg->md->size[1],
        inimg->md->size[2]
    };
    if (inimg->md->naxis < 3) {
        sizes[2] = 1;
    }
    if (inimg->md->naxis < 2) {
        sizes[1] = 1;
    }
    double *normarray = (double *)
        calloc(sizes[sliceaxis],
               sizeof(double));
    for (uint32_t i = 0;
         i < sizes[0]; i++)
    {
        for (uint32_t j = 0;
             j < sizes[1]; j++)
        {
            for (uint32_t k = 0;
                 k < sizes[2]; k++)
            {
                uint64_t idx =
                    (uint64_t) k
                    * sizes[1] * sizes[0]
                    + (uint64_t) j * sizes[0]
                    + i;
                double v = 0;
                switch (
                    inimg->mdt->datatype)
                {
                case _DATATYPE_FLOAT:
                    v = inimg->im->array.F[
                        idx];
                    break;
                case _DATATYPE_DOUBLE:
                    v = inimg->im->array.D[
                        idx];
                    break;
                }
                uint32_t coords[3] =
                    {i, j, k};
                normarray[
                    coords[sliceaxis]]
                    += v * v;
            }
        }
    }
    for (uint32_t i = 0;
         i < sizes[sliceaxis]; i++)
    {
        outimg->im->array.F[i] =
            sqrt(normarray[i]);
    }
    free(normarray);
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
errno_t image_slicenorm(
    const char *inname,
    const char *outname,
    uint8_t sliceaxis
)
{
    IMGID idin =
        imgid_make_from_name(inname);
    IMGID idout =
        imgid_make_from_name(outname);
    errno_t ret = image_slicenorm_IMGID(
        &idin, &idout, sliceaxis);
    imgid_free(&idin);
    imgid_free(&idout);
    return ret;
}
#endif

void image_norm_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *inimg, IMAGE *outimg
)
{
    if (fps && fps->md->processinfo_change_cnt
        != processinfo_change_cnt_local)
    {
        fps_to_processinfo(fps, processinfo);
        processinfo_change_cnt_local =
            fps->md->processinfo_change_cnt;
    }
    if (!norm_sliceaxis) {
        return;
    }
    IMGID idin = imgid_make();
    idin.im = inimg;
    idin.md = &inimg->md[0];
    imgid_update_creationparams(&idin);

    IMGID idout = imgid_make();
    idout.im = outimg;
    idout.md = &outimg->md[0];
    imgid_update_creationparams(&idout);

    image_slicenorm_IMGID(
        &idin, &idout, *norm_sliceaxis);
    imgid_free(&idin);
    imgid_free(&idout);
}


static FPS_CLI_BINDING bindings[] = {
    NORMSLICE_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    NORMSLICE_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "normslice", "image norm by slice",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_normslice(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "normslice", "image norm by slice",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID idin =
        imgid_make_from_name(norm_inimname);
    resolveIMGID(
        &idin, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    IMGID idout =
        imgid_make_from_name(norm_outimname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_norm_compute(
        data.fpsptr, processinfo,
        idin.im, idout.im);
    processinfo_update_output_stream(
        processinfo, idout.im, idin.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    imgid_free(&idin);
    imgid_free(&idout);
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_arith__image_normslice()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    NORMSLICE_PARAMS,
    compute_function
)
#endif
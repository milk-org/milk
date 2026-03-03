#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "image_merge3D.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *immerge_inimname0 = NULL;
char     *immerge_inimname1 = NULL;
char     *immerge_outimname = NULL;
uint32_t *immerge_mergeaxis = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "immerge",
    .cmdkey      = "immerge",
    .description = "merge images along axis"
};

static uint64_t processinfo_change_cnt_local;

errno_t image_marge(
    IMGID inimg0, IMGID inimg1,
    IMGID *outimg, uint8_t mergeaxis
)
{
#ifndef FPS_STANDALONE
    resolveIMGID(
        &inimg0, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    resolveIMGID(
        &inimg1, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    resolveIMGID(
        outimg, ERRMODE_NULL,
        data.image, data.NB_MAX_IMAGE);
#endif
    if (outimg->ID == -1) {
        imgid_copy(&inimg0, outimg);
    }
    if (mergeaxis < 3) {
        uint32_t s0 =
            (inimg0.md->size[mergeaxis] == 0)
            ? 1 : inimg0.md->size[mergeaxis];
        uint32_t s1 =
            (inimg1.md->size[mergeaxis] == 0)
            ? 1 : inimg1.md->size[mergeaxis];
        outimg->mdt->size[mergeaxis] =
            s0 + s1;
    } else {
        return RETURN_FAILURE;
    }
    outimg->mdt->naxis =
        (outimg->mdt->size[2] > 1) ? 3
        : ((outimg->mdt->size[1] > 1) ? 2
           : 1);
#ifndef FPS_STANDALONE
    createimagefromIMGID(outimg);
#else
    outimg->im =
        (IMAGE *) malloc(sizeof(IMAGE));
    strncpy(outimg->name,
            immerge_outimname, 79);
    ImageStreamIO_createIm_gpu(
        outimg->im, outimg->name,
        outimg->mdt->naxis,
        outimg->mdt->size,
        outimg->mdt->datatype,
        -1, 1, 10, 0, 0, 0);
    outimg->md = outimg->im->md;
#endif
    size_t ts = ImageStreamIO_typesize(
        outimg->mdt->datatype);
    if (mergeaxis
        == outimg->mdt->naxis - 1)
    {
        size_t sz0 =
            ts * inimg0.md->nelement;
        memcpy(outimg->im->array.raw,
               inimg0.im->array.raw, sz0);
        memcpy(
            ((char *)
             outimg->im->array.raw)
            + sz0,
            inimg1.im->array.raw,
            ts * inimg1.md->nelement);
    } else {
        uint64_t b0 =
            inimg0.mdt->size[0];
        uint64_t b1 =
            inimg1.mdt->size[0];
        if (mergeaxis == 1) {
            b0 *= inimg0.mdt->size[1];
            b1 *= inimg1.mdt->size[1];
        }
        uint64_t po = 0, p0 = 0, p1 = 0;
        while (po
               < outimg->md->nelement)
        {
            memcpy(
                ((char *)
                 outimg->im->array.raw)
                + po * ts,
                ((char *)
                 inimg0.im->array.raw)
                + p0 * ts,
                ts * b0);
            p0 += b0;
            po += b0;
            memcpy(
                ((char *)
                 outimg->im->array.raw)
                + po * ts,
                ((char *)
                 inimg1.im->array.raw)
                + p1 * ts,
                ts * b1);
            p1 += b1;
            po += b1;
        }
    }
    return RETURN_SUCCESS;
}

void image_merge_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *inimg0, IMAGE *inimg1,
    IMAGE *outimg
)
{
    if (fps && fps->md->processinfo_change_cnt
        != processinfo_change_cnt_local)
    {
        fps_to_processinfo(fps, processinfo);
        processinfo_change_cnt_local =
            fps->md->processinfo_change_cnt;
    }
    if (!immerge_mergeaxis) {
        return;
    }
    IMGID id0 = imgid_make();
    id0.im = inimg0;
    id0.md = &inimg0->md[0];
    imgid_update_creationparams(&id0);

    IMGID id1 = imgid_make();
    id1.im = inimg1;
    id1.md = &inimg1->md[0];
    imgid_update_creationparams(&id1);

    IMGID idout = imgid_make();
    idout.im = outimg;
    idout.md = &outimg->md[0];
    imgid_update_creationparams(&idout);

    image_marge(
        id0, id1, &idout,
        *immerge_mergeaxis);

    imgid_free(&id0);
    imgid_free(&id1);
    imgid_free(&idout);
}


static FPS_CLI_BINDING bindings[] = {
    IMMERGE_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    IMMERGE_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "immerge", "merge images along axis",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_immerge(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "immerge", "merge images along axis",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID id0 = imgid_make_from_name(
        immerge_inimname0);
    resolveIMGID(
        &id0, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    IMGID id1 = imgid_make_from_name(
        immerge_inimname1);
    resolveIMGID(
        &id1, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    IMGID idout = imgid_make_from_name(
        immerge_outimname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_merge_compute(
        data.fpsptr, processinfo,
        id0.im, id1.im, idout.im);
    processinfo_update_output_stream(
        processinfo, idout.im, id0.im);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    imgid_free(&id0);
    imgid_free(&id1);
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
CLIADDCMD_COREMOD_arith__image_merge()
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
    IMMERGE_PARAMS,
    compute_function
)
#endif

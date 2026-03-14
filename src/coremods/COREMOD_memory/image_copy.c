/**
 * @file    image_copy.c
 * @brief   image copy, rename, copy to shm
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"
#include "stream_sem.h"
#include "image_copy.h"

/* forward decls */
imageID copy_image_ID(
    const char *name,
    const char *newname, int shared);
imageID copy_image_ID_IMGID(
    IMGID *imgin, IMGID *imgout, int shared);
imageID chname_image_ID(
    const char *ID_name,
    const char *new_name);
imageID chname_image_ID_IMGID(
    IMGID *imgin, const char *new_name);
errno_t COREMOD_MEMORY_cp2shm(
    const char *IDname,
    const char *IDshmname);
errno_t COREMOD_MEMORY_cp2shm_IMGID(
    IMGID *imgin, IMGID *imgout);


/* ================================================================
 *  COMMON PARAMS (2 string args)
 * ============================================================= */

static char p_srcname[FUNCTION_PARAMETER_STRMAXLEN]
    = "im1";
static char p_dstname[FUNCTION_PARAMETER_STRMAXLEN]
    = "im4";

#define FPS_PARAMS_2STR(X) \
    X(".srcname", p_srcname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "source image") \
    X(".dstname", p_dstname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_OUTPUT, \
      "destination image")


/* ================================================================
 *  CMD 1: cp
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_cp = {
    .fps_name    = "cp",
    .cmdkey      = "cp",
    .description = "copy image"
};

static CLICMDDATA CLIcmddata_cp = {
    "", "", CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms_cp = {0};

static __attribute__((constructor))
void init_cms_cp(void)
{
    strncpy(CLIcmddata_cp.key,
            FPS_app_info_cp.cmdkey,
            sizeof(CLIcmddata_cp.key) - 1);
    strncpy(CLIcmddata_cp.description,
            FPS_app_info_cp.description,
            sizeof(
                CLIcmddata_cp.description
            ) - 1);
    if (CLIcmddata_cp.cmdsettings == NULL) {
        CLIcmddata_cp.cmdsettings = &cms_cp;
    }
}

static errno_t compute_cp()
{
    copy_image_ID(p_srcname, p_dstname, 0);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: mv
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_mv = {
    .fps_name    = "mv",
    .cmdkey      = "mv",
    .description = "change image name"
};

static CLICMDDATA CLIcmddata_mv = {
    "", "", CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms_mv = {0};

static __attribute__((constructor))
void init_cms_mv(void)
{
    strncpy(CLIcmddata_mv.key,
            FPS_app_info_mv.cmdkey,
            sizeof(CLIcmddata_mv.key) - 1);
    strncpy(CLIcmddata_mv.description,
            FPS_app_info_mv.description,
            sizeof(
                CLIcmddata_mv.description
            ) - 1);
    if (CLIcmddata_mv.cmdsettings == NULL) {
        CLIcmddata_mv.cmdsettings = &cms_mv;
    }
}

static errno_t compute_mv()
{
    chname_image_ID(p_srcname, p_dstname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: imcp2shm (primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imcp2shm",
    .cmdkey      = "imcp2shm",
    .description =
        "copy image to shared memory"
};

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS_2STR(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS_2STR(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS cms_shm = {0};

static __attribute__((constructor))
void init_cms_shm(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms_shm;
    }
}

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    COREMOD_MEMORY_cp2shm(
        p_srcname, p_dstname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_cp(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_cp,
        farg, &CLIcmddata_cp,
        my_bindings, nb_bindings,
        compute_cp);
}

static errno_t CLIfunction_mv(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_mv,
        farg, &CLIcmddata_mv,
        my_bindings, nb_bindings,
        compute_mv);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__image_copy()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_cp, CLIfunction_cp);
        CLIcmddata_cp.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_mv, CLIfunction_mv);
        CLIcmddata_mv.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
imageID copy_image_ID_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    int shared
)
{
    resolveIMGID(imgin, ERRMODE_ABORT, dcimg, dcnimg);

    uint32_t naxis = imgin->md[0].naxis;
    uint32_t size[3];
    for(uint32_t i = 0; i < naxis; i++)
    {
        size[i] = imgin->md[0].size[i];
    }
    uint8_t  datatype = imgin->md[0].datatype;
    uint64_t nelement = imgin->md[0].nelement;

    resolveIMGID(imgout, ERRMODE_NULL, dcimg, dcnimg);

    int newim = 0;
    if(imgout->ID != -1)
    {
        // verify imgout has the right size and type
        if(imgin->md[0].nelement != imgout->md[0].nelement)
        {
            fprintf(stderr,
                    "ERROR [copy_image_ID_IMGID]: images %s and %s do not have "
                    "the same size -> deleting and re-creating image\n",
                    imgin->name,
                    imgout->name);
            newim = 1;
        }

        if(imgin->md[0].datatype != imgout->md[0].datatype)
        {
            fprintf(stderr,
                    "ERROR [copy_image_ID_IMGID]: images %s and %s do not have "
                    "the same type -> deleting and re-creating image\n",
                    imgin->name,
                    imgout->name);
            newim = 1;
        }

        if(newim == 1)
        {
            delete_image_ID(imgout->name, DELETE_IMAGE_ERRMODE_WARNING);
            imgout->ID = -1;
        }
    }

    if(imgout->ID == -1)
    {
        imgout->mdt->naxis = naxis;
        for(uint32_t i = 0; i < naxis; i++)
        {
            imgout->mdt->size[i] =
                size[i];
        }
        imgout->mdt->datatype = datatype;
        imgout->mdt->shared = shared;
        imgout->mdt->NBkw =
            NB_KEYWNODE_MAX;
        imgout->im = (IMAGE *) calloc(
            1, sizeof(IMAGE));
        imgid_mkimage(imgout);
    }

    imgout->md[0].write = 1;

    memcpy(imgout->im->array.raw,
           imgin->im->array.raw,
           ImageStreamIO_typesize(datatype) * nelement);

    COREMOD_MEMORY_image_set_sempost_byID(imgout->ID, -1);

    imgout->md[0].write = 0;
    imgout->md[0].cnt0++;

    return imgout->ID;
}

imageID copy_image_ID(
    const char *restrict name,
    const char *restrict newname,
    int shared
)
{
    IMGID imgin  = imgid_make_from_name(name);
    IMGID imgout = imgid_make_from_name(newname);

    return copy_image_ID_IMGID(&imgin, &imgout, shared);
}

imageID chname_image_ID_IMGID(
    IMGID *imgin,
    const char *new_name
)
{
    resolveIMGID(imgin, ERRMODE_ABORT, dcimg, dcnimg);

    if((image_ID(new_name, dcimg, dcnimg) == -1) && (variable_ID(new_name) == -1))
    {
        strcpy(imgin->im->name, new_name);
        strcpy(imgin->name, new_name);
    }
    else
    {
        printf("Cannot change name %s -> %s : new name already in use\n",
               imgin->name,
               new_name);
    }

    if(dcmemmon == 1)
    {
        list_image_ID_ncurses();
    }

    return imgin->ID;
}

imageID chname_image_ID(
    const char *restrict ID_name,
    const char *restrict new_name
)
{
    IMGID imgin = imgid_make_from_name(ID_name);

    return chname_image_ID_IMGID(&imgin, new_name);
}

/** copy an image to shared memory
 *
 *
 */
errno_t COREMOD_MEMORY_cp2shm_IMGID(
    IMGID *imgin,
    IMGID *imgout
)
{
    resolveIMGID(imgin, ERRMODE_ABORT, dcimg, dcnimg);

    uint32_t naxis = imgin->md[0].naxis;
    uint32_t size[3];
    for(uint32_t k = 0; k < naxis; k++)
    {
        size[k] = imgin->md[0].size[k];
    }
    uint8_t datatype = imgin->md[0].datatype;

    int shmOK = 1;
    resolveIMGID(imgout, ERRMODE_NULL, dcimg, dcnimg);
    if(imgout->ID != -1)
    {
        // verify type and size
        if(imgin->md[0].naxis != imgout->md[0].naxis)
        {
            shmOK = 0;
        }
        if(shmOK == 1)
        {
            for(int axis = 0; axis < imgin->md[0].naxis; axis++)
                if(imgin->md[0].size[axis] !=
                        imgout->md[0].size[axis])
                {
                    shmOK = 0;
                }
        }
        if(imgin->md[0].datatype != imgout->md[0].datatype)
        {
            shmOK = 0;
        }

        if(shmOK == 0)
        {
            delete_image_ID(imgout->name, DELETE_IMAGE_ERRMODE_WARNING);
            imgout->ID = -1;
        }
    }

    if(imgout->ID == -1)
    {
        imgout->mdt->naxis = naxis;
        for(uint32_t k = 0; k < naxis; k++)
        {
            imgout->mdt->size[k] =
                size[k];
        }
        imgout->mdt->datatype = datatype;
        imgout->mdt->shared = 1;
        imgout->im = (IMAGE *) calloc(
            1, sizeof(IMAGE));
        imgid_mkimage(imgout);
    }

    imgout->md[0].write = 1;

    memcpy(imgout->im->array.raw, imgin->im->array.raw,
           ImageStreamIO_typesize(datatype) * imgin->md[0].nelement);

    COREMOD_MEMORY_image_set_sempost_byID(imgout->ID, -1);
    imgout->md[0].cnt0++;
    imgout->md[0].write = 0;

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_cp2shm(
    const char *restrict IDname,
    const char *restrict IDshmname
)
{
    IMGID imgin  = imgid_make_from_name(IDname);
    IMGID imgout = imgid_make_from_name(IDshmname);

    return COREMOD_MEMORY_cp2shm_IMGID(&imgin, &imgout);
}


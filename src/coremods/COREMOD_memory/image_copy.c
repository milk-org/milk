/**
 * @file    image_copy.c
 * @brief   Image copy, rename, and copy-to-SHM
 *
 * Provides three CLI commands:
 *
 *  - cp       — copy image (local)
 *  - mv       — rename image in-place
 *  - imcp2shm — copy image into shared memory
 *
 * Each operation has an IMGID API and a string API.
 * Copy and cp2shm verify size/type compatibility,
 * deleting and re-creating the destination if there
 * is a mismatch.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

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
    const char *newname,
    int        shared);
imageID copy_image_ID_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    int   shared);
/**
 * @brief Rename an image in the image array.
 *
 * Updates the name field without copying data.
 */
imageID chname_image_ID(
    const char *ID_name,
    const char *new_name);
imageID chname_image_ID_IMGID(
    IMGID      *imgin,
    const char *new_name);
errno_t COREMOD_MEMORY_cp2shm(
    const char *IDname,
    const char *IDshmname);
errno_t COREMOD_MEMORY_cp2shm_IMGID(
    IMGID *imgin,
    IMGID *imgout);


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

static FPS_APP_INFO FPS_app_info_cp =
{
    .fps_name    = "cp",
    .cmdkey      = "cp",
    .description = "copy image",
    .description_long =
    "Copy an image stream to a new name or location. Creates a deep copy of all pixel data and metadata. Can also rename an existing image in the process table."
};

static CLICMDDATA CLIcmddata_cp =
{
    "", "", CLICMD_FIELDS_NOPARAM
};

FPS_CMDSETTINGS_INIT(cp, CLIcmddata_cp, FPS_app_info_cp)

static errno_t __attribute__((unused)) compute_cp()
{
    copy_image_ID(p_srcname, p_dstname, 0);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: mv
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_mv =
{
    .fps_name    = "mv",
    .cmdkey      = "mv",
    .description = "change image name",
    .description_long =
    "Copy an image stream to a new name or location. Creates a deep copy of all pixel data and metadata. Can also rename an existing image in the process table."
};

static CLICMDDATA CLIcmddata_mv =
{
    "", "", CLICMD_FIELDS_NOPARAM
};

FPS_CMDSETTINGS_INIT(mv, CLIcmddata_mv, FPS_app_info_mv)

static errno_t __attribute__((unused)) compute_mv()
{
    chname_image_ID(p_srcname, p_dstname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: imcp2shm (primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "imcp2shm",
    .cmdkey      = "imcp2shm",
    .description =
    "copy image to shared memory",
    .description_long =
    "Copy an image stream to a new name or location. Creates a deep copy of all pixel data and metadata. Can also rename an existing image in the process table."
};

static FPS_CLI_BINDING my_bindings[] =
{
    FPS_PARAMS_2STR(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] =
{
    FPS_PARAMS_2STR(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata =
{
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(shm, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
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
/**
 * @brief Copy image data to a new or existing image
 *
 * Resolves input image; if output exists, checks
 * for size/type mismatch (re-creates if needed).
 * Copies raw pixel data via memcpy and posts all
 * output semaphores.
 *
 * @param imgin   Source image
 * @param imgout  Destination image
 * @param shared  1 for shared memory, 0 for local
 * @return Output image ID
 */
imageID copy_image_ID_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    int   shared)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);

    uint32_t naxis = imgin->md[0].naxis;
    if(imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }
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

    __builtin_memcpy(
        imgout->im->array.raw,
        imgin->im->array.raw,
        ImageStreamIO_typesize(datatype) * nelement);

    imgout->md[0].cnt0++;
    imgout->md[0].write = 0;

    COREMOD_MEMORY_image_set_sempost_byID(
        imgout->ID, -1);

    return imgout->ID;
}

/**
 * @brief Copy image by name (string API)
 *
 * @param name     Source image name
 * @param newname  Destination image name
 * @param shared   1 for shared memory output
 * @return Output image ID
 */
imageID copy_image_ID(
    const char *restrict name,
    const char *restrict newname,
    int                  shared)
{
    IMGID imgin  = imgid_make_from_name(name);
    IMGID imgout = imgid_make_from_name(newname);

    return copy_image_ID_IMGID(&imgin, &imgout, shared);
}

/**
 * @brief Rename image in-place (IMGID API)
 *
 * Changes the name field of the image. Fails if
 * the new name is already in use by another image
 * or variable.
 *
 * @param imgin     Image to rename
 * @param new_name  New name string
 * @return Image ID
 */
imageID chname_image_ID_IMGID(
    IMGID      *imgin,
    const char *new_name)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);

    if((!imgid_exists(new_name)) && (variable_ID(new_name) == -1))
    {
        snprintf(imgin->im->name,
                 STRINGMAXLEN_IMAGE_NAME,
                 "%s", new_name);
        if(imgin->ID == -1)
        {
            return RETURN_FAILURE;
        }
        snprintf(imgin->name,
                 STRINGMAXLEN_IMAGE_NAME,
                 "%s", new_name);
    }
    else
    {
        printf("Cannot change name %s -> %s : new name already in use\n",
               imgin->name,
               new_name);
    }



    return imgin->ID;
}

/**
 * @brief Rename an image in the image array.
 *
 * Updates the name field without copying data.
 */
imageID chname_image_ID(
    const char *restrict ID_name,
    const char *restrict new_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);

    return chname_image_ID_IMGID(&imgin, new_name);
}

/**
 * @brief Copy image into shared memory (IMGID API)
 *
 * Creates a shared-memory image matching the source
 * dimensions/type, copies pixel data, and posts
 * semaphores. If destination exists but mismatches,
 * it is deleted and re-created.
 *
 * @param imgin   Source image (any allocation)
 * @param imgout  Destination (will be shared)
 * @return RETURN_SUCCESS
 */
errno_t COREMOD_MEMORY_cp2shm_IMGID(
    IMGID *imgin,
    IMGID *imgout)
{
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);

    uint32_t naxis = imgin->md[0].naxis;
    if(imgin->ID == -1)
    {
        return RETURN_FAILURE;
    }
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

    __builtin_memcpy(
        imgout->im->array.raw,
        imgin->im->array.raw,
        ImageStreamIO_typesize(datatype)
        * imgin->md[0].nelement);

    imgout->md[0].cnt0++;
    imgout->md[0].write = 0;

    COREMOD_MEMORY_image_set_sempost_byID(
        imgout->ID, -1);

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_cp2shm(
    const char *restrict IDname,
    const char *restrict IDshmname)
{
    IMGID imgin  = imgid_make_from_name(IDname);
    IMGID imgout = imgid_make_from_name(IDshmname);

    return COREMOD_MEMORY_cp2shm_IMGID(&imgin, &imgout);
}

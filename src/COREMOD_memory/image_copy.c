// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_copy.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"
#include "stream_sem.h"
#include "image_copy.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID copy_image_ID(const char *name, const char *newname, int shared);
imageID copy_image_ID_IMGID(IMGID *imgin, IMGID *imgout, int shared);

imageID chname_image_ID(const char *ID_name, const char *new_name);
imageID chname_image_ID_IMGID(IMGID *imgin, const char *new_name);

errno_t COREMOD_MEMORY_cp2shm(const char *IDname, const char *IDshmname);
errno_t COREMOD_MEMORY_cp2shm_IMGID(IMGID *imgin, IMGID *imgout);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t copy_image_ID__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    copy_image_ID(data.cmdargtoken[1].val.string,
                  data.cmdargtoken[2].val.string,
                  0);

    return CLICMD_SUCCESS;
}

static errno_t chname_image_ID__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    chname_image_ID(data.cmdargtoken[1].val.string,
                    data.cmdargtoken[2].val.string);

    return CLICMD_SUCCESS;
}

static errno_t COREMOD_MEMORY_cp2shm__cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) == 0)
    {
        COREMOD_MEMORY_cp2shm(data.cmdargtoken[1].val.string,
                              data.cmdargtoken[2].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t image_copy_addCLIcmd()
{
    RegisterCLIcommand(
        "cp",
        __FILE__,
        copy_image_ID__cli,
        "copy image",
        "source, dest",
        "cp im1 im4",
        "long copy_image_ID(const char *name, const char *newname, 0)");

  /*  RegisterCLIcommand(
        "cpsh",
        __FILE__,
        copy_image_ID_sharedmem__cli,
        "copy image - create in shared mem if does not exist",
        "source, dest",
        "cp im1 im4",
        "long copy_image_ID(const char *name, const char *newname, 1)");
*/

    RegisterCLIcommand(
        "mv",
        __FILE__,
        chname_image_ID__cli,
        "change image name",
        "source, dest",
        "mv im1 im4",
        "long chname_image_ID(const char *name, const char *newname)");

    RegisterCLIcommand("imcp2shm",
                       __FILE__,
                       COREMOD_MEMORY_cp2shm__cli,
                       "copy image ot shared memory",
                       "<image> <shared mem image>",
                       "imcp2shm im1 ims1",
                       "long COREMOD_MEMORY_cp2shm(const char *IDname, const "
                       "char *IDshmname)");

    return RETURN_SUCCESS;
}

imageID copy_image_ID_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    int shared
)
{
    resolveIMGID(imgin, ERRMODE_ABORT);

    uint32_t naxis = imgin->md[0].naxis;
    uint32_t size[3];
    for(uint32_t i = 0; i < naxis; i++)
    {
        size[i] = imgin->md[0].size[i];
    }
    uint8_t  datatype = imgin->md[0].datatype;
    uint64_t nelement = imgin->md[0].nelement;

    resolveIMGID(imgout, ERRMODE_NULL);

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
        create_image_ID(imgout->name,
                        naxis,
                        size,
                        datatype,
                        shared,
                        NB_KEYWNODE_MAX,
                        0,
                        &imgout->ID);
        resolveIMGID(imgout, ERRMODE_ABORT);
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
    IMGID imgin  = mkIMGID_from_name(name);
    IMGID imgout = mkIMGID_from_name(newname);

    return copy_image_ID_IMGID(&imgin, &imgout, shared);
}

imageID chname_image_ID_IMGID(
    IMGID *imgin,
    const char *new_name
)
{
    resolveIMGID(imgin, ERRMODE_ABORT);

    if((image_ID(new_name) == -1) && (variable_ID(new_name) == -1))
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

    if(data.MEM_MONITOR == 1)
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
    IMGID imgin = mkIMGID_from_name(ID_name);

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
    resolveIMGID(imgin, ERRMODE_ABORT);

    uint32_t naxis = imgin->md[0].naxis;
    uint32_t size[3];
    for(uint32_t k = 0; k < naxis; k++)
    {
        size[k] = imgin->md[0].size[k];
    }
    uint8_t datatype = imgin->md[0].datatype;

    int shmOK = 1;
    resolveIMGID(imgout, ERRMODE_NULL);
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
        create_image_ID(imgout->name, naxis, size, datatype, 1, 0, 0, &imgout->ID);
        resolveIMGID(imgout, ERRMODE_ABORT);
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
    IMGID imgin  = mkIMGID_from_name(IDname);
    IMGID imgout = mkIMGID_from_name(IDshmname);

    return COREMOD_MEMORY_cp2shm_IMGID(&imgin, &imgout);
}

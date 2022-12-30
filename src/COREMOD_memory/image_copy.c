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
#include "variable_ID.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID copy_image_ID(const char *name, const char *newname, int shared);

imageID chname_image_ID(const char *ID_name, const char *new_name);

errno_t COREMOD_MEMORY_cp2shm(const char *IDname, const char *IDshmname);

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

static errno_t copy_image_ID_sharedmem__cli()
{
    if(data.cmdargtoken[1].type != CLIARG_IMG)
    {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return CLICMD_INVALID_ARG;
    }

    copy_image_ID(data.cmdargtoken[1].val.string,
                  data.cmdargtoken[2].val.string,
                  1);

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

    RegisterCLIcommand(
        "cpsh",
        __FILE__,
        copy_image_ID_sharedmem__cli,
        "copy image - create in shared mem if does not exist",
        "source, dest",
        "cp im1 im4",
        "long copy_image_ID(const char *name, const char *newname, 1)");

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

imageID copy_image_ID(const char *name, const char *newname, int shared)
{
    imageID   ID;
    imageID   IDout;
    long      naxis;
    uint32_t *size = NULL;
    uint8_t   datatype;
    long      nelement;
    long      i;
    int       newim = 0;

    ID = image_ID(name);
    if(ID == -1)
    {
        PRINT_ERROR("image \"%s\" does not exist", name);
        exit(0);
    }
    naxis = data.image[ID].md[0].naxis;

    size = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(size == NULL)
    {
        PRINT_ERROR("malloc error");
        exit(0);
    }

    for(i = 0; i < naxis; i++)
    {
        size[i] = data.image[ID].md[0].size[i];
    }
    datatype = data.image[ID].md[0].datatype;

    nelement = data.image[ID].md[0].nelement;

    IDout = image_ID(newname);

    if(IDout != -1)
    {
        // verify newname has the right size and type
        if(data.image[ID].md[0].nelement != data.image[IDout].md[0].nelement)
        {
            fprintf(stderr,
                    "ERROR [copy_image_ID]: images %s and %s do not have "
                    "the same size -> deleting and re-creating image\n",
                    name,
                    newname);
            newim = 1;
        }
        if(data.image[ID].md[0].datatype != data.image[IDout].md[0].datatype)
        {
            fprintf(stderr,
                    "ERROR [copy_image_ID]: images %s and %s do not have "
                    "the same type -> deleting and re-creating image\n",
                    name,
                    newname);
            newim = 1;
        }

        if(newim == 1)
        {
            delete_image_ID(newname, DELETE_IMAGE_ERRMODE_WARNING);
            IDout = -1;
        }
    }

    if(IDout == -1)
    {
        create_image_ID(newname,
                        naxis,
                        size,
                        datatype,
                        shared,
                        data.NBKEYWORD_DFT,
                        0,
                        NULL);
        IDout = image_ID(newname);
    }
    else
    {
        // verify newname has the right size and type
        if(data.image[ID].md[0].nelement != data.image[IDout].md[0].nelement)
        {
            fprintf(stderr,
                    "ERROR [copy_image_ID]: images %s and %s do not "
                    "have the same size\n",
                    name,
                    newname);
            exit(0);
        }
        if(data.image[ID].md[0].datatype != data.image[IDout].md[0].datatype)
        {
            fprintf(stderr,
                    "ERROR [copy_image_ID]: images %s and %s do not "
                    "have the same type\n",
                    name,
                    newname);
            exit(0);
        }
    }
    data.image[IDout].md[0].write = 1;

    memcpy(data.image[IDout].array.raw,
           data.image[ID].array.raw,
           ImageStreamIO_typesize(datatype) * nelement);

    COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);

    data.image[IDout].md[0].write = 0;
    data.image[IDout].md[0].cnt0++;

    free(size);

    return IDout;
}

imageID chname_image_ID(const char *ID_name, const char *new_name)
{
    imageID ID;

    ID = -1;
    if((image_ID(new_name) == -1) && (variable_ID(new_name) == -1))
    {
        ID = image_ID(ID_name);
        strcpy(data.image[ID].name, new_name);
        //      if ( Debug > 0 ) { printf("change image name %s -> %s\n",ID_name,new_name);}
    }
    else
    {
        printf("Cannot change name %s -> %s : new name already in use\n",
               ID_name,
               new_name);
    }

    if(data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    return ID;
}

/** copy an image to shared memory
 *
 *
 */
errno_t COREMOD_MEMORY_cp2shm(const char *IDname, const char *IDshmname)
{
    imageID   ID;
    imageID   IDshm;
    uint8_t   datatype;
    long      naxis;
    uint32_t *sizearray;
    long      k;
    int       axis;
    int       shmOK;

    ID    = image_ID(IDname);
    naxis = data.image[ID].md[0].naxis;

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    datatype  = data.image[ID].md[0].datatype;
    for(k = 0; k < naxis; k++)
    {
        sizearray[k] = data.image[ID].md[0].size[k];
    }

    shmOK = 1;
    IDshm = read_sharedmem_image(IDshmname);
    if(IDshm != -1)
    {
        // verify type and size
        if(data.image[ID].md[0].naxis != data.image[IDshm].md[0].naxis)
        {
            shmOK = 0;
        }
        if(shmOK == 1)
        {
            for(axis = 0; axis < data.image[IDshm].md[0].naxis; axis++)
                if(data.image[ID].md[0].size[axis] !=
                        data.image[IDshm].md[0].size[axis])
                {
                    shmOK = 0;
                }
        }
        if(data.image[ID].md[0].datatype != data.image[IDshm].md[0].datatype)
        {
            shmOK = 0;
        }

        if(shmOK == 0)
        {
            delete_image_ID(IDshmname, DELETE_IMAGE_ERRMODE_WARNING);
            IDshm = -1;
        }
    }

    if(IDshm == -1)
    {
        create_image_ID(IDshmname, naxis, sizearray, datatype, 1, 0, 0, &IDshm);
    }
    free(sizearray);

    //data.image[IDshm].md[0].nelement = data.image[ID].md[0].nelement;
    //printf("======= %ld %ld ============\n", data.image[ID].md[0].nelement, data.image[IDshm].md[0].nelement);

    data.image[IDshm].md[0].write = 1;

    memcpy(data.image[IDshm].array.raw, data.image[ID].array.raw,
           ImageStreamIO_typesize(datatype) * data.image[ID].md[0].nelement);

    COREMOD_MEMORY_image_set_sempost_byID(IDshm, -1);
    data.image[IDshm].md[0].cnt0++;
    data.image[IDshm].md[0].write = 0;

    return RETURN_SUCCESS;
}

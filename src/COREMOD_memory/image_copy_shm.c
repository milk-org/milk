#include "CommandLineInterface/CLIcore.h"

#include "create_image.h"
#include "read_shmim.h"


// Local variables pointers
static char *inimname;
static char *outimname;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".in_name", "input image", "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname
    },
    {
        CLIARG_STR, ".out_name", "output stream", "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname
    }
};


// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "imcpshm",
    "copy image to shm",
    CLICMD_FIELDS_DEFAULTS
};




// Computation code
static errno_t image_copy_shm(
    IMGID img,
    char *outshmname
)
{
    resolveIMGID(&img, ERRMODE_ABORT);
    imageID ID = img.ID;

    uint8_t naxis = data.image[ID].md[0].naxis;
    uint32_t *sizearray = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    uint8_t datatype = data.image[ID].md[0].datatype;
    for(uint8_t k = 0; k < naxis; k++)
    {
        sizearray[k] = data.image[ID].md[0].size[k];
    }
    uint16_t NBkw = data.image[ID].md[0].NBkw;

    int shmOK = 1;

    DEBUG_TRACEPOINT("reading = %s", outshmname);
    imageID IDshm = read_sharedmem_image(outshmname);
    DEBUG_TRACEPOINT("IDshm = %ld", IDshm);

    if(IDshm != -1)
    {
        // verify type and size
        if(data.image[ID].md[0].naxis != data.image[IDshm].md[0].naxis)
        {
            shmOK = 0;
        }
        if(shmOK == 1)
        {
            for(uint8_t axis = 0; axis < data.image[IDshm].md[0].naxis; axis++)
                if(data.image[ID].md[0].size[axis] != data.image[IDshm].md[0].size[axis])
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
            delete_image_ID(outshmname);
            IDshm = -1;
        }
    }


    if(IDshm == -1)
    {
        DEBUG_TRACEPOINT("Creating image");
        IDshm = create_image_ID(outshmname, naxis, sizearray, datatype, 1, NBkw, 0);
    }
    free(sizearray);

    //data.image[IDshm].md[0].nelement = data.image[ID].md[0].nelement;
    //printf("======= %ld %ld ============\n", data.image[ID].md[0].nelement, data.image[IDshm].md[0].nelement);

    DEBUG_TRACEPOINT("Writing memory");

    data.image[IDshm].md[0].write = 1;

    char      *ptr1;
    char      *ptr2;

    switch(datatype)
    {
        case _DATATYPE_FLOAT :
            ptr1 = (char *) data.image[ID].array.F;
            ptr2 = (char *) data.image[IDshm].array.F;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_FLOAT * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_DOUBLE :
            ptr1 = (char *) data.image[ID].array.D;
            ptr2 = (char *) data.image[IDshm].array.D;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_DOUBLE * data.image[ID].md[0].nelement);
            break;


        case _DATATYPE_INT8 :
            ptr1 = (char *) data.image[ID].array.SI8;
            ptr2 = (char *) data.image[IDshm].array.SI8;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_INT8 * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_UINT8 :
            ptr1 = (char *) data.image[ID].array.UI8;
            ptr2 = (char *) data.image[IDshm].array.UI8;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_UINT8 * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_INT16 :
            ptr1 = (char *) data.image[ID].array.SI16;
            ptr2 = (char *) data.image[IDshm].array.SI16;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_INT16 * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_UINT16 :
            ptr1 = (char *) data.image[ID].array.UI16;
            ptr2 = (char *) data.image[IDshm].array.UI16;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_UINT16 * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_INT32 :
            ptr1 = (char *) data.image[ID].array.SI32;
            ptr2 = (char *) data.image[IDshm].array.SI32;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_INT32 * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_UINT32 :
            ptr1 = (char *) data.image[ID].array.UI32;
            ptr2 = (char *) data.image[IDshm].array.UI32;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_UINT32 * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_INT64 :
            ptr1 = (char *) data.image[ID].array.SI64;
            ptr2 = (char *) data.image[IDshm].array.SI64;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_INT64 * data.image[ID].md[0].nelement);
            break;

        case _DATATYPE_UINT64 :
            ptr1 = (char *) data.image[ID].array.UI64;
            ptr2 = (char *) data.image[IDshm].array.UI64;
            memcpy((void *) ptr2, (void *) ptr1,
                   SIZEOF_DATATYPE_UINT64 * data.image[ID].md[0].nelement);
            break;

        default :
            printf("data type not supported\n");
            break;
    }

    // copy keywords
    ptr1 = (char *) data.image[ID].kw;
    ptr2 = (char *) data.image[IDshm].kw;
    memcpy((void *) ptr2, (void *) ptr1,
                   sizeof(IMAGE_KEYWORD) * NBkw);

    COREMOD_MEMORY_image_set_sempost_byID(IDshm, -1);
    data.image[IDshm].md[0].cnt0++;
    data.image[IDshm].md[0].write = 0;



    return RETURN_SUCCESS;
}


// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_copy_shm(
        makeIMGID(inimname),
        outimname
    );

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


INSERT_STD_FPSCLIfunctions

errno_t CLIADDCMD_COREMOD_memory__image_copy_shm()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

#include "CommandLineInterface/CLIcore.h"

#include "create_image.h"
#include "read_shmim.h"

// Local variables pointers
static char *inimname;
static char *outimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".out_name",
        "output stream",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    }
};

// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "imcpshm", "copy image to shm", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

// Computation code
static errno_t image_copy_shm(IMGID img, char *outshmname)
{
    resolveIMGID(&img, ERRMODE_ABORT);
    imageID ID = img.ID;

    uint8_t   naxis     = data.image[ID].md[0].naxis;
    uint32_t *sizearray = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    uint8_t   datatype  = data.image[ID].md[0].datatype;
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
            delete_image_ID(outshmname, DELETE_IMAGE_ERRMODE_WARNING);
            IDshm = -1;
        }
    }

    if(IDshm == -1)
    {
        DEBUG_TRACEPOINT("Creating image");
        create_image_ID(outshmname,
                        naxis,
                        sizearray,
                        datatype,
                        1,
                        NBkw,
                        0,
                        &IDshm);
    }
    free(sizearray);

    //data.image[IDshm].md[0].nelement = data.image[ID].md[0].nelement;
    //printf("======= %ld %ld ============\n", data.image[ID].md[0].nelement, data.image[IDshm].md[0].nelement);

    DEBUG_TRACEPOINT("Writing memory");

    data.image[IDshm].md[0].write = 1;

    char *ptr1;
    char *ptr2;

    memcpy(data.image[IDshm].array.raw,
           data.image[ID].array.raw,
           ImageStreamIO_typesize(datatype));

    // copy keywords
    memcpy(data.image[IDshm].kw, data.image[ID].kw, sizeof(IMAGE_KEYWORD) * NBkw);

    COREMOD_MEMORY_image_set_sempost_byID(IDshm, -1);
    data.image[IDshm].md[0].cnt0++;
    data.image[IDshm].md[0].write = 0;

    return RETURN_SUCCESS;
}

// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_copy_shm(mkIMGID_from_name(inimname), outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

errno_t
CLIADDCMD_COREMOD_memory__image_copy_shm()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

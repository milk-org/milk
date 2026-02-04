/**
 * @file    read_shmim.c
 * @brief   read shared memory stream
 */

#include <fcntl.h>    // open
#include <sys/mman.h> // mmap
#include <sys/stat.h>
#include <unistd.h> // close

#include "CLIcore.h"
#include "image_ID.h"
#include "image_keyword_list.h"
#include "list_image.h"
#include "imageID.h"

// Local variables pointers
static char *insname;

// List of arguments to function
static CLICMDARGDEF farg[] = {{
        CLIARG_STR_NOT_IMG,
        ".in_sname",
        "input stream",
        "ims1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &insname,
        NULL
    }
};

// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "readshmim", "read shared memory image", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}








imageID read_sharedmem_image(
    const char * restrict sname,
    IMAGE *imagearray,
    long NB_images
)
{
    IMGID img = imgid_make_from_name(sname);
    resolveIMGID(&img, ERRMODE_ABORT, imagearray, NB_images);
    imgid_connect(&img, IMGID_CONNECT_NOCHECK);
    if (img.ID == -1)
    {
        return -1;
    }

    return RegisterIMGID(&img, imagearray, NB_images);
}



// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    read_sharedmem_image(insname, data.image, data.NB_MAX_IMAGE);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

errno_t
CLIADDCMD_COREMOD_memory__read_sharedmem_image()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

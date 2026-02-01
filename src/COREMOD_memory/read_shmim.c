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





imageID RegisterIMGID(
    IMGID *img,
    IMAGE *imagearray,
    long NB_images
)
{
    imageID ID = -1;

    if (imagearray == NULL)
    {
        // If no array provided, we close the image and return 0 (success) or -1 (fail)
        // This corresponds to non-CLI mode check
        if (img->ID != -1)
        {
            ImageStreamIO_closeIm(img->im);
            free(img->im);
            img->ID = 0;
            return 0;
        }
        return -1;
    }

    // Check if already loaded
    ID = image_ID(img->name, imagearray, NB_images);
    if(ID != -1)
    {
        // Already loaded: close the one we just opened and point to the existing one
        if (img->im != NULL)
        {
            ImageStreamIO_closeIm(img->im);
            free(img->im);
        }
        img->ID = ID;
        img->im = &imagearray[ID];
        img->md = &imagearray[ID].md[0];
        img->createcnt = imagearray[ID].createcnt;
        updateIMGIDcreationparams(img);
    }
    else
    {
        // Not loaded: find slot and move it
        ID = next_avail_image_ID(-1);
        if (ID != -1)
        {
            // We assume imagearray has enough space and ID is valid index
            // Move content
            memcpy(&imagearray[ID], img->im, sizeof(IMAGE));
            // Free temporary structure
            free(img->im);

            img->ID = ID;
            img->im = &imagearray[ID];
            img->md = &imagearray[ID].md[0];
            // img.createcnt = data.image[img.ID].createcnt; // Should be set? ImageStreamIO doesn't set createcnt?
            // Actually createcnt is in IMAGE struct, so it was copied.

            imagearray[ID].used = 1; // next_avail_image_ID sets this, but just to be sure if we used different logic

            updateIMGIDcreationparams(img);
        }
        else
        {
            // No space available
            if (img->im != NULL)
            {
                ImageStreamIO_closeIm(img->im);
                free(img->im);
            }
            img->ID = -1;
        }
    }

    return ID;
}


imageID read_sharedmem_image(
    const char * restrict sname,
    IMAGE *imagearray,
    long NB_images
)
{
    IMGID img = read_sharedmem_img(sname);
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

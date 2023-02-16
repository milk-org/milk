/** @file im3Dto2D.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID image_basic_3Dto2D(const char *__restrict IDname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_3Dto2D_cli() // collapse first 2 axis into one
{
    if(CLI_checkarg(1, CLIARG_IMG) == 0)
    {
        image_basic_3Dto2D(data.cmdargtoken[1].val.string);
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

errno_t __attribute__((cold)) im3Dto2D_addCLIcmd()
{

    RegisterCLIcommand("im3Dto2D",
                       __FILE__,
                       image_basic_3Dto2D_cli,
                       "collapse first 2 axis of 3D image (in place)",
                       "<image name>",
                       "im3Dto2D im1",
                       "long image_basic_3Dto2D(const char *IDname)");

    return RETURN_SUCCESS;
}

/* ----------------------------------------------------------------------
 *
 * turns a 3D image into a 2D image by collapsing first 2 axis
 *
 *
 * ---------------------------------------------------------------------- */

imageID image_basic_3Dto2D_byID(imageID ID)
{
    if(data.image[ID].md[0].naxis != 3)
    {
        printf("ERROR: image needs to have 3 axis\n");
    }
    else
    {
        data.image[ID].md[0].size[0] *= data.image[ID].md[0].size[1];
        data.image[ID].md[0].size[1] = data.image[ID].md[0].size[2];
        data.image[ID].md[0].naxis   = 2;
    }

    return ID;
}

imageID image_basic_3Dto2D(const char *__restrict IDname)
{
    imageID ID;

    ID = image_ID(IDname);
    image_basic_3Dto2D_byID(ID);

    return ID;
}

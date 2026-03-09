/**
 * @file clearall.c
 * @brief Clearall module
 */

/** @file clearall.c
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "delete_image.h"
#include "delete_variable.h"
#include "image_ID.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t clearall();

// ==========================================
#ifndef MILK_NO_CLI
// Command line interface wrapper function(s)
// ==========================================

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t clearall_addCLIcmd()
{

    RegisterCLIcommand("rmall",
                       __FILE__,
                       clearall,
                       "remove all images",
                       "no argument",
                       "rmall",
                       "int clearall()");

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */
errno_t clearall()
{
    imageID ID;

    // clear images
    for(ID = 0; ID < dcnimg; ID++)
    {
        if(dcimg[ID].used == 1)
        {
            delete_image_ID(dcimg[ID].name, DELETE_IMAGE_ERRMODE_WARNING);
        }
    }

    // clear variables
    for(ID = 0; ID < dcnvar; ID++)
    {
        if(dcvar[ID].used == 1)
        {
            delete_variable_ID(dcvar[ID].name);
        }
    }

    // clear FPS

    for(int fpsindex = 0; fpsindex < dcnfps; fpsindex++)
    {
        DEBUG_TRACEPOINT("clear FPS %d", fpsindex);
        dcfpsarr[fpsindex].SMfd = -1;
        if(dcfpsarr[fpsindex].parray != NULL)
        {
            dcfpsarr[fpsindex].parray = NULL;
        }
        if(dcfpsarr[fpsindex].md != NULL)
        {
            dcfpsarr[fpsindex].md = NULL;
        }
    }

    return RETURN_SUCCESS;
}


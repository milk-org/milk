/** @file cubecollapse.c
 */

#include <sched.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID cube_collapse(const char *__restrict ID_in_name,
                      const char *__restrict ID_out_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_cubecollapse_cli()
{
    if(0 + CLI_checkarg(1, 4) + CLI_checkarg(2, 3) == 0)
    {
        cube_collapse(data.cmdargtoken[1].val.string,
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

errno_t __attribute__((cold)) cubecollapse_addCLIcmd()
{

    RegisterCLIcommand(
        "cubecollapse",
        __FILE__,
        image_basic_cubecollapse_cli,
        "collapse a cube along z",
        "cubecollapse <inim> <outim>",
        "cubecollapse im1 outim",
        "long cube_collapse(const char *ID_in_name, const char *ID_out_name)");

    return RETURN_SUCCESS;
}

imageID cube_collapse(const char *__restrict ID_in_name,
                      const char *__restrict ID_out_name)
{
    imageID IDin;
    imageID IDout;
    long    xsize, ysize, ksize;
    long    ii, kk;

    IDin  = image_ID(ID_in_name);
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    ksize = data.image[IDin].md[0].size[2];

    create_2Dimage_ID(ID_out_name, xsize, ysize, &IDout);

    for(ii = 0; ii < xsize * ysize; ii++)
    {
        for(kk = 0; kk < ksize; kk++)
        {
            data.image[IDout].array.F[ii] +=
                data.image[IDin].array.F[kk * xsize * ysize + ii];
        }
    }

    return (IDout);
}

/** @file imswapaxis2D.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID image_basic_SwapAxis2D(const char *__restrict IDin_name,
                               const char *__restrict IDout_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_SwapAxis2D_cli() // swap axis of a 2D image
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) == 0)
    {
        image_basic_SwapAxis2D(data.cmdargtoken[1].val.string,
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

errno_t __attribute__((cold)) imswapaxis2D_addCLIcmd()
{

    RegisterCLIcommand("imswapaxis2D",
                       __FILE__,
                       image_basic_SwapAxis2D_cli,
                       "Swap axis of a 2D image",
                       "<input image> <output image>",
                       "imswapaxis2D im1 im2",
                       "long image_basic_SwapAxis2D(const char *IDin_name, "
                       "const char *IDout_name)");

    return RETURN_SUCCESS;
}

imageID image_basic_SwapAxis2D_byID(imageID IDin,
                                    const char *__restrict IDout_name)
{
    imageID IDout = -1;

    if(data.image[IDin].md[0].naxis != 2)
    {
        printf("ERROR: image needs to have 2 axis\n");
    }
    else
    {
        create_2Dimage_ID(IDout_name,
                          data.image[IDin].md[0].size[1],
                          data.image[IDin].md[0].size[0],
                          &IDout);

        for(uint32_t ii = 0; ii < data.image[IDin].md[0].size[0]; ii++)
            for(uint32_t jj = 0; jj < data.image[IDin].md[0].size[1]; jj++)
            {
                data.image[IDout]
                .array.F[ii * data.image[IDin].md[0].size[1] + jj] =
                    data.image[IDin]
                    .array.F[jj * data.image[IDin].md[0].size[0] + ii];
            }
    }

    return IDout;
}

imageID image_basic_SwapAxis2D(const char *__restrict IDin_name,
                               const char *__restrict IDout_name)
{
    imageID IDin;
    imageID IDout = -1;

    IDin = image_ID(IDin_name);
    image_basic_SwapAxis2D_byID(IDin, IDout_name);

    return IDout;
}

/**
 * @file imswapaxis2D.c
 * @brief Imswapaxis2d module
 */

/** @file imswapaxis2D.c
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

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

    if(dcimg[IDin].md[0].naxis != 2)
    {
        printf("ERROR: image needs to have 2 axis\n");
    }
    else
    {
        create_2Dimage_ID(IDout_name,
                          dcimg[IDin].md[0].size[1],
                          dcimg[IDin].md[0].size[0],
                          &IDout);

        for(uint32_t ii = 0; ii < dcimg[IDin].md[0].size[0]; ii++)
            for(uint32_t jj = 0; jj < dcimg[IDin].md[0].size[1]; jj++)
            {
                dcimg[IDout]
                .array.F[ii * dcimg[IDin].md[0].size[1] + jj] =
                    dcimg[IDin]
                    .array.F[jj * dcimg[IDin].md[0].size[0] + ii];
            }
    }

    return IDout;
}

imageID image_basic_SwapAxis2D(const char *__restrict IDin_name,
                               const char *__restrict IDout_name)
{
    imageID IDin;
    imageID IDout = -1;

    IDin = image_ID(IDin_name, dcimg, dcnimg);
    image_basic_SwapAxis2D_byID(IDin, IDout_name);

    return IDout;
}

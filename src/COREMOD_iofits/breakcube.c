/**
 * @file    breakcube.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID break_cube(const char *restrict ID_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t break_cube_cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) == 0)
    {
        break_cube(data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }

    return CLICMD_SUCCESS;

    break_cube(data.cmdargtoken[1].val.string);

    return CLICMD_SUCCESS;
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t breakcube_addCLIcmd()
{

    RegisterCLIcommand("breakcube",
                       __FILE__,
                       break_cube_cli,
                       "break cube into individual images (slices)",
                       "<input image>",
                       "breakcube imc",
                       "int break_cube(char *ID_name)");

    return RETURN_SUCCESS;
}

imageID break_cube(const char *restrict ID_name)
{
    imageID  ID;
    uint32_t naxes[3];
    long     i;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    naxes[2] = data.image[ID].md[0].size[2];

    for (uint32_t kk = 0; kk < naxes[2]; kk++)
    {
        long ID1;

        CREATE_IMAGENAME(framename, "%s_%5u", ID_name, kk);

        for (i = 0; i < (long) strlen(framename); i++)
        {
            if (framename[i] == ' ')
            {
                framename[i] = '0';
            }
        }
        create_2Dimage_ID(framename, naxes[0], naxes[1], &ID1);
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
            for (uint32_t jj = 0; jj < naxes[1]; jj++)
            {
                data.image[ID1].array.F[jj * naxes[0] + ii] =
                    data.image[ID]
                        .array.F[kk * naxes[0] * naxes[1] + jj * naxes[0] + ii];
            }
    }

    return ID;
}

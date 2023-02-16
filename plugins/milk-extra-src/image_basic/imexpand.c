/** @file imexpand.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID
basic_expand(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_expand3D(
    const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_expand_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 2) ==
            0)
    {
        basic_expand(data.cmdargtoken[1].val.string,
                     data.cmdargtoken[2].val.string,
                     data.cmdargtoken[3].val.numl,
                     data.cmdargtoken[4].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t image_basic_expand3D_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 2) + CLI_checkarg(5, 2) ==
            0)
    {
        basic_expand3D(data.cmdargtoken[1].val.string,
                       data.cmdargtoken[2].val.string,
                       data.cmdargtoken[3].val.numl,
                       data.cmdargtoken[4].val.numl,
                       data.cmdargtoken[5].val.numl);
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

errno_t imexpand_addCLIcmd()
{

    RegisterCLIcommand("imexpand",
                       __FILE__,
                       image_basic_expand_cli,
                       "expand 2D image",
                       "<image in> <output image> <x factor> <y factor>",
                       "imexpand im1 im2 2 2",
                       "long basic_expand(const char *ID_name, const char "
                       "*ID_name_out, int n1, int n2)");

    RegisterCLIcommand(
        "imexpand3D",
        __FILE__,
        image_basic_expand3D_cli,
        "expand 3D image",
        "<image in> <output image> <x factor> <y factor> <z factor>",
        "imexpand3D im1 im2 2 2 2",
        "long basic_expand3D(const char *ID_name, const char *ID_name_out, int "
        "n1, int n2, int n3)");

    return RETURN_SUCCESS;
}

/* expand image by factor n1 along x axis and n2 along y axis */
imageID
basic_expand(const char *ID_name, const char *ID_name_out, int n1, int n2)
{
    DEBUG_TRACE_FSTART();

    imageID ID;
    imageID ID_out; /* ID for the output image */
    long    ii, jj;
    long    naxes[2], naxes_out[2];
    int     i, j;

    ID = image_ID(ID_name);

    naxes[0]     = data.image[ID].md[0].size[0];
    naxes[1]     = data.image[ID].md[0].size[1];
    naxes_out[0] = naxes[0] * n1;
    naxes_out[1] = naxes[1] * n2;

    FUNC_CHECK_RETURN(
        create_2Dimage_ID(ID_name_out, naxes_out[0], naxes_out[1], &ID_out));

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
            for(i = 0; i < n1; i++)
                for(j = 0; j < n2; j++)
                {
                    data.image[ID_out]
                    .array.F[(jj * n2 + j) * naxes_out[0] + ii * n1 + i] =
                        data.image[ID].array.F[jj * naxes[0] + ii];
                }

    DEBUG_TRACE_FEXIT();
    return (ID_out);
}

/* expand image by factor n1 along x axis and n2 along y axis */
imageID basic_expand3D(
    const char *ID_name, const char *ID_name_out, int n1, int n2, int n3)
{
    imageID ID;
    imageID ID_out; /* ID for the output image */
    long    ii, jj, kk;
    long    naxes[3], naxes_out[3];
    int     i, j, k;

    ID = image_ID(ID_name);

    naxes[0] = data.image[ID].md[0].size[0];
    if(data.image[ID].md[0].naxis > 1)
    {
        naxes[1] = data.image[ID].md[0].size[1];
    }
    else
    {
        naxes[1] = 1;
    }
    if(data.image[ID].md[0].naxis == 3)
    {
        naxes[2] = data.image[ID].md[0].size[2];
    }
    else
    {
        naxes[2] = 1;
    }
    naxes_out[0] = naxes[0] * n1;
    naxes_out[1] = naxes[1] * n2;
    naxes_out[2] = naxes[2] * n3;

    printf(" %ld %ld %ld -> %ld %ld %ld\n",
           naxes[0],
           naxes[1],
           naxes[2],
           naxes_out[0],
           naxes_out[1],
           naxes_out[2]);

    create_3Dimage_ID(ID_name_out,
                      naxes_out[0],
                      naxes_out[1],
                      naxes_out[2],
                      &ID_out);
    list_image_ID();

    for(kk = 0; kk < naxes[2]; kk++)
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
                for(i = 0; i < n1; i++)
                    for(j = 0; j < n2; j++)
                        for(k = 0; k < n3; k++)
                        {
                            data.image[ID_out]
                            .array
                            .F[(kk * n3 + k) * naxes_out[0] * naxes_out[1] +
                                             (jj * n2 + j) * naxes_out[0] + ii * n1 + i] =
                                   data.image[ID]
                                   .array.F[kk * naxes[0] * naxes[1] +
                                               jj * naxes[0] + ii];
                        }
    return (ID_out);
}

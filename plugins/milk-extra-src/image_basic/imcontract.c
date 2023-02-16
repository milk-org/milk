/** @file imcontract.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID
basic_contract(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_contract3D(
    const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_contract_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 2) ==
            0)
    {
        basic_contract(data.cmdargtoken[1].val.string,
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

static errno_t image_basic_contract3D_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 2) + CLI_checkarg(5, 2) ==
            0)
    {
        basic_contract3D(data.cmdargtoken[1].val.string,
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

errno_t imcontract_addCLIcmd()
{

    RegisterCLIcommand("imcontract",
                       __FILE__,
                       image_basic_contract_cli,
                       "image binning",
                       "<inim> <outim> <binx> <biny>",
                       "imcontract im1 outim 4 4",
                       "long basic_contract(const char *ID_name, const char "
                       "*ID_name_out, int n1, int n2)");

    RegisterCLIcommand("imcontract3D",
                       __FILE__,
                       image_basic_contract3D_cli,
                       "image binning (3D)",
                       "<inim> <outim> <binx> <biny> <binz>",
                       "imcontracteD im1 outim 4 4 1",
                       "long basic_contract3D(const char *ID_name, const char "
                       "*ID_name_out, int n1, int n2, int n3)");

    return RETURN_SUCCESS;
}

imageID
basic_contract(const char *ID_name, const char *ID_name_out, int n1, int n2)
{
    imageID  ID;
    imageID  ID_out; /* ID for the output image */
    uint32_t naxes[2], naxes_out[2];
    int      i, j;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    naxes_out[0] = naxes[0] / n1;
    naxes_out[1] = naxes[1] / n2;

    //  printf("%ld %ld  ->  %ld %ld\n",naxes[0],naxes[1],naxes_out[0],naxes_out[1]);
    create_2Dimage_ID(ID_name_out, naxes_out[0], naxes_out[1], &ID_out);

    for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
        for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
            for(i = 0; i < n1; i++)
                for(j = 0; j < n2; j++)
                {
                    data.image[ID_out].array.F[jj * naxes_out[0] + ii] +=
                        data.image[ID]
                        .array.F[(jj * n2 + j) * naxes[0] + ii * n1 + i];
                }

    return (ID_out);
}

imageID basic_contract3D(
    const char *ID_name, const char *ID_name_out, int n1, int n2, int n3)
{
    DEBUG_TRACE_FSTART();

    imageID   ID;
    imageID   ID_out; /* ID for the output image */
    uint32_t  naxes[3];
    uint32_t *naxes_out;
    uint8_t   datatype;

    ID       = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    naxes[2] = data.image[ID].md[0].size[2];

    naxes_out = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxes_out == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }
    naxes_out[0] = naxes[0] / n1;
    naxes_out[1] = naxes[1] / n2;
    naxes_out[2] = naxes[2] / n3;

    if(naxes_out[2] == 1)
    {
        create_2Dimage_ID(ID_name_out, naxes_out[0], naxes_out[1], NULL);
    }
    else
    {
        printf("(%ld x %ld x %ld)  ->  (%ld x %ld x %ld)\n",
               (long) naxes[0],
               (long) naxes[1],
               (long) naxes[2],
               (long) naxes_out[0],
               (long) naxes_out[1],
               (long) naxes_out[2]);
        create_image_ID(ID_name_out, 3, naxes_out, datatype, 0, 0, 0, NULL);
    }

    ID_out = image_ID(ID_name_out);

    switch(datatype)
    {
        case _DATATYPE_FLOAT:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    data.image[ID_out]
                                    .array.F[kk * naxes_out[0] * naxes_out[1] +
                                                jj * naxes_out[0] + ii] +=
                                                 data.image[ID]
                                                 .array
                                                 .F[(kk * n3 + k) * naxes[0] * naxes[1] +
                                                                  (jj * n2 + j) * naxes[0] + ii * n1 +
                                                                  i];
                                }
            break;
        case _DATATYPE_DOUBLE:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    data.image[ID_out]
                                    .array.D[kk * naxes_out[0] * naxes_out[1] +
                                                jj * naxes_out[0] + ii] +=
                                                 data.image[ID]
                                                 .array
                                                 .D[(kk * n3 + k) * naxes[0] * naxes[1] +
                                                                  (jj * n2 + j) * naxes[0] + ii * n1 +
                                                                  i];
                                }
            break;
        case _DATATYPE_COMPLEX_FLOAT:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    data.image[ID_out]
                                    .array
                                    .CF[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .re += data.image[ID]
                                           .array
                                           .CF[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .re;
                                    data.image[ID_out]
                                    .array
                                    .CF[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .im += data.image[ID]
                                           .array
                                           .CF[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .im;
                                }
            break;
        case _DATATYPE_COMPLEX_DOUBLE:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    data.image[ID_out]
                                    .array
                                    .CD[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .re += data.image[ID]
                                           .array
                                           .CD[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .re;
                                    data.image[ID_out]
                                    .array
                                    .CD[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .im += data.image[ID]
                                           .array
                                           .CD[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .im;
                                }
            break;
    }

    free(naxes_out);

    DEBUG_TRACE_FEXIT();

    return (ID_out);
}

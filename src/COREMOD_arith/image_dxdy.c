/**
 * @file    image_dxdy.c
 * @brief   spatial derivatives
 *
 *
 */
#include <assert.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"



imageID arith_image_dx(
    const char *ID_name,
    const char *IDout_name
)
{
    imageID ID;
    imageID IDout;
    uint32_t *naxes = NULL;
    uint8_t naxis;
    uint8_t datatype;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;
    naxis = data.image[ID].md[0].naxis;
    if(naxis != 2)
    {
        PRINT_ERROR("Function only supports 2-D images\n");
        abort();
    }

    assert(naxis != 0);
    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc error. size %d", (int) naxis);
        abort();
    }

    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    create_image_ID(IDout_name, naxis, naxes, datatype, data.SHARED_DFT,
                            data.NBKEYWORD_DFT, 0, &IDout);
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for(uint32_t ii = 1; ii < naxes[0] - 1; ii++)
            data.image[IDout].array.F[jj * naxes[0] + ii] =
                (data.image[ID].array.F[jj * naxes[0] + ii + 1] - data.image[ID].array.F[jj *
                        naxes[0] + ii - 1]) / 2.0;
        data.image[IDout].array.F[jj * naxes[0]] =
            data.image[ID].array.F[jj * naxes[0] + 1] - data.image[ID].array.F[jj *
                    naxes[0]];
        data.image[IDout].array.F[jj * naxes[0] + naxes[0] - 1] =
            data.image[ID].array.F[jj * naxes[0] + naxes[0] - 1] - data.image[ID].array.F[jj
                    * naxes[0] + naxes[0] - 2];
    }

    free(naxes);

    return IDout;
}




imageID arith_image_dy(
    const char *ID_name,
    const char *IDout_name
)
{
    imageID ID;
    imageID IDout;
    uint32_t *naxes = NULL;
    uint8_t naxis;
    uint8_t datatype;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;
    naxis = data.image[ID].md[0].naxis;
    if(naxis != 2)
    {
        PRINT_ERROR("Function only supports 2-D images\n");
        abort();
    }

    assert(naxis != 0);
    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc error. size %d", (int) naxis);
        abort();
    }

    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    create_image_ID(IDout_name, naxis, naxes, datatype, data.SHARED_DFT,
                            data.NBKEYWORD_DFT, 0, &IDout);
    for(uint32_t ii = 0; ii < naxes[0]; ii++)
    {
        for(uint32_t jj = 1; jj < naxes[1] - 1; jj++)
        {
            data.image[IDout].array.F[jj * naxes[0] + ii] = (data.image[ID].array.F[(jj + 1)
                    * naxes[0] + ii] - data.image[ID].array.F[(jj - 1) * naxes[0] + ii]) / 2.0;
        }

        data.image[IDout].array.F[ii] = data.image[ID].array.F[1 * naxes[0] + ii] -
                                        data.image[ID].array.F[ii];

        data.image[IDout].array.F[(naxes[1] - 1)*naxes[0] + ii] =
            data.image[ID].array.F[(naxes[1] - 1) * naxes[0] + ii] -
            data.image[ID].array.F[(naxes[1] - 2) * naxes[0] + ii];

    }

    free(naxes);

    return IDout;
}

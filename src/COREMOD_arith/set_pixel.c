/**
 * @file    set_pixel.c
 * @brief   set single pixel value
 *
 *
 */


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

imageID arith_set_pixel(
    const char *ID_name,
    double      value,
    long        x,
    long        y
)
{
    imageID  ID;
    uint32_t naxes[2];
    uint8_t  datatype;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    //  printf("Setting pixel %ld %ld of image %s [%ld] to %f\n", x, y, ID_name, ID, (float) value);

    data.image[ID].md[0].write = 1;
    if(datatype == _DATATYPE_FLOAT)
    {
        data.image[ID].array.F[y * naxes[0] + x] = (float) value;
        //    printf("float -> %f\n", data.image[ID].array.F[y*naxes[0]+x]);
    }
    else if(datatype == _DATATYPE_DOUBLE)
    {
        data.image[ID].array.D[y * naxes[0] + x] = value;
    }
    else if(datatype == _DATATYPE_UINT8)
    {
        data.image[ID].array.UI8[y * naxes[0] + x] = (uint8_t) value;
    }
    else if(datatype == _DATATYPE_UINT16)
    {
        data.image[ID].array.UI16[y * naxes[0] + x] = (uint16_t) value;
    }
    else if(datatype == _DATATYPE_UINT32)
    {
        data.image[ID].array.UI32[y * naxes[0] + x] = (uint32_t) value;
    }
    else if(datatype == _DATATYPE_UINT64)
    {
        data.image[ID].array.UI64[y * naxes[0] + x] = (uint64_t) value;
    }
    else if(datatype == _DATATYPE_INT8)
    {
        data.image[ID].array.SI8[y * naxes[0] + x] = (int8_t) value;
    }
    else if(datatype == _DATATYPE_INT16)
    {
        data.image[ID].array.SI16[y * naxes[0] + x] = (int16_t) value;
    }
    else if(datatype == _DATATYPE_INT32)
    {
        data.image[ID].array.SI32[y * naxes[0] + x] = (int32_t) value;
    }
    else if(datatype == _DATATYPE_INT64)
    {
        data.image[ID].array.SI64[y * naxes[0] + x] = (int64_t) value;
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        exit(EXIT_FAILURE);
    }
    data.image[ID].md[0].write = 0;
    data.image[ID].md[0].cnt0++;
    COREMOD_MEMORY_image_set_sempost(ID_name, -1);

    return ID;
}



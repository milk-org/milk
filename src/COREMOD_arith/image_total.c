/**
 * @file    image_total.c
 * @brief   sum image pixels
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

double arith_image_total(const char *ID_name)
{
    long double lvalue; // uses long double internally
    imageID ID;
    uint64_t nelement;
    uint8_t datatype;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    nelement = data.image[ID].md[0].nelement;

    lvalue = 0.0;

    if (datatype == _DATATYPE_FLOAT)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.F[ii];
        }
    }
    else if (datatype == _DATATYPE_DOUBLE)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.D[ii];
        }
    }
    else if (datatype == _DATATYPE_UINT8)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.UI8[ii];
        }
    }
    else if (datatype == _DATATYPE_UINT16)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.UI16[ii];
        }
    }
    else if (datatype == _DATATYPE_UINT32)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.UI32[ii];
        }
    }
    else if (datatype == _DATATYPE_UINT64)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.UI64[ii];
        }
    }
    else if (datatype == _DATATYPE_INT8)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.SI8[ii];
        }
    }
    else if (datatype == _DATATYPE_INT16)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.SI16[ii];
        }
    }
    else if (datatype == _DATATYPE_INT32)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.SI32[ii];
        }
    }
    else if (datatype == _DATATYPE_INT64)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)data.image[ID].array.SI64[ii];
        }
    }
    else
    {
        PRINT_ERROR("invalid data type");
        exit(0);
    }

    double value;
    value = (double)lvalue;

    return (value);
}

double arith_image_sumsquare(const char *ID_name)
{
    long double lvalue; // uses long double internally
    imageID ID;
    uint64_t nelement;
    uint8_t datatype;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    nelement = data.image[ID].md[0].nelement;

    lvalue = 0.0;

    if (datatype == _DATATYPE_FLOAT)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.F[ii] * data.image[ID].array.F[ii]);
        }
    }
    else if (datatype == _DATATYPE_DOUBLE)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.D[ii] * data.image[ID].array.D[ii]);
        }
    }
    else if (datatype == _DATATYPE_UINT8)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.UI8[ii] * data.image[ID].array.UI8[ii]);
        }
    }
    else if (datatype == _DATATYPE_UINT16)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.UI16[ii] * data.image[ID].array.UI16[ii]);
        }
    }
    else if (datatype == _DATATYPE_UINT32)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.UI32[ii] * data.image[ID].array.UI32[ii]);
        }
    }
    else if (datatype == _DATATYPE_UINT64)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.UI64[ii] * data.image[ID].array.UI64[ii]);
        }
    }
    else if (datatype == _DATATYPE_INT8)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.SI8[ii] * data.image[ID].array.SI8[ii]);
        }
    }
    else if (datatype == _DATATYPE_INT16)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.SI16[ii] * data.image[ID].array.SI16[ii]);
        }
    }
    else if (datatype == _DATATYPE_INT32)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.SI32[ii] * data.image[ID].array.SI32[ii]);
        }
    }
    else if (datatype == _DATATYPE_INT64)
    {
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(data.image[ID].array.SI64[ii] * data.image[ID].array.SI64[ii]);
        }
    }
    else
    {
        PRINT_ERROR("invalid data type");
        exit(0);
    }

    double value;
    value = (double)lvalue;

    return (value);
}

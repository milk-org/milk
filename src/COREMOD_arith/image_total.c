/**
 * @file    image_total.c
 * @brief   sum image pixels
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_total.h"

#include "COREMOD_memory/COREMOD_memory.h"

double arith_image_total_IMGID(IMGID *imgin)
{
    long double lvalue; // uses long double internally
    uint64_t    nelement;
    uint8_t     datatype;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype = imgin->md[0].datatype;

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

    if(datatype == _DATATYPE_FLOAT)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.F[ii];
        }
    }
    else if(datatype == _DATATYPE_DOUBLE)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.D[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT8)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.UI8[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT16)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.UI16[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT32)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.UI32[ii];
        }
    }
    else if(datatype == _DATATYPE_UINT64)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.UI64[ii];
        }
    }
    else if(datatype == _DATATYPE_INT8)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.SI8[ii];
        }
    }
    else if(datatype == _DATATYPE_INT16)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.SI16[ii];
        }
    }
    else if(datatype == _DATATYPE_INT32)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.SI32[ii];
        }
    }
    else if(datatype == _DATATYPE_INT64)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double) imgin->im->array.SI64[ii];
        }
    }
    else
    {
        PRINT_ERROR("invalid data type");
        exit(0);
    }

    double value;
    value = (double) lvalue;

    return (value);
}

double arith_image_total(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_total_IMGID(&imgin);
}

double arith_image_sumsquare_IMGID(IMGID *imgin)
{
    long double lvalue; // uses long double internally
    uint64_t    nelement;
    uint8_t     datatype;

    resolveIMGID(imgin, ERRMODE_ABORT);
    datatype = imgin->md[0].datatype;

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

    if(datatype == _DATATYPE_FLOAT)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.F[ii] *
                                    imgin->im->array.F[ii]);
        }
    }
    else if(datatype == _DATATYPE_DOUBLE)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.D[ii] *
                                    imgin->im->array.D[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT8)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.UI8[ii] *
                                    imgin->im->array.UI8[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT16)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.UI16[ii] *
                                    imgin->im->array.UI16[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT32)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.UI32[ii] *
                                    imgin->im->array.UI32[ii]);
        }
    }
    else if(datatype == _DATATYPE_UINT64)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.UI64[ii] *
                                    imgin->im->array.UI64[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT8)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.SI8[ii] *
                                    imgin->im->array.SI8[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT16)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.SI16[ii] *
                                    imgin->im->array.SI16[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT32)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.SI32[ii] *
                                    imgin->im->array.SI32[ii]);
        }
    }
    else if(datatype == _DATATYPE_INT64)
    {
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            lvalue += (long double)(imgin->im->array.SI64[ii] *
                                    imgin->im->array.SI64[ii]);
        }
    }
    else
    {
        PRINT_ERROR("invalid data type");
        exit(0);
    }

    double value;
    value = (double) lvalue;

    return (value);
}

double arith_image_sumsquare(const char *ID_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    return arith_image_sumsquare_IMGID(&imgin);
}

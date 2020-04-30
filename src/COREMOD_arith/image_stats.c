/**
 * @file    image_stats.c
 * @brief   simple stats functions
 *
 *
 */


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_tools/COREMOD_tools.h"

#include "image_total.h"



double arith_image_mean(
    const char *ID_name
)
{
    double value;
    imageID ID;

    ID = image_ID(ID_name);

    value = (double)(arith_image_total(ID_name) / data.image[ID].md[0].nelement);

    return(value);
}




double arith_image_min(
    const char *ID_name
)
{
    imageID ID;
    uint64_t nelement;
    uint8_t datatype;
    int OK = 0;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    nelement = data.image[ID].md[0].nelement;


    if(datatype == _DATATYPE_FLOAT)
    {
        float value;

        value = data.image[ID].array.F[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            float value1 = data.image[ID].array.F[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_DOUBLE)
    {
        double value;

        value = data.image[ID].array.D[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            double value1 = data.image[ID].array.D[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return (value);
    }

    if(datatype == _DATATYPE_UINT8)
    {
        uint8_t value;

        value = data.image[ID].array.UI8[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            uint8_t value1 = data.image[ID].array.UI8[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_UINT16)
    {
        uint16_t value;

        value = data.image[ID].array.UI16[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            uint16_t value1 = data.image[ID].array.UI16[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_UINT32)
    {
        uint32_t value;

        value = data.image[ID].array.UI32[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            uint32_t value1 = data.image[ID].array.UI32[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_UINT64)
    {
        uint64_t value;

        value = data.image[ID].array.UI64[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            uint64_t value1 = data.image[ID].array.UI64[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT8)
    {
        int8_t value;

        value = data.image[ID].array.SI8[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            int8_t value1 = data.image[ID].array.SI8[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT16)
    {
        int16_t value;

        value = (double) data.image[ID].array.SI16[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            int16_t value1 = data.image[ID].array.SI16[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT32)
    {
        int32_t value;

        value = data.image[ID].array.SI32[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            int32_t value1 = data.image[ID].array.SI32[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT64)
    {
        int64_t value;

        value = data.image[ID].array.SI64[0];
        for(uint64_t ii = 0; ii < nelement; ii++)
        {
            int64_t value1 = data.image[ID].array.SI64[ii];
            if(value1 < value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(OK == 0)
    {
        printf("Error : Invalid data format for arith_image_min\n");
    }

    return(0);
}





double arith_image_max(
    const char *ID_name
)
{
    imageID ID;
    long    ii;
    long    nelement;
    uint8_t datatype;
    int     OK = 0;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    nelement = data.image[ID].md[0].nelement;

    if(datatype == _DATATYPE_FLOAT)
    {
        float value, value1;

        value = data.image[ID].array.F[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.F[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_DOUBLE)
    {
        double value, value1;

        value = data.image[ID].array.D[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.D[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return (value);
    }

    if(datatype == _DATATYPE_UINT8)
    {
        uint8_t value, value1;

        value = data.image[ID].array.UI8[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.UI8[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_UINT16)
    {
        uint16_t value, value1;

        value = data.image[ID].array.UI16[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.UI16[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_UINT32)
    {
        uint32_t value, value1;

        value = data.image[ID].array.UI32[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.UI32[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_UINT64)
    {
        uint64_t value, value1;

        value = data.image[ID].array.UI64[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.UI64[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT8)
    {
        int8_t value, value1;

        value = data.image[ID].array.SI8[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.SI8[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT16)
    {
        int16_t value, value1;

        value = (double) data.image[ID].array.SI16[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.SI16[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT32)
    {
        int32_t value, value1;

        value = data.image[ID].array.SI32[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.SI32[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }

    if(datatype == _DATATYPE_INT64)
    {
        int64_t value, value1;

        value = data.image[ID].array.SI64[0];
        for(ii = 0; ii < nelement; ii++)
        {
            value1 = data.image[ID].array.SI64[ii];
            if(value1 > value)
            {
                value = value1;
            }
        }
        OK = 1;
        return ((double) value);
    }
    if(OK == 0)
    {
        printf("Error : Invalid data format for arith_image_max\n");
    }

    return(0);
}





double arith_image_percentile(
    const char *ID_name,
    double      fraction
)
{
    imageID  ID;
    long     ii;
    double   value = 0;
    long    *arrayL = NULL;
    float   *arrayF = NULL;
    double  *arrayD = NULL;
    unsigned short *arrayU = NULL;
    long     nelement;
    uint8_t  datatype;
    int      atypeOK = 1;

    ID = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    nelement = data.image[ID].md[0].nelement;


    switch(datatype)
    {

        case _DATATYPE_FLOAT :
            arrayF = (float *) malloc(sizeof(float) * nelement);
            if(arrayF == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            memcpy(arrayF, data.image[ID].array.F, sizeof(float)*nelement);
            quick_sort_float(arrayF, nelement);
            value = (double) arrayF[(long)(fraction * nelement)];
            free(arrayF);
            break;

        case _DATATYPE_DOUBLE :
            arrayD = (double *) malloc(sizeof(double) * nelement);
            if(arrayD == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            memcpy(arrayD, data.image[ID].array.D, sizeof(double)*nelement);
            quick_sort_double(arrayD, nelement);
            value = arrayD[(long)(fraction * nelement)];
            free(arrayD);
            break;



        case _DATATYPE_UINT8 :
            arrayU = (unsigned short *) malloc(sizeof(unsigned short) * nelement);
            if(arrayU == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayU[ii] = data.image[ID].array.UI8[ii];
            }
            quick_sort_ushort(arrayU, nelement);
            value = arrayU[(long)(fraction * nelement)];
            free(arrayU);
            break;

        case _DATATYPE_UINT16 :
            arrayU = (unsigned short *) malloc(sizeof(unsigned short) * nelement);
            if(arrayU == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayU[ii] = data.image[ID].array.UI16[ii];
            }
            quick_sort_ushort(arrayU, nelement);
            value = arrayU[(long)(fraction * nelement)];
            free(arrayU);
            break;

        case _DATATYPE_UINT32 :
            arrayL = (long *) malloc(sizeof(long) * nelement);
            if(arrayU == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayL[ii] = data.image[ID].array.UI32[ii];
            }
            quick_sort_long(arrayL, nelement);
            value = arrayL[(long)(fraction * nelement)];
            free(arrayL);
            break;

        case _DATATYPE_UINT64 :
            arrayL = (long *) malloc(sizeof(long) * nelement);
            if(arrayU == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayL[ii] = data.image[ID].array.UI64[ii];
            }
            quick_sort_long(arrayL, nelement);
            value = arrayL[(long)(fraction * nelement)];
            free(arrayL);
            break;


        case _DATATYPE_INT8 :
            arrayL = (long *) malloc(sizeof(long) * nelement);
            if(arrayL == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayL[ii] = (long) data.image[ID].array.SI8[ii];
            }
            quick_sort_long(arrayL, nelement);
            value = (double) arrayL[(long)(fraction * nelement)];
            free(arrayL);
            break;

        case _DATATYPE_INT16 :
            arrayL = (long *) malloc(sizeof(long) * nelement);
            if(arrayL == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayL[ii] = (long) data.image[ID].array.SI16[ii];
            }
            quick_sort_long(arrayL, nelement);
            value = (double) arrayL[(long)(fraction * nelement)];
            free(arrayL);
            break;

        case _DATATYPE_INT32 :
            arrayL = (long *) malloc(sizeof(long) * nelement);
            if(arrayL == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayL[ii] = (long) data.image[ID].array.SI32[ii];
            }
            quick_sort_long(arrayL, nelement);
            value = (double) arrayL[(long)(fraction * nelement)];
            free(arrayL);
            break;

        case _DATATYPE_INT64 :
            arrayL = (long *) malloc(sizeof(long) * nelement);
            if(arrayL == NULL)
            {
                PRINT_ERROR("malloc() error");
                exit(EXIT_FAILURE);
            }
            for(ii = 0; ii < nelement; ii++)
            {
                arrayL[ii] = (long) data.image[ID].array.SI64[ii];
            }
            quick_sort_long(arrayL, nelement);
            value = (double) arrayL[(long)(fraction * nelement)];
            free(arrayL);
            break;

        default:
            PRINT_ERROR("Image type not supported");
            atypeOK = 0;
            break;
    }

    if(atypeOK == 0)
    {
        exit(EXIT_FAILURE);
    }

    return(value);
}


double arith_image_median(
    const char *ID_name
)
{
    double value = 0.0;

    value = arith_image_percentile(ID_name, 0.5);

    return(value);
}

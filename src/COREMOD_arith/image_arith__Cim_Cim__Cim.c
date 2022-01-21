/**
 * @file    image_arith__Cim_Cim__Cim.c
 * @brief   arith functions
 *
 * input : complex image, complex image
 * output: complex image
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"

errno_t
arith_image_Cadd(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    uint8_t atype1, atype2;
    imageID ID1;
    imageID ID2;

    ID1    = image_ID(ID1_name);
    ID2    = image_ID(ID2_name);
    atype1 = data.image[ID1].md[0].datatype;
    atype2 = data.image[ID2].md[0].datatype;

    if ((atype1 == _DATATYPE_COMPLEX_FLOAT) &&
        (atype2 == _DATATYPE_COMPLEX_FLOAT))
        {
            arith_image_function_CF_CF__CF(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPadd_CF_CF);
            return RETURN_SUCCESS;
        }

    if ((atype1 == _DATATYPE_COMPLEX_DOUBLE) &&
        (atype2 == _DATATYPE_COMPLEX_DOUBLE))
        {
            arith_image_function_CD_CD__CD(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPadd_CD_CD);
            return RETURN_SUCCESS;
        }
    PRINT_ERROR("data types do not match");

    return RETURN_FAILURE;
}

errno_t
arith_image_Csub(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    uint8_t datatype1, datatype2;
    imageID ID1;
    imageID ID2;

    ID1       = image_ID(ID1_name);
    ID2       = image_ID(ID2_name);
    datatype1 = data.image[ID1].md[0].datatype;
    datatype2 = data.image[ID2].md[0].datatype;

    if ((datatype1 == _DATATYPE_COMPLEX_FLOAT) &&
        (datatype2 == _DATATYPE_COMPLEX_FLOAT))
        {
            arith_image_function_CF_CF__CF(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPsub_CF_CF);
            return RETURN_SUCCESS;
        }

    if ((datatype1 == _DATATYPE_COMPLEX_DOUBLE) &&
        (datatype2 == _DATATYPE_COMPLEX_DOUBLE))
        {
            arith_image_function_CD_CD__CD(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPsub_CD_CD);
            return RETURN_SUCCESS;
        }
    PRINT_ERROR("data types do not match");

    return RETURN_FAILURE;
}

errno_t arith_image_Cmult(const char *ID1_name,
                          const char *ID2_name,
                          const char *ID_out)
{
    uint8_t datatype1, datatype2;
    imageID ID1;
    imageID ID2;

    ID1       = image_ID(ID1_name);
    ID2       = image_ID(ID2_name);
    datatype1 = data.image[ID1].md[0].datatype;
    datatype2 = data.image[ID2].md[0].datatype;

    if ((datatype1 == _DATATYPE_COMPLEX_FLOAT) &&
        (datatype2 == _DATATYPE_COMPLEX_FLOAT))
        {
            arith_image_function_CF_CF__CF(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPmult_CF_CF);
            return RETURN_SUCCESS;
        }

    if ((datatype1 == _DATATYPE_COMPLEX_DOUBLE) &&
        (datatype2 == _DATATYPE_COMPLEX_DOUBLE))
        {
            arith_image_function_CD_CD__CD(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPmult_CD_CD);
            return RETURN_SUCCESS;
        }
    PRINT_ERROR("data types do not match");

    return RETURN_FAILURE;
}

int arith_image_Cdiv(const char *ID1_name,
                     const char *ID2_name,
                     const char *ID_out)
{
    uint8_t datatype1, datatype2;
    imageID ID1;
    imageID ID2;

    ID1       = image_ID(ID1_name);
    ID2       = image_ID(ID2_name);
    datatype1 = data.image[ID1].md[0].datatype;
    datatype2 = data.image[ID2].md[0].datatype;

    if ((datatype1 == _DATATYPE_COMPLEX_FLOAT) &&
        (datatype2 == _DATATYPE_COMPLEX_FLOAT))
        {
            arith_image_function_CF_CF__CF(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPdiv_CF_CF);
            return RETURN_SUCCESS;
        }

    if ((datatype1 == _DATATYPE_COMPLEX_DOUBLE) &&
        (datatype2 == _DATATYPE_COMPLEX_DOUBLE))
        {
            arith_image_function_CD_CD__CD(ID1_name,
                                           ID2_name,
                                           ID_out,
                                           &CPdiv_CD_CD);
            return RETURN_SUCCESS;
        }
    PRINT_ERROR("data types do not match");

    return RETURN_FAILURE;
}

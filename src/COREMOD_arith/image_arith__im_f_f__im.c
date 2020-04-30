/**
 * @file    image_arith__im_f_f__im.c
 * @brief   arith functions
 * 
 * input : image, float, float
 * output: image
 *
 */


#include <math.h>


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"






int arith_image_trunc(const char *ID_name, double f1, double f2,
                      const char *ID_out)
{
    arith_image_function_1ff_1(ID_name, f1, f2, ID_out, &Ptrunc);
    return(0);
}

int arith_image_trunc_inplace(const char *ID_name, double f1, double f2)
{
    arith_image_function_1ff_1_inplace(ID_name, f1, f2, &Ptrunc);
    return(0);
}
int arith_image_trunc_inplace_byID(long ID, double f1, double f2)
{
    arith_image_function_1ff_1_inplace_byID(ID, f1, f2, &Ptrunc);
    return(0);
}




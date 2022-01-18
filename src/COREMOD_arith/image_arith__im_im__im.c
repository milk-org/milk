/**
 * @file    image_arith__im_im__im.c
 * @brief   arith functions
 *
 * input : image, image
 * output: image
 *
 */

#include <math.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "CommandLineInterface/CLIcore.h"

#include "imfunctions.h"
#include "mathfuncs.h"

int arith_image_fmod(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Pfmod);
    return (0);
}

int arith_image_pow(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Ppow);
    return (0);
}

int arith_image_add(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Padd);
    return (0);
}

int arith_image_sub(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Psub);
    return (0);
}

int arith_image_mult(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Pmult);
    return (0);
}

int arith_image_div(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Pdiv);
    return (0);
}

int arith_image_minv(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Pminv);
    return (0);
}

int arith_image_maxv(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Pmaxv);
    return (0);
}

int arith_image_testlt(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Ptestlt);
    return (0);
}

int arith_image_testmt(const char *ID1_name, const char *ID2_name, const char *ID_out)
{
    arith_image_function_2_1(ID1_name, ID2_name, ID_out, &Ptestmt);
    return (0);
}

int arith_image_fmod_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pfmod);
    return (0);
}

int arith_image_pow_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Ppow);
    return (0);
}

int arith_image_add_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Padd);
    return (0);
}

int arith_image_sub_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Psub);
    return (0);
}

int arith_image_mult_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pmult);
    return (0);
}

int arith_image_div_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pdiv);
    return (0);
}

int arith_image_minv_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pminv);
    return (0);
}

int arith_image_maxv_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Pmaxv);
    return (0);
}

int arith_image_testlt_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Ptestlt);
    return (0);
}

int arith_image_testmt_inplace(const char *ID1_name, const char *ID2_name)
{
    arith_image_function_2_1_inplace(ID1_name, ID2_name, &Ptestmt);
    return (0);
}

int arith_image_fmod_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pfmod);
    return (0);
}

int arith_image_pow_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Ppow);
    return (0);
}

int arith_image_add_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Padd);
    return (0);
}

int arith_image_sub_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Psub);
    return (0);
}

int arith_image_mult_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pmult);
    return (0);
}
int arith_image_div_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pdiv);
    return (0);
}

int arith_image_minv_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pminv);
    return (0);
}

int arith_image_maxv_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Pmaxv);
    return (0);
}

int arith_image_testlt_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Ptestlt);
    return (0);
}

int arith_image_testmt_inplace_byID(long ID1, long ID2)
{
    arith_image_function_2_1_inplace_byID(ID1, ID2, &Ptestmt);
    return (0);
}

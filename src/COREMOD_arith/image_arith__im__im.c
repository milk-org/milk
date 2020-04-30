/**
 * @file    image_arith__im__im.c
 * @brief   arith functions
 * 
 * input : image
 * output: image
 *
 */


#include <math.h>


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"






int arith_image_acos_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pacos);
    return(0);
}


int arith_image_asin_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pasin);
    return(0);
}


int arith_image_atan_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Patan);
    return(0);
}


int arith_image_ceil_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pceil);
    return(0);
}


int arith_image_cos_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pcos);
    return(0);
}


int arith_image_cosh_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pcosh);
    return(0);
}


int arith_image_exp_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pexp);
    return(0);
}


int arith_image_fabs_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pfabs);
    return(0);
}


int arith_image_floor_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pfloor);
    return(0);
}


int arith_image_ln_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Pln);
    return(0);
}


int arith_image_log_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Plog);
    return(0);
}


int arith_image_sqrt_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Psqrt);
    return(0);
}


int arith_image_sin_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Psin);
    return(0);
}


int arith_image_sinh_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Psinh);
    return(0);
}

int arith_image_tan_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Ptan);
    return(0);
}

int arith_image_tanh_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Ptanh);
    return(0);
}

int arith_image_positive_byID(long ID, long IDout)
{
    arith_image_function_1_1_byID(ID, IDout, &Ppositive);
    return(0);
}




int arith_image_acos(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pacos);
    return(0);
}

int arith_image_asin(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pasin);
    return(0);
}

int arith_image_atan(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Patan);
    return(0);
}

int arith_image_ceil(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pceil);
    return(0);
}

int arith_image_cos(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pcos);
    return(0);
}

int arith_image_cosh(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pcosh);
    return(0);
}

int arith_image_exp(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pexp);
    return(0);
}

int arith_image_fabs(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pfabs);
    return(0);
}

int arith_image_floor(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pfloor);
    return(0);
}

int arith_image_ln(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Pln);
    return(0);
}

int arith_image_log(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Plog);
    return(0);
}

int arith_image_sqrt(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Psqrt);
    return(0);
}

int arith_image_sin(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Psin);
    return(0);
}

int arith_image_sinh(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Psinh);
    return(0);
}

int arith_image_tan(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Ptan);
    return(0);
}

int arith_image_tanh(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Ptanh);
    return(0);
}

int arith_image_positive(const char *ID_name, const char *ID_out)
{
    arith_image_function_1_1(ID_name, ID_out, &Ppositive);
    return(0);
}








int arith_image_acos_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pacos);
    return(0);
}
int arith_image_asin_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pasin);
    return(0);
}
int arith_image_atan_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Patan);
    return(0);
}
int arith_image_ceil_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pceil);
    return(0);
}
int arith_image_cos_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pcos);
    return(0);
}
int arith_image_cosh_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pcosh);
    return(0);
}
int arith_image_exp_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pexp);
    return(0);
}
int arith_image_fabs_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pfabs);
    return(0);
}
int arith_image_floor_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pfloor);
    return(0);
}
int arith_image_ln_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Pln);
    return(0);
}
int arith_image_log_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Plog);
    return(0);
}
int arith_image_sqrt_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Psqrt);
    return(0);
}
int arith_image_sin_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Psin);
    return(0);
}
int arith_image_sinh_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Psinh);
    return(0);
}
int arith_image_tan_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Ptan);
    return(0);
}
int arith_image_tanh_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Ptanh);
    return(0);
}
int arith_image_positive_inplace_byID(long ID)
{
    arith_image_function_1_1_inplace_byID(ID, &Ppositive);
    return(0);
}



int arith_image_acos_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pacos);
    return(0);
}
int arith_image_asin_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pasin);
    return(0);
}
int arith_image_atan_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Patan);
    return(0);
}
int arith_image_ceil_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pceil);
    return(0);
}
int arith_image_cos_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pcos);
    return(0);
}
int arith_image_cosh_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pcosh);
    return(0);
}
int arith_image_exp_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pexp);
    return(0);
}
int arith_image_fabs_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pfabs);
    return(0);
}
int arith_image_floor_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pfloor);
    return(0);
}
int arith_image_ln_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Pln);
    return(0);
}
int arith_image_log_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Plog);
    return(0);
}
int arith_image_sqrt_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Psqrt);
    return(0);
}
int arith_image_sin_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Psin);
    return(0);
}
int arith_image_sinh_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Psinh);
    return(0);
}
int arith_image_tan_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Ptan);
    return(0);
}
int arith_image_tanh_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Ptanh);
    return(0);
}
int arith_image_positive_inplace(const char *ID_name)
{
    arith_image_function_1_1_inplace(ID_name, &Ppositive);
    return(0);
}





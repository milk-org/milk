/**
 * @file    image_arith__im_f__im.c
 * @brief   arith functions
 *
 * input : image, float
 * output: image
 *
 */

#include <math.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "CommandLineInterface/CLIcore.h"

#include "imfunctions.h"
#include "mathfuncs.h"

int arith_image_cstfmod(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Pfmod);
    return (0);
}

int arith_image_cstadd(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Padd);
    return (0);
}

int arith_image_cstsub(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Psub);
    return (0);
}

int arith_image_cstsubm(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Psubm);
    return (0);
}

int arith_image_cstmult(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Pmult);
    return (0);
}

int arith_image_cstdiv(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Pdiv);
    return (0);
}

int arith_image_cstdiv1(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Pdiv1);
    return (0);
}

int arith_image_cstpow(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Ppow);
    return (0);
}

int arith_image_cstmaxv(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Pmaxv);
    return (0);
}

int arith_image_cstminv(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Pminv);
    return (0);
}

int arith_image_csttestlt(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Ptestlt);
    return (0);
}

int arith_image_csttestmt(const char *ID_name, double f1, const char *ID_out)
{
    arith_image_function_1f_1(ID_name, f1, ID_out, &Ptestmt);
    return (0);
}

int arith_image_cstfmod_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pfmod);
    return (0);
}

int arith_image_cstadd_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Padd);
    return (0);
}

int arith_image_cstsub_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Psub);
    return (0);
}

int arith_image_cstmult_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pmult);
    return (0);
}

int arith_image_cstdiv_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pdiv);
    return (0);
}

int arith_image_cstdiv1_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pdiv1);
    return (0);
}

int arith_image_cstpow_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Ppow);
    return (0);
}

int arith_image_cstmaxv_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pmaxv);
    return (0);
}

int arith_image_cstminv_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Pminv);
    return (0);
}

int arith_image_csttestlt_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Ptestlt);
    return (0);
}

int arith_image_csttestmt_inplace(const char *ID_name, double f1)
{
    arith_image_function_1f_1_inplace(ID_name, f1, &Ptestmt);
    return (0);
}

int arith_image_cstfmod_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pfmod);
    return (0);
}

int arith_image_cstadd_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Padd);
    return (0);
}

int arith_image_cstsub_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Psub);
    return (0);
}

int arith_image_cstmult_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pmult);
    return (0);
}

int arith_image_cstdiv_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pdiv);
    return (0);
}

int arith_image_cstdiv1_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pdiv1);
    return (0);
}

int arith_image_cstpow_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Ppow);
    return (0);
}

int arith_image_cstmaxv_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pmaxv);
    return (0);
}

int arith_image_cstminv_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Pminv);
    return (0);
}

int arith_image_csttestlt_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Ptestlt);
    return (0);
}

int arith_image_csttestmt_inplace_byID(long ID, double f1)
{
    arith_image_function_1f_1_inplace_byID(ID, f1, &Ptestmt);
    return (0);
}

/**
 * @file    image_arith__im_f_f__im.c
 * @brief   arith functions
 *
 * input : image, float, float
 * output: image
 *
 */

#include <math.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "CommandLineInterface/CLIcore.h"

#include "imfunctions.h"
#include "mathfuncs.h"

// ==========================================
// Forward declaration(s)
// ==========================================

int arith_image_trunc(const char *ID_name, double f1, double f2, const char *ID_out);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t arith_image_trunc_cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_FLOAT) + CLI_checkarg(3, CLIARG_FLOAT) +
            CLI_checkarg(4, CLIARG_STR_NOT_IMG) ==
        0)
    {
        arith_image_trunc(data.cmdargtoken[1].val.string, data.cmdargtoken[2].val.numf, data.cmdargtoken[3].val.numf,
                          data.cmdargtoken[4].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t image_arith__im_f_f__im_addCLIcmd()
{

    RegisterCLIcommand("imtrunc", __FILE__, arith_image_trunc_cli, "trucate pixel values",
                       "<input image> <min> <max> <output image>", "imtrunc im 0.0 1.0 out",
                       "arith_image_trunc(const char *ID_name, double f1, double f2, const char *ID_out)");

    return RETURN_SUCCESS;
}

int arith_image_trunc(const char *ID_name, double f1, double f2, const char *ID_out)
{
    arith_image_function_1ff_1(ID_name, f1, f2, ID_out, &Ptrunc);
    return (0);
}

int arith_image_trunc_inplace(const char *ID_name, double f1, double f2)
{
    arith_image_function_1ff_1_inplace(ID_name, f1, f2, &Ptrunc);
    return (0);
}
int arith_image_trunc_inplace_byID(long ID, double f1, double f2)
{
    arith_image_function_1ff_1_inplace_byID(ID, f1, f2, &Ptrunc);
    return (0);
}

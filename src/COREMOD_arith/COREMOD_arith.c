/**
 * @file    COREMOD_arith.c
 * @brief   Arithmeric operations on images
 *
 * Addition, multiplication and much more
 *
 *
 */


/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT ""

// Module short description
#define MODULE_DESCRIPTION       "Image arithmetic operations"







/* ================================================================== */
/* ================================================================== */
/*            DEPENDANCIES                                            */
/* ================================================================== */
/* ================================================================== */




#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>

#include <fitsio.h>


#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
#endif


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"


//#include "COREMOD_arith/COREMOD_arith.h"


#include "set_pixel.h"
#include "image_crop.h"
#include "image_merge3D.h"
#include "image_total.h"
#include "image_stats.h"
#include "image_dxdy.h"
#include "imfunctions.h"

#include "mathfuncs.h"
#include "image_arith__im__im.h"
#include "image_arith__im_im__im.h"
#include "image_arith__im_f__im.h"
#include "image_arith__im_f_f__im.h"

#include "execute_arith.h"



/* ================================================================== */
/* ================================================================== */
/*           MACROS, DEFINES                                          */
/* ================================================================== */
/* ================================================================== */


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif







/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(COREMOD_arith)





/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */


// CLI commands
//
// function CLI_checkarg used to check arguments
// 1: float
// 2: long
// 3: string
// 4: existing image
//


static errno_t arith_image_extract2D_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_LONG)
            + CLI_checkarg(6, CLIARG_LONG)
            == 0)
    {
        arith_image_extract2D(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.numl,
            data.cmdargtoken[6].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t arith_image_extract3D_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_LONG)
            + CLI_checkarg(6, CLIARG_LONG)
            + CLI_checkarg(7, CLIARG_LONG)
            + CLI_checkarg(8, CLIARG_LONG)
            == 0)
    {
        arith_image_extract3D(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.numl,
            data.cmdargtoken[6].val.numl,
            data.cmdargtoken[7].val.numl,
            data.cmdargtoken[8].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}


static errno_t arith_set_pixel_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_FLOAT)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            == 0)
    {
        arith_set_pixel(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numf,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}


static errno_t arith_set_pixel_1Drange_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_FLOAT)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            == 0)
    {
        arith_set_pixel_1Drange(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numf,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}


static errno_t arith_set_row_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_FLOAT)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        arith_set_row(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numf,
            data.cmdargtoken[3].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}


static errno_t arith_set_col_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_FLOAT)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        arith_set_col(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numf,
            data.cmdargtoken[3].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}


static errno_t arith_image_zero_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            == 0)
    {
        arith_image_zero(data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}


static errno_t arith_image_trunc_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_FLOAT)
            + CLI_checkarg(3, CLIARG_FLOAT)
            + CLI_checkarg(4, CLIARG_STR_NOT_IMG)
            == 0)
    {
        arith_image_trunc(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numf,
            data.cmdargtoken[3].val.numf,
            data.cmdargtoken[4].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}


static errno_t arith_image_merge3D_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, CLIARG_STR_NOT_IMG)
            == 0)
    {
        arith_image_merge3D(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}






static errno_t init_module_CLI()
{

    RegisterCLIcommand(
        "extractim",
        __FILE__,
        arith_image_extract2D_cli,
        "crop 2D image",
        "<input image> <output image> <sizex> <sizey> <xstart> <ystart>",
        "extractim im ime 256 256 100 100",
        "int arith_image_extract2D(const char *in_name, const char *out_name, long size_x, long size_y, long xstart, long ystart)");



    RegisterCLIcommand(
        "extract3Dim",
        __FILE__,
        arith_image_extract3D_cli,
        "crop 3D image",
        "<input image> <output image> <sizex> <sizey> <sizez> <xstart> <ystart> <zstart>",
        "extractim im ime 256 256 5 100 100 0",
        "int arith_image_extract3D(const char *in_name, const char *out_name, long size_x, long size_y, long size_z, long xstart, long ystart, long zstart)");


    RegisterCLIcommand(
        "setpix",
        __FILE__,
        arith_set_pixel_cli,
        "set pixel value",
        "<input image> <value> <x> <y>",
        "setpix im 1.24 100 100",
        "int arith_set_pixel(const char *ID_name, double value, long x, long y)");


    RegisterCLIcommand(
        "setpix1Drange",
        __FILE__,
        arith_set_pixel_1Drange_cli,
        "set pixel value for 1D area",
        "<input image> <value> <first pix> <last pix>",
        "setpix im 1.24 10 200",
        "int arith_set_pixel_1Drange(const char *ID_name, double value, long x, long y)");


    RegisterCLIcommand(
        "setrow",
        __FILE__,
        arith_set_row_cli,
        "set pixel row value",
        "<input image> <value> <row>",
        "setrow im 1.24 100",
        "int arith_set_row(const char *ID_name, double value, long y)");


    RegisterCLIcommand(
        "setcol",
        __FILE__,
        arith_set_col_cli,
        "set pixel column value",
        "<input image> <value> <col>",
        "setcol im 1.24 100",
        "int arith_set_col(const char *ID_name, double value, long x)");


    RegisterCLIcommand(
        "imzero",
        __FILE__,
        arith_image_zero_cli,
        "set pixels to zero",
        "<input image>",
        "imzero im",
        "int arith_image_zero(const char *ID_name)");


    RegisterCLIcommand(
        "imtrunc",
        __FILE__,
        arith_image_trunc_cli,
        "trucate pixel values",
        "<input image> <min> <max> <output image>",
        "imtrunc im 0.0 1.0 out",
        "arith_image_trunc(const char *ID_name, double f1, double f2, const char *ID_out)");


    RegisterCLIcommand(
        "merge3d",
        __FILE__,
        arith_image_merge3D_cli,
        "merge two 3D cubes into one",
        "<input cube 1> <input cube 2> <output cube>",
        "merge3d imc1 imc2 imcout",
        "long arith_image_merge3D(const char *ID_name1, const char *ID_name2, const char *IDout_name)");



    // add atexit functions here

    return RETURN_SUCCESS;
}






errno_t init_COREMOD_arith()
{
    init_module_CLI();

    return RETURN_SUCCESS;
}


















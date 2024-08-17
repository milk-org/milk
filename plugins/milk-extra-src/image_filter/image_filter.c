/**
 * @file    image_filter.c
 * @brief   Image filtering
 *
 */


// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "imgfilt"

// Module short description
#define MODULE_DESCRIPTION "Image filtering"

#include "CommandLineInterface/CLIcore.h"

#include "fconvolve.h"
#include "gaussfilter.h"
#include "im2Dfilter_1pixbblurr.h"



// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(image_filter)



//long fconvolve(const char *ID_in, const char *ID_ke, const char *ID_out);

static errno_t init_module_CLI()
{
    gaussfilter_addCLIcmd();
    fconvolve_addCLIcmd();

    CLIADDCMD_image_filter__im2Dfilter_1pixblurr();

    // add atexit functions here

    return RETURN_SUCCESS;
}



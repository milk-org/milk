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

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "CLIcore.h"

#    include "fconvolve.h"
#    include "gaussfilter.h"
#    include "im2Dfilter_1pixbblurr.h"


// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//

static errno_t init_module_CLI()
{
    gaussfilter_addCLIcmd();
    CLIADDCMD_image_filter__fconvolve();

    CLIADDCMD_image_filter__im2Dfilter_1pixblurr();

    // add atexit functions here

    return RETURN_SUCCESS;
}

MILK_MODULE(image_filter, init_module_CLI, NULL);

#endif /* MILK_NO_CLI */

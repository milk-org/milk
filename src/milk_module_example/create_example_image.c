/**
 * @file    create_example_image.c
 * @brief   example : create image
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"

// required for create_2Dimage_ID
#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t
milk_module_example__create_image_with_value(const char *restrict imname,
        double value);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

/** @brief Example CLI function
 *
 * Command Line Interface (CLI) wrapper to function\n
 * This is referred to as a "CLI function", written to connect a command
 * on the CLI prompt to a function.\n
 * A CLI function will check arguments entered on the prompt, and pass
 * them to the function.
 *
 * Naming conventions:
 * - CLI function <modulename>__<function>__cli()
 * - Execution function <modulename>__<function>()
 *
 *
 *
 * ### Checking if arguments are valid
 *
 * Each argument is checked by calling CLI_checkarg(i, argtype), which
 * checks if argument number <i> conforms to type <argtype>.\n
 *
 * Types are defined in CLIcore.h. Common types are:
 * - CLIARG_FLOAT             floating point number
 * - CLIARG_LONG              integer (int or long)
 * - CLIARG_STR_NOT_IMG       string, not existing image
 * - CLIARG_IMG               existing image
 * - CLIARG_STR               string
 */
static errno_t milk_module_example__create_image_with_value__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(2, CLIARG_FLOAT) ==
            0)
    {
        // If arguments meet requirements, command is executed
        //
        milk_module_example__create_image_with_value(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numf);

        return CLICMD_SUCCESS;
    }
    else
    {
        // If arguments do not pass test, errror code returned
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t create_example_image_addCLIcmd()
{

    RegisterCLIcommand(
        "createim", // function call name from CLI
        __FILE__,   // this file, used to track where function comes from
        milk_module_example__create_image_with_value__cli, // function to call
        "creates image with specified value",              // short description
        "<image> <value>",                                 // arguments
        "createim im1 3.5",                                // example use
        "milk_module_example__create_image_with_value(char *imname, double "
        "value)"); // source code call

    return RETURN_SUCCESS;
}

// By convention, function name starts with <modulename>__
//
errno_t
milk_module_example__create_image_with_value(const char *restrict imname,
        double value)
{
    uint32_t xsize =
        128; // by convention, pixel index variables are uint32_t type
    uint32_t ysize = 256;

    // overall image size is, by convention, uint64_t type
    uint64_t xysize = xsize * ysize;

    // create 2D image
    // store image index in variable ID
    // by default, the image will be single precision floating point, so
    // it is accessed as array.F
    imageID ID = create_2Dimage_ID(imname, xsize, ysize);

    // set each pixel to value
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        data.image[ID].array.F[ii] = value;
    }

    return RETURN_SUCCESS;
}

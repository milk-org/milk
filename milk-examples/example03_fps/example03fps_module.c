/**
 * @file example03fps_module.c
 * @brief Combined Milk CLI module for writer03 and processor03.
 * 
 * This file serves as the main entry point for the shared object 'example03fps.so'.
 * It aggregates multiple FPS-enabled commands into a single loadable unit.
 */

#include "CLIcore.h"

/** @brief Default shortname for the module (used as command prefix in milk CLI). */
#define MODULE_SHORTNAME_DEFAULT "ex03"

/** @brief Human-readable description of the module. */
#define MODULE_DESCRIPTION "Example 03 FPS Writer and Processor"

/* Forward declarations of command registration functions defined in respective *_module.c files */
errno_t CLIADDCMD_writer03();
errno_t CLIADDCMD_processor03();

/**
 * @brief Module Initialization logic.
 * 
 * This function is called by the INIT_MODULE_LIB constructor. 
 * It registers all CLI commands contained within this shared object.
 */
static errno_t init_module_CLI() {
    CLIADDCMD_writer03();
    CLIADDCMD_processor03();
    return RETURN_SUCCESS;
}

/** 
 * @brief Boilerplate module constructor/destructor generation.
 * Registers 'example03fps' with the Milk framework.
 */
INIT_MODULE_LIB(example03fps)
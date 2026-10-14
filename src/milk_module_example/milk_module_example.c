// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_module_example.c
 * @brief   example milk module
 *
 * milk module source code template\n
 * Demonstates how to add modules to milk and connect functions.\n
 *
 *
 * To compile with module :
 * > cd _build
 * > cmake .. -DEXTRAMODULES="milk_module_example" -DINSTALLMAKEDEFAULT=ON
 *
 *
 * To load, type "mload milkmilk_module_example" in CLI\n
 *
 *
 *  Files
 *
 * - CMakeLists.txt         : cmake input file for module
 *
 * - milk_module_example.c  : module main C file, includes binding code to milk
 * - milk_module_example.h  : function prototypes to be included by other modules
 *
 *
 * Several examples are provided to demonstrate code features.
 * Each example builds on the previous one(s), demonstrating additional capabilities.
 *
 * ## Simple function example
 *
 * Simple function example. No FPS, no processinfo.
 *
 * Files:
 * - simplefunc.c
 * - simplefunc.h
 *
 *
 * ## Function parameter structure (FPS) example
 *
 *
 *
 *
 * - create_example_image.c : source code, .c file
 * - create_example_image.h : source code, .h file
 *
 */


/* ================================================================== */
/* ================================================================== */
/*  MODULE INFO                                                       */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "modex"

// Module short description
#define MODULE_DESCRIPTION "Example module: template for creating new modules"


#include "milk_runnable.h"

//
// Each header declares the runnable_modex_* units listed in the registry below
//
#include "milk_module_example.h"


// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_modex_imsum1,        &runnable_modex_imsum2, &runnable_modex_streamupdate,
    &runnable_modex_streamprocess, &runnable_modex_fpscli,
};

MILK_MODULE_RUNNABLES(milk_module_example, NULL, runnables);

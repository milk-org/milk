/**
 * @file    clustering.c
 * @brief   cluster imates and data
 *
 *
 */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "clustering"

// Module short description
#define MODULE_DESCRIPTION "Cluster images and data"

#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CommandLineInterface/CLIcore.h"

#include "cubecluster.h"
#include "mindiffscan.h"

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(clustering)

static errno_t init_module_CLI()
{

    CLIADDCMD_clustering__imcube_mkcluster();

    CLIADDCMD_clustering__imcube_mindiffscan();

    // add atexit functions here

    return RETURN_SUCCESS;
}

/**
 * @file    templatemodule.c
 * @brief   template module
 *
 * Follow this template to write your C code
 * The template includes examples of frequently used coding practices
 * and doxygen-based documentation
 *
 * Source code includes notes (comments):
 * - CODING STANDARD NOTE : note about coding practices and standards
 * - DOCUMENTATION NOTE : how to document code
 *
 *
 * ## Other files of interest
 * Each module should include :
 * - souce code (.c file)
 * - header file (.h file)
 *
 * ## Change log
 * - 20180120  Guyon   Added mode documentation
 * - 20170813  Guyon   Added some documentation
 *
 * @author  O. Guyon
 *
 * @bug No known bugs.
 *
 */

/** @defgroup RTfunctions Functions with high priority scheduler */

/// CODING STANDARD NOTE: code indented by : bash -c "astyle --indent-classes -Y"

/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

// System includes
#include <math.h>
#include <stdio.h>

/// CODING STANDARD NOTE: document any unusual head file include

#include <strangefile.h> // module strangefile does strange things, which we need in this module

// frequently included
#include "00CORE/00CORE.h"
#include "CLIcore.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include <fitsio.h>

// CODING STANDARD NOTE: include function prototypes for this module
#include "templatemodule/templatemodule.h"

/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

// CODING STANDARD NOTE: start with defines
// CODING STANDARD NOTE: #define constants should be in all CAPS
// CODING STANDARD NOTE: do not use names starting or ending with "_" (reserved for system)
#define TWOPLUSTWO             4
#define NAME_STRING_MAXSIZE    100
#define FNAME_STRING_MAXSIZE   200
#define COMMAND_STRING_MAXSIZE 200

// CODING STANDARD NOTE: list function macros after defines
// CODING STANDARD NOTE: function macro start with module name when specific to current module
#define ABS(x)                  (((x) < 0) ? -(x) : (x))
#define EXAMPLEMODULE_MAX(a, b) ((a < b) ? (b) : (a))

// CODING STANDARD NOTE: list typedefs after function macros

// CODING STANDARD NOTE: list enums after typedefs

/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */

// CODING STANDARD NOTE: Function, typedef, and variable names, as well as struct, union, and enum tag names should be in lower case
// CODING STANDARD NOTE:

// externs

// non-static globals

// static globals

/* =============================================================================================== */
/* =============================================================================================== */
/*                           FUNCTIONS TIED TO COMMAND LINE INTERFACE (CLI)                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name CLI bindings */

// CLI commands
//
// function CLI_checkarg used to check arguments
// CLI_checkarg ( CLI argument index , type code )
//
// type codes:
// 1: float
// 2: long
// 3: string, not existing image
// 4: existing image
// 5: string
//

// CODING STANDARD NOTE: CLI function name should be function name + "_cli", with no argument
int_fast8_t templatemodule_examplefunction00_cli()
{
    if (CLI_checkarg(1, 2) == 0)
        {
            templatemodule_examplefunction00(data.cmdargtoken[1].val.numl);
            return 0;
        }
    else
        return 1;
}

int_fast8_t templatemodule_examplefunction01_cli()
{
    if (CLI_checkarg(1, 1) + CLI_checkarg(2, 2) == 0)
        {
            templatemodule_examplefunction01(data.cmdargtoken[1].val.numl,
                                             data.cmdargtoken[2].val.numf,
                                             NULL);
            return 0;
        }
    else
        return 1;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                    MODULE INITIALIZATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name Module initialization */

int_fast8_t init_templatemodule()
{
    FILE *fp;

    strcpy(data.module[data.NBmodule].name, __FILE__);
    strcpy(data.module[data.NBmodule].info, "AO loop control");
    data.NBmodule++;

    // CODING STANDARD NOTE: follow this template to link function calls to the command line interface
    // CODING STANDARD NOTE: arg1 : function name in command line
    // CODING STANDARD NOTE: arg2 : module name (= __FILE__)
    // CODING STANDARD NOTE: arg3 : C call
    // CODING STANDARD NOTE: arg4 : one-line description
    // CODING STANDARD NOTE: arg5 : arguments. <arg1 [type]> <arg2 [type]> ...
    // CODING STANDARD NOTE: arg6 : example call from CLI
    // CODING STANDARD NOTE: arg7 : C call syntax

    RegisterCLIcommand("clicmdname",
                       __FILE__,
                       templatemodule_examplefunction_cli,
                       "function purpose",
                       "<mode [int]>",
                       "clicmdname 3",
                       "int templatemodule_examplefunc(int mode)");

    // CODING STANDARD NOTE: Link as many functions as desired

    // CODING STANDARD NOTE: Add atexit function(s) here (OPTIONAL)
    // atexit((void*) templatemodule_clearmem());
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTION(S) SOURCE CODE                                      */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name TEMPLATEMODULE functions */

/* =============================================================================================== */
/* =============================================================================================== */
/** @name TEMPLATEMODULE - 1. FIRST GROUP OF FUNCTIONS                                             */
/* =============================================================================================== */
/* =============================================================================================== */

// CODING STANDARD NOTE: minimal required documentation for doxygen
/**
 *  ## Purpose
 *
 * This function does absolutely nothing useful
 *
 *
 * ## Arguments
 *
 * @param[in]	mode
 * 		mode sets up what function does
 * -	does nothing
 * -	also does nothing
 *
 */
// CODING STANDARD NOTE: function name start with module name
int templatemodule_examplefunc00(int mode)
{
    // CODING STANDARD NOTE: pointer qualifier '*' with variable rather than with the type
    float *farray;

    int return_value;

    // CODING STANDARD NOTE: choose human-readable variable names
    // CODING STANDARD NOTE: unrelated declarations should be on separate lines
    long iipix, jjpix; // pixel coordinates
    long iteration;
    long NBiteration;
    long n2;

    FILE *fp_test;

    farray = (float *) malloc(sizeof(float) * 10);
    if (farray == NULL)
        printERROR(__FILE__, __func__, __LINE__, "malloc returns zero value");

    // CODING STANDARD NOTE: how to write an infinite loop
    // CODING STANDARD NOTE: Do not write infinite loop with while statement
    for (;;)
        {
            // infinite loop
        }
    free(farray);

    fp_test = fopen("testfile.log", "w");
    if (fp_test == NULL)
        printERROR(__FILE__,
                   __func__,
                   __LINE__,
                   "Cannot open file testfile.log");
    fclose(fp_test);

    if (mode == 2)
        {
            // CODING STANDARD NOTE: reduce variable scope as much as possible
            // CODING STANDARD NOTE: variables used inside code block declared at beginning of code block
            char imagename[NAME_STRING_MAXSIZE];
            char command[COMMAND_STRING_MAXSIZE];

            // CODING STANDARD NOTE: Always test return value of std functions
            // CODING STANDARD NOTE: Use functions in 00CORE.h :
            // CODING STANDARD NOTE: 		printERROR(const char *file, const char *func, int line, char *errmessage)
            // CODING STANDARD NOTE: 		printWARNING(const char *file, const char *func, int line, char *warnmessage)
            // CODING STANDARD NOTE:  printERROR will exit code, printWARNING will issue warning and continue
            if (sprintf(name, "image1", loop) < 1)
                printERROR(__FILE__,
                           __func__,
                           __LINE__,
                           "sprintf wrote <1 char");
            if (sprintf(command, "ls %s.fits", name) < 1)
                printERROR(__FILE__,
                           __func__,
                           __LINE__,
                           "sprintf wrote <1 char");
            if (system(command) != 0)
                printERROR(__FILE__,
                           __func__,
                           __LINE__,
                           "system() returns non-zero value");
        }

    fp_test = fopen("testfile.log", "r");
    if (fp_test == NULL)
        printERROR(__FILE__,
                   __func__,
                   __LINE__,
                   "Cannot Read file testfile.log");

    // CODING STANDARD NOTE: include field width limits in fscanf and sscanf calls
    if (fscanf(fp_test, "%8ld", &n2) != 1)
        printERROR(__FILE__, __func__, __LINE__, "fscanf returns value != 1");
    fclose(fp_test);

    // CODING STANDARD NOTE: Other test prototypes:
    // CODING STANDARD NOTE: if(fread(...) < 1) printERROR(__FILE__,__func__,__LINE__, "fread() returns <1 value");

    free(farray);

    return (0);
}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name TEMPLATEMODULE - 2. SECOND GROUP OF FUNCTIONS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

/* =============================================================================================== */
/** @name TEMPLATEMODULE - 2.1. SECOND GROUP OF FUNCTIONS - SUBGROUP1                              */
/* =============================================================================================== */

// DOCUMENTATION NOTE: put @brief statement in .h
// DOCUMENTATION NOTE: put short argument description in .h
// DOCUMENTATION NOTE: put detailed function documentation in .c

/**
 * ## Purpose
 *
 * This function demonstrates use of seteuid, sched_setscheduler \n
 * Note that fields "Use", "return", "not" and "warning" are optional
 *
 * ## Arguments
 *
 * Brief description of arguments is in .h file \n
 * A more detailed description of arguments in provided here \n
 *
 *
 *
 * ## Use
 *
 * Use section is optional
 * to use this function, call it as
 * @code
 * templatemodule_examplefunc01(v1, n1, farray)
 * @endcode
 *
 * ---
 *
 *
 * @return number of iteration [int]
 * @note sched_setscheduler and seteuid not supported under OS-X
 * @warning This function does nothing useful
 *
 * \ingroup RTfunctions
 *
 *
 * ## Details
 *
 */
int templatemodule_examplefunc01(const char *namein,
                                 float       val1,
                                 int         n1,
                                 float *restrict farray)
{
    /// ---
    /// # Code Description
    ///
    int Niteration;

    int RT_priority =
        95; // any number from 0-99. Higher number = higher priority
    struct sched_param schedpar;

    int retval;

    schedpar.sched_priority = RT_priority;

    /// ## Set up priviledges

#ifndef __MACH__                    // Do not run code below if OS-X
    iretval = seteuid(euid_called); //This goes up to maximum privileges
    if (retval != 0)
        printERROR(__FILE__,
                   __func__,
                   __LINE__,
                   "seteuid() returns non-zero value");

    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster

    retval = seteuid(euid_real); //Go back to normal privileges
    if (retval != 0)
        printERROR(__FILE__,
                   __func__,
                   __LINE__,
                   "seteuid() returns non-zero value");
#endif

    /// ## Execute loop

    // code here
    for (;;) // preferred way to write infinite loop
        {
        }

    /// ---
    return (Niteration);
}

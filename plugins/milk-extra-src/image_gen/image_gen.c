/**
 * @file    image_gen.c
 * @brief   Generate frequently used image(s)
 *
 * Creates images for misc applications
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
#define MODULE_SHORTNAME_DEFAULT "imgen"

// Module short description
#define MODULE_DESCRIPTION                                                     \
    "Creating images (shapes, useful functions and patterns)"

#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CFITSIO
#include <fitsio.h> /* required by every program that uses CFITSIO  */
#endif

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

#include "COREMOD_arith/COREMOD_arith.h"
#ifdef USE_CFITSIO
#include "COREMOD_iofits/COREMOD_iofits.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

#include "statistic/statistic.h"

#ifndef MILK_NO_CLI
#include "image_gen/image_gen.h"

#include "mkdisk.h"
#include "mkpolygon.h"
#include "mkrandomim.h"
#include "mkspdisk.h"
#include "voronoi.h"

#define OMP_NELEMENT_LIMIT 1000000

#define SWAP(x, y)                                                             \
    tmp = (x);                                                                 \
    x   = (y);                                                                 \
    y   = tmp;

#define PI 3.14159265358979323846264338328

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(image_gen)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/* Placeholder for CLICMD_FIELDS_DEFAULTS macro which
 * hardcodes 'farg'. The constructor init_xx() functions
 * below overwrite nbarg and funcfpscliarg at runtime. */
static CLICMDARGDEF farg[] = {
    {CLIARG_FLOAT64, "", "", "", 0, NULL, NULL}
};

#include "fps.h"

/* ==================================================================
 * CLI Command Registrations
 * ================================================================== */

extern errno_t CLIADDCMD_image_gen__segs2wfmodes(void);
extern errno_t CLIADDCMD_image_gen__mkrect(void);
extern errno_t CLIADDCMD_image_gen__mkgridpix(void);
extern errno_t CLIADDCMD_image_gen__mkline(void);
extern errno_t CLIADDCMD_image_gen__mkdist(void);
extern errno_t CLIADDCMD_image_gen__mkrndim(void);
extern errno_t CLIADDCMD_image_gen__mkhexsegpup(void);
extern errno_t CLIADDCMD_image_gen__mkrndgim(void);
extern errno_t CLIADDCMD_image_gen__mkslopexy(void);
extern errno_t CLIADDCMD_image_gen__mklincoord(void);
extern errno_t CLIADDCMD_image_gen__im2coord(void);
extern errno_t CLIADDCMD_image_gen__mkfiberclpoverlap(void);
extern errno_t CLIADDCMD_image_gen__mkgauss(void);
extern errno_t CLIADDCMD_image_gen__mkdisk(void);
extern errno_t CLIADDCMD_image_gen__mkpolygon(void);
extern errno_t CLIADDCMD_image_gen__mkspdisk(void);
extern errno_t CLIADDCMD_image_gen__voronoi(void);
extern errno_t CLIADDCMD_image_gen__mkrandomim(void);

/* ===== Module init ===== */

static errno_t init_module_CLI()
{
    CLIADDCMD_image_gen__segs2wfmodes();
    CLIADDCMD_image_gen__mkrect();
    CLIADDCMD_image_gen__mkgridpix();
    CLIADDCMD_image_gen__mkline();
    CLIADDCMD_image_gen__mkdist();
    CLIADDCMD_image_gen__mkrndim();
    CLIADDCMD_image_gen__mkhexsegpup();
    CLIADDCMD_image_gen__mkrndgim();
    CLIADDCMD_image_gen__mkslopexy();
    CLIADDCMD_image_gen__mklincoord();
    CLIADDCMD_image_gen__im2coord();
    CLIADDCMD_image_gen__mkfiberclpoverlap();
    CLIADDCMD_image_gen__mkgauss();
    CLIADDCMD_image_gen__mkdisk();
    CLIADDCMD_image_gen__mkpolygon();
    CLIADDCMD_image_gen__mkspdisk();
    CLIADDCMD_image_gen__voronoi();
    CLIADDCMD_image_gen__mkrandomim();

    // add atexit functions here

    return RETURN_SUCCESS;
}

#endif /* MILK_NO_CLI */

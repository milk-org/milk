/**
 * @file image_multicrop2D.h
 * @brief Header for the multicrop 2D function.
 */

#ifndef COREMOD_ARITH_MULTICROP2D_H
#define COREMOD_ARITH_MULTICROP2D_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#define MAXNB_CROPWINDOW 8

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char     *multicrop_insname;
extern char     *multicrop_outsname;
extern uint32_t *multicrop_outxsize;
extern uint32_t *multicrop_outysize;

extern int64_t  *multicrop_wactive[MAXNB_CROPWINDOW];
extern int64_t  *multicrop_waddmode[MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropxstart[MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropxsize[MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropystart[MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropysize[MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wbinfact[MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropxpos[MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropypos[MAXNB_CROPWINDOW];

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void image_multicrop2D_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *imgin, IMAGE *imgout);
errno_t image_multicrop2D_validate();

errno_t CLIADDCMD_COREMODE_arith__multicrop2D();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define MULTICROP_WPARAMS(X, wn) \
    X(CLIARG_ONOFF,  FPTYPE_ONOFF,  int64_t,  ".w"#wn".active",      "crop window active",        "0",  0, &multicrop_wactive[wn],      (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_ONOFF,  FPTYPE_ONOFF,  int64_t,  ".w"#wn".addmode",     "1:add, 0:replace",          "0",  0, &multicrop_waddmode[wn],     (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32, uint32_t, ".w"#wn".cropxstart",  "crop x coord start",        "30", 30, &multicrop_wcropxstart[wn],  (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32, uint32_t, ".w"#wn".cropxsize",   "crop x coord size",         "30", 30, &multicrop_wcropxsize[wn],   (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32, uint32_t, ".w"#wn".cropystart",  "crop y coord start",        "30", 30, &multicrop_wcropystart[wn],  (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32, uint32_t, ".w"#wn".cropysize",   "crop y coord size",         "30", 30, &multicrop_wcropysize[wn],   (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32, uint32_t, ".w"#wn".cropxpos",    "x placement in output",     "30", 30, &multicrop_wcropxpos[wn],    (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32, uint32_t, ".w"#wn".cropypos",    "y placement in output",     "30", 30, &multicrop_wcropypos[wn],    (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32, uint32_t, ".w"#wn".cropbinfact", "binning factor",            "1",  1, &multicrop_wbinfact[wn],     (void*)&val, CLIARG_HIDDEN_DEFAULT)

#define MULTICROP2D_PARAMS(X) \
    X(CLIARG_IMG,    FPTYPE_STREAMNAME, char*,    ".insname",  "input stream name",  "inim",  "inim", &multicrop_insname,  (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_STR,    FPTYPE_STREAMNAME, char*,    ".outsname", "output stream name", "outim", "outim",&multicrop_outsname,  (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".outxsize", "output x size",      "200",   200,    &multicrop_outxsize, (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".outysize", "output y size",      "200",   200,    &multicrop_outysize, (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    MULTICROP_WPARAMS(X, 0) MULTICROP_WPARAMS(X, 1) MULTICROP_WPARAMS(X, 2) MULTICROP_WPARAMS(X, 3) \
    MULTICROP_WPARAMS(X, 4) MULTICROP_WPARAMS(X, 5) MULTICROP_WPARAMS(X, 6) MULTICROP_WPARAMS(X, 7)

#define MULTICROP2D_HELPTEXT \
    "multicrop2D: crop 2D image, multiple windows\n" \
    "===========================================\n" \
    "Extracts multiple rectangular regions from an input stream and places\n" \
    "them into a single output stream.\n\n" \
    "Parameters:\n" \
    "  .insname  : Input stream\n" \
    "  .outsname : Output stream\n" \
    "  .wXX.*    : Per-window settings (active, start, size, pos, bin)\n"

#endif
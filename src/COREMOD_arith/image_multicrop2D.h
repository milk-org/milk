/**
 * @file image_multicrop2D.h
 * @brief Multi-window 2D cropping from stream.
 */

#ifndef COREMOD_ARITH_MULTICROP2D_H
#define COREMOD_ARITH_MULTICROP2D_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#define MAXNB_CROPWINDOW 8

extern char     *multicrop_insname;
extern char     *multicrop_outsname;
extern uint32_t *multicrop_outxsize;
extern uint32_t *multicrop_outysize;

extern int64_t  *multicrop_wactive[
    MAXNB_CROPWINDOW];
extern int64_t  *multicrop_waddmode[
    MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropxstart[
    MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropxsize[
    MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropystart[
    MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropysize[
    MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wbinfact[
    MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropxpos[
    MAXNB_CROPWINDOW];
extern uint32_t *multicrop_wcropypos[
    MAXNB_CROPWINDOW];

void image_multicrop2D_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *imgin, IMAGE *imgout
);
errno_t image_multicrop2D_validate();

errno_t
CLIADDCMD_COREMODE_arith__multicrop2D();

/**
 * V2 format nested X-macro for per-window
 * parameters.
 * X(keyword, ptr, type,
 *   is_primary, fpflag, descr)
 */
#define MULTICROP_WPARAMS(X, wn) \
    X(".w"#wn".active",              \
      &multicrop_wactive[wn],        \
      FPTYPE_ONOFF, 0,               \
      FPFLAG_DEFAULT_INPUT,          \
      "crop window active")          \
    X(".w"#wn".addmode",             \
      &multicrop_waddmode[wn],       \
      FPTYPE_ONOFF, 0,               \
      FPFLAG_DEFAULT_INPUT,          \
      "1:add, 0:replace")           \
    X(".w"#wn".cropxstart",          \
      &multicrop_wcropxstart[wn],    \
      FPTYPE_UINT32, 0,              \
      FPFLAG_DEFAULT_INPUT,          \
      "crop x coord start")         \
    X(".w"#wn".cropxsize",           \
      &multicrop_wcropxsize[wn],     \
      FPTYPE_UINT32, 0,              \
      FPFLAG_DEFAULT_INPUT,          \
      "crop x coord size")          \
    X(".w"#wn".cropystart",          \
      &multicrop_wcropystart[wn],    \
      FPTYPE_UINT32, 0,              \
      FPFLAG_DEFAULT_INPUT,          \
      "crop y coord start")         \
    X(".w"#wn".cropysize",           \
      &multicrop_wcropysize[wn],     \
      FPTYPE_UINT32, 0,              \
      FPFLAG_DEFAULT_INPUT,          \
      "crop y coord size")          \
    X(".w"#wn".cropxpos",            \
      &multicrop_wcropxpos[wn],      \
      FPTYPE_UINT32, 0,              \
      FPFLAG_DEFAULT_INPUT,          \
      "x placement in output")      \
    X(".w"#wn".cropypos",            \
      &multicrop_wcropypos[wn],      \
      FPTYPE_UINT32, 0,              \
      FPFLAG_DEFAULT_INPUT,          \
      "y placement in output")      \
    X(".w"#wn".cropbinfact",         \
      &multicrop_wbinfact[wn],       \
      FPTYPE_UINT32, 0,              \
      FPFLAG_DEFAULT_INPUT,          \
      "binning factor")

#define MULTICROP2D_PARAMS(X) \
    X(".insname", &multicrop_insname,    \
      FPTYPE_STREAMNAME, 1,              \
      (FPFLAG_DEFAULT_INPUT              \
       | FPFLAG_CLI_INPUT),              \
      "input stream name")               \
    X(".outsname", &multicrop_outsname,  \
      FPTYPE_STREAMNAME, 1,              \
      (FPFLAG_DEFAULT_INPUT              \
       | FPFLAG_CLI_INPUT),              \
      "output stream name")              \
    X(".outxsize", &multicrop_outxsize,  \
      FPTYPE_UINT32, 1,                  \
      (FPFLAG_DEFAULT_INPUT              \
       | FPFLAG_CLI_INPUT),              \
      "output x size")                   \
    X(".outysize", &multicrop_outysize,  \
      FPTYPE_UINT32, 1,                  \
      (FPFLAG_DEFAULT_INPUT              \
       | FPFLAG_CLI_INPUT),              \
      "output y size")                   \
    MULTICROP_WPARAMS(X, 0)              \
    MULTICROP_WPARAMS(X, 1)              \
    MULTICROP_WPARAMS(X, 2)              \
    MULTICROP_WPARAMS(X, 3)              \
    MULTICROP_WPARAMS(X, 4)              \
    MULTICROP_WPARAMS(X, 5)              \
    MULTICROP_WPARAMS(X, 6)              \
    MULTICROP_WPARAMS(X, 7)

#define MULTICROP2D_HELPTEXT \
    "multicrop2D: multi-window 2D crop\n"

#endif
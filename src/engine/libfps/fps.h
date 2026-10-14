// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FPS_H
#define FPS_H

#include <linux/limits.h> // For PATH_MAX (but maybe would belong in a global milk compilation header?) // For some reason this fails only for standalone execs, and even not all of them!!
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milkDebugTools.h"

/* Type definitions, structs, constants, and flags */
#include "fps_types.h"

/* Additional dependencies not in fps_types.h */
#include "timeutils.h"
#include "processinfo_signals.h"
#include "processtools.h"


typedef long imageID;
typedef long variableID;

#include "IMGID.h"

int function_parameter_printlist(FPS_PARAM *funcparamarray, long NBparamMAX);

/* Core FPS operations (connect, params, entries) */
#include "fps_core.h"

/* Additional API sub-headers */
#include "fps_GetFileName.h"
#include "fps_load.h"
#include "fps_loadstream.h"
#include "fps_outlog.h"
// #include "fps_printlist.h" // Removed
#include "fps_save2disk.h"
#include "fps_scan.h"
#include "fps_shmdirname.h"
#include "fps_WriteParameterToDisk.h"

#include "fps_local_store.h"

/* Shared help-message color palette and flag init */
#include "milk_help.h"

/* Legacy color aliases -- map to milk_help.h palette.
 * Used by non-help code (e.g. runtime status lines).
 * New code should use MH_* macros instead.
 */
#ifndef COLORRESET
#    define COLORRESET MH_RST
#endif
#ifndef COLORARGCLI
#    define COLORARGCLI MH_DFLT
#endif
#ifndef COLORARGnotCLI
#    define COLORARGnotCLI MH_ARG
#endif
#ifndef COLORCOMMAND
#    define COLORCOMMAND MH_CMD
#endif
#ifndef COLORHEADER
#    define COLORHEADER MH_HDR
#endif
#ifndef COLOROPTION
#    define COLOROPTION MH_OPT
#endif
#ifndef COLORPRIMARY
#    define COLORPRIMARY MH_TITLE
#endif
#ifndef COLORNORMAL
#    define COLORNORMAL "\033[0;37m"
#    define COLORERROR MH_ERR
#endif

#endif // FPS_H

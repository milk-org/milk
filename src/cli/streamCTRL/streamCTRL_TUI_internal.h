// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _STREAMCTRL_TUI_INTERNAL_H
#define _STREAMCTRL_TUI_INTERNAL_H

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;


#endif

#include <sys/stat.h>
#include <sys/select.h>
#include <fcntl.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "streamCTRL_defs.h"
#include "streamCTRL_ansi.h"
#include "streamCTRL_TUIcompat.h"


#define TAB_LIST(X)                                 \
    X(DISPLAY_MODE_HELP, "h", "Help")               \
    X(DISPLAY_MODE_SUMMARY, "F2", "summary")        \
    X(DISPLAY_MODE_WRITE, "F3", "write PIDs")       \
    X(DISPLAY_MODE_READ, "F4", "read PIDs")         \
    X(DISPLAY_MODE_SPTRACE, "F5", "process traces") \
    X(DISPLAY_MODE_FUSER, "F6", "access")

#define TAB_COUNT_ONE(mode, key, label) +1
#define TAB_COUNT (0 TAB_LIST(TAB_COUNT_ONE))

#ifndef SHAREDSHMDIR
#    define SHAREDSHMDIR "/dev/shm"


#endif

#include "streamCTRL_TUI.h"

#include "streamCTRL_find_streams.h"
#include "streamCTRL_print_inode.h"
#include "streamCTRL_print_procpid.h"
#include "streamCTRL_print_trace.h"
#include "streamCTRL_scan.h"
#include "streamCTRL_utilfuncs.h"

struct streamCTRL_TUI_parameters
{
    int        loopOK;
    int        dindexSelected;
    int        DisplayDetailLevel;
    int        DisplayMode;
    int        NBsindex;
    int        SORTING;
    int        DISPLAY_ALL_SEMS;
    struct tm *uttime_lastScan;
    int        fuserScan;
    int        SORT_TOGGLE;
    float      frequ;                   // Hz
    int        sort_col;                // 0=none, 1..7=column id
    int        sort_dir;                // 0=ascending, 1=descending
    long       ssindex[streamNBID_MAX]; // sorted index array
};

extern struct streamCTRL_TUI_parameters sTUIparam;

extern short unsigned int wrow, wcol;

extern STREAMINFO *g_streaminfo_qsort;
extern IMAGE      *g_sort_images;
extern int         g_sort_col;
extern int         g_sort_dir;

struct streamCTRL_TUI_state
{
    long   doffsetindex;
    int    monstrlen;
    char  *monstring;
    int    DispName_NBchar;
    int    DispSize_NBchar;
    int    Dispcnt0_NBchar;
    int    Dispfreq_NBchar;
    int    DispPID_NBchar;
    int    PIDmax;
    char **PIDname_array;
    ino_t  inodeselected;

    int    NBupstreaminodeMAX;
    ino_t *upstreaminode;
    int    NBupstreaminode;

    int    NBupstreamprocMAX;
    pid_t *upstreamproc;
    int    NBupstreamproc;

    long long loopcnt;

    /** Terminal row where the first data entry starts (1-based).
     *  Set by the header renderer so the input handler can
     *  map mouse click row to display index. */
    int body_start_row;
};

// Functions implemented in streamCTRL_TUI_input.c
errno_t streamCTRL_keyinput_process(int                          ch,
                                    streamCTRLarg_struct        *streamCTRLdata,
                                    struct streamCTRL_TUI_state *state);

// Functions implemented in streamCTRL_TUI_render.c
void streamCTRL_render_screen(streamCTRLarg_struct        *streamCTRLdata,
                              struct streamCTRL_TUI_state *state);

// Render helpers that are currently static in streamCTRL_TUI.c but needed in streamCTRL_TUI_render.c


extern volatile sig_atomic_t sc_sigINT;
#include <signal.h>


#endif

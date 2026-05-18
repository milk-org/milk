#ifndef _PROCCTRL_TUI_INTERNAL_H
#define _PROCCTRL_TUI_INTERNAL_H

#include <stdio.h>
#include <time.h>
#include <sys/resource.h>

#include "procCTRL_TUI.h"
#include "processtools.h"
#include "processinfo_scan_shm.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_create.h"
#include "milkDebugTools.h"

extern short unsigned int wrow;
extern short unsigned int wcol;


typedef struct {
    PROCINFOPROC *procinfoproc;
    PROCSCAN_SHM *scan_shm;
    FILE *flog;
    char procdname[STRINGMAXLEN_DIRNAME];
    float frequ;
    int pindexSelected;
    int pindexActiveSelected;
    long doffsetindex;
    int freeze;
    int loopOK;
    int Xexit;
    char monstring[200];
    int monstringlen;
    int last_ch;
    float tool_cpu_pcnt;
    float actual_fps;
    
    struct timespec t_last_scan;
    struct timespec t_now;
    struct rusage usage_prev;
    struct rusage usage_cur;
    struct timespec t_usage_prev;
    struct timespec t_usage_cur;
    struct timespec t_disp_prev;
    struct timespec t_disp_cur;
    STRINGLISTENTRY *CPUsetList;
} procctrl_context_t;

// Render function
void procctrl_render_frame(
    procctrl_context_t *ctx,
    int NBactive);

// Input handling function
void procctrl_handle_keyboard_event(
    procctrl_context_t *ctx,
    int ch,
    int NBactive);

/* Sort context (defined in procCTRL_TUI_sort.c) */
extern int sort_ctx_m;
extern int sort_ctx_col;
extern int sort_ctx_dir;
extern PROCSCAN_SHM *sort_ctx_scan_shm;
extern PROCESSINFOLIST *sort_ctx_pinfolist;
int proc_comp(const void *a, const void *b);

#endif // _PROCCTRL_TUI_INTERNAL_H

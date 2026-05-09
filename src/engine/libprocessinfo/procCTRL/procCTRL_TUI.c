/**
 * @file procCTRL_TUI.c
 * @brief Procctrl tui module
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>

#include <sys/types.h>
#include <unistd.h>

#include <fcntl.h>
#include <sys/mman.h>

#include <dirent.h>

#include <locale.h>
#include <wchar.h>



#include "timeutils.h"
#include "quicksort.h"

#include "procCTRL_TUIcompat.h"
#include "milkDebugTools.h"

static char local_shmdir[STRINGMAXLEN_DIRNAME];
#define SHAREDPROCDIR local_shmdir

#include <processtools.h>
#include "processinfo_signals.h"

#include "processinfo_setup.h"
#include "processinfo_procdirname.h"
#include "processinfo_SIGexit.h"
#include "processinfo_shm_create.h"
#include "processinfo_shm_list_create.h"
#include "processinfo_exec_start.h"
#include "processinfo_exec_end.h"


#include "procCTRL_PIDcollectSystemInfo.h"
#include "procCTRL_GetCPUloads.h"
#include "procCTRL_GetNumberCPUs.h"
#include "procCTRL_processinfo_scan.h"

#include "procCTRL_TUI.h"


int procCTRL_debug_mode = 0;
char procCTRL_logfile[1024] = "";

static short unsigned int wrow, wcol;

extern PROCESSINFOLIST *pinfolist;

#include "processinfo_scan_shm.h"
static int processinfo_CPUsets_List(STRINGLISTENTRY *CPUsetList, int has_cset)
{
    if(has_cset == 0) return 0;
    
    char fname[STRINGMAXLEN_FULLFILENAME];
    snprintf(fname, sizeof(fname), "%s/.csetlist.%ld", SHAREDPROCDIR, (long) getpid());
    
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "cset set -l | awk '/root/{stop=1} stop==1{print $0}' > %s", fname);
    if(system(cmd) != 0) {
        return 0;
    }

    FILE *fp = fopen(fname, "r");
    if (!fp) return 0;

    char line[200];
    char word[200], word1[200];
    int NBset = 0;
    int setindex = 0;

    while(NBset < 1000 && fgets(line, 199, fp) != NULL) {
        sscanf(line, "%199s %199s", word, word1);
        strncpy(CPUsetList[setindex].name, word,
                sizeof(CPUsetList[setindex].name) - 1);
        CPUsetList[setindex].name[
            sizeof(CPUsetList[setindex].name) - 1] = '\0';
        strncpy(CPUsetList[setindex].description, word1,
                sizeof(CPUsetList[setindex].description) - 1);
        CPUsetList[setindex].description[
            sizeof(CPUsetList[setindex].description) - 1]
            = '\0';
        setindex++;
        NBset++;
    }
    fclose(fp);
    remove(fname);
    return NBset;
}

static int __attribute__((unused)) processinfo_SelectFromList(STRINGLISTENTRY *StringList, int NBelem)
{
    int selected = 0;
    int inputOK = 0;
    char buff[100];
    char *p;

    printf("%d entries in list:\n", NBelem);
    for(int i = 0; i < NBelem; i++) {
        printf("   %3d   : %16s   %s\n", i, StringList[i].name, StringList[i].description);
    }

    while(inputOK == 0) {
        printf("\nEnter a number: ");
        fflush(stdout);
        if (fgets(buff, sizeof(buff), stdin)) {
            selected = strtol(buff, &p, 10);
            if(selected >= 0 && selected < NBelem) inputOK = 1;
            else printf("\nError: invalid number.\n");
        }
    }
    printf("Selected entry : %s\n", StringList[selected].name);
    return selected;
}

/* Sort context and proc_comp() are in
 * procCTRL_TUI_sort.c */
extern int sort_ctx_m;
extern int sort_ctx_col;
extern int sort_ctx_dir;
extern PROCSCAN_SHM *sort_ctx_scan_shm;
extern PROCESSINFOLIST *sort_ctx_pinfolist;
extern int proc_comp(
    const void *a, const void *b);


static inline void *link_scan_shm(const char *name, size_t size) {
    int fd = open(name, O_RDWR);
    if (fd == -1) return MAP_FAILED;
    void *ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return ptr;
}

#include <sys/resource.h>

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
} procctrl_context_t;

static void procctrl_render_frame(procctrl_context_t *ctx, int NBactive);

static void procctrl_update_stats(procctrl_context_t *ctx) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->t_now);
    double elapsed = (ctx->t_now.tv_sec - ctx->t_last_scan.tv_sec) + (ctx->t_now.tv_nsec - ctx->t_last_scan.tv_nsec) * 1e-9;
    if (elapsed >= (ctx->procinfoproc->twaitus * 1e-6)) {
        processinfo_scan_step(ctx->procinfoproc);
        ctx->t_last_scan = ctx->t_now;
        
        getrusage(RUSAGE_SELF, &ctx->usage_cur);
        clock_gettime(CLOCK_MONOTONIC, &ctx->t_usage_cur);
        double t_diff = (ctx->t_usage_cur.tv_sec - ctx->t_usage_prev.tv_sec) + (ctx->t_usage_cur.tv_nsec - ctx->t_usage_prev.tv_nsec) * 1e-9;
        double u_diff = (ctx->usage_cur.ru_utime.tv_sec - ctx->usage_prev.ru_utime.tv_sec) + (ctx->usage_cur.ru_utime.tv_usec - ctx->usage_prev.ru_utime.tv_usec) * 1e-6;
        double s_diff = (ctx->usage_cur.ru_stime.tv_sec - ctx->usage_prev.ru_stime.tv_sec) + (ctx->usage_cur.ru_stime.tv_usec - ctx->usage_prev.ru_stime.tv_usec) * 1e-6;
        if (t_diff > 0) ctx->tool_cpu_pcnt = 100.0 * (u_diff + s_diff) / t_diff;
        ctx->usage_prev = ctx->usage_cur;
        ctx->t_usage_prev = ctx->t_usage_cur;
    }

    clock_gettime(CLOCK_MONOTONIC, &ctx->t_disp_cur);
    double d_elapsed = (ctx->t_disp_cur.tv_sec - ctx->t_disp_prev.tv_sec) + (ctx->t_disp_cur.tv_nsec - ctx->t_disp_prev.tv_nsec) * 1e-9;
    if (d_elapsed > 0) ctx->actual_fps = 0.9 * ctx->actual_fps + 0.1 * (1.0 / d_elapsed);
    ctx->t_disp_prev = ctx->t_disp_cur;
}

static void procctrl_handle_keyboard_event(procctrl_context_t *ctx, int ch, int NBactive) {
    ctx->last_ch = ch;
    if (ctx->flog) {
        char tbuf[64];
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        struct tm *tm_info = gmtime(&ts.tv_sec);
        size_t len = strftime(tbuf, sizeof(tbuf), "%Y%m%dT%H:%M:%S", tm_info);
        snprintf(tbuf + len, sizeof(tbuf) - len, ".%06ld", ts.tv_nsec / 1000);
        fprintf(ctx->flog, "%s Input: %d\\n", tbuf, ch);
        fflush(ctx->flog);
    }

    if (ch == 545 || ch == 560 || ch == 443 || ch == 564 || ch == 554) {
         ctx->procinfoproc->DisplayMode--;
         if (ctx->procinfoproc->DisplayMode < 1) ctx->procinfoproc->DisplayMode = 6;
         if (ctx->flog) { fprintf(ctx->flog, "  -> Mode changed to %d\\n", ctx->procinfoproc->DisplayMode); fflush(ctx->flog); }
    }
    else if (ch == 561 || ch == 566 || ch == 444 || ch == 565 || ch == 569) {
         ctx->procinfoproc->DisplayMode++;
         if (ctx->procinfoproc->DisplayMode > 6) ctx->procinfoproc->DisplayMode = 1;
         if (ctx->flog) { fprintf(ctx->flog, "  -> Mode changed to %d\\n", ctx->procinfoproc->DisplayMode); fflush(ctx->flog); }
    }
    else if (ch >= '0' && ch <= '9') {
         int cidx = ch - '0';
         int m = ctx->procinfoproc->DisplayMode;
         if (m >= 0 && m < 10)
             ctx->procinfoproc->col_visible[m][cidx] = !ctx->procinfoproc->col_visible[m][cidx];
    }
    else if (ch == ANSI_KEY_F2) ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
    else if (ch == ANSI_KEY_F3) ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_RESOURCES;
    else if (ch == ANSI_KEY_F4) ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TRIGGER;
    else if (ch == ANSI_KEY_F5) ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TIMING;
    else if (ch == ANSI_KEY_F6) ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_PROCINFO;
    else if (ch == 'h')      ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_HELP;
    else if (ch == ANSI_KEY_CTRL_LEFT) {
        ctx->procinfoproc->DisplayMode--;
        if(ctx->procinfoproc->DisplayMode < 1) ctx->procinfoproc->DisplayMode = 6;
    }
    else if (ch == ANSI_KEY_CTRL_RIGHT) {
        ctx->procinfoproc->DisplayMode++;
        if(ctx->procinfoproc->DisplayMode > 6) ctx->procinfoproc->DisplayMode = 1;
    }
    else if (ch == 'x' || ch == 3) { ctx->loopOK = 0; ctx->Xexit = 1; }
    else if (ch == 'f') { ctx->freeze = !ctx->freeze; }
    else if (ch == '+' || ch == '=') { ctx->frequ *= 1.2; if (ctx->frequ > 1000.0) ctx->frequ = 1000.0; }
    else if (ch == '-') { ctx->frequ /= 1.2; if (ctx->frequ < 0.1) ctx->frequ = 0.1; }
    else if (ch == ' ' && ctx->pindexSelected >= 0) {
         ctx->procinfoproc->selectedarray[ctx->pindexSelected] = !ctx->procinfoproc->selectedarray[ctx->pindexSelected];
    }
    else if (ch == ANSI_KEY_UP && NBactive > 0) {
         ctx->pindexActiveSelected--; if (ctx->pindexActiveSelected < 0) ctx->pindexActiveSelected = 0;
    }
    else if (ch == ANSI_KEY_DOWN && NBactive > 0) {
         ctx->pindexActiveSelected++; if (ctx->pindexActiveSelected >= NBactive) ctx->pindexActiveSelected = NBactive - 1;
    }
    else if (ch == ANSI_KEY_LEFT) {
         ctx->procinfoproc->selected_col--;
         if (ctx->procinfoproc->selected_col < 1) ctx->procinfoproc->selected_col = 9;
    }
    else if (ch == ANSI_KEY_RIGHT) {
         ctx->procinfoproc->selected_col++;
         if (ctx->procinfoproc->selected_col > 9) ctx->procinfoproc->selected_col = 1;
    }
    else if (ch == 'r' || ch == 'R') {
        int all = (ch == 'R');
        for(int i=0; i<PROCESSINFOLISTSIZE; i++) {
            if (all || i == ctx->pindexSelected) {
                if (pinfolist->active[i] == 2 || pinfolist->active[i] == 3) pinfolist->active[i] = 0;
            }
        }
    }
    else if ((ch == 'T' || ch == 'K' || ch == 'I') && NBactive > 0) {
        int sig = (ch == 'T') ? SIGTERM : (ch == 'K') ? SIGKILL : SIGINT;
        int any_sel = 0;
        for(int i=0; i<PROCESSINFOLISTSIZE; i++) if(ctx->procinfoproc->selectedarray[i]) {
            kill(pinfolist->PIDarray[i], sig);
            any_sel = 1;
        }
        if(!any_sel && ctx->pindexSelected >= 0) kill(pinfolist->PIDarray[ctx->pindexSelected], sig);
    }
    else if (ch == 's') {
         int m = ctx->procinfoproc->DisplayMode;
         ctx->procinfoproc->sort_mode[m] = m;
         if (ctx->procinfoproc->sort_col[m] == ctx->procinfoproc->selected_col) {
             ctx->procinfoproc->sort_dir[m] = !ctx->procinfoproc->sort_dir[m];
         } else {
             ctx->procinfoproc->sort_col[m] = ctx->procinfoproc->selected_col;
             ctx->procinfoproc->sort_dir[m] = 0;
         }
    }
    else if (ch == 'S') {
         int m_curr = ctx->procinfoproc->DisplayMode;
         int smod = ctx->procinfoproc->sort_mode[m_curr];
         int scol = ctx->procinfoproc->sort_col[m_curr];
         int sdir = ctx->procinfoproc->sort_dir[m_curr];
         for(int m=0; m<10; m++) {
             ctx->procinfoproc->sort_mode[m] = smod;
             ctx->procinfoproc->sort_col[m] = scol;
             ctx->procinfoproc->sort_dir[m] = sdir;
         }
    }
    else if ((ch == 'p' || ch == 19 || ch == 'e') && NBactive > 0) {
        int val = (ch == 'p') ? -1 : (ch == 19) ? 2 : 3;
        int any_sel = 0;
        for(int i=0; i<PROCESSINFOLISTSIZE; i++) if(ctx->procinfoproc->selectedarray[i]) {
            if (val == -1 && ctx->procinfoproc->pinfoarray[i]) ctx->procinfoproc->pinfoarray[i]->CTRLval = (ctx->procinfoproc->pinfoarray[i]->CTRLval == 0) ? 1 : 0;
            else if (ctx->procinfoproc->pinfoarray[i]) ctx->procinfoproc->pinfoarray[i]->CTRLval = val;
            any_sel = 1;
        }
        if(!any_sel && ctx->pindexSelected >= 0 && ctx->procinfoproc->pinfoarray[ctx->pindexSelected]) {
            if (val == -1) ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->CTRLval = (ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->CTRLval == 0) ? 1 : 0;
            else ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->CTRLval = val;
        }
    }
    else if ((ch == 'z' || ch == 'Z') && NBactive > 0) {
        int all = (ch == 'Z');
        for(int i=0; i<PROCESSINFOLISTSIZE; i++) if((all || ctx->procinfoproc->selectedarray[i]) && ctx->procinfoproc->pinfoarray[i]) {
            ctx->procinfoproc->pinfoarray[i]->loopcnt = 0;
        }
        if(!all && !ctx->procinfoproc->selectedarray[ctx->pindexSelected] && ctx->pindexSelected >= 0 && ctx->procinfoproc->pinfoarray[ctx->pindexSelected]) {
            ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->loopcnt = 0;
        }
    }
}



static void procctrl_render_frame(procctrl_context_t *ctx, int NBactive) {
    int m = ctx->procinfoproc->DisplayMode;
    int pindexSelected = ctx->pindexSelected;
    int pindexActiveSelected = ctx->pindexActiveSelected;
        if (ctx->freeze == 0) {
            sc_frame_clear();
            TUI_clearscreen(&wrow, &wcol);
            snprintf(ctx->monstring, ctx->monstringlen, "Mode %d   PRESS x TO STOP MONITOR", ctx->procinfoproc->DisplayMode);
            TUI_print_header(ctx->monstring, '-');
            TUI_newline();
            
            {
                struct stat shm_stat;
                char shm_list_fname[STRINGMAXLEN_FULLFILENAME];
                WRITE_FULLFILENAME(shm_list_fname, "%s/processinfo.list.shm", ctx->procdname);
                if(stat(shm_list_fname, &shm_stat) == 0) {
                    char timestr[100];
                    struct tm *tm_info = gmtime(&shm_stat.st_mtime);
                    strftime(timestr, 100, "%Y-%m-%d %H:%M:%S", tm_info);
                    TUI_printfw("List: %-25s (Upd: %s)  ToolCPU: %5.2f%% (%4.1f fps)", shm_list_fname, timestr, ctx->tool_cpu_pcnt, ctx->actual_fps);
                    if (ctx->flog) TUI_printfw(" K:%d", ctx->last_ch);
                    TUI_newline();
                }
            }
            
            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP) screenprint_setcolor(2);
            TUI_printfw("[h] Help");
            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP) screenprint_unsetcolor(2);
            TUI_printfw("   ");

            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_CTRL) screenprint_setcolor(2);
            TUI_printfw("[F2] CTRL");
            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_CTRL) screenprint_unsetcolor(2);
            TUI_printfw("   ");

            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES) screenprint_setcolor(2);
            TUI_printfw("[F3] Resources");
            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES) screenprint_unsetcolor(2);
            TUI_printfw("   ");

            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER) screenprint_setcolor(2);
            TUI_printfw("[F4] Triggering");
            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER) screenprint_unsetcolor(2);
            TUI_printfw("   ");

            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TIMING) screenprint_setcolor(2);
            TUI_printfw("[F5] Timing");
            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TIMING) screenprint_unsetcolor(2);
            TUI_printfw("   ");

            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_PROCINFO) screenprint_setcolor(2);
            TUI_printfw("[F6] PROCINFO");
            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_PROCINFO) screenprint_unsetcolor(2);
            TUI_newline();

            // Column visibility status line
            {
                const char *colnames[10] = {NULL};
                int nbcol = 0;
                switch(ctx->procinfoproc->DisplayMode) {
                    case PROCCTRL_DISPLAYMODE_CTRL:
                        {
                            static const char *names[] = {"", "idx", "status", "pid", "tstart", "pname", "state", "lcnt", "msg", ""};
                            for(int i=0; i<10; i++) colnames[i] = names[i];
                            nbcol = 8;
                        }
                        break;
                    case PROCCTRL_DISPLAYMODE_RESOURCES:
                        {
                            static const char *names[] = {"", "idx", "status", "pid", "pname", "prio", "cpu", "mem", "thr", ""};
                            for(int i=0; i<10; i++) colnames[i] = names[i];
                            nbcol = 8;
                        }
                        break;
                    case PROCCTRL_DISPLAYMODE_TRIGGER:
                        {
                            static const char *names[] = {"", "idx", "status", "pid", "pname", "tmode", "tstream", "tsem", "tcnt", "tmiss"};
                            for(int i=0; i<10; i++) colnames[i] = names[i];
                            nbcol = 9;
                        }
                        break;
                    case PROCCTRL_DISPLAYMODE_TIMING:
                        {
                            static const char *names[] = {"", "idx", "status", "pid", "pname", "freq", "exec", "over", "", ""};
                            for(int i=0; i<10; i++) colnames[i] = names[i];
                            nbcol = 7;
                        }
                        break;
                    case PROCCTRL_DISPLAYMODE_PROCINFO:
                        {
                            static const char *names[] = {"", "idx", "status", "pid", "pname", "RT", "loopMax", "trig", "timeout", "timing"};
                            for(int i=0; i<10; i++) colnames[i] = names[i];
                            nbcol = 9;
                        }
                        break;
                }

                if (nbcol > 0) {
                    int m = ctx->procinfoproc->DisplayMode;
                    if (ctx->procinfoproc->selected_col > nbcol) ctx->procinfoproc->selected_col = nbcol;
                    if (ctx->procinfoproc->selected_col < 1) ctx->procinfoproc->selected_col = 1;

                    for(int i=1; i<=nbcol; i++) {
                        char colinfo[40];
                        if (i == ctx->procinfoproc->sort_col[m]) {
                            snprintf(colinfo, 40, "%d:%s%c", i, colnames[i], ctx->procinfoproc->sort_dir[m] ? 'v' : '^');
                        } else {
                            snprintf(colinfo, 40, "%d:%s", i, colnames[i]);
                        }

                        if (i == ctx->procinfoproc->selected_col) screenprint_setcolor(10);
                        if (ctx->procinfoproc->col_visible[m][i]) {
                            screenprint_setbold(); // Bold font, no background highlight
                            TUI_printfw("%s ", colinfo);
                            screenprint_unsetbold();
                        } else {
                            TUI_printfw("%s ", colinfo); // Normal text
                        }
                        if (i == ctx->procinfoproc->selected_col) screenprint_unsetcolor(10);
                    }
                    TUI_newline();
                }
                TUI_newline();
            }

            if (ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP) {
                TUI_printfw("MILK Process Control (procCTRL) - HELP");
                TUI_newline();
                TUI_printfw("Navigation:  ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("F2-F6"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Modes (CTRL, Res, Trig, Tim, PInfo)   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("UP/DN"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Selection");
                TUI_newline();
                TUI_printfw("             ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("SPACE"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Select process (batch)                ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("1-9"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw("   : Toggle Cols");
                TUI_newline();
                TUI_printfw("             ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("LEFT/RGHT"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Highlight Column                      ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("^L/^R"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Cycle Tabs");
                TUI_newline();
                TUI_printfw("Control:     ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("p"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Pause/Resume   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("^S"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Step   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("e"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Exit   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("T/K/I"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : TERM/KILL/INT");
                TUI_newline();
                TUI_printfw("             ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("r/R"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Rem Log (Sel/All)   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("z/Z"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Zero Cnt (Sel/All)");
                TUI_newline();
                TUI_printfw("Other:       ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("f"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Freeze   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("s/S"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Sort (Tab/All)   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("+/-"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Update freq   ");
                screenprint_setcolor(1); screenprint_setbold(); TUI_printfw("x"); screenprint_unsetcolor(1); screenprint_unsetbold(); TUI_printfw(" : Exit procCTRL");
                TUI_newline();
            }
            else {
                int dispindexMax = wrow - 6;

                int margin_dn = 2;
                int margin_up = 2;

                if (margin_dn >= dispindexMax) {
                    margin_dn = dispindexMax - 1;
                }
                if (margin_dn < 0) {
                    margin_dn = 0;
                }
                if (margin_up >= dispindexMax) {
                    margin_up = dispindexMax - 1;
                }
                if (margin_up < 0) {
                    margin_up = 0;
                }

                while (ctx->pindexActiveSelected - ctx->doffsetindex > dispindexMax - 1 - margin_dn) {
                    ctx->doffsetindex++;
                }

                while (ctx->pindexActiveSelected < ctx->doffsetindex + margin_up) {
                    ctx->doffsetindex--;
                }

                if (ctx->pindexActiveSelected < ctx->doffsetindex) {
                    ctx->doffsetindex = ctx->pindexActiveSelected;
                }
                if (ctx->pindexActiveSelected >= ctx->doffsetindex + dispindexMax) {
                    ctx->doffsetindex = ctx->pindexActiveSelected - dispindexMax + 1;
                }

                if (ctx->doffsetindex < 0) {
                    ctx->doffsetindex = 0;
                }

                int lastindex = ctx->doffsetindex + dispindexMax;
                if (lastindex > NBactive) {
                    lastindex = NBactive;
                }

                for(int dispindex = ctx->doffsetindex; dispindex < lastindex; dispindex++) {
                    if (dispindex == ctx->doffsetindex && ctx->doffsetindex > 0) {
                        screenprint_setbold();
                        screenprint_setcolor(3);
                        TUI_printfw("      ^^^^ %d more entries above ^^^^", (int)(ctx->doffsetindex + 1));
                        screenprint_unsetcolor(3);
                        screenprint_unsetbold();
                        TUI_newline();
                        continue;
                    }

                    if (dispindex == lastindex - 1 && lastindex < NBactive) {
                        screenprint_setbold();
                        screenprint_setcolor(3);
                        TUI_printfw("      vvvv %d more entries below vvvv", (int)(NBactive - lastindex + 1));
                        screenprint_unsetcolor(3);
                        screenprint_unsetbold();
                        TUI_newline();
                        continue;
                    }

                if (ctx->scan_shm == NULL) break;
                int pindex;
                if (ctx->procinfoproc->sort_col[m] > 0) pindex = ctx->procinfoproc->local_sorted_pindex[dispindex];
                else pindex = ctx->scan_shm->sorted_pindex[dispindex];
                
                if (pindex >= 0 && pindex < PROCESSINFOLISTSIZE && (ctx->procinfoproc->pinfommapped[pindex] || pinfolist->active[pindex] != 0)) {
                        if (pindex == ctx->pindexSelected) screenprint_setreverse();
                        
                        if (ctx->procinfoproc->selectedarray[pindex]) TUI_printfw("* "); else TUI_printfw("  ");
                        
                        int m = ctx->procinfoproc->DisplayMode;

                        // Column 1: idx
                        if (ctx->procinfoproc->col_visible[m][1]) {
                            if (ctx->procinfoproc->selected_col == 1) screenprint_setcolor(10);
                            TUI_printfw("%4d ", pindex);
                            if (ctx->procinfoproc->selected_col == 1) screenprint_unsetcolor(10);
                        }

                        // Column 2: status
                        if (ctx->procinfoproc->col_visible[m][2]) {
                            if (ctx->procinfoproc->selected_col == 2) screenprint_setcolor(10);
                            if (pinfolist->active[pindex] == 1) {
                                screenprint_setcolor(6); TUI_printfw("%-10s ", "ACTIVE"); screenprint_unsetcolor(6);
                            } else if (pinfolist->active[pindex] == 2) {
                                screenprint_setcolor(7); TUI_printfw("%-10s ", "STOPPED"); screenprint_unsetcolor(7);
                            } else if (pinfolist->active[pindex] == 3) {
                                screenprint_setcolor(8); TUI_printfw("%-10s ", "CRASHED"); screenprint_unsetcolor(8);
                            } else {
                                TUI_printfw("%-10s ", "OFF");
                            }
                            if (ctx->procinfoproc->selected_col == 2) screenprint_unsetcolor(10);
                        }
                        
                        // Column 3: pid
                        if (ctx->procinfoproc->col_visible[m][3]) {
                            if (ctx->procinfoproc->selected_col == 3) screenprint_setcolor(10);
                            pid_t pid = pinfolist->PIDarray[pindex];
                            int pid_exists = (kill(pid, 0) == 0);
                            if (!pid_exists) screenprint_setcolor(4);
                            char state = ctx->scan_shm->pinfodisp[pindex].state;
                            if (state == 0) state = ' ';
                            TUI_printfw("%7d %c ", pid, state);
                            if (!pid_exists) screenprint_unsetcolor(4);
                            if (ctx->procinfoproc->selected_col == 3) screenprint_unsetcolor(10);
                        }

                        if (ctx->scan_shm && pindex < PROCESSINFOLISTSIZE && pinfolist->active[pindex] == 1 && ctx->scan_shm->request_scan[pindex] == 0) {
                            ctx->scan_shm->request_scan[pindex] = 1;
                        }

                        if (m == PROCCTRL_DISPLAYMODE_CTRL) {
                            // 4: tstart
                            if (ctx->procinfoproc->col_visible[m][4]) {
                                if (ctx->procinfoproc->selected_col == 4) screenprint_setcolor(10);
                                char tbuf[30];
                                time_t sec = (time_t)pinfolist->createtime[pindex];
                                long usec = (long)((pinfolist->createtime[pindex] - sec) * 1000000);
                                struct tm *tm_info = gmtime(&sec);
                                strftime(tbuf, 20, "%Y%m%dT%H:%M:%S", tm_info);
                                snprintf(tbuf + 19,
                                         sizeof(tbuf) - 19,
                                         ".%06ld", usec);
                                TUI_printfw("%s ", tbuf);
                                if (ctx->procinfoproc->selected_col == 4) screenprint_unsetcolor(10);
                            }

                            // 5: pname
                            if (ctx->procinfoproc->col_visible[m][5]) {
                                if (ctx->procinfoproc->selected_col == 5) screenprint_setcolor(10);
                                TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
                                if (ctx->procinfoproc->selected_col == 5) screenprint_unsetcolor(10);
                            }

                            int   ctrlval = (ctx->procinfoproc->pinfoarray[pindex]) ? ctx->procinfoproc->pinfoarray[pindex]->CTRLval : 0;
                            long  loopcnt = ctx->scan_shm->pinfodisp[pindex].loopcnt;
                            char *desc    = ctx->scan_shm->pinfodisp[pindex].statusmsg;

                            // 6: state
                            if (ctx->procinfoproc->col_visible[m][6]) {
                                if (ctx->procinfoproc->selected_col == 6) screenprint_setcolor(10);
                                if (ctrlval == 1) {
                                     screenprint_setcolor(3); screenprint_setblink(); TUI_printfw("C1"); screenprint_unsetcolor(3); screenprint_unsetblink(); TUI_printfw(" ");
                                } else {
                                     TUI_printfw("C%d ", ctrlval); 
                                }
                                if (ctx->procinfoproc->selected_col == 6) screenprint_unsetcolor(10);
                            }

                            // 7: lcnt
                            if (ctx->procinfoproc->col_visible[m][7]) {
                                if (ctx->procinfoproc->selected_col == 7) screenprint_setcolor(10);
                                if (loopcnt != ctx->procinfoproc->loopcntarray[pindex]) {
                                     screenprint_setcolor(6); TUI_printfw("%10ld ", loopcnt); screenprint_unsetcolor(6);
                                } else {
                                     TUI_printfw("%10ld ", loopcnt); 
                                }
                                ctx->procinfoproc->loopcntarray[pindex] = loopcnt;
                                if (ctx->procinfoproc->selected_col == 7) screenprint_unsetcolor(10);
                            }

                            // 8: msg
                            if (ctx->procinfoproc->col_visible[m][8]) {
                                if (ctx->procinfoproc->selected_col == 8) screenprint_setcolor(10);
                                TUI_printfw("%-30s ", desc); 
                                if (ctx->procinfoproc->selected_col == 8) screenprint_unsetcolor(10);
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_RESOURCES) {
                            // 4: pname
                            if (ctx->procinfoproc->col_visible[m][4]) {
                                if (ctx->procinfoproc->selected_col == 4) screenprint_setcolor(10);
                                TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
                                if (ctx->procinfoproc->selected_col == 4) screenprint_unsetcolor(10);
                            }
                            // 5: prio
                            if (ctx->procinfoproc->col_visible[m][5]) {
                                if (ctx->procinfoproc->selected_col == 5) screenprint_setcolor(10);
                                TUI_printfw("P%2d ", ctx->scan_shm->pinfodisp[pindex].rt_priority); 
                                if (ctx->procinfoproc->selected_col == 5) screenprint_unsetcolor(10);
                            }
                            // 6: cpu
                            if (ctx->procinfoproc->col_visible[m][6]) {
                                if (ctx->procinfoproc->selected_col == 6) screenprint_setcolor(10);
                                TUI_printfw("CPU:%5.1f%% ", ctx->scan_shm->pinfodisp[pindex].subprocCPUloadarray_timeaveraged[0]);
                                if (ctx->procinfoproc->selected_col == 6) screenprint_unsetcolor(10);
                            }
                            // 7: mem
                            if (ctx->procinfoproc->col_visible[m][7]) {
                                if (ctx->procinfoproc->selected_col == 7) screenprint_setcolor(10);
                                TUI_printfw("MEM:%7ldkB ", ctx->scan_shm->pinfodisp[pindex].VmRSSarray[0]);
                                if (ctx->procinfoproc->selected_col == 7) screenprint_unsetcolor(10);
                            }
                            // 8: thr
                            if (ctx->procinfoproc->col_visible[m][8]) {
                                if (ctx->procinfoproc->selected_col == 8) screenprint_setcolor(10);
                                TUI_printfw("Thr:%3d ", ctx->scan_shm->pinfodisp[pindex].threads);
                                if (ctx->procinfoproc->selected_col == 8) screenprint_unsetcolor(10);
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_TRIGGER) {
                            // 4: pname
                            if (ctx->procinfoproc->col_visible[m][4]) {
                                if (ctx->procinfoproc->selected_col == 4) screenprint_setcolor(10);
                                TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
                                if (ctx->procinfoproc->selected_col == 4) screenprint_unsetcolor(10);
                            }
                            // 5: tmode
                            if (ctx->procinfoproc->col_visible[m][5]) {
                                if (ctx->procinfoproc->selected_col == 5) screenprint_setcolor(10);
                                TUI_printfw("TR:%d ", ctx->scan_shm->pinfodisp[pindex].triggermode);
                                if (ctx->procinfoproc->selected_col == 5) screenprint_unsetcolor(10);
                            }
                            // 6: tstream
                            if (ctx->procinfoproc->col_visible[m][6]) {
                                if (ctx->procinfoproc->selected_col == 6) screenprint_setcolor(10);
                                TUI_printfw("%-15s ", ctx->scan_shm->pinfodisp[pindex].triggerstreamname);
                                if (ctx->procinfoproc->selected_col == 6) screenprint_unsetcolor(10);
                            }
                            // 7: tsem
                            if (ctx->procinfoproc->col_visible[m][7]) {
                                if (ctx->procinfoproc->selected_col == 7) screenprint_setcolor(10);
                                TUI_printfw("S:%d ", ctx->scan_shm->pinfodisp[pindex].triggersem);
                                if (ctx->procinfoproc->selected_col == 7) screenprint_unsetcolor(10);
                            }
                            // 8: tcnt
                            if (ctx->procinfoproc->col_visible[m][8]) {
                                if (ctx->procinfoproc->selected_col == 8) screenprint_setcolor(10);
                                TUI_printfw("CNT:%ld ", (long)ctx->scan_shm->pinfodisp[pindex].triggerstreamcnt);
                                if (ctx->procinfoproc->selected_col == 8) screenprint_unsetcolor(10);
                            }
                            // 9: tmiss
                            if (ctx->procinfoproc->col_visible[m][9]) {
                                if (ctx->procinfoproc->selected_col == 9) screenprint_setcolor(10);
                                TUI_printfw("M:%d/%ld ", ctx->scan_shm->pinfodisp[pindex].triggermissedframe, (long)ctx->scan_shm->pinfodisp[pindex].triggermissedframe_cumul);
                                if (ctx->procinfoproc->selected_col == 9) screenprint_unsetcolor(10);
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_TIMING) {
                            // 4: pname
                            if (ctx->procinfoproc->col_visible[m][4]) {
                                if (ctx->procinfoproc->selected_col == 4) screenprint_setcolor(10);
                                TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
                                if (ctx->procinfoproc->selected_col == 4) screenprint_unsetcolor(10);
                            }
                            if (ctx->scan_shm->pinfodisp[pindex].MeasureTiming) {
                                double freq = 0.0;
                                if (ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns > 0) freq = 1.0e9 / ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns;
                                double exec_ms = 1.0e-6 * ctx->scan_shm->pinfodisp[pindex].dtmedian_exec_ns;
                                double overhead = 0.0;
                                if (ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns > 0) overhead = 100.0 * ctx->scan_shm->pinfodisp[pindex].dtmedian_exec_ns / ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns;
                                // 5: freq
                                if (ctx->procinfoproc->col_visible[m][5]) {
                                    if (ctx->procinfoproc->selected_col == 5) screenprint_setcolor(10);
                                    TUI_printfw("%8.2fHz ", freq);
                                    if (ctx->procinfoproc->selected_col == 5) screenprint_unsetcolor(10);
                                }
                                // 6: exec
                                if (ctx->procinfoproc->col_visible[m][6]) {
                                    if (ctx->procinfoproc->selected_col == 6) screenprint_setcolor(10);
                                    TUI_printfw("Exec:%7.3fms ", exec_ms);
                                    if (ctx->procinfoproc->selected_col == 6) screenprint_unsetcolor(10);
                                }
                                // 7: over
                                if (ctx->procinfoproc->col_visible[m][7]) {
                                    if (ctx->procinfoproc->selected_col == 7) screenprint_setcolor(10);
                                    TUI_printfw("(%5.1f%%) ", overhead);
                                    if (ctx->procinfoproc->selected_col == 7) screenprint_unsetcolor(10);
                                }
                            } else {
                                TUI_printfw("--- Timing Disabled ---");
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_PROCINFO) {
                            // 4: pname
                            if (ctx->procinfoproc->col_visible[m][4]) {
                                if (ctx->procinfoproc->selected_col == 4) screenprint_setcolor(10);
                                TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
                                if (ctx->procinfoproc->selected_col == 4) screenprint_unsetcolor(10);
                            }
                            // 5: RT
                            if (ctx->procinfoproc->col_visible[m][5]) {
                                if (ctx->procinfoproc->selected_col == 5) screenprint_setcolor(10);
                                TUI_printfw("RT:%2d ", ctx->scan_shm->pinfodisp[pindex].rt_priority);
                                if (ctx->procinfoproc->selected_col == 5) screenprint_unsetcolor(10);
                            }
                            // 6: loopMax
                            if (ctx->procinfoproc->col_visible[m][6]) {
                                if (ctx->procinfoproc->selected_col == 6) screenprint_setcolor(10);
                                TUI_printfw("Lmax:%7ld ", ctx->scan_shm->pinfodisp[pindex].loopcntMax);
                                if (ctx->procinfoproc->selected_col == 6) screenprint_unsetcolor(10);
                            }
                            // 7: trig
                            if (ctx->procinfoproc->col_visible[m][7]) {
                                if (ctx->procinfoproc->selected_col == 7) screenprint_setcolor(10);
                                TUI_printfw("Trig:%d ", ctx->scan_shm->pinfodisp[pindex].triggermode);
                                if (ctx->procinfoproc->selected_col == 7) screenprint_unsetcolor(10);
                            }
                            // 8: timeout
                            if (ctx->procinfoproc->col_visible[m][8]) {
                                if (ctx->procinfoproc->selected_col == 8) screenprint_setcolor(10);
                                double tout = ctx->scan_shm->pinfodisp[pindex].triggertimeout.tv_sec + 1e-9 * ctx->scan_shm->pinfodisp[pindex].triggertimeout.tv_nsec;
                                TUI_printfw("TO:%5.2f ", tout);
                                if (ctx->procinfoproc->selected_col == 8) screenprint_unsetcolor(10);
                            }
                            // 9: timing
                            if (ctx->procinfoproc->col_visible[m][9]) {
                                if (ctx->procinfoproc->selected_col == 9) screenprint_setcolor(10);
                                TUI_printfw("Tim:%d ", ctx->scan_shm->pinfodisp[pindex].MeasureTiming);
                                if (ctx->procinfoproc->selected_col == 9) screenprint_unsetcolor(10);
                            }
                        }
                        else {
                            TUI_printfw("(Mode %d not impl)", m);
                        }

                        if (pindex == ctx->pindexSelected) screenprint_unsetreverse();
                        TUI_newline();
                    }
                }
            }
            TUI_cleartobottom();
            sc_frame_flush();
        }
}


errno_t processinfo_CTRLscreen()
{
    if (getenv("PROCCTRL_DEBUG")) procCTRL_debug_mode = 1;
    if (procCTRL_debug_mode) printf("DEBUG: processinfo_CTRLscreen start\n");

    procctrl_context_t ctx;
    memset(&ctx, 0, sizeof(procctrl_context_t));

    if (strlen(procCTRL_logfile) > 0) {
        ctx.flog = fopen(procCTRL_logfile, "a");
        if (ctx.flog) {
            fprintf(ctx.flog, "\n--- processinfo_CTRLscreen started ---\n");
            fflush(ctx.flog);
        }
    }

    if (ctx.flog) { fprintf(ctx.flog, "Checking for daemon...\n"); fflush(ctx.flog); }
    if (system("pgrep \"milk-procCTRL-s\" > /dev/null") != 0) {
        fprintf(stderr, "\nWARNING: milk-procCTRL-scan daemon is not running.\n");
        printf("Start it now in tmux session 'milk-procCTRL-scan'? [y/n] ");
        fflush(stdout);
        char response = 'n';
        if (scanf(" %c", &response) == 1 && (response == 'y' || response == 'Y')) {
            printf("Launching milk-procCTRL-scan...\n");
            if(system("tmux new-session -d -s milk-procCTRL-scan 'milk-procCTRL-scan'") < 0) {}
            sleep(1);
            if (system("pgrep \"milk-procCTRL-s\" > /dev/null") != 0) {
                fprintf(stderr, "ERROR: Failed to launch milk-procCTRL-scan daemon.\n");
                if (ctx.flog) fclose(ctx.flog);
                return RETURN_FAILURE;
            }
        } else {
            fprintf(stderr, "ERROR: milk-procCTRL-scan daemon is required for this tool.\n");
            if (ctx.flog) fclose(ctx.flog);
            return RETURN_FAILURE;
        }
    }

    processinfo_procdirname(local_shmdir);
    processinfo_procdirname(ctx.procdname);

    if (ctx.flog) { fprintf(ctx.flog, "Allocating procinfoproc...\n"); fflush(ctx.flog); }
    ctx.procinfoproc = (PROCINFOPROC *) calloc(1, sizeof(PROCINFOPROC));
    if(ctx.procinfoproc == NULL) {
        PRINT_ERROR("calloc returns NULL pointer");
        if (ctx.flog) fclose(ctx.flog);
        return RETURN_FAILURE;
    }

    ctx.procinfoproc->NBcpus = GetNumberCPUs(ctx.procinfoproc);
    GetCPUloads(ctx.procinfoproc); 

    if(system("which cset > /dev/null 2>&1") == 0) ctx.procinfoproc->has_cset = 1;
    else ctx.procinfoproc->has_cset = 0;

    for(int m=0; m<10; m++) {
        for(int i=0; i<10; i++) ctx.procinfoproc->col_visible[m][i] = 1;
        ctx.procinfoproc->sort_col[m] = 0;
        ctx.procinfoproc->sort_dir[m] = 0;
        ctx.procinfoproc->sort_mode[m] = m;
    }
    ctx.procinfoproc->selected_col = 1;

    STRINGLISTENTRY *CPUsetList = (STRINGLISTENTRY *) malloc(sizeof(STRINGLISTENTRY) * 1000);
    int NBCPUset __attribute__((unused)) = processinfo_CPUsets_List(CPUsetList, ctx.procinfoproc->has_cset);

    if (ctx.flog) { fprintf(ctx.flog, "Connecting to process list...\n"); fflush(ctx.flog); }
    if(processinfo_shm_list_create() == -1) {
        printf("==== ERROR: CANNOT ACCESS PROCESS LIST ====\n");
        if (ctx.flog) fclose(ctx.flog);
        return RETURN_FAILURE;
    }
    ctx.procinfoproc->pinfolist = pinfolist;
    
    char scan_shm_name[STRINGMAXLEN_FULLFILENAME];
    snprintf(scan_shm_name, sizeof(scan_shm_name), "%s/%s", ctx.procdname, PROCESSINFO_SCAN_SHM_NAME);
    if (ctx.flog) { fprintf(ctx.flog, "Linking to scan SHM: %s\n", scan_shm_name); fflush(ctx.flog); }
    ctx.scan_shm = (PROCSCAN_SHM *) link_scan_shm(scan_shm_name, sizeof(PROCSCAN_SHM));
    if (ctx.scan_shm == MAP_FAILED) {
        printf("WARNING: Could not link to scan SHM %s. Stats may be missing.\n", scan_shm_name);
        ctx.scan_shm = NULL;
    }

    TUI_set_screenprintmode(SCREENPRINT_NCURSES);
    if(getenv("MILK_TUIPRINT_STDIO")) TUI_set_screenprintmode(SCREENPRINT_STDIO);
    
    if (ctx.flog) { fprintf(ctx.flog, "Initializing terminal...\n"); fflush(ctx.flog); }
    ansi_raw_mode_enter();
    TUI_init_terminal(&wrow, &wcol);
    if(wrow < 10) wrow = 10;

    if (ctx.flog) { fprintf(ctx.flog, "Allocating pinfodisp buffer (250MB)\n"); fflush(ctx.flog); }
    ctx.procinfoproc->pinfodisp = (PROCESSINFODISP *) calloc(PROCESSINFOLISTSIZE, sizeof(PROCESSINFODISP));
    if (ctx.procinfoproc->pinfodisp == NULL) {
        ansi_raw_mode_exit();
        fprintf(stderr, "FATAL ERROR: Could not allocate 250MB process info buffer.\n");
        if (ctx.flog) fclose(ctx.flog);
        return RETURN_FAILURE;
    }
    ctx.procinfoproc->NBpinfodisp = PROCESSINFOLISTSIZE;
    
    ctx.procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
    ctx.procinfoproc->loop = 1;
    ctx.procinfoproc->twaitus = 1000000;

    int backstderr = -1, newstderr = -1;
    if (procCTRL_debug_mode == 0) {
        fflush(stderr);
        backstderr = dup(STDERR_FILENO);
        newstderr  = open("/dev/null", O_WRONLY);
        dup2(newstderr, STDERR_FILENO);
        close(newstderr);
    }

    ctx.t_last_scan.tv_sec = 0; ctx.t_last_scan.tv_nsec = 0;
    ctx.frequ = 32.0;
    ctx.pindexSelected = -1;
    ctx.pindexActiveSelected = 0;
    ctx.doffsetindex = 0;
    ctx.freeze = 0;
    ctx.loopOK = 1;
    ctx.Xexit = 0;
    ctx.monstringlen = 200;
    ctx.last_ch = -1;

    getrusage(RUSAGE_SELF, &ctx.usage_prev);
    clock_gettime(CLOCK_MONOTONIC, &ctx.t_usage_prev);
    ctx.tool_cpu_pcnt = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &ctx.t_disp_prev);
    ctx.actual_fps = 0.0;

    if (ctx.flog) { fprintf(ctx.flog, "Entering main loop.\n"); fflush(ctx.flog); }
    sc_frame_clear();

    while(ctx.loopOK) {
        if(processinfo_signal_SEGV) { 
            if (ctx.flog) { fprintf(ctx.flog, "SEGV signal received!\n"); fflush(ctx.flog); }
            ctx.loopOK=0; break; 
        }

        procctrl_update_stats(&ctx);

        usleep((long)(1000000.0 / ctx.frequ));
        int ch = get_singlechar_nonblock();
        
        int NBactive = (ctx.scan_shm) ? ctx.scan_shm->NBactive : 0;
        int m = ctx.procinfoproc->DisplayMode;
        
        if (ctx.procinfoproc->sort_col[m] > 0 && ctx.scan_shm != NULL) {
            sort_ctx_m = ctx.procinfoproc->sort_mode[m];
            sort_ctx_col = ctx.procinfoproc->sort_col[m];
            sort_ctx_dir = ctx.procinfoproc->sort_dir[m];
            sort_ctx_scan_shm = ctx.scan_shm;
            sort_ctx_pinfolist = pinfolist;
            
            for(int i=0; i<NBactive; i++) ctx.procinfoproc->local_sorted_pindex[i] = ctx.scan_shm->sorted_pindex[i];
            qsort(ctx.procinfoproc->local_sorted_pindex, NBactive, sizeof(int), proc_comp);
        }

        if (ctx.pindexActiveSelected >= NBactive && NBactive > 0) ctx.pindexActiveSelected = NBactive - 1;
        
        if (NBactive > 0 && ctx.scan_shm != NULL) {
            if (ctx.procinfoproc->sort_col[m] > 0) ctx.pindexSelected = ctx.procinfoproc->local_sorted_pindex[ctx.pindexActiveSelected];
            else ctx.pindexSelected = ctx.scan_shm->sorted_pindex[ctx.pindexActiveSelected];
        }
        else ctx.pindexSelected = -1;

        if (ch != -1) {
            procctrl_handle_keyboard_event(&ctx, ch, NBactive);
        }

        procctrl_render_frame(&ctx, NBactive);
    }

    ansi_raw_mode_exit();
    if (ctx.scan_shm) munmap(ctx.scan_shm, sizeof(PROCSCAN_SHM));
    for(long i=0; i<PROCESSINFOLISTSIZE; i++) if(ctx.procinfoproc->pinfommapped[i]) {
        if (ctx.procinfoproc->pinfoarray[i] != NULL && ctx.procinfoproc->pinfoarray[i] != (PROCESSINFO*)MAP_FAILED)
            processinfo_shm_close(ctx.procinfoproc->pinfoarray[i], ctx.procinfoproc->fdarray[i]);
    }
    free(ctx.procinfoproc->pinfodisp);
    free(ctx.procinfoproc);
    free(CPUsetList);

    if (procCTRL_debug_mode == 0) {
        fflush(stderr);
        dup2(backstderr, STDERR_FILENO);
        close(backstderr);
    }
    if (ctx.flog) { fprintf(ctx.flog, "--- processinfo_CTRLscreen ended ---\n"); fclose(ctx.flog); }
    return RETURN_SUCCESS;
}

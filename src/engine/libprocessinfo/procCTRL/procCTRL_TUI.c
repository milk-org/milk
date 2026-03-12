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
#include <sys/file.h>
#include <sys/stat.h>

#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <math.h>

#include <sys/types.h>
#include <unistd.h>

#include <ctype.h>
#include <fcntl.h>
#include <ncurses.h>
#include <sys/mman.h>

#include <dirent.h>

#include <locale.h>
#include <wchar.h>

#include <pthread.h>

#include "timeutils.h"
#include "quicksort.h"

#include "TUItools.h"
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
    
    char fname[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fname, "%s/.csetlist.%ld", SHAREDPROCDIR, (long) getpid());
    
    char cmd[1024];
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
        sscanf(line, "%s %s", word, word1);
        strcpy(CPUsetList[setindex].name, word);
        strcpy(CPUsetList[setindex].description, word1);
        setindex++;
        NBset++;
    }
    fclose(fp);
    remove(fname);
    return NBset;
}

static int processinfo_SelectFromList(STRINGLISTENTRY *StringList, int NBelem)
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

static int sort_ctx_m = 0;
static int sort_ctx_col = 0;
static int sort_ctx_dir = 0;
static PROCSCAN_SHM *sort_ctx_scan_shm = NULL;
static PROCESSINFOLIST *sort_ctx_pinfolist = NULL;

static int proc_comp(const void *a, const void *b) {
    int idx1 = *(const int *)a;
    int idx2 = *(const int *)b;
    int res = 0;

    if (!sort_ctx_scan_shm || !sort_ctx_pinfolist) return 0;

    PROCESSINFODISP *p1 = &sort_ctx_scan_shm->pinfodisp[idx1];
    PROCESSINFODISP *p2 = &sort_ctx_scan_shm->pinfodisp[idx2];

    switch(sort_ctx_col) {
        case 1: // idx
            res = (idx1 < idx2) ? -1 : (idx1 > idx2) ? 1 : 0;
            break;
        case 2: // status
            res = (sort_ctx_pinfolist->active[idx1] < sort_ctx_pinfolist->active[idx2]) ? -1 : 
                  (sort_ctx_pinfolist->active[idx1] > sort_ctx_pinfolist->active[idx2]) ? 1 : 0;
            break;
        case 3: // pid
            res = (sort_ctx_pinfolist->PIDarray[idx1] < sort_ctx_pinfolist->PIDarray[idx2]) ? -1 : 
                  (sort_ctx_pinfolist->PIDarray[idx1] > sort_ctx_pinfolist->PIDarray[idx2]) ? 1 : 0;
            break;
        default:
            if (sort_ctx_m == PROCCTRL_DISPLAYMODE_CTRL) {
                switch(sort_ctx_col) {
                    case 4: // tstart
                        res = (sort_ctx_pinfolist->createtime[idx1] < sort_ctx_pinfolist->createtime[idx2]) ? -1 : 
                              (sort_ctx_pinfolist->createtime[idx1] > sort_ctx_pinfolist->createtime[idx2]) ? 1 : 0;
                        break;
                    case 5: // pname
                        res = strcmp(sort_ctx_pinfolist->pnamearray[idx1], sort_ctx_pinfolist->pnamearray[idx2]);
                        break;
                    case 6: // state (ctrlval)
                        {
                            int v1 = p1->loopstat; 
                            int v2 = p2->loopstat;
                            res = (v1 < v2) ? -1 : (v1 > v2) ? 1 : 0;
                        }
                        break;
                    case 7: // lcnt
                        res = (p1->loopcnt < p2->loopcnt) ? -1 : (p1->loopcnt > p2->loopcnt) ? 1 : 0;
                        break;
                    case 8: // msg
                        res = strcmp(p1->statusmsg, p2->statusmsg);
                        break;
                }
            } else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_RESOURCES) {
                switch(sort_ctx_col) {
                    case 4: // pname
                        res = strcmp(sort_ctx_pinfolist->pnamearray[idx1], sort_ctx_pinfolist->pnamearray[idx2]);
                        break;
                    case 5: // prio
                        res = (p1->rt_priority < p2->rt_priority) ? -1 : (p1->rt_priority > p2->rt_priority) ? 1 : 0;
                        break;
                    case 6: // cpu
                        res = (p1->subprocCPUloadarray_timeaveraged[0] < p2->subprocCPUloadarray_timeaveraged[0]) ? -1 : 
                              (p1->subprocCPUloadarray_timeaveraged[0] > p2->subprocCPUloadarray_timeaveraged[0]) ? 1 : 0;
                        break;
                    case 7: // mem
                        res = (p1->VmRSSarray[0] < p2->VmRSSarray[0]) ? -1 : 
                              (p1->VmRSSarray[0] > p2->VmRSSarray[0]) ? 1 : 0;
                        break;
                    case 8: // thr
                        res = (p1->threads < p2->threads) ? -1 : 
                              (p1->threads > p2->threads) ? 1 : 0;
                        break;
                }
            } else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_TRIGGER) {
                switch(sort_ctx_col) {
                    case 4: // pname
                        res = strcmp(sort_ctx_pinfolist->pnamearray[idx1], sort_ctx_pinfolist->pnamearray[idx2]);
                        break;
                    case 5: // tmode
                        res = (p1->triggermode < p2->triggermode) ? -1 : 
                              (p1->triggermode > p2->triggermode) ? 1 : 0;
                        break;
                    case 6: // tstream
                        res = strcmp(p1->triggerstreamname, p2->triggerstreamname);
                        break;
                    case 7: // tsem
                        res = (p1->triggersem < p2->triggersem) ? -1 : 
                              (p1->triggersem > p2->triggersem) ? 1 : 0;
                        break;
                    case 8: // tcnt
                        res = (p1->triggerstreamcnt < p2->triggerstreamcnt) ? -1 : 
                              (p1->triggerstreamcnt > p2->triggerstreamcnt) ? 1 : 0;
                        break;
                    case 9: // tmiss
                        res = (p1->triggermissedframe_cumul < p2->triggermissedframe_cumul) ? -1 : 
                              (p1->triggermissedframe_cumul > p2->triggermissedframe_cumul) ? 1 : 0;
                        break;
                }
            } else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_TIMING) {
                switch(sort_ctx_col) {
                    case 4: // pname
                        res = strcmp(sort_ctx_pinfolist->pnamearray[idx1], sort_ctx_pinfolist->pnamearray[idx2]);
                        break;
                    case 5: // freq
                        {
                            double f1 = (p1->dtmedian_iter_ns > 0) ? 1.0e9 / p1->dtmedian_iter_ns : 0;
                            double f2 = (p2->dtmedian_iter_ns > 0) ? 1.0e9 / p2->dtmedian_iter_ns : 0;
                            res = (f1 < f2) ? -1 : (f1 > f2) ? 1 : 0;
                        }
                        break;
                    case 6: // exec
                        res = (p1->dtmedian_exec_ns < p2->dtmedian_exec_ns) ? -1 : 
                              (p1->dtmedian_exec_ns > p2->dtmedian_exec_ns) ? 1 : 0;
                        break;
                    case 7: // over
                        {
                            double o1 = (p1->dtmedian_iter_ns > 0) ? 100.0 * p1->dtmedian_exec_ns / p1->dtmedian_iter_ns : 0;
                            double o2 = (p2->dtmedian_iter_ns > 0) ? 100.0 * p2->dtmedian_exec_ns / p2->dtmedian_iter_ns : 0;
                            res = (o1 < o2) ? -1 : (o1 > o2) ? 1 : 0;
                        }
                        break;
                }
            } else if (sort_ctx_m == PROCCTRL_DISPLAYMODE_PROCINFO) {
                switch(sort_ctx_col) {
                    case 4: // pname
                        res = strcmp(sort_ctx_pinfolist->pnamearray[idx1], sort_ctx_pinfolist->pnamearray[idx2]);
                        break;
                    case 5: // RT
                        res = (p1->rt_priority < p2->rt_priority) ? -1 : (p1->rt_priority > p2->rt_priority) ? 1 : 0;
                        break;
                    case 6: // loopMax
                        res = (p1->loopcntMax < p2->loopcntMax) ? -1 : (p1->loopcntMax > p2->loopcntMax) ? 1 : 0;
                        break;
                    case 7: // trig
                        res = (p1->triggermode < p2->triggermode) ? -1 : (p1->triggermode > p2->triggermode) ? 1 : 0;
                        break;
                    case 8: // timeout
                        {
                            double t1 = p1->triggertimeout.tv_sec + 1e-9 * p1->triggertimeout.tv_nsec;
                            double t2 = p2->triggertimeout.tv_sec + 1e-9 * p2->triggertimeout.tv_nsec;
                            res = (t1 < t2) ? -1 : (t1 > t2) ? 1 : 0;
                        }
                        break;
                    case 9: // timing
                        res = (p1->MeasureTiming < p2->MeasureTiming) ? -1 : (p1->MeasureTiming > p2->MeasureTiming) ? 1 : 0;
                        break;
                }
            }
            break;
    }

    if (sort_ctx_dir == 1) res = -res;
    return res;
}

static inline void *link_scan_shm(const char *name, size_t size) {
    int fd = open(name, O_RDWR);
    if (fd == -1) return MAP_FAILED;
    void *ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return ptr;
}

#include <sys/resource.h>
#include <ctype.h>

errno_t processinfo_CTRLscreen()
{
    if (getenv("PROCCTRL_DEBUG")) procCTRL_debug_mode = 1;
    if (procCTRL_debug_mode) printf("DEBUG: processinfo_CTRLscreen start\n");

    FILE *flog = NULL;
    if (strlen(procCTRL_logfile) > 0) {
        flog = fopen(procCTRL_logfile, "a");
        if (flog) {
            fprintf(flog, "\n--- processinfo_CTRLscreen started ---\n");
            fflush(flog);
        }
    }

    if (flog) { fprintf(flog, "Checking for daemon...\n"); fflush(flog); }
    if (system("pgrep \"milk-procCTRL-s\" > /dev/null") != 0) {
        fprintf(stderr, "\nWARNING: milk-procCTRL-scan daemon is not running.\n");
        printf("Start it now in tmux session 'milk-procCTRL-scan'? [y/n] ");
        fflush(stdout);
        char response = 'n';
        if (scanf(" %c", &response) == 1 && (response == 'y' || response == 'Y')) {
            printf("Launching milk-procCTRL-scan...\n");
            system("tmux new-session -d -s milk-procCTRL-scan 'milk-procCTRL-scan'");
            sleep(1);
            if (system("pgrep \"milk-procCTRL-s\" > /dev/null") != 0) {
                fprintf(stderr, "ERROR: Failed to launch milk-procCTRL-scan daemon.\n");
                if (flog) fclose(flog);
                return RETURN_FAILURE;
            }
        } else {
            fprintf(stderr, "ERROR: milk-procCTRL-scan daemon is required for this tool.\n");
            if (flog) fclose(flog);
            return RETURN_FAILURE;
        }
    }

    processinfo_procdirname(local_shmdir);
    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    if (flog) { fprintf(flog, "Allocating procinfoproc...\n"); fflush(flog); }
    PROCINFOPROC *procinfoproc = (PROCINFOPROC *) calloc(1, sizeof(PROCINFOPROC));
    if(procinfoproc == NULL) {
        PRINT_ERROR("calloc returns NULL pointer");
        if (flog) fclose(flog);
        return RETURN_FAILURE;
    }

    procinfoproc->NBcpus = GetNumberCPUs(procinfoproc);
    GetCPUloads(procinfoproc); 

    if(system("which cset > /dev/null 2>&1") == 0) procinfoproc->has_cset = 1;
    else procinfoproc->has_cset = 0;

    // Initialize column visibility and sorting
    for(int m=0; m<10; m++) {
        for(int i=0; i<10; i++) procinfoproc->col_visible[m][i] = 1;
        procinfoproc->sort_col[m] = 0;
        procinfoproc->sort_dir[m] = 0;
        procinfoproc->sort_mode[m] = m;
    }
    procinfoproc->selected_col = 1;

    STRINGLISTENTRY *CPUsetList = (STRINGLISTENTRY *) malloc(sizeof(STRINGLISTENTRY) * 1000);
    int NBCPUset = processinfo_CPUsets_List(CPUsetList, procinfoproc->has_cset);

    if (flog) { fprintf(flog, "Connecting to process list...\n"); fflush(flog); }
    if(processinfo_shm_list_create() == -1) {
        printf("==== ERROR: CANNOT ACCESS PROCESS LIST ====\n");
        if (flog) fclose(flog);
        return RETURN_FAILURE;
    }
    procinfoproc->pinfolist = pinfolist;
    
    char scan_shm_name[STRINGMAXLEN_FULLFILENAME];
    snprintf(scan_shm_name, sizeof(scan_shm_name), "%s/%s", procdname, PROCESSINFO_SCAN_SHM_NAME);
    if (flog) { fprintf(flog, "Linking to scan SHM: %s\n", scan_shm_name); fflush(flog); }
    PROCSCAN_SHM *scan_shm = (PROCSCAN_SHM *) link_scan_shm(scan_shm_name, sizeof(PROCSCAN_SHM));
    if (scan_shm == MAP_FAILED) {
        printf("WARNING: Could not link to scan SHM %s. Stats may be missing.\n", scan_shm_name);
        scan_shm = NULL;
    }

    TUI_set_screenprintmode(SCREENPRINT_NCURSES);
    if(getenv("MILK_TUIPRINT_STDIO")) TUI_set_screenprintmode(SCREENPRINT_STDIO);
    
    if (flog) { fprintf(flog, "Initializing terminal...\n"); fflush(flog); } // Corrected 'frow' to 'flog'
    TUI_init_terminal(&wrow, &wcol);
    if(wrow < 10) wrow = 10;

    if (flog) { fprintf(flog, "Allocating pinfodisp buffer (250MB)\n"); fflush(flog); }
    procinfoproc->pinfodisp = (PROCESSINFODISP *) calloc(PROCESSINFOLISTSIZE, sizeof(PROCESSINFODISP));
    if (procinfoproc->pinfodisp == NULL) {
        TUI_exit();
        fprintf(stderr, "FATAL ERROR: Could not allocate 250MB process info buffer.\n");
        if (flog) fclose(flog);
        return RETURN_FAILURE;
    }
    procinfoproc->NBpinfodisp = PROCESSINFOLISTSIZE;
    
    procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
    procinfoproc->loop = 1;
    procinfoproc->twaitus = 1000000;

    int backstderr = -1, newstderr = -1;
    if (procCTRL_debug_mode == 0) {
        fflush(stderr);
        backstderr = dup(STDERR_FILENO);
        newstderr  = open("/dev/null", O_WRONLY);
        dup2(newstderr, STDERR_FILENO);
        close(newstderr);
    }

    struct timespec t_last_scan = {0,0}, t_now;
    float frequ = 32.0;
    int pindexSelected = -1;
    int pindexActiveSelected = 0;
    long doffsetindex = 0;
    int freeze = 0;
    int loopOK = 1;
    int Xexit = 0;
    char monstring[200];
    int monstringlen = 200;
    int last_ch = -1;

    struct rusage usage_prev, usage_cur;
    struct timespec t_usage_prev, t_usage_cur;
    getrusage(RUSAGE_SELF, &usage_prev);
    clock_gettime(CLOCK_MONOTONIC, &t_usage_prev);
    float tool_cpu_pcnt = 0.0;

    struct timespec t_disp_prev, t_disp_cur;
    clock_gettime(CLOCK_MONOTONIC, &t_disp_prev);
    float actual_fps = 0.0;

    if (flog) { fprintf(flog, "Entering main loop.\n"); fflush(flog); }
    clear();

    while(loopOK) {
        if(processinfo_signal_SEGV) { 
            if (flog) { fprintf(flog, "SEGV signal received!\n"); fflush(flog); }
            loopOK=0; break; 
        }

        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed = (t_now.tv_sec - t_last_scan.tv_sec) + (t_now.tv_nsec - t_last_scan.tv_nsec) * 1e-9;
        if (elapsed >= (procinfoproc->twaitus * 1e-6)) {
            processinfo_scan_step(procinfoproc);
            t_last_scan = t_now;
            
            getrusage(RUSAGE_SELF, &usage_cur);
            clock_gettime(CLOCK_MONOTONIC, &t_usage_cur);
            double t_diff = (t_usage_cur.tv_sec - t_usage_prev.tv_sec) + (t_usage_cur.tv_nsec - t_usage_prev.tv_nsec) * 1e-9;
            double u_diff = (usage_cur.ru_utime.tv_sec - usage_prev.ru_utime.tv_sec) + (usage_cur.ru_utime.tv_usec - usage_prev.ru_utime.tv_usec) * 1e-6;
            double s_diff = (usage_cur.ru_stime.tv_sec - usage_prev.ru_stime.tv_sec) + (usage_cur.ru_stime.tv_usec - usage_prev.ru_stime.tv_usec) * 1e-6;
            if (t_diff > 0) tool_cpu_pcnt = 100.0 * (u_diff + s_diff) / t_diff;
            usage_prev = usage_cur;
            t_usage_prev = t_usage_cur;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_disp_cur);
        double d_elapsed = (t_disp_cur.tv_sec - t_disp_prev.tv_sec) + (t_disp_cur.tv_nsec - t_disp_prev.tv_nsec) * 1e-9;
        if (d_elapsed > 0) actual_fps = 0.9 * actual_fps + 0.1 * (1.0 / d_elapsed);
        t_disp_prev = t_disp_cur;

        usleep((long)(1000000.0 / frequ));
        int ch = getch();
        
        int NBactive = (scan_shm) ? scan_shm->NBactive : 0;
        int m = procinfoproc->DisplayMode;
        
        // Sorting logic
        if (procinfoproc->sort_col[m] > 0 && scan_shm != NULL) {
            sort_ctx_m = procinfoproc->sort_mode[m];
            sort_ctx_col = procinfoproc->sort_col[m];
            sort_ctx_dir = procinfoproc->sort_dir[m];
            sort_ctx_scan_shm = scan_shm;
            sort_ctx_pinfolist = pinfolist;
            
            for(int i=0; i<NBactive; i++) procinfoproc->local_sorted_pindex[i] = scan_shm->sorted_pindex[i];
            qsort(procinfoproc->local_sorted_pindex, NBactive, sizeof(int), proc_comp);
        }

        if (pindexActiveSelected >= NBactive && NBactive > 0) pindexActiveSelected = NBactive - 1;
        
        if (NBactive > 0 && scan_shm != NULL) {
            if (procinfoproc->sort_col[m] > 0) pindexSelected = procinfoproc->local_sorted_pindex[pindexActiveSelected];
            else pindexSelected = scan_shm->sorted_pindex[pindexActiveSelected];
        }
        else pindexSelected = -1;

        if (ch != -1) {
            last_ch = ch;
            if (flog) {
                char tbuf[30];
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                struct tm *tm_info = gmtime(&ts.tv_sec);
                strftime(tbuf, 20, "%Y%m%dT%H:%M:%S", tm_info);
                sprintf(tbuf + 19, ".%06ld", ts.tv_nsec / 1000);
                fprintf(flog, "%s Input: %d\n", tbuf, ch);
                fflush(flog);
            }

            // PRIMARY INPUT HANDLER
            if (ch == 545 || ch == 560 || ch == 443 || ch == 564 || ch == 554) { // CTRL+LEFT
                 procinfoproc->DisplayMode--;
                 if (procinfoproc->DisplayMode < 1) procinfoproc->DisplayMode = 6;
                 if (flog) { fprintf(flog, "  -> Mode changed to %d\n", procinfoproc->DisplayMode); fflush(flog); }
            }
            else if (ch == 561 || ch == 566 || ch == 444 || ch == 565 || ch == 569) { // CTRL+RIGHT
                 procinfoproc->DisplayMode++;
                 if (procinfoproc->DisplayMode > 6) procinfoproc->DisplayMode = 1;
                 if (flog) { fprintf(flog, "  -> Mode changed to %d\n", procinfoproc->DisplayMode); fflush(flog); }
            }
            else if (ch >= '0' && ch <= '9') {
                 int cidx = ch - '0';
                 int m = procinfoproc->DisplayMode;
                 if (m >= 0 && m < 10)
                     procinfoproc->col_visible[m][cidx] = !procinfoproc->col_visible[m][cidx];
            }
            else if (ch == KEY_F(2)) procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
            else if (ch == KEY_F(3)) procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_RESOURCES;
            else if (ch == KEY_F(4)) procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TRIGGER;
            else if (ch == KEY_F(5)) procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TIMING;
            else if (ch == KEY_F(6)) procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_PROCINFO;
            else if (ch == 'h')      procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_HELP;
            else if (ch == 'x') { loopOK = 0; Xexit = 1; }
            else if (ch == 'f') { freeze = !freeze; }
            else if (ch == '+' || ch == '=') { frequ *= 1.2; if (frequ > 1000.0) frequ = 1000.0; }
            else if (ch == '-') { frequ /= 1.2; if (frequ < 0.1) frequ = 0.1; }
            else if (ch == ' ' && pindexSelected >= 0) {
                 procinfoproc->selectedarray[pindexSelected] = !procinfoproc->selectedarray[pindexSelected];
            }
            else if (ch == KEY_UP && NBactive > 0) {
                 pindexActiveSelected--; if (pindexActiveSelected < 0) pindexActiveSelected = 0;
            }
            else if (ch == KEY_DOWN && NBactive > 0) {
                 pindexActiveSelected++; if (pindexActiveSelected >= NBactive) pindexActiveSelected = NBactive - 1;
            }
            else if (ch == KEY_LEFT) {
                 procinfoproc->selected_col--;
                 if (procinfoproc->selected_col < 1) procinfoproc->selected_col = 9;
            }
            else if (ch == KEY_RIGHT) {
                 procinfoproc->selected_col++;
                 if (procinfoproc->selected_col > 9) procinfoproc->selected_col = 1;
            }
            else if (ch == 'r' || ch == 'R') {
                int all = (ch == 'R');
                for(int i=0; i<PROCESSINFOLISTSIZE; i++) {
                    if (all || i == pindexSelected) {
                        if (pinfolist->active[i] == 2 || pinfolist->active[i] == 3) pinfolist->active[i] = 0;
                    }
                }
            }
            else if ((ch == 'T' || ch == 'K' || ch == 'I') && NBactive > 0) {
                int sig = (ch == 'T') ? SIGTERM : (ch == 'K') ? SIGKILL : SIGINT;
                int any_sel = 0;
                for(int i=0; i<PROCESSINFOLISTSIZE; i++) if(procinfoproc->selectedarray[i]) {
                    kill(pinfolist->PIDarray[i], sig);
                    any_sel = 1;
                }
                if(!any_sel && pindexSelected >= 0) kill(pinfolist->PIDarray[pindexSelected], sig);
            }
            else if (ch == 's') {
                 int m = procinfoproc->DisplayMode;
                 procinfoproc->sort_mode[m] = m; // Use current mode logic
                 if (procinfoproc->sort_col[m] == procinfoproc->selected_col) {
                     procinfoproc->sort_dir[m] = !procinfoproc->sort_dir[m];
                 } else {
                     procinfoproc->sort_col[m] = procinfoproc->selected_col;
                     procinfoproc->sort_dir[m] = 0; // Default ascending
                 }
            }
            else if (ch == 'S') {
                 int m_curr = procinfoproc->DisplayMode;
                 int smod = procinfoproc->sort_mode[m_curr];
                 int scol = procinfoproc->sort_col[m_curr];
                 int sdir = procinfoproc->sort_dir[m_curr];
                 for(int m=0; m<10; m++) {
                     procinfoproc->sort_mode[m] = smod;
                     procinfoproc->sort_col[m] = scol;
                     procinfoproc->sort_dir[m] = sdir;
                 }
            }
            else if ((ch == 'p' || ch == 19 || ch == 'e') && NBactive > 0) { // 19 is CTRL+S
                int val = (ch == 'p') ? -1 : (ch == 19) ? 2 : 3;
                int any_sel = 0;
                for(int i=0; i<PROCESSINFOLISTSIZE; i++) if(procinfoproc->selectedarray[i]) {
                    if (val == -1 && procinfoproc->pinfoarray[i]) procinfoproc->pinfoarray[i]->CTRLval = (procinfoproc->pinfoarray[i]->CTRLval == 0) ? 1 : 0;
                    else if (procinfoproc->pinfoarray[i]) procinfoproc->pinfoarray[i]->CTRLval = val;
                    any_sel = 1;
                }
                if(!any_sel && pindexSelected >= 0 && procinfoproc->pinfoarray[pindexSelected]) {
                    if (val == -1) procinfoproc->pinfoarray[pindexSelected]->CTRLval = (procinfoproc->pinfoarray[pindexSelected]->CTRLval == 0) ? 1 : 0;
                    else procinfoproc->pinfoarray[pindexSelected]->CTRLval = val;
                }
            }
            else if ((ch == 'z' || ch == 'Z') && NBactive > 0) {
                int all = (ch == 'Z');
                for(int i=0; i<PROCESSINFOLISTSIZE; i++) if((all || procinfoproc->selectedarray[i]) && procinfoproc->pinfoarray[i]) {
                    procinfoproc->pinfoarray[i]->loopcnt = 0;
                }
                if(!all && !procinfoproc->selectedarray[pindexSelected] && pindexSelected >= 0 && procinfoproc->pinfoarray[pindexSelected]) {
                    procinfoproc->pinfoarray[pindexSelected]->loopcnt = 0;
                }
            }
            else if (ch == KEY_RESIZE) {
                clear();
            }
        }

        if (freeze == 0) {
            erase();
            snprintf(monstring, monstringlen, "Mode %d   PRESS x TO STOP MONITOR", procinfoproc->DisplayMode);
            TUI_print_header(monstring, '-');
            TUI_newline();
            
            {
                struct stat shm_stat;
                char shm_list_fname[STRINGMAXLEN_FULLFILENAME];
                WRITE_FULLFILENAME(shm_list_fname, "%s/processinfo.list.shm", procdname);
                if(stat(shm_list_fname, &shm_stat) == 0) {
                    char timestr[100];
                    struct tm *tm_info = gmtime(&shm_stat.st_mtime);
                    strftime(timestr, 100, "%Y-%m-%d %H:%M:%S", tm_info);
                    TUI_printfw("List: %-25s (Upd: %s)  ToolCPU: %5.2f%% (%4.1f fps)", shm_list_fname, timestr, tool_cpu_pcnt, actual_fps);
                    if (flog) TUI_printfw(" K:%d", last_ch);
                    TUI_newline();
                }
            }
            
            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP) attron(COLOR_PAIR(2));
            TUI_printfw("[h] Help");
            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP) attroff(COLOR_PAIR(2));
            TUI_printfw("   ");

            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_CTRL) attron(COLOR_PAIR(2));
            TUI_printfw("[F2] CTRL");
            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_CTRL) attroff(COLOR_PAIR(2));
            TUI_printfw("   ");

            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES) attron(COLOR_PAIR(2));
            TUI_printfw("[F3] Resources");
            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES) attroff(COLOR_PAIR(2));
            TUI_printfw("   ");

            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER) attron(COLOR_PAIR(2));
            TUI_printfw("[F4] Triggering");
            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER) attroff(COLOR_PAIR(2));
            TUI_printfw("   ");

            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TIMING) attron(COLOR_PAIR(2));
            TUI_printfw("[F5] Timing");
            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TIMING) attroff(COLOR_PAIR(2));
            TUI_printfw("   ");

            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_PROCINFO) attron(COLOR_PAIR(2));
            TUI_printfw("[F6] PROCINFO");
            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_PROCINFO) attroff(COLOR_PAIR(2));
            TUI_newline();

            // Column visibility status line
            {
                const char *colnames[10] = {NULL};
                int nbcol = 0;
                switch(procinfoproc->DisplayMode) {
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
                    int m = procinfoproc->DisplayMode;
                    if (procinfoproc->selected_col > nbcol) procinfoproc->selected_col = nbcol;
                    if (procinfoproc->selected_col < 1) procinfoproc->selected_col = 1;

                    for(int i=1; i<=nbcol; i++) {
                        char colinfo[40];
                        if (i == procinfoproc->sort_col[m]) {
                            snprintf(colinfo, 40, "%d:%s%c", i, colnames[i], procinfoproc->sort_dir[m] ? 'v' : '^');
                        } else {
                            snprintf(colinfo, 40, "%d:%s", i, colnames[i]);
                        }

                        if (i == procinfoproc->selected_col) attron(COLOR_PAIR(10));
                        if (procinfoproc->col_visible[m][i]) {
                            attron(A_BOLD); // Bold font, no background highlight
                            TUI_printfw("%s ", colinfo);
                            attroff(A_BOLD);
                        } else {
                            TUI_printfw("%s ", colinfo); // Normal text
                        }
                        if (i == procinfoproc->selected_col) attroff(COLOR_PAIR(10));
                    }
                    TUI_newline();
                }
                TUI_newline();
            }

            if (procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP) {
                TUI_printfw("MILK Process Control (procCTRL) - HELP");
                TUI_newline();
                TUI_printfw("Navigation:  ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("F2-F6"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Modes (CTRL, Res, Trig, Tim, PInfo)   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("UP/DN"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Selection");
                TUI_newline();
                TUI_printfw("             ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("SPACE"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Select process (batch)                ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("1-9"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw("   : Toggle Cols");
                TUI_newline();
                TUI_printfw("             ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("LEFT/RGHT"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Highlight Column                      ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("^L/^R"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Cycle Tabs");
                TUI_newline();
                TUI_printfw("Control:     ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("p"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Pause/Resume   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("^S"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Step   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("e"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Exit   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("T/K/I"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : TERM/KILL/INT");
                TUI_newline();
                TUI_printfw("             ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("r/R"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Rem Log (Sel/All)   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("z/Z"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Zero Cnt (Sel/All)");
                TUI_newline();
                TUI_printfw("Other:       ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("f"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Freeze   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("s/S"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Sort (Tab/All)   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("+/-"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Update freq   ");
                attron(COLOR_PAIR(1) | A_BOLD); TUI_printfw("x"); attroff(COLOR_PAIR(1) | A_BOLD); TUI_printfw(" : Exit procCTRL");
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

                while (pindexActiveSelected - doffsetindex > dispindexMax - 1 - margin_dn) {
                    doffsetindex++;
                }

                while (pindexActiveSelected < doffsetindex + margin_up) {
                    doffsetindex--;
                }

                if (pindexActiveSelected < doffsetindex) {
                    doffsetindex = pindexActiveSelected;
                }
                if (pindexActiveSelected >= doffsetindex + dispindexMax) {
                    doffsetindex = pindexActiveSelected - dispindexMax + 1;
                }

                if (doffsetindex < 0) {
                    doffsetindex = 0;
                }

                int lastindex = doffsetindex + dispindexMax;
                if (lastindex > NBactive) {
                    lastindex = NBactive;
                }

                for(int dispindex = doffsetindex; dispindex < lastindex; dispindex++) {
                    if (dispindex == doffsetindex && doffsetindex > 0) {
                        attron(A_BOLD);
                        screenprint_setcolor(3);
                        TUI_printfw("      ^^^^ %d more entries above ^^^^", (int)(doffsetindex + 1));
                        screenprint_unsetcolor(3);
                        attroff(A_BOLD);
                        TUI_newline();
                        continue;
                    }

                    if (dispindex == lastindex - 1 && lastindex < NBactive) {
                        attron(A_BOLD);
                        screenprint_setcolor(3);
                        TUI_printfw("      vvvv %d more entries below vvvv", (int)(NBactive - lastindex + 1));
                        screenprint_unsetcolor(3);
                        attroff(A_BOLD);
                        TUI_newline();
                        continue;
                    }

                if (scan_shm == NULL) break;
                int pindex;
                if (procinfoproc->sort_col[m] > 0) pindex = procinfoproc->local_sorted_pindex[dispindex];
                else pindex = scan_shm->sorted_pindex[dispindex];
                
                if (pindex >= 0 && pindex < PROCESSINFOLISTSIZE && (procinfoproc->pinfommapped[pindex] || pinfolist->active[pindex] != 0)) {
                        if (pindex == pindexSelected) attron(A_REVERSE);
                        
                        if (procinfoproc->selectedarray[pindex]) TUI_printfw("* "); else TUI_printfw("  ");
                        
                        int m = procinfoproc->DisplayMode;

                        // Column 1: idx
                        if (procinfoproc->col_visible[m][1]) {
                            if (procinfoproc->selected_col == 1) attron(COLOR_PAIR(10));
                            TUI_printfw("%4d ", pindex);
                            if (procinfoproc->selected_col == 1) attroff(COLOR_PAIR(10));
                        }

                        // Column 2: status
                        if (procinfoproc->col_visible[m][2]) {
                            if (procinfoproc->selected_col == 2) attron(COLOR_PAIR(10));
                            if (pinfolist->active[pindex] == 1) {
                                screenprint_setcolor(6); TUI_printfw("% -10s ", "ACTIVE"); screenprint_unsetcolor(6);
                            } else if (pinfolist->active[pindex] == 2) {
                                screenprint_setcolor(7); TUI_printfw("% -10s ", "STOPPED"); screenprint_unsetcolor(7);
                            } else if (pinfolist->active[pindex] == 3) {
                                screenprint_setcolor(8); TUI_printfw("% -10s ", "CRASHED"); screenprint_unsetcolor(8);
                            } else {
                                TUI_printfw("% -10s ", "OFF");
                            }
                            if (procinfoproc->selected_col == 2) attroff(COLOR_PAIR(10));
                        }
                        
                        // Column 3: pid
                        if (procinfoproc->col_visible[m][3]) {
                            if (procinfoproc->selected_col == 3) attron(COLOR_PAIR(10));
                            pid_t pid = pinfolist->PIDarray[pindex];
                            int pid_exists = (kill(pid, 0) == 0);
                            if (!pid_exists) attron(COLOR_PAIR(4));
                            char state = scan_shm->pinfodisp[pindex].state;
                            if (state == 0) state = ' ';
                            TUI_printfw("%7d %c ", pid, state);
                            if (!pid_exists) attroff(COLOR_PAIR(4));
                            if (procinfoproc->selected_col == 3) attroff(COLOR_PAIR(10));
                        }

                        if (scan_shm && pindex < PROCESSINFOLISTSIZE && pinfolist->active[pindex] == 1 && scan_shm->request_scan[pindex] == 0) {
                            scan_shm->request_scan[pindex] = 1;
                        }

                        if (m == PROCCTRL_DISPLAYMODE_CTRL) {
                            // 4: tstart
                            if (procinfoproc->col_visible[m][4]) {
                                if (procinfoproc->selected_col == 4) attron(COLOR_PAIR(10));
                                char tbuf[30];
                                time_t sec = (time_t)pinfolist->createtime[pindex];
                                long usec = (long)((pinfolist->createtime[pindex] - sec) * 1000000);
                                struct tm *tm_info = gmtime(&sec);
                                strftime(tbuf, 20, "%Y%m%dT%H:%M:%S", tm_info);
                                sprintf(tbuf + 19, ".%06ld", usec);
                                TUI_printfw("%s ", tbuf);
                                if (procinfoproc->selected_col == 4) attroff(COLOR_PAIR(10));
                            }

                            // 5: pname
                            if (procinfoproc->col_visible[m][5]) {
                                if (procinfoproc->selected_col == 5) attron(COLOR_PAIR(10));
                                TUI_printfw("% -25s ", pinfolist->pnamearray[pindex]);
                                if (procinfoproc->selected_col == 5) attroff(COLOR_PAIR(10));
                            }

                            int   ctrlval = (procinfoproc->pinfoarray[pindex]) ? procinfoproc->pinfoarray[pindex]->CTRLval : 0;
                            long  loopcnt = scan_shm->pinfodisp[pindex].loopcnt;
                            char *desc    = scan_shm->pinfodisp[pindex].statusmsg;

                            // 6: state
                            if (procinfoproc->col_visible[m][6]) {
                                if (procinfoproc->selected_col == 6) attron(COLOR_PAIR(10));
                                if (ctrlval == 1) {
                                     attron(COLOR_PAIR(3) | A_BLINK); TUI_printfw("C1"); attroff(COLOR_PAIR(3) | A_BLINK); TUI_printfw(" ");
                                } else {
                                     TUI_printfw("C%d ", ctrlval); 
                                }
                                if (procinfoproc->selected_col == 6) attroff(COLOR_PAIR(10));
                            }

                            // 7: lcnt
                            if (procinfoproc->col_visible[m][7]) {
                                if (procinfoproc->selected_col == 7) attron(COLOR_PAIR(10));
                                if (loopcnt != procinfoproc->loopcntarray[pindex]) {
                                     screenprint_setcolor(6); TUI_printfw("%10ld ", loopcnt); screenprint_unsetcolor(6);
                                } else {
                                     TUI_printfw("%10ld ", loopcnt); 
                                }
                                procinfoproc->loopcntarray[pindex] = loopcnt;
                                if (procinfoproc->selected_col == 7) attroff(COLOR_PAIR(10));
                            }

                            // 8: msg
                            if (procinfoproc->col_visible[m][8]) {
                                if (procinfoproc->selected_col == 8) attron(COLOR_PAIR(10));
                                TUI_printfw("% -30s ", desc); 
                                if (procinfoproc->selected_col == 8) attroff(COLOR_PAIR(10));
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_RESOURCES) {
                            // 4: pname
                            if (procinfoproc->col_visible[m][4]) {
                                if (procinfoproc->selected_col == 4) attron(COLOR_PAIR(10));
                                TUI_printfw("% -25s ", pinfolist->pnamearray[pindex]);
                                if (procinfoproc->selected_col == 4) attroff(COLOR_PAIR(10));
                            }
                            // 5: prio
                            if (procinfoproc->col_visible[m][5]) {
                                if (procinfoproc->selected_col == 5) attron(COLOR_PAIR(10));
                                TUI_printfw("P%2d ", scan_shm->pinfodisp[pindex].rt_priority); 
                                if (procinfoproc->selected_col == 5) attroff(COLOR_PAIR(10));
                            }
                            // 6: cpu
                            if (procinfoproc->col_visible[m][6]) {
                                if (procinfoproc->selected_col == 6) attron(COLOR_PAIR(10));
                                TUI_printfw("CPU:%5.1f%% ", scan_shm->pinfodisp[pindex].subprocCPUloadarray_timeaveraged[0]);
                                if (procinfoproc->selected_col == 6) attroff(COLOR_PAIR(10));
                            }
                            // 7: mem
                            if (procinfoproc->col_visible[m][7]) {
                                if (procinfoproc->selected_col == 7) attron(COLOR_PAIR(10));
                                TUI_printfw("MEM:%7ldkB ", scan_shm->pinfodisp[pindex].VmRSSarray[0]);
                                if (procinfoproc->selected_col == 7) attroff(COLOR_PAIR(10));
                            }
                            // 8: thr
                            if (procinfoproc->col_visible[m][8]) {
                                if (procinfoproc->selected_col == 8) attron(COLOR_PAIR(10));
                                TUI_printfw("Thr:%3d ", scan_shm->pinfodisp[pindex].threads);
                                if (procinfoproc->selected_col == 8) attroff(COLOR_PAIR(10));
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_TRIGGER) {
                            // 4: pname
                            if (procinfoproc->col_visible[m][4]) {
                                if (procinfoproc->selected_col == 4) attron(COLOR_PAIR(10));
                                TUI_printfw("% -25s ", pinfolist->pnamearray[pindex]);
                                if (procinfoproc->selected_col == 4) attroff(COLOR_PAIR(10));
                            }
                            // 5: tmode
                            if (procinfoproc->col_visible[m][5]) {
                                if (procinfoproc->selected_col == 5) attron(COLOR_PAIR(10));
                                TUI_printfw("TR:%d ", scan_shm->pinfodisp[pindex].triggermode);
                                if (procinfoproc->selected_col == 5) attroff(COLOR_PAIR(10));
                            }
                            // 6: tstream
                            if (procinfoproc->col_visible[m][6]) {
                                if (procinfoproc->selected_col == 6) attron(COLOR_PAIR(10));
                                TUI_printfw("%-15s ", scan_shm->pinfodisp[pindex].triggerstreamname);
                                if (procinfoproc->selected_col == 6) attroff(COLOR_PAIR(10));
                            }
                            // 7: tsem
                            if (procinfoproc->col_visible[m][7]) {
                                if (procinfoproc->selected_col == 7) attron(COLOR_PAIR(10));
                                TUI_printfw("S:%d ", scan_shm->pinfodisp[pindex].triggersem);
                                if (procinfoproc->selected_col == 7) attroff(COLOR_PAIR(10));
                            }
                            // 8: tcnt
                            if (procinfoproc->col_visible[m][8]) {
                                if (procinfoproc->selected_col == 8) attron(COLOR_PAIR(10));
                                TUI_printfw("CNT:%ld ", (long)scan_shm->pinfodisp[pindex].triggerstreamcnt);
                                if (procinfoproc->selected_col == 8) attroff(COLOR_PAIR(10));
                            }
                            // 9: tmiss
                            if (procinfoproc->col_visible[m][9]) {
                                if (procinfoproc->selected_col == 9) attron(COLOR_PAIR(10));
                                TUI_printfw("M:%d/%ld ", scan_shm->pinfodisp[pindex].triggermissedframe, (long)scan_shm->pinfodisp[pindex].triggermissedframe_cumul);
                                if (procinfoproc->selected_col == 9) attroff(COLOR_PAIR(10));
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_TIMING) {
                            // 4: pname
                            if (procinfoproc->col_visible[m][4]) {
                                if (procinfoproc->selected_col == 4) attron(COLOR_PAIR(10));
                                TUI_printfw("% -25s ", pinfolist->pnamearray[pindex]);
                                if (procinfoproc->selected_col == 4) attroff(COLOR_PAIR(10));
                            }
                            if (scan_shm->pinfodisp[pindex].MeasureTiming) {
                                double freq = 0.0;
                                if (scan_shm->pinfodisp[pindex].dtmedian_iter_ns > 0) freq = 1.0e9 / scan_shm->pinfodisp[pindex].dtmedian_iter_ns;
                                double exec_ms = 1.0e-6 * scan_shm->pinfodisp[pindex].dtmedian_exec_ns;
                                double overhead = 0.0;
                                if (scan_shm->pinfodisp[pindex].dtmedian_iter_ns > 0) overhead = 100.0 * scan_shm->pinfodisp[pindex].dtmedian_exec_ns / scan_shm->pinfodisp[pindex].dtmedian_iter_ns;
                                // 5: freq
                                if (procinfoproc->col_visible[m][5]) {
                                    if (procinfoproc->selected_col == 5) attron(COLOR_PAIR(10));
                                    TUI_printfw("%8.2fHz ", freq);
                                    if (procinfoproc->selected_col == 5) attroff(COLOR_PAIR(10));
                                }
                                // 6: exec
                                if (procinfoproc->col_visible[m][6]) {
                                    if (procinfoproc->selected_col == 6) attron(COLOR_PAIR(10));
                                    TUI_printfw("Exec:%7.3fms ", exec_ms);
                                    if (procinfoproc->selected_col == 6) attroff(COLOR_PAIR(10));
                                }
                                // 7: over
                                if (procinfoproc->col_visible[m][7]) {
                                    if (procinfoproc->selected_col == 7) attron(COLOR_PAIR(10));
                                    TUI_printfw("(%5.1f%%) ", overhead);
                                    if (procinfoproc->selected_col == 7) attroff(COLOR_PAIR(10));
                                }
                            } else {
                                TUI_printfw("--- Timing Disabled ---");
                            }
                        }
                        else if (m == PROCCTRL_DISPLAYMODE_PROCINFO) {
                            // 4: pname
                            if (procinfoproc->col_visible[m][4]) {
                                if (procinfoproc->selected_col == 4) attron(COLOR_PAIR(10));
                                TUI_printfw("% -25s ", pinfolist->pnamearray[pindex]);
                                if (procinfoproc->selected_col == 4) attroff(COLOR_PAIR(10));
                            }
                            // 5: RT
                            if (procinfoproc->col_visible[m][5]) {
                                if (procinfoproc->selected_col == 5) attron(COLOR_PAIR(10));
                                TUI_printfw("RT:%2d ", scan_shm->pinfodisp[pindex].rt_priority);
                                if (procinfoproc->selected_col == 5) attroff(COLOR_PAIR(10));
                            }
                            // 6: loopMax
                            if (procinfoproc->col_visible[m][6]) {
                                if (procinfoproc->selected_col == 6) attron(COLOR_PAIR(10));
                                TUI_printfw("Lmax:%7ld ", scan_shm->pinfodisp[pindex].loopcntMax);
                                if (procinfoproc->selected_col == 6) attroff(COLOR_PAIR(10));
                            }
                            // 7: trig
                            if (procinfoproc->col_visible[m][7]) {
                                if (procinfoproc->selected_col == 7) attron(COLOR_PAIR(10));
                                TUI_printfw("Trig:%d ", scan_shm->pinfodisp[pindex].triggermode);
                                if (procinfoproc->selected_col == 7) attroff(COLOR_PAIR(10));
                            }
                            // 8: timeout
                            if (procinfoproc->col_visible[m][8]) {
                                if (procinfoproc->selected_col == 8) attron(COLOR_PAIR(10));
                                double tout = scan_shm->pinfodisp[pindex].triggertimeout.tv_sec + 1e-9 * scan_shm->pinfodisp[pindex].triggertimeout.tv_nsec;
                                TUI_printfw("TO:%5.2f ", tout);
                                if (procinfoproc->selected_col == 8) attroff(COLOR_PAIR(10));
                            }
                            // 9: timing
                            if (procinfoproc->col_visible[m][9]) {
                                if (procinfoproc->selected_col == 9) attron(COLOR_PAIR(10));
                                TUI_printfw("Tim:%d ", scan_shm->pinfodisp[pindex].MeasureTiming);
                                if (procinfoproc->selected_col == 9) attroff(COLOR_PAIR(10));
                            }
                        }
                        else {
                            TUI_printfw("(Mode %d not impl)", m);
                        }

                        if (pindex == pindexSelected) attroff(A_REVERSE);
                        TUI_newline();
                    }
                }
            }
            refresh();
        }
    }

    TUI_exit();
    if (scan_shm) munmap(scan_shm, sizeof(PROCSCAN_SHM));
    for(long i=0; i<PROCESSINFOLISTSIZE; i++) if(procinfoproc->pinfommapped[i]) {
        if (procinfoproc->pinfoarray[i] != NULL && procinfoproc->pinfoarray[i] != (PROCESSINFO*)MAP_FAILED)
            processinfo_shm_close(procinfoproc->pinfoarray[i], procinfoproc->fdarray[i]);
    }
    free(procinfoproc->pinfodisp);
    free(procinfoproc);
    free(CPUsetList);

    if (procCTRL_debug_mode == 0) {
        fflush(stderr);
        dup2(backstderr, STDERR_FILENO);
        close(backstderr);
    }
    if (flog) { fprintf(flog, "--- processinfo_CTRLscreen ended ---\n"); fclose(flog); }
    return RETURN_SUCCESS;
}
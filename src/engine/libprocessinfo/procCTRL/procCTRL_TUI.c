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





#include <sys/mman.h>






#include "procCTRL_TUIcompat.h"
#include "processinfo_internal.h"

static char local_shmdir[STRINGMAXLEN_DIRNAME];
#define SHAREDPROCDIR local_shmdir

#include "processinfo_signals.h"

#include "processinfo_shm_list_create.h"


#include "procCTRL_PIDcollectSystemInfo.h"
#include "procCTRL_GetCPUloads.h"
#include "procCTRL_GetNumberCPUs.h"
#include "procCTRL_processinfo_scan.h"



int procCTRL_debug_mode = 0;
char procCTRL_logfile[1024] = "";

short unsigned int wrow, wcol;



/**
 * @brief List available CPU sets.
 *
 * Scans /dev/cpuset (or cgroup hierarchy) for
 * available CPU isolation sets.
 */
static int processinfo_CPUsets_List(
    STRINGLISTENTRY *CPUsetList,
    int has_cset)
{
    if(has_cset == 0)
    {
        return 0;
    }

    char fname[STRINGMAXLEN_FULLFILENAME];
    snprintf(fname, sizeof(fname), "%s/.csetlist.%ld", SHAREDPROCDIR, (long) getpid());

    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "cset set -l | awk '/root/{stop=1} stop==1{print $0}' > %s", fname);
    if(system(cmd) != 0)
    {
        return 0;
    }

    FILE *fp = fopen(fname, "r");
    if(!fp)
    {
        return 0;
    }

    char line[200];
    char word[200], word1[200];
    int NBset = 0;
    int setindex = 0;

    while(NBset < 1000 && fgets(line, 199, fp) != NULL)
    {
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

/**
 * @brief Interactive CLI selector from a string list.
 *
 * Prints entries and prompts the user to select one
 * by number. Currently unused but retained for
 * potential future interactive mode.
 */
static int __attribute__(
    (unused)) processinfo_SelectFromList(STRINGLISTENTRY *StringList,
            int NBelem)
{
    int selected = 0;
    int inputOK = 0;
    char buff[100];
    char *p;

    printf("%d entries in list:\n", NBelem);
    for(int i = 0; i < NBelem; i++)
    {
        printf("   %3d   : %16s   %s\n", i, StringList[i].name, StringList[i].description);
    }

    while(inputOK == 0)
    {
        printf("\nEnter a number: ");
        fflush(stdout);
        if(fgets(buff, sizeof(buff), stdin))
        {
            selected = strtol(buff, &p, 10);
            if(selected >= 0 && selected < NBelem)
            {
                inputOK = 1;
            }
            else
            {
                printf("\nError: invalid number.\n");
            }
        }
    }
    printf("Selected entry : %s\n", StringList[selected].name);
    return selected;
}




/**
 * @brief Connect to the process scanner shared memory.
 *
 * Maps the SHM segment written by milk-procCTRL-scan
 * to access live process listing data.
 */
static inline void *link_scan_shm(const char *name, size_t size)
{
    int fd = open(name, O_RDWR);
    if(fd == -1)
    {
        return MAP_FAILED;
    }
    void *ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return ptr;
}

#include "procCTRL_TUI_internal.h"

/**
 * @brief Update per-process statistics.
 *
 * Reads timing counters and computes loop rates,
 * CPU usage percentages, and latency metrics.
 */
static void procctrl_update_stats(procctrl_context_t *ctx)
{
    clock_gettime(CLOCK_MONOTONIC, &ctx->t_now);
    double elapsed = (ctx->t_now.tv_sec - ctx->t_last_scan.tv_sec) + (ctx->t_now.tv_nsec -
                     ctx->t_last_scan.tv_nsec) * 1e-9;
    if(elapsed >= (ctx->procinfoproc->twaitus * 1e-6))
    {
        processinfo_scan_step(ctx->procinfoproc);
        ctx->t_last_scan = ctx->t_now;

        getrusage(RUSAGE_SELF, &ctx->usage_cur);
        clock_gettime(CLOCK_MONOTONIC, &ctx->t_usage_cur);
        double t_diff = (ctx->t_usage_cur.tv_sec - ctx->t_usage_prev.tv_sec) +
                        (ctx->t_usage_cur.tv_nsec - ctx->t_usage_prev.tv_nsec) * 1e-9;
        double u_diff = (ctx->usage_cur.ru_utime.tv_sec - ctx->usage_prev.ru_utime.tv_sec) +
                        (ctx->usage_cur.ru_utime.tv_usec - ctx->usage_prev.ru_utime.tv_usec) * 1e-6;
        double s_diff = (ctx->usage_cur.ru_stime.tv_sec - ctx->usage_prev.ru_stime.tv_sec) +
                        (ctx->usage_cur.ru_stime.tv_usec - ctx->usage_prev.ru_stime.tv_usec) * 1e-6;
        if(t_diff > 0)
        {
            ctx->tool_cpu_pcnt = 100.0 * (u_diff + s_diff) / t_diff;
        }
        ctx->usage_prev = ctx->usage_cur;
        ctx->t_usage_prev = ctx->t_usage_cur;
    }

    clock_gettime(CLOCK_MONOTONIC, &ctx->t_disp_cur);
    double d_elapsed = (ctx->t_disp_cur.tv_sec - ctx->t_disp_prev.tv_sec) +
                       (ctx->t_disp_cur.tv_nsec - ctx->t_disp_prev.tv_nsec) * 1e-9;
    if(d_elapsed > 0)
    {
        ctx->actual_fps = 0.9 * ctx->actual_fps + 0.1 * (1.0 / d_elapsed);
    }
    ctx->t_disp_prev = ctx->t_disp_cur;
}

/**
 * @brief Initialize the procCTRL TUI.
 *
 * Sets up terminal, allocates display buffers, and
 * connects to scanner SHM.
 */
static errno_t procctrl_init(procctrl_context_t *ctx)
{
    if(strlen(procCTRL_logfile) > 0)
    {
        ctx->flog = fopen(procCTRL_logfile, "a");
        if(ctx->flog)
        {
            fprintf(ctx->flog, "\n--- processinfo_CTRLscreen started ---\n");
            fflush(ctx->flog);
        }
    }

    if(ctx->flog)
    {
        fprintf(ctx->flog, "Checking for daemon...\n");
        fflush(ctx->flog);
    }
    if(system("pgrep \"milk-procCTRL-s\" > /dev/null") != 0)
    {
        PRINT_WARNING(
            "milk-procCTRL-scan daemon is not running");
        printf("Start it now in tmux session 'milk-procCTRL-scan'? [y/n] ");
        fflush(stdout);
        char response = 'n';
        if(scanf(" %c", &response) == 1 && (response == 'y' || response == 'Y'))
        {
            printf("Launching milk-procCTRL-scan...\n");
            if(system("tmux new-session -d -s milk-procCTRL-scan 'milk-procCTRL-scan'") < 0) {}
            sleep(1);
            if(system("pgrep \"milk-procCTRL-s\" > /dev/null") != 0)
            {
                PRINT_ERROR("ERROR: Failed to launch milk-procCTRL-scan daemon.");
                if(ctx->flog)
                {
                    fclose(ctx->flog);
                }
                return RETURN_FAILURE;
            }
        }
        else
        {
            PRINT_ERROR("ERROR: milk-procCTRL-scan daemon is required for this tool.");
            if(ctx->flog)
            {
                fclose(ctx->flog);
            }
            return RETURN_FAILURE;
        }
    }

    processinfo_procdirname(local_shmdir);
    processinfo_procdirname(ctx->procdname);

    if(ctx->flog)
    {
        fprintf(ctx->flog, "Allocating procinfoproc...\n");
        fflush(ctx->flog);
    }
    ctx->procinfoproc = (PROCINFOPROC *) calloc(1, sizeof(PROCINFOPROC));
    if(ctx->procinfoproc == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer");
        if(ctx->flog)
        {
            fclose(ctx->flog);
        }
        return RETURN_FAILURE;
    }

    ctx->procinfoproc->NBcpus = GetNumberCPUs(ctx->procinfoproc);
    GetCPUloads(ctx->procinfoproc);

    if(system("which cset > /dev/null 2>&1") == 0)
    {
        ctx->procinfoproc->has_cset = 1;
    }
    else
    {
        ctx->procinfoproc->has_cset = 0;
    }

    for(int m = 0; m < 10; m++)
    {
        for(int i = 0; i < 10; i++)
        {
            ctx->procinfoproc->col_visible[m][i] = 1;
        }
        ctx->procinfoproc->sort_col[m] = 0;
        ctx->procinfoproc->sort_dir[m] = 0;
        ctx->procinfoproc->sort_mode[m] = m;
    }
    ctx->procinfoproc->selected_col = 1;

    ctx->CPUsetList = (STRINGLISTENTRY *) malloc(sizeof(STRINGLISTENTRY) * 1000);
    int NBCPUset __attribute__((unused)) = processinfo_CPUsets_List(ctx->CPUsetList,
                                           ctx->procinfoproc->has_cset);

    if(ctx->flog)
    {
        fprintf(ctx->flog, "Connecting to process list...\n");
        fflush(ctx->flog);
    }
    {
        long pindex_unused;
        if(processinfo_shm_list_create(&pindex_unused)
                != RETURN_SUCCESS)
        {
            printf("==== ERROR: CANNOT ACCESS PROCESS LIST ====\n");
            if(ctx->flog)
            {
                fclose(ctx->flog);
            }
            return RETURN_FAILURE;
        }
    }
    ctx->procinfoproc->pinfolist = pinfolist;

    char scan_shm_name[STRINGMAXLEN_FULLFILENAME];
    snprintf(scan_shm_name, sizeof(scan_shm_name), "%s/%s", ctx->procdname, PROCESSINFO_SCAN_SHM_NAME);
    if(ctx->flog)
    {
        fprintf(ctx->flog, "Linking to scan SHM: %s\n", scan_shm_name);
        fflush(ctx->flog);
    }
    ctx->scan_shm = (PROCSCAN_SHM *) link_scan_shm(scan_shm_name, sizeof(PROCSCAN_SHM));
    if(ctx->scan_shm == MAP_FAILED)
    {
        printf("WARNING: Could not link to scan SHM %s. Stats may be missing.\n", scan_shm_name);
        ctx->scan_shm = NULL;
    }

    TUI_set_screenprintmode(SCREENPRINT_NCURSES);
    if(getenv("MILK_TUIPRINT_STDIO"))
    {
        TUI_set_screenprintmode(SCREENPRINT_STDIO);
    }

    if(ctx->flog)
    {
        fprintf(ctx->flog, "Initializing terminal...\n");
        fflush(ctx->flog);
    }
    ansi_raw_mode_enter();
    TUI_init_terminal(&wrow, &wcol);
    if(wrow < 10)
    {
        wrow = 10;
    }

    if(ctx->flog)
    {
        fprintf(ctx->flog, "Allocating pinfodisp buffer (250MB)\n");
        fflush(ctx->flog);
    }
    ctx->procinfoproc->pinfodisp = (PROCESSINFODISP *) calloc(PROCESSINFOLISTSIZE,
                                   sizeof(PROCESSINFODISP));
    if(ctx->procinfoproc->pinfodisp == NULL)
    {
        ansi_raw_mode_exit();
        PRINT_ERROR("FATAL ERROR: Could not allocate 250MB process info buffer.");
        if(ctx->flog)
        {
            fclose(ctx->flog);
        }
        return RETURN_FAILURE;
    }
    ctx->procinfoproc->NBpinfodisp = PROCESSINFOLISTSIZE;

    ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
    ctx->procinfoproc->loop = 1;
    ctx->procinfoproc->twaitus = 1000000;

    ctx->t_last_scan.tv_sec = 0;
    ctx->t_last_scan.tv_nsec = 0;
    ctx->frequ = 32.0;
    ctx->pindexSelected = -1;
    ctx->pindexActiveSelected = 0;
    ctx->doffsetindex = 0;
    ctx->freeze = 0;
    ctx->loopOK = 1;
    ctx->Xexit = 0;
    ctx->monstringlen = 200;
    ctx->last_ch = -1;

    getrusage(RUSAGE_SELF, &ctx->usage_prev);
    clock_gettime(CLOCK_MONOTONIC, &ctx->t_usage_prev);
    ctx->tool_cpu_pcnt = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &ctx->t_disp_prev);
    ctx->actual_fps = 0.0;

    return RETURN_SUCCESS;
}

/**
 * @brief Clean up procCTRL TUI resources.
 *
 * Restores terminal settings and frees display
 * buffers.
 */
static void procctrl_cleanup(procctrl_context_t *ctx)
{
    ansi_raw_mode_exit();
    if(ctx->scan_shm)
    {
        munmap(ctx->scan_shm, sizeof(PROCSCAN_SHM));
    }
    if(ctx->procinfoproc)
    {
        for(long i = 0; i < PROCESSINFOLISTSIZE; i++) if(ctx->procinfoproc->pinfommapped[i])
            {
                if(ctx->procinfoproc->pinfoarray[i] != NULL
                        && ctx->procinfoproc->pinfoarray[i] != (PROCESSINFO *)MAP_FAILED)
                {
                    processinfo_shm_close(ctx->procinfoproc->pinfoarray[i], ctx->procinfoproc->fdarray[i]);
                }
            }
        free(ctx->procinfoproc->pinfodisp);
        free(ctx->procinfoproc);
    }
    if(ctx->CPUsetList)
    {
        free(ctx->CPUsetList);
    }

    if(ctx->flog)
    {
        fprintf(ctx->flog, "--- processinfo_CTRLscreen ended ---\n");
        fclose(ctx->flog);
    }
}
/**
 * @brief Main execution loop for the process info TUI control screen.
 */
errno_t processinfo_CTRLscreen()
{
    if(getenv("PROCCTRL_DEBUG"))
    {
        procCTRL_debug_mode = 1;
    }
    if(procCTRL_debug_mode)
    {
        printf("DEBUG: processinfo_CTRLscreen start\n");
    }

    procctrl_context_t ctx;
    memset(&ctx, 0, sizeof(procctrl_context_t));

    if(procctrl_init(&ctx) != RETURN_SUCCESS)
    {
        return RETURN_FAILURE;
    }

    int backstderr = -1;
    if(procCTRL_debug_mode == 0)
    {
        fflush(stderr);
        backstderr = dup(STDERR_FILENO);
        int newstderr = open("/dev/null", O_WRONLY);
        if(newstderr != -1)
        {
            dup2(newstderr, STDERR_FILENO);
            close(newstderr);
        }
    }

    if(ctx.flog)
    {
        fprintf(ctx.flog, "Entering main loop.\n");
        fflush(ctx.flog);
    }
    sc_frame_clear();

    while(ctx.loopOK)
    {
        if(processinfo_signal_SEGV)
        {
            if(ctx.flog)
            {
                fprintf(ctx.flog, "SEGV signal received!\n");
                fflush(ctx.flog);
            }
            ctx.loopOK = 0;
            break;
        }

        procctrl_update_stats(&ctx);

        usleep((long)(1000000.0 / ctx.frequ));
        int ch = get_singlechar_nonblock();

        int NBactive = (ctx.scan_shm) ? ctx.scan_shm->NBactive : 0;
        int m = ctx.procinfoproc->DisplayMode;

        if(ctx.procinfoproc->sort_col[m] > 0 && ctx.scan_shm != NULL)
        {
            sort_ctx_m = ctx.procinfoproc->sort_mode[m];
            sort_ctx_col = ctx.procinfoproc->sort_col[m];
            sort_ctx_dir = ctx.procinfoproc->sort_dir[m];
            sort_ctx_scan_shm = ctx.scan_shm;
            sort_ctx_pinfolist = pinfolist;

            for(int i = 0; i < NBactive; i++)
            {
                ctx.procinfoproc->local_sorted_pindex[i] = ctx.scan_shm->sorted_pindex[i];
            }
            qsort(ctx.procinfoproc->local_sorted_pindex, NBactive, sizeof(int), proc_comp);
        }

        if(ctx.pindexActiveSelected >= NBactive && NBactive > 0)
        {
            ctx.pindexActiveSelected = NBactive - 1;
        }

        if(NBactive > 0 && ctx.scan_shm != NULL)
        {
            if(ctx.procinfoproc->sort_col[m] > 0)
            {
                ctx.pindexSelected = ctx.procinfoproc->local_sorted_pindex[ctx.pindexActiveSelected];
            }
            else
            {
                ctx.pindexSelected = ctx.scan_shm->sorted_pindex[ctx.pindexActiveSelected];
            }
        }
        else
        {
            ctx.pindexSelected = -1;
        }

        if(ch != -1)
        {
            procctrl_handle_keyboard_event(&ctx, ch, NBactive);
        }

        procctrl_render_frame(&ctx, NBactive);
    }

    procctrl_cleanup(&ctx);

    if(procCTRL_debug_mode == 0)
    {
        fflush(stderr);
        dup2(backstderr, STDERR_FILENO);
        close(backstderr);
    }

    return RETURN_SUCCESS;
}

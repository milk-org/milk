#include <sys/stat.h>

#include "procCTRL_TUI_internal.h"
#include "procCTRL_TUIcompat.h"

/**
 * @brief Render the procCTRL header bar.
 *
 * Shows title, timestamp, and system CPU load.
 */
static void procctrl_render_header(procctrl_context_t *ctx)
{
    struct stat shm_stat;
    char shm_list_fname[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(shm_list_fname, "%s/processinfo.list.shm", ctx->procdname);
    if(stat(shm_list_fname, &shm_stat) == 0)
    {
        char timestr[100];
        struct tm *tm_info = gmtime(&shm_stat.st_mtime);
        strftime(timestr, 100, "%Y-%m-%d %H:%M:%S", tm_info);
        TUI_printfw("List: %-25s (Upd: %s)  ToolCPU: %5.2f%% (%4.1f fps)", shm_list_fname, timestr,
                    ctx->tool_cpu_pcnt, ctx->actual_fps);
        if(ctx->flog)
        {
            TUI_printfw(" K:%d", ctx->last_ch);
        }
        TUI_newline();
    }
}

/**
 * @brief Render the mode selection tabs.
 *
 * Highlights the active display mode (ctrl,
 * resources, trigger, timing, info).
 */
static void procctrl_render_mode_tabs(procctrl_context_t *ctx)
{
    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP)
    {
        screenprint_setcolor(2);
    }
    TUI_printfw("[h] Help");
    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP)
    {
        screenprint_unsetcolor(2);
    }
    TUI_printfw("   ");

    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_CTRL)
    {
        screenprint_setcolor(2);
    }
    TUI_printfw("[F2] CTRL");
    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_CTRL)
    {
        screenprint_unsetcolor(2);
    }
    TUI_printfw("   ");

    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES)
    {
        screenprint_setcolor(2);
    }
    TUI_printfw("[F3] Resources");
    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_RESOURCES)
    {
        screenprint_unsetcolor(2);
    }
    TUI_printfw("   ");

    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER)
    {
        screenprint_setcolor(2);
    }
    TUI_printfw("[F4] Triggering");
    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TRIGGER)
    {
        screenprint_unsetcolor(2);
    }
    TUI_printfw("   ");

    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TIMING)
    {
        screenprint_setcolor(2);
    }
    TUI_printfw("[F5] Timing");
    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_TIMING)
    {
        screenprint_unsetcolor(2);
    }
    TUI_printfw("   ");

    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_PROCINFO)
    {
        screenprint_setcolor(2);
    }
    TUI_printfw("[F6] PROCINFO");
    if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_PROCINFO)
    {
        screenprint_unsetcolor(2);
    }
    TUI_newline();
}

/**
 * @brief Render column headers for the process list.
 *
 * Adapts columns to the current display mode.
 */
static void procctrl_render_column_headers(procctrl_context_t *ctx)
{
    const char *colnames[10] = {NULL};
    int nbcol = 0;
    int m = ctx->procinfoproc->DisplayMode;

    switch(m)
    {
    case PROCCTRL_DISPLAYMODE_CTRL:
    {
        static const char *names[] = {"", "idx", "status", "pid", "tstart", "pname", "state", "lcnt", "msg", ""};
        for(int i = 0; i < 10; i++)
        {
            colnames[i] = names[i];
        }
        nbcol = 8;
    }
    break;
    case PROCCTRL_DISPLAYMODE_RESOURCES:
    {
        static const char *names[] = {"", "idx", "status", "pid", "pname", "prio", "cpu", "mem", "thr", ""};
        for(int i = 0; i < 10; i++)
        {
            colnames[i] = names[i];
        }
        nbcol = 8;
    }
    break;
    case PROCCTRL_DISPLAYMODE_TRIGGER:
    {
        static const char *names[] = {"", "idx", "status", "pid", "pname", "tmode", "tstream", "tsem", "tcnt", "tmiss"};
        for(int i = 0; i < 10; i++)
        {
            colnames[i] = names[i];
        }
        nbcol = 9;
    }
    break;
    case PROCCTRL_DISPLAYMODE_TIMING:
    {
        static const char *names[] = {"", "idx", "status", "pid", "pname", "freq", "exec", "over", "", ""};
        for(int i = 0; i < 10; i++)
        {
            colnames[i] = names[i];
        }
        nbcol = 7;
    }
    break;
    case PROCCTRL_DISPLAYMODE_PROCINFO:
    {
        static const char *names[] = {"", "idx", "status", "pid", "pname", "RT", "loopMax", "trig", "timeout", "timing"};
        for(int i = 0; i < 10; i++)
        {
            colnames[i] = names[i];
        }
        nbcol = 9;
    }
    break;
    }

    if(nbcol > 0)
    {
        if(ctx->procinfoproc->selected_col > nbcol)
        {
            ctx->procinfoproc->selected_col = nbcol;
        }
        if(ctx->procinfoproc->selected_col < 1)
        {
            ctx->procinfoproc->selected_col = 1;
        }

        for(int i = 1; i <= nbcol; i++)
        {
            char colinfo[40];
            if(i == ctx->procinfoproc->sort_col[m])
            {
                snprintf(colinfo, 40, "%d:%s%c", i, colnames[i], ctx->procinfoproc->sort_dir[m] ? 'v' : '^');
            }
            else
            {
                snprintf(colinfo, 40, "%d:%s", i, colnames[i]);
            }

            if(i == ctx->procinfoproc->selected_col)
            {
                screenprint_setcolor(10);
            }
            if(ctx->procinfoproc->col_visible[m][i])
            {
                screenprint_setbold(); // Bold font, no background highlight
                TUI_printfw("%s ", colinfo);
                screenprint_unsetbold();
            }
            else
            {
                TUI_printfw("%s ", colinfo); // Normal text
            }
            if(i == ctx->procinfoproc->selected_col)
            {
                screenprint_unsetcolor(10);
            }
        }
        TUI_newline();
    }
    TUI_newline();
}

/**
 * @brief Render the help overlay.
 *
 * Shows available key bindings and navigation
 * commands.
 */
static void procctrl_render_help(void)
{
    TUI_printfw("MILK Process Control (procCTRL) - HELP");
    TUI_newline();
    TUI_printfw("Navigation:  ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("F2-F6");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Modes (CTRL, Res, Trig, Tim, PInfo)   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("UP/DN");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Selection");
    TUI_newline();
    TUI_printfw("             ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("SPACE");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Select process (batch)                ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("1-9");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw("   : Toggle Cols");
    TUI_newline();
    TUI_printfw("             ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("LEFT/RGHT");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Highlight Column                      ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("^L/^R");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Cycle Tabs");
    TUI_newline();
    TUI_printfw("Control:     ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("p");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Pause/Resume   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("^S");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Step   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("e");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Exit   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("T/K/I");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : TERM/KILL/INT");
    TUI_newline();
    TUI_printfw("             ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("r/R");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Rem Log (Sel/All)   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("z/Z");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Zero Cnt (Sel/All)");
    TUI_newline();
    TUI_printfw("Other:       ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("f");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Freeze   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("s/S");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Sort (Tab/All)   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("+/-");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Update freq   ");
    screenprint_setcolor(1);
    screenprint_setbold();
    TUI_printfw("x");
    screenprint_unsetcolor(1);
    screenprint_unsetbold();
    TUI_printfw(" : Exit procCTRL");
    TUI_newline();
}

/**
 * @brief Render a process row in ctrl mode.
 *
 * Shows PID, name, loop state, and control actions.
 */
static void procctrl_render_row_ctrl(
    procctrl_context_t *ctx,
    int                m,
    int                pindex)
{
    // 4: tstart
    if(ctx->procinfoproc->col_visible[m][4])
    {
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_setcolor(10);
        }
        char tbuf[30];
        time_t sec = (time_t)pinfolist->createtime[pindex];
        long usec = (long)((pinfolist->createtime[pindex] - sec) * 1000000);
        struct tm *tm_info = gmtime(&sec);
        strftime(tbuf, 20, "%Y%m%dT%H:%M:%S", tm_info);
        snprintf(tbuf + 19,
                 sizeof(tbuf) - 19,
                 ".%06ld", usec);
        TUI_printfw("%s ", tbuf);
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_unsetcolor(10);
        }
    }

    // 5: pname
    if(ctx->procinfoproc->col_visible[m][5])
    {
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_unsetcolor(10);
        }
    }

    int   ctrlval = (ctx->procinfoproc->pinfoarray[pindex]) ?
                    ctx->procinfoproc->pinfoarray[pindex]->CTRLval : 0;
    long  loopcnt = ctx->scan_shm->pinfodisp[pindex].loopcnt;
    char *desc    = ctx->scan_shm->pinfodisp[pindex].statusmsg;

    // 6: state
    if(ctx->procinfoproc->col_visible[m][6])
    {
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_setcolor(10);
        }
        if(ctrlval == 1)
        {
            screenprint_setcolor(3);
            screenprint_setblink();
            TUI_printfw("C1");
            screenprint_unsetcolor(3);
            screenprint_unsetblink();
            TUI_printfw(" ");
        }
        else
        {
            TUI_printfw("C%d ", ctrlval);
        }
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_unsetcolor(10);
        }
    }

    // 7: lcnt
    if(ctx->procinfoproc->col_visible[m][7])
    {
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_setcolor(10);
        }
        if(loopcnt != ctx->procinfoproc->loopcntarray[pindex])
        {
            screenprint_setcolor(6);
            TUI_printfw("%10ld ", loopcnt);
            screenprint_unsetcolor(6);
        }
        else
        {
            TUI_printfw("%10ld ", loopcnt);
        }
        ctx->procinfoproc->loopcntarray[pindex] = loopcnt;
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_unsetcolor(10);
        }
    }

    // 8: msg
    if(ctx->procinfoproc->col_visible[m][8])
    {
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("%-30s ", desc);
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_unsetcolor(10);
        }
    }
}

/**
 * @brief Render a process row in resources mode.
 *
 * Shows memory, CPU affinity, and thread count.
 */
static void procctrl_render_row_resources(
    procctrl_context_t *ctx,
    int                m,
    int                pindex)
{
    // 4: pname
    if(ctx->procinfoproc->col_visible[m][4])
    {
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 5: prio
    if(ctx->procinfoproc->col_visible[m][5])
    {
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("P%2d ", ctx->scan_shm->pinfodisp[pindex].rt_priority);
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 6: cpu
    if(ctx->procinfoproc->col_visible[m][6])
    {
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("CPU:%5.1f%% ", ctx->scan_shm->pinfodisp[pindex].subprocCPUloadarray_timeaveraged[0]);
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 7: mem
    if(ctx->procinfoproc->col_visible[m][7])
    {
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("MEM:%7ldkB ", ctx->scan_shm->pinfodisp[pindex].VmRSSarray[0]);
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 8: thr
    if(ctx->procinfoproc->col_visible[m][8])
    {
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("Thr:%3d ", ctx->scan_shm->pinfodisp[pindex].threads);
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_unsetcolor(10);
        }
    }
}

/**
 * @brief Render a process row in trigger mode.
 *
 * Shows trigger stream, semaphore index, and
 * timeout settings.
 */
static void procctrl_render_row_trigger(
    procctrl_context_t *ctx,
    int                m,
    int                pindex)
{
    // 4: pname
    if(ctx->procinfoproc->col_visible[m][4])
    {
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 5: tmode
    if(ctx->procinfoproc->col_visible[m][5])
    {
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("TR:%d ", ctx->scan_shm->pinfodisp[pindex].triggermode);
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 6: tstream
    if(ctx->procinfoproc->col_visible[m][6])
    {
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("%-15s ", ctx->scan_shm->pinfodisp[pindex].triggerstreamname);
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 7: tsem
    if(ctx->procinfoproc->col_visible[m][7])
    {
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("S:%d ", ctx->scan_shm->pinfodisp[pindex].triggersem);
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 8: tcnt
    if(ctx->procinfoproc->col_visible[m][8])
    {
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("CNT:%ld ", (long)ctx->scan_shm->pinfodisp[pindex].triggerstreamcnt);
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 9: tmiss
    if(ctx->procinfoproc->col_visible[m][9])
    {
        if(ctx->procinfoproc->selected_col == 9)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("M:%d/%ld ", ctx->scan_shm->pinfodisp[pindex].triggermissedframe,
                    (long)ctx->scan_shm->pinfodisp[pindex].triggermissedframe_cumul);
        if(ctx->procinfoproc->selected_col == 9)
        {
            screenprint_unsetcolor(10);
        }
    }
}

/**
 * @brief Render a process row in timing mode.
 *
 * Shows loop rate, latency, and jitter metrics.
 */
static void procctrl_render_row_timing(
    procctrl_context_t *ctx,
    int                m,
    int                pindex)
{
    // 4: pname
    if(ctx->procinfoproc->col_visible[m][4])
    {
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_unsetcolor(10);
        }
    }
    if(ctx->scan_shm->pinfodisp[pindex].MeasureTiming)
    {
        double freq = 0.0;
        if(ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns > 0)
        {
            freq = 1.0e9 / ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns;
        }
        double exec_ms = 1.0e-6 * ctx->scan_shm->pinfodisp[pindex].dtmedian_exec_ns;
        double overhead = 0.0;
        if(ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns > 0)
        {
            overhead = 100.0 * ctx->scan_shm->pinfodisp[pindex].dtmedian_exec_ns /
                       ctx->scan_shm->pinfodisp[pindex].dtmedian_iter_ns;
        }
        // 5: freq
        if(ctx->procinfoproc->col_visible[m][5])
        {
            if(ctx->procinfoproc->selected_col == 5)
            {
                screenprint_setcolor(10);
            }
            TUI_printfw("%8.2fHz ", freq);
            if(ctx->procinfoproc->selected_col == 5)
            {
                screenprint_unsetcolor(10);
            }
        }
        // 6: exec
        if(ctx->procinfoproc->col_visible[m][6])
        {
            if(ctx->procinfoproc->selected_col == 6)
            {
                screenprint_setcolor(10);
            }
            TUI_printfw("Exec:%7.3fms ", exec_ms);
            if(ctx->procinfoproc->selected_col == 6)
            {
                screenprint_unsetcolor(10);
            }
        }
        // 7: over
        if(ctx->procinfoproc->col_visible[m][7])
        {
            if(ctx->procinfoproc->selected_col == 7)
            {
                screenprint_setcolor(10);
            }
            TUI_printfw("(%5.1f%%) ", overhead);
            if(ctx->procinfoproc->selected_col == 7)
            {
                screenprint_unsetcolor(10);
            }
        }
    }
    else
    {
        TUI_printfw("--- Timing Disabled ---");
    }
}

/**
 * @brief Render a process row in procinfo mode.
 *
 * Shows executable path, start time, and
 * status message.
 */
static void procctrl_render_row_procinfo(
    procctrl_context_t *ctx,
    int                m,
    int                pindex)
{
    // 4: pname
    if(ctx->procinfoproc->col_visible[m][4])
    {
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("%-25s ", pinfolist->pnamearray[pindex]);
        if(ctx->procinfoproc->selected_col == 4)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 5: RT
    if(ctx->procinfoproc->col_visible[m][5])
    {
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("RT:%2d ", ctx->scan_shm->pinfodisp[pindex].rt_priority);
        if(ctx->procinfoproc->selected_col == 5)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 6: loopMax
    if(ctx->procinfoproc->col_visible[m][6])
    {
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("Lmax:%7ld ", ctx->scan_shm->pinfodisp[pindex].loopcntMax);
        if(ctx->procinfoproc->selected_col == 6)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 7: trig
    if(ctx->procinfoproc->col_visible[m][7])
    {
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("Trig:%d ", ctx->scan_shm->pinfodisp[pindex].triggermode);
        if(ctx->procinfoproc->selected_col == 7)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 8: timeout
    if(ctx->procinfoproc->col_visible[m][8])
    {
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_setcolor(10);
        }
        double tout = ctx->scan_shm->pinfodisp[pindex].triggertimeout.tv_sec + 1e-9 *
                      ctx->scan_shm->pinfodisp[pindex].triggertimeout.tv_nsec;
        TUI_printfw("TO:%5.2f ", tout);
        if(ctx->procinfoproc->selected_col == 8)
        {
            screenprint_unsetcolor(10);
        }
    }
    // 9: timing
    if(ctx->procinfoproc->col_visible[m][9])
    {
        if(ctx->procinfoproc->selected_col == 9)
        {
            screenprint_setcolor(10);
        }
        TUI_printfw("Tim:%d ", ctx->scan_shm->pinfodisp[pindex].MeasureTiming);
        if(ctx->procinfoproc->selected_col == 9)
        {
            screenprint_unsetcolor(10);
        }
    }
}

/**
 * @brief Render the full process list.
 *
 * Iterates visible processes and dispatches to
 * the mode-specific row renderer.
 */
static void procctrl_render_process_list(
    procctrl_context_t *ctx,
    int                NBactive)
{
    int dispindexMax = wrow - 6;

    int margin_dn = 2;
    int margin_up = 2;

    if(margin_dn >= dispindexMax)
    {
        margin_dn = dispindexMax - 1;
    }
    if(margin_dn < 0)
    {
        margin_dn = 0;
    }
    if(margin_up >= dispindexMax)
    {
        margin_up = dispindexMax - 1;
    }
    if(margin_up < 0)
    {
        margin_up = 0;
    }

    while(ctx->pindexActiveSelected - ctx->doffsetindex > dispindexMax - 1 - margin_dn)
    {
        ctx->doffsetindex++;
    }

    while(ctx->pindexActiveSelected < ctx->doffsetindex + margin_up)
    {
        ctx->doffsetindex--;
    }

    if(ctx->pindexActiveSelected < ctx->doffsetindex)
    {
        ctx->doffsetindex = ctx->pindexActiveSelected;
    }
    if(ctx->pindexActiveSelected >= ctx->doffsetindex + dispindexMax)
    {
        ctx->doffsetindex = ctx->pindexActiveSelected - dispindexMax + 1;
    }

    if(ctx->doffsetindex < 0)
    {
        ctx->doffsetindex = 0;
    }

    int lastindex = ctx->doffsetindex + dispindexMax;
    if(lastindex > NBactive)
    {
        lastindex = NBactive;
    }

    for(int dispindex = ctx->doffsetindex; dispindex < lastindex; dispindex++)
    {
        if(dispindex == ctx->doffsetindex && ctx->doffsetindex > 0)
        {
            screenprint_setbold();
            screenprint_setcolor(3);
            TUI_printfw("      ^^^^ %d more entries above ^^^^", (int)(ctx->doffsetindex + 1));
            screenprint_unsetcolor(3);
            screenprint_unsetbold();
            TUI_newline();
            continue;
        }

        if(dispindex == lastindex - 1 && lastindex < NBactive)
        {
            screenprint_setbold();
            screenprint_setcolor(3);
            TUI_printfw("      vvvv %d more entries below vvvv", (int)(NBactive - lastindex + 1));
            screenprint_unsetcolor(3);
            screenprint_unsetbold();
            TUI_newline();
            continue;
        }

        if(ctx->scan_shm == NULL)
        {
            break;
        }
        int pindex;
        int m = ctx->procinfoproc->DisplayMode;

        if(ctx->procinfoproc->sort_col[m] > 0)
        {
            pindex = ctx->procinfoproc->local_sorted_pindex[dispindex];
        }
        else
        {
            pindex = ctx->scan_shm->sorted_pindex[dispindex];
        }

        if(pindex >= 0 && pindex < PROCESSINFOLISTSIZE && (ctx->procinfoproc->pinfommapped[pindex]
                || pinfolist->active[pindex] != 0))
        {
            if(pindex == ctx->pindexSelected)
            {
                screenprint_setreverse();
            }

            if(ctx->procinfoproc->selectedarray[pindex])
            {
                TUI_printfw("* ");
            }
            else
            {
                TUI_printfw("  ");
            }

            // Column 1: idx
            if(ctx->procinfoproc->col_visible[m][1])
            {
                if(ctx->procinfoproc->selected_col == 1)
                {
                    screenprint_setcolor(10);
                }
                TUI_printfw("%4d ", pindex);
                if(ctx->procinfoproc->selected_col == 1)
                {
                    screenprint_unsetcolor(10);
                }
            }

            // Column 2: status
            if(ctx->procinfoproc->col_visible[m][2])
            {
                if(ctx->procinfoproc->selected_col == 2)
                {
                    screenprint_setcolor(10);
                }
                if(pinfolist->active[pindex] == 1)
                {
                    screenprint_setcolor(6);
                    TUI_printfw("%-10s ", "ACTIVE");
                    screenprint_unsetcolor(6);
                }
                else if(pinfolist->active[pindex] == 2)
                {
                    screenprint_setcolor(7);
                    TUI_printfw("%-10s ", "STOPPED");
                    screenprint_unsetcolor(7);
                }
                else if(pinfolist->active[pindex] == 3)
                {
                    screenprint_setcolor(8);
                    TUI_printfw("%-10s ", "CRASHED");
                    screenprint_unsetcolor(8);
                }
                else
                {
                    TUI_printfw("%-10s ", "OFF");
                }
                if(ctx->procinfoproc->selected_col == 2)
                {
                    screenprint_unsetcolor(10);
                }
            }

            // Column 3: pid
            if(ctx->procinfoproc->col_visible[m][3])
            {
                if(ctx->procinfoproc->selected_col == 3)
                {
                    screenprint_setcolor(10);
                }
                pid_t pid = pinfolist->PIDarray[pindex];
                int pid_exists = (kill(pid, 0) == 0);
                if(!pid_exists)
                {
                    screenprint_setcolor(4);
                }
                char state = ctx->scan_shm->pinfodisp[pindex].state;
                if(state == 0)
                {
                    state = ' ';
                }
                TUI_printfw("%7d %c ", pid, state);
                if(!pid_exists)
                {
                    screenprint_unsetcolor(4);
                }
                if(ctx->procinfoproc->selected_col == 3)
                {
                    screenprint_unsetcolor(10);
                }
            }

            if(ctx->scan_shm && pindex < PROCESSINFOLISTSIZE && pinfolist->active[pindex] == 1
                    && ctx->scan_shm->request_scan[pindex] == 0)
            {
                ctx->scan_shm->request_scan[pindex] = 1;
            }

            switch(m)
            {
            case PROCCTRL_DISPLAYMODE_CTRL:
                procctrl_render_row_ctrl(ctx, m, pindex);
                break;
            case PROCCTRL_DISPLAYMODE_RESOURCES:
                procctrl_render_row_resources(ctx, m, pindex);
                break;
            case PROCCTRL_DISPLAYMODE_TRIGGER:
                procctrl_render_row_trigger(ctx, m, pindex);
                break;
            case PROCCTRL_DISPLAYMODE_TIMING:
                procctrl_render_row_timing(ctx, m, pindex);
                break;
            case PROCCTRL_DISPLAYMODE_PROCINFO:
                procctrl_render_row_procinfo(ctx, m, pindex);
                break;
            default:
                TUI_printfw("(Mode %d not impl)", m);
                break;
            }

            if(pindex == ctx->pindexSelected)
            {
                screenprint_unsetreverse();
            }
            TUI_newline();
        }
    }
}

/**
 * @brief Render one complete procCTRL frame.
 *
 * Composes header, tabs, column headers, process
 * list, and status bar into a single screen update.
 */
void procctrl_render_frame(
    procctrl_context_t *ctx,
    int                NBactive)
{
    if(ctx->freeze == 0)
    {
        sc_frame_clear();
        TUI_clearscreen(&wrow, &wcol);
        snprintf(ctx->monstring, ctx->monstringlen, "Mode %d   PRESS x TO STOP MONITOR",
                 ctx->procinfoproc->DisplayMode);
        TUI_print_header(ctx->monstring, '-');
        TUI_newline();

        procctrl_render_header(ctx);
        procctrl_render_mode_tabs(ctx);
        procctrl_render_column_headers(ctx);

        if(ctx->procinfoproc->DisplayMode == PROCCTRL_DISPLAYMODE_HELP)
        {
            procctrl_render_help();
        }
        else
        {
            procctrl_render_process_list(ctx, NBactive);
        }

        TUI_cleartobottom();
        sc_frame_flush();
    }
}

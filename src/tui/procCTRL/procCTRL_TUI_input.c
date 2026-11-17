// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "procCTRL_TUI_internal.h"
#include "procCTRL_ansi.h"

#include <signal.h>
#include <unistd.h>

/** ident_set - capture the identity of registry slot @pindex. */
static void ident_set(procctrl_ident_t *id, int pindex)
{
    id->pindex = pindex;
    id->pid    = pinfolist->PIDarray[pindex];
    id->ctime  = pinfolist->createtime[pindex];
}

/** ident_live - 1 while slot @id->pindex still holds the same entry (PID and createtime). */
static int ident_live(const procctrl_ident_t *id)
{
    return id->pindex >= 0 && id->pindex < PROCESSINFOLISTSIZE &&
           pinfolist->active[id->pindex] != 0 && pinfolist->PIDarray[id->pindex] == id->pid &&
           pinfolist->createtime[id->pindex] == id->ctime;
}

/** procctrl_order - the row order on screen (sorted locally, or the scanner's order). */
static const int *procctrl_order(const procctrl_context_t *ctx)
{
    int m = ctx->procinfoproc->DisplayMode;
    return (ctx->procinfoproc->sort_col[m] > 0) ? ctx->procinfoproc->local_sorted_pindex
                                                : ctx->scan_shm->sorted_pindex;
}

/** procctrl_pid_start_epoch - start time of @pid in seconds since the epoch (/proc), or 0. */
static double procctrl_pid_start_epoch(pid_t pid)
{
    static double btime = -1.0;
    char          buf[1024];
    if (btime < 0.0)
    {
        FILE *f = fopen("/proc/stat", "r");
        btime   = 0.0;
        while (f != NULL && fgets(buf, sizeof(buf), f) != NULL)
        {
            if (strncmp(buf, "btime ", 6) == 0)
            {
                btime = atof(buf + 6);
            }
        }
        if (f != NULL)
        {
            fclose(f);
        }
    }
    snprintf(buf, sizeof(buf), "/proc/%d/stat", (int) pid);
    FILE *f = fopen(buf, "r");
    if (f == NULL)
    {
        return 0.0;
    }
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n]  = '\0';
    char *p = strrchr(buf, ')'); /* comm may contain spaces and parentheses */
    for (int field = 2; p != NULL && field < 22; field++)
    {
        p = strchr(p + 1, ' ');
    }
    return (p == NULL) ? 0.0 : btime + atof(p + 1) / (double) sysconf(_SC_CLK_TCK);
}

/**
 * procctrl_ident_kill - signal @id only if it is still the same live process: same slot entry,
 * active, and a PID that started no later than the entry was created (2 s for btime granularity).
 */
static void procctrl_ident_kill(const procctrl_ident_t *id, int sig)
{
    if (!ident_live(id) || pinfolist->active[id->pindex] != 1)
    {
        return;
    }
    double st = procctrl_pid_start_epoch(id->pid);
    if (st > 0.0 && st <= id->ctime + 2.0)
    {
        kill(id->pid, sig);
    }
}

/** procctrl_ident_ctrlval - set (or toggle pause, @val -1) CTRLval of a still-live entry. */
static void procctrl_ident_ctrlval(procctrl_context_t *ctx, const procctrl_ident_t *id, int val)
{
    PROCESSINFO *pi = ident_live(id) ? ctx->procinfoproc->pinfoarray[id->pindex] : NULL;
    if (pi != NULL)
    {
        pi->CTRLval = (val == -1) ? ((pi->CTRLval == 0) ? 1 : 0) : val;
    }
}

/**
 * procctrl_follow_cursor - put the cursor back on its identity after a rescan or sort; when it is
 * gone, adopt the row under the cursor (cur_found = 0: no action this frame). Drops dead marks.
 */
void procctrl_follow_cursor(procctrl_context_t *ctx, int NBactive)
{
    int k = 0;
    for (int i = 0; i < ctx->nsel; i++)
    {
        if (ident_live(&ctx->sel[i]))
        {
            ctx->sel[k++] = ctx->sel[i];
        }
    }
    ctx->nsel = k;

    const int *order = procctrl_order(ctx);
    if (ident_live(&ctx->cur))
    {
        for (int a = 0; a < NBactive; a++)
        {
            if (order[a] == ctx->cur.pindex)
            {
                ctx->pindexActiveSelected = a;
                ctx->pindexSelected       = ctx->cur.pindex;
                ctx->cur_found            = 1;
                return;
            }
        }
    }
    ctx->pindexSelected = order[ctx->pindexActiveSelected];
    ident_set(&ctx->cur, ctx->pindexSelected);
    ctx->cur_found = 0;
}

/** procctrl_is_selected - 1 if the entry now in slot @pindex carries a mark. */
int procctrl_is_selected(const procctrl_context_t *ctx, int pindex)
{
    for (int i = 0; i < ctx->nsel; i++)
    {
        if (ctx->sel[i].pindex == pindex && ident_live(&ctx->sel[i]))
        {
            return 1;
        }
    }
    return 0;
}

/** procctrl_move - move the cursor to row @a and capture its identity at once. */
static void procctrl_move(procctrl_context_t *ctx, int a)
{
    ctx->pindexActiveSelected = a;
    ctx->pindexSelected       = procctrl_order(ctx)[a];
    ident_set(&ctx->cur, ctx->pindexSelected);
    ctx->cur_found = 1;
}


/**
 * @brief Handle a keyboard event in the procCTRL TUI.
 *
 * Dispatches key presses to navigation, mode
 * switching, or process control actions.
 */
void procctrl_handle_keyboard_event(procctrl_context_t *ctx, int ch, int NBactive)
{
    ctx->last_ch = ch;
    if (ctx->flog)
    {
        char            tbuf[64];
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        struct tm *tm_info = gmtime(&ts.tv_sec);
        size_t     len     = strftime(tbuf, sizeof(tbuf), "%Y%m%dT%H:%M:%S", tm_info);
        snprintf(tbuf + len, sizeof(tbuf) - len, ".%06ld", ts.tv_nsec / 1000);
        fprintf(ctx->flog, "%s Input: %d\\n", tbuf, ch);
        fflush(ctx->flog);
    }

    if (ch == 545 || ch == 560 || ch == 443 || ch == 564 || ch == 554)
    {
        ctx->procinfoproc->DisplayMode--;
        if (ctx->procinfoproc->DisplayMode < 1)
        {
            ctx->procinfoproc->DisplayMode = 6;
        }
        if (ctx->flog)
        {
            fprintf(ctx->flog, "  -> Mode changed to %d\\n", ctx->procinfoproc->DisplayMode);
            fflush(ctx->flog);
        }
    }
    else if (ch == 561 || ch == 566 || ch == 444 || ch == 565 || ch == 569)
    {
        ctx->procinfoproc->DisplayMode++;
        if (ctx->procinfoproc->DisplayMode > 6)
        {
            ctx->procinfoproc->DisplayMode = 1;
        }
        if (ctx->flog)
        {
            fprintf(ctx->flog, "  -> Mode changed to %d\\n", ctx->procinfoproc->DisplayMode);
            fflush(ctx->flog);
        }
    }
    else if (ch >= '0' && ch <= '9')
    {
        int cidx = ch - '0';
        int m    = ctx->procinfoproc->DisplayMode;
        if (m >= 0 && m < 10)
        {
            ctx->procinfoproc->col_visible[m][cidx] = !ctx->procinfoproc->col_visible[m][cidx];
        }
    }
    else if (ch == ANSI_KEY_F2)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
    }
    else if (ch == ANSI_KEY_F3)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_RESOURCES;
    }
    else if (ch == ANSI_KEY_F4)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TRIGGER;
    }
    else if (ch == ANSI_KEY_F5)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TIMING;
    }
    else if (ch == ANSI_KEY_F6)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_PROCINFO;
    }
    else if (ch == 'h')
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_HELP;
    }
    else if (ch == ANSI_KEY_CTRL_LEFT)
    {
        ctx->procinfoproc->DisplayMode--;
        if (ctx->procinfoproc->DisplayMode < 1)
        {
            ctx->procinfoproc->DisplayMode = 6;
        }
    }
    else if (ch == ANSI_KEY_CTRL_RIGHT)
    {
        ctx->procinfoproc->DisplayMode++;
        if (ctx->procinfoproc->DisplayMode > 6)
        {
            ctx->procinfoproc->DisplayMode = 1;
        }
    }
    else if (ch == 'x' || ch == 3)
    {
        ctx->loopOK = 0;
        ctx->Xexit  = 1;
    }
    else if (ch == 'f')
    {
        ctx->freeze = !ctx->freeze;
    }
    else if (ch == '+' || ch == '=')
    {
        ctx->frequ *= 1.2;
        if (ctx->frequ > 1000.0)
        {
            ctx->frequ = 1000.0;
        }
    }
    else if (ch == '-')
    {
        ctx->frequ /= 1.2;
        if (ctx->frequ < 0.1)
        {
            ctx->frequ = 0.1;
        }
    }
    else if (ch == ' ' && ctx->cur_found)
    {
        int i = 0;
        while (i < ctx->nsel && ctx->sel[i].pindex != ctx->cur.pindex)
        {
            i++;
        }
        if (i < ctx->nsel)
        {
            ctx->sel[i] = ctx->sel[--ctx->nsel]; /* unmark */
        }
        else if (ctx->nsel < PROCCTRL_SEL_MAX)
        {
            ctx->sel[ctx->nsel++] = ctx->cur;
        }
    }
    else if (ch == 'u')
    {
        ctx->nsel = 0;
    }
    else if (ch == ANSI_KEY_UP && NBactive > 0 && ctx->scan_shm != NULL)
    {
        procctrl_move(ctx, (ctx->pindexActiveSelected > 0) ? ctx->pindexActiveSelected - 1 : 0);
    }
    else if (ch == ANSI_KEY_DOWN && NBactive > 0 && ctx->scan_shm != NULL)
    {
        procctrl_move(ctx, (ctx->pindexActiveSelected + 1 < NBactive)
                               ? ctx->pindexActiveSelected + 1
                               : NBactive - 1);
    }
    else if (ch == ANSI_KEY_LEFT)
    {
        ctx->procinfoproc->selected_col--;
        if (ctx->procinfoproc->selected_col < 1)
        {
            ctx->procinfoproc->selected_col = 9;
        }
    }
    else if (ch == ANSI_KEY_RIGHT)
    {
        ctx->procinfoproc->selected_col++;
        if (ctx->procinfoproc->selected_col > 9)
        {
            ctx->procinfoproc->selected_col = 1;
        }
    }
    else if (ch == 'r')
    {
        /* free the selected stopped/crashed slot, only if it still holds that process */
        int i = ctx->pindexSelected;
        if (i >= 0 && i < PROCESSINFOLISTSIZE &&
            (pinfolist->active[i] == 2 || pinfolist->active[i] == 3))
        {
            processinfo_list_clear_slot(i, pinfolist->PIDarray[i]);
        }
    }
    else if (ch == 'R')
    {
        processinfo_list_reap(PROCESSINFO_REAP_FREE_SLOTS); /* every slot whose process is gone */
    }
    else if ((ch == 'T' || ch == 'K' || ch == 'I') && NBactive > 0)
    {
        int sig = (ch == 'T') ? SIGTERM : (ch == 'K') ? SIGKILL : SIGINT;
        for (int i = 0; i < ctx->nsel; i++)
        {
            procctrl_ident_kill(&ctx->sel[i], sig);
        }
        if (ctx->nsel == 0 && ctx->cur_found)
        {
            procctrl_ident_kill(&ctx->cur, sig);
        }
    }
    else if (ch == 's')
    {
        int m                           = ctx->procinfoproc->DisplayMode;
        ctx->procinfoproc->sort_mode[m] = m;
        if (ctx->procinfoproc->sort_col[m] == ctx->procinfoproc->selected_col)
        {
            ctx->procinfoproc->sort_dir[m] = !ctx->procinfoproc->sort_dir[m];
        }
        else
        {
            ctx->procinfoproc->sort_col[m] = ctx->procinfoproc->selected_col;
            ctx->procinfoproc->sort_dir[m] = 0;
        }
    }
    else if (ch == 'S')
    {
        int m_curr = ctx->procinfoproc->DisplayMode;
        int smod   = ctx->procinfoproc->sort_mode[m_curr];
        int scol   = ctx->procinfoproc->sort_col[m_curr];
        int sdir   = ctx->procinfoproc->sort_dir[m_curr];
        for (int m = 0; m < 10; m++)
        {
            ctx->procinfoproc->sort_mode[m] = smod;
            ctx->procinfoproc->sort_col[m]  = scol;
            ctx->procinfoproc->sort_dir[m]  = sdir;
        }
    }
    else if ((ch == 'p' || ch == 19 || ch == 'e') && NBactive > 0)
    {
        int val = (ch == 'p') ? -1 : (ch == 19) ? 2 : 3;
        for (int i = 0; i < ctx->nsel; i++)
        {
            procctrl_ident_ctrlval(ctx, &ctx->sel[i], val);
        }
        if (ctx->nsel == 0 && ctx->cur_found)
        {
            procctrl_ident_ctrlval(ctx, &ctx->cur, val);
        }
    }
    else if (ch == 'Z' && NBactive > 0)
    {
        for (int i = 0; i < PROCESSINFOLISTSIZE; i++)
        {
            if (ctx->procinfoproc->pinfoarray[i])
            {
                ctx->procinfoproc->pinfoarray[i]->loopcnt = 0;
            }
        }
    }
    else if (ch == 'z' && NBactive > 0)
    {
        /* marked entries plus the cursor, each only while it is still the same entry */
        for (int i = 0; i <= ctx->nsel; i++)
        {
            const procctrl_ident_t *id =
                (i < ctx->nsel) ? &ctx->sel[i] : (ctx->cur_found ? &ctx->cur : NULL);
            if (id != NULL && ident_live(id) && ctx->procinfoproc->pinfoarray[id->pindex] != NULL)
            {
                ctx->procinfoproc->pinfoarray[id->pindex]->loopcnt = 0;
            }
        }
    }
}

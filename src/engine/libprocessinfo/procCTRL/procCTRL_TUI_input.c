#include "procCTRL_TUI_internal.h"
#include "procCTRL_ansi.h"

/**
 * @brief Handle a keyboard event in the procCTRL TUI.
 *
 * Dispatches key presses to navigation, mode
 * switching, or process control actions.
 */
void procctrl_handle_keyboard_event(
    procctrl_context_t *ctx,
    int                ch,
    int                NBactive)
{
    ctx->last_ch = ch;
    if(ctx->flog)
    {
        char tbuf[64];
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        struct tm *tm_info = gmtime(&ts.tv_sec);
        size_t len = strftime(tbuf, sizeof(tbuf), "%Y%m%dT%H:%M:%S", tm_info);
        snprintf(tbuf + len, sizeof(tbuf) - len, ".%06ld", ts.tv_nsec / 1000);
        fprintf(ctx->flog, "%s Input: %d\\n", tbuf, ch);
        fflush(ctx->flog);
    }

    if(ch == 545 || ch == 560 || ch == 443 || ch == 564 || ch == 554)
    {
        ctx->procinfoproc->DisplayMode--;
        if(ctx->procinfoproc->DisplayMode < 1)
        {
            ctx->procinfoproc->DisplayMode = 6;
        }
        if(ctx->flog)
        {
            fprintf(ctx->flog, "  -> Mode changed to %d\\n", ctx->procinfoproc->DisplayMode);
            fflush(ctx->flog);
        }
    }
    else if(ch == 561 || ch == 566 || ch == 444 || ch == 565 || ch == 569)
    {
        ctx->procinfoproc->DisplayMode++;
        if(ctx->procinfoproc->DisplayMode > 6)
        {
            ctx->procinfoproc->DisplayMode = 1;
        }
        if(ctx->flog)
        {
            fprintf(ctx->flog, "  -> Mode changed to %d\\n", ctx->procinfoproc->DisplayMode);
            fflush(ctx->flog);
        }
    }
    else if(ch >= '0' && ch <= '9')
    {
        int cidx = ch - '0';
        int m = ctx->procinfoproc->DisplayMode;
        if(m >= 0 && m < 10)
        {
            ctx->procinfoproc->col_visible[m][cidx] = !ctx->procinfoproc->col_visible[m][cidx];
        }
    }
    else if(ch == ANSI_KEY_F2)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_CTRL;
    }
    else if(ch == ANSI_KEY_F3)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_RESOURCES;
    }
    else if(ch == ANSI_KEY_F4)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TRIGGER;
    }
    else if(ch == ANSI_KEY_F5)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_TIMING;
    }
    else if(ch == ANSI_KEY_F6)
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_PROCINFO;
    }
    else if(ch == 'h')
    {
        ctx->procinfoproc->DisplayMode = PROCCTRL_DISPLAYMODE_HELP;
    }
    else if(ch == ANSI_KEY_CTRL_LEFT)
    {
        ctx->procinfoproc->DisplayMode--;
        if(ctx->procinfoproc->DisplayMode < 1)
        {
            ctx->procinfoproc->DisplayMode = 6;
        }
    }
    else if(ch == ANSI_KEY_CTRL_RIGHT)
    {
        ctx->procinfoproc->DisplayMode++;
        if(ctx->procinfoproc->DisplayMode > 6)
        {
            ctx->procinfoproc->DisplayMode = 1;
        }
    }
    else if(ch == 'x' || ch == 3)
    {
        ctx->loopOK = 0;
        ctx->Xexit = 1;
    }
    else if(ch == 'f')
    {
        ctx->freeze = !ctx->freeze;
    }
    else if(ch == '+' || ch == '=')
    {
        ctx->frequ *= 1.2;
        if(ctx->frequ > 1000.0)
        {
            ctx->frequ = 1000.0;
        }
    }
    else if(ch == '-')
    {
        ctx->frequ /= 1.2;
        if(ctx->frequ < 0.1)
        {
            ctx->frequ = 0.1;
        }
    }
    else if(ch == ' ' && ctx->pindexSelected >= 0)
    {
        ctx->procinfoproc->selectedarray[ctx->pindexSelected] =
            !ctx->procinfoproc->selectedarray[ctx->pindexSelected];
    }
    else if(ch == ANSI_KEY_UP && NBactive > 0)
    {
        ctx->pindexActiveSelected--;
        if(ctx->pindexActiveSelected < 0)
        {
            ctx->pindexActiveSelected = 0;
        }
    }
    else if(ch == ANSI_KEY_DOWN && NBactive > 0)
    {
        ctx->pindexActiveSelected++;
        if(ctx->pindexActiveSelected >= NBactive)
        {
            ctx->pindexActiveSelected = NBactive - 1;
        }
    }
    else if(ch == ANSI_KEY_LEFT)
    {
        ctx->procinfoproc->selected_col--;
        if(ctx->procinfoproc->selected_col < 1)
        {
            ctx->procinfoproc->selected_col = 9;
        }
    }
    else if(ch == ANSI_KEY_RIGHT)
    {
        ctx->procinfoproc->selected_col++;
        if(ctx->procinfoproc->selected_col > 9)
        {
            ctx->procinfoproc->selected_col = 1;
        }
    }
    else if(ch == 'r' || ch == 'R')
    {
        int all = (ch == 'R');
        for(int i = 0; i < PROCESSINFOLISTSIZE; i++)
        {
            if(all || i == ctx->pindexSelected)
            {
                if(pinfolist->active[i] == 2 || pinfolist->active[i] == 3)
                {
                    pinfolist->active[i] = 0;
                }
            }
        }
    }
    else if((ch == 'T' || ch == 'K' || ch == 'I') && NBactive > 0)
    {
        int sig = (ch == 'T') ? SIGTERM : (ch == 'K') ? SIGKILL : SIGINT;
        int any_sel = 0;
        for(int i = 0; i < PROCESSINFOLISTSIZE; i++) if(ctx->procinfoproc->selectedarray[i])
            {
                kill(pinfolist->PIDarray[i], sig);
                any_sel = 1;
            }
        if(!any_sel && ctx->pindexSelected >= 0)
        {
            kill(pinfolist->PIDarray[ctx->pindexSelected], sig);
        }
    }
    else if(ch == 's')
    {
        int m = ctx->procinfoproc->DisplayMode;
        ctx->procinfoproc->sort_mode[m] = m;
        if(ctx->procinfoproc->sort_col[m] == ctx->procinfoproc->selected_col)
        {
            ctx->procinfoproc->sort_dir[m] = !ctx->procinfoproc->sort_dir[m];
        }
        else
        {
            ctx->procinfoproc->sort_col[m] = ctx->procinfoproc->selected_col;
            ctx->procinfoproc->sort_dir[m] = 0;
        }
    }
    else if(ch == 'S')
    {
        int m_curr = ctx->procinfoproc->DisplayMode;
        int smod = ctx->procinfoproc->sort_mode[m_curr];
        int scol = ctx->procinfoproc->sort_col[m_curr];
        int sdir = ctx->procinfoproc->sort_dir[m_curr];
        for(int m = 0; m < 10; m++)
        {
            ctx->procinfoproc->sort_mode[m] = smod;
            ctx->procinfoproc->sort_col[m] = scol;
            ctx->procinfoproc->sort_dir[m] = sdir;
        }
    }
    else if((ch == 'p' || ch == 19 || ch == 'e') && NBactive > 0)
    {
        int val = (ch == 'p') ? -1 : (ch == 19) ? 2 : 3;
        int any_sel = 0;
        for(int i = 0; i < PROCESSINFOLISTSIZE; i++) if(ctx->procinfoproc->selectedarray[i])
            {
                if(val == -1 && ctx->procinfoproc->pinfoarray[i])
                {
                    ctx->procinfoproc->pinfoarray[i]->CTRLval = (ctx->procinfoproc->pinfoarray[i]->CTRLval == 0) ? 1 :
                            0;
                }
                else if(ctx->procinfoproc->pinfoarray[i])
                {
                    ctx->procinfoproc->pinfoarray[i]->CTRLval = val;
                }
                any_sel = 1;
            }
        if(!any_sel && ctx->pindexSelected >= 0 && ctx->procinfoproc->pinfoarray[ctx->pindexSelected])
        {
            if(val == -1)
            {
                ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->CTRLval =
                    (ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->CTRLval == 0) ? 1 : 0;
            }
            else
            {
                ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->CTRLval = val;
            }
        }
    }
    else if((ch == 'z' || ch == 'Z') && NBactive > 0)
    {
        int all = (ch == 'Z');
        for(int i = 0; i < PROCESSINFOLISTSIZE; i++) if((all || ctx->procinfoproc->selectedarray[i])
                    && ctx->procinfoproc->pinfoarray[i])
            {
                ctx->procinfoproc->pinfoarray[i]->loopcnt = 0;
            }
        if(!all && !ctx->procinfoproc->selectedarray[ctx->pindexSelected] && ctx->pindexSelected >= 0
                && ctx->procinfoproc->pinfoarray[ctx->pindexSelected])
        {
            ctx->procinfoproc->pinfoarray[ctx->pindexSelected]->loopcnt = 0;
        }
    }
}

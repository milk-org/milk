/**
 * @file    fpsCTRL_TUI_process_user_key.c
 * @brief   TUI key input processing
 */


#include "fps.h"
#include "fpsCTRL_TUIcompat.h"
#include "engine/libfpsseq/fpsseq.h"

#include "fps_CONFstart.h"
#include "fps_FPSremove.h"

#define ctrl(x) ((x) & 0x1f)

/**
 * fpsCTRL_inline_edit_param - edit a parameter inline
 * within the TUI.
 *
 * Shows a prompt at the bottom of the screen with
 * parameter name, type, and current value. Accepts
 * text input character-by-character in raw mode.
 * ESC aborts, ENTER confirms.
 *
 * @fps:      FPS array
 * @fps_idx: index into fps array
 * @p_idx:   parameter index within the FPS
 *
 * Return: 0 on success or abort
 */
static int fpsCTRL_inline_edit_param(FPS *fps, int fps_idx, int p_idx)
{
    char curval[200];
    functionparameter_GetParamValueString(&fps[fps_idx].parray[p_idx], curval, 200);

    char typestr[STRINGMAXLEN_FPSTYPE];
    functionparameter_GetTypeString(fps[fps_idx].parray[p_idx].type, typestr);

    /* Strip FPS name prefix from keyword */
    const char *display_kw = fps[fps_idx].parray[p_idx].keywordfull;
    int         prefix_len = strlen(fps[fps_idx].md->name);
    if (strncmp(display_kw, fps[fps_idx].md->name, prefix_len) == 0 &&
        display_kw[prefix_len] == '.')
    {
        display_kw += prefix_len + 1;
    }

    /* Flush current frame, then draw edit bar */
    sc_frame_flush();

    /* Show cursor */
    if (write(STDOUT_FILENO, "\033[?25h", 6) < 0)
    {
    }

    /* Position at bottom of terminal */
    {
        char posbuf[32];
        int  n = snprintf(posbuf, sizeof(posbuf), "\033[%d;1H", sc_term_rows);
        if (n > 0)
        {
            if (write(STDOUT_FILENO, posbuf, (size_t) n) < 0)
            {
            }
        }
    }

    /* Clear the bottom line */
    if (write(STDOUT_FILENO, "\033[2K", 4) < 0)
    {
    }

    /* Print prompt with param context */
    {
        char prompt[512];
        int  n;
        if (!(fps[fps_idx].parray[p_idx].fpflag & FPFLAG_WRITESTATUS))
        {
            n = snprintf(prompt, sizeof(prompt),
                         "\033[1;31m"
                         " [%s] %s = %s  (read-only)"
                         "\033[0m",
                         typestr, display_kw, curval);
            if (n > 0)
            {
                if (write(STDOUT_FILENO, prompt, (size_t) n) < 0)
                {
                }
            }
            /* Wait for any key, then return */
            usleep(800000);
            /* Drain any buffered keys */
            while (ansi_get_key() != ANSI_KEY_NONE)
            {
            }
            /* Hide cursor */
            if (write(STDOUT_FILENO, "\033[?25l", 6) < 0)
            {
            }
            return 0;
        }

        n = snprintf(prompt, sizeof(prompt),
                     "\033[1;36m"
                     " [%s] %s"
                     "\033[0m"
                     " (was: "
                     "\033[33m%s\033[0m"
                     ") new value: ",
                     typestr, display_kw, curval);
        if (n > 0)
        {
            if (write(STDOUT_FILENO, prompt, (size_t) n) < 0)
            {
            }
        }
    }

    /* Read input character-by-character */
    char buf[200];
    int  bufpos  = 0;
    int  maxlen  = (int) sizeof(buf) - 1;
    int  aborted = 0;

    for (;;)
    {
        usleep(10000);
        int key = ansi_get_key();

        if (key == ANSI_KEY_NONE)
        {
            continue;
        }

        /* ESC — abort */
        if (key == 27)
        {
            aborted = 1;
            break;
        }

        /* ENTER — confirm */
        if (key == 10 || key == 13)
        {
            break;
        }

        /* Backspace (127 or ctrl-h) */
        if (key == 127 || key == 8)
        {
            if (bufpos > 0)
            {
                bufpos--;
                if (write(STDOUT_FILENO, "\b \b", 3) < 0)
                {
                }
            }
            continue;
        }

        /* Ctrl+U — clear input */
        if (key == ctrl('u'))
        {
            while (bufpos > 0)
            {
                bufpos--;
                if (write(STDOUT_FILENO, "\b \b", 3) < 0)
                {
                }
            }
            continue;
        }

        /* Ignore non-printable / special keys */
        if (key < 32 || key > 126)
        {
            continue;
        }

        /* Printable char */
        if (bufpos < maxlen)
        {
            buf[bufpos++] = (char) key;
            char echo     = (char) key;
            if (write(STDOUT_FILENO, &echo, 1) < 0)
            {
            }
        }
    }
    buf[bufpos] = '\0';

    /* Hide cursor */
    if (write(STDOUT_FILENO, "\033[?25l", 6) < 0)
    {
    }

    if (!aborted && bufpos > 0)
    {
        if (functionparameter_SetParamValue_fromString(&fps[fps_idx], p_idx, buf) != 0)
        {
            /* Show error briefly */
            char errmsg[] = "\033[1;31m  ERROR: invalid value"
                            "\033[0m";
            if (write(STDOUT_FILENO, errmsg, sizeof(errmsg) - 1) < 0)
            {
            }
            usleep(500000);
        }
        else
        {
            /* processinfo change tracking */
            if (strncmp(fps[fps_idx].parray[p_idx].keywordfull, ".procinfo.", 10) == 0)
            {
                fps[fps_idx].md->processinfo_change_cnt++;
            }

            /* Notify GUI */
            fps[fps_idx].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

            /* Save to disk if needed */
            if (fps[fps_idx].parray[p_idx].fpflag & FPFLAG_SAVEONCHANGE)
            {
                functionparameter_WriteParameterToDisk(&fps[fps_idx], p_idx, "setval",
                                                       "UserInputSetParamValue");

                functionparameter_SaveFPS2disk(&fps[fps_idx]);
            }
        }
    }

    return 0;
}

/**
 * @brief Process a keyboard event in the fpsCTRL TUI.
 *
 * Dispatches key presses to navigation, editing,
 * or command handlers based on the current mode.
 */
int fpsCTRL_TUI_process_user_key(int                   ch,
                                 FPS                  *fps,
                                 KEYWORD_TREE_NODE    *keywnode,
                                 FPSCTRL_TASK_ENTRY   *fpsctrltasklist __attribute__((unused)),
                                 FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist __attribute__((unused)),
                                 FPSCTRL_PROCESS_VARS *fpsCTRLvar)
{
    int loopOK = 1;
    int fps_idx;
    int p_idx;

    if (ch != -1)
    {
        if (fpsCTRLvar->NBfps > 0)
        {
            int selected_fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            if (fps[selected_fpsindex].md == NULL)
            {
                // If the selected FPS is invalid (was deleted), ignore commands that operate on it,
                // but allow navigation and quit commands.
                if (ch == ' ' || ch == 'T' || ch == ctrl('t') || ch == 'E' || ch == 'R' ||
                    ch == 'r' || ch == 'C' || ch == 'O' || ch == 'c' || ch == ctrl('a') ||
                    ch == ctrl('e') || ch == ctrl('r') || ch == ctrl('o'))
                {
                    return loopOK;
                }
            }
        }

        if (fpsCTRLvar->search_mode > 0)
        {
            if (ch == 27) // ESC
            {
                fpsCTRLvar->search_mode      = 0;
                fpsCTRLvar->search_string[0] = '\0';
                return loopOK;
            }
            else if (ch == 10 || ch == 13) // ENTER
            {
                fpsCTRLvar->search_mode = 2; // Lock in search
                return loopOK;
            }
            else if (ch == 127 || ch == 8) // Backspace
            {
                int len = strlen(fpsCTRLvar->search_string);
                if (len > 0)
                {
                    fpsCTRLvar->search_string[len - 1] = '\0';
                }
                return loopOK;
            }
            else if (ch >= 32 && ch <= 126)
            {
                int len = strlen(fpsCTRLvar->search_string);
                if (len < sizeof(fpsCTRLvar->search_string) - 1)
                {
                    fpsCTRLvar->search_string[len]     = (char) ch;
                    fpsCTRLvar->search_string[len + 1] = '\0';
                }
                return loopOK;
            }
            // If not printable and not enter/esc/backspace, allow fall-through
            // for navigation keys (UP/DOWN/LEFT/RIGHT/PGUP/PGDN)
            if (ch != ANSI_KEY_UP && ch != ANSI_KEY_DOWN && ch != ANSI_KEY_PGUP &&
                ch != ANSI_KEY_PGDN && ch != ANSI_KEY_LEFT && ch != ANSI_KEY_RIGHT)
            {
                return loopOK; // Swallow other commands
            }
        }

        /* CTRL+LEFT/RIGHT handled below in the switch */
        if (ch == 'v' || ch == 'V')
        {
            fpsCTRLvar->fpsCTRL_DisplayVerbose = !fpsCTRLvar->fpsCTRL_DisplayVerbose;
            if (write(STDOUT_FILENO, "\033[2J", 4) < 0)
            {
            }
        }

        switch (ch)
        {
        case 3:   // Ctrl+C
        case 'x': // Exit control screen
            loopOK = 0;
            break;

        case 27: // ESC
            fpsCTRLvar->search_mode      = 0;
            fpsCTRLvar->search_string[0] = '\0';
            break;

        case '/': // Enter search mode
            fpsCTRLvar->search_mode      = 1;
            fpsCTRLvar->search_string[0] = '\0';
            break;

        case 'S': // Toggle sort mode (legacy)
            fpsCTRLvar->sort_mode = (fpsCTRLvar->sort_mode + 1) % 2;
            break;

        case ']': // Next sort column
            fpsCTRLvar->sort_mode = (fpsCTRLvar->sort_mode + 1) % 3;
            break;

        case '[': // Toggle sort direction (future)
            break;

        case 'y': // Yank to tmux buffer
        {
            fps_idx           = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            p_idx             = keywnode[fpsCTRLvar->nodeSelected].pindex;
            char yankstr[512] = { 0 };
            if (keywnode[fpsCTRLvar->nodeSelected].leaf == 1)
            {
                char valstr[200] = { 0 };
                functionparameter_GetParamValueString(&fps[fps_idx].parray[p_idx], valstr, 200);
                snprintf(yankstr, sizeof(yankstr), "%s = %s",
                         fps[fps_idx].parray[p_idx].keywordfull, valstr);
            }
            else
            {
                snprintf(yankstr, sizeof(yankstr), "%s",
                         keywnode[fpsCTRLvar->nodeSelected].keywordfull);
            }
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "tmux set-buffer \"%s\" 2>/dev/null", yankstr);
            if (system(cmd) != 0)
            {
            }
        }
        break;

        case '\t':
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
            {
                char names[20][FPSSEQ_NAME_MAX];
                int  count = milkseq_list(names, 20);
                if (count > 0)
                {
                    int current_idx = -1;
                    if (fpsCTRLvar->milkseq_state != NULL)
                    {
                        for (int seq_idx = 0; seq_idx < count; seq_idx++)
                        {
                            if (strcmp(names[seq_idx], fpsCTRLvar->milkseq_name) == 0)
                            {
                                current_idx = seq_idx;
                                break;
                            }
                        }
                    }
                    int next_idx = (current_idx + 1) % count;
                    if (fpsCTRLvar->milkseq_state != NULL)
                    {
                        milkseq_disconnect((MILKSEQ_STATE *) fpsCTRLvar->milkseq_state);
                        fpsCTRLvar->milkseq_state = NULL;
                    }
                    fpsCTRLvar->milkseq_state = milkseq_connect(names[next_idx]);
                    if (fpsCTRLvar->milkseq_state)
                    {
                        strncpy(fpsCTRLvar->milkseq_name, names[next_idx],
                                sizeof(fpsCTRLvar->milkseq_name) - 1);
                        fpsCTRLvar->milkseq_name[sizeof(fpsCTRLvar->milkseq_name) - 1] = '\0';
                    }
                    else
                    {
                        fpsCTRLvar->milkseq_name[0] = '\0';
                    }
                }
            }
            break;

            // ============ SCREENS

        case ANSI_KEY_RESIZE:
            TUI_clearscreen(NULL, NULL);
            break;

        case 'h': // help
            fpsCTRLvar->fpsCTRL_DisplayMode = 1;
            if (write(STDOUT_FILENO, "\033[2J", 4) < 0)
            {
            }
            break;

        case '?': // fps log
            fpsCTRLvar->fpsCTRL_DisplayMode = 2;
            if (write(STDOUT_FILENO, "\033[2J", 4) < 0)
            {
            }
            break;

        case ANSI_KEY_F2: // control
            fpsCTRLvar->fpsCTRL_DisplayMode = 3;
            if (write(STDOUT_FILENO, "\033[2J", 4) < 0)
            {
            }
            break;

        case ANSI_KEY_F3: // scheduler
            fpsCTRLvar->fpsCTRL_DisplayMode = 4;
            if (write(STDOUT_FILENO, "\033[2J", 4) < 0)
            {
            }
            break;

        case ANSI_KEY_CTRL_RIGHT:
            fpsCTRLvar->fpsCTRL_DisplayMode++;
            if (fpsCTRLvar->fpsCTRL_DisplayMode > 4)
            {
                fpsCTRLvar->fpsCTRL_DisplayMode = 1;
            }
            if (write(STDOUT_FILENO, "\033[2J", 4) < 0)
            {
            }
            break;

        case ANSI_KEY_CTRL_LEFT:
            fpsCTRLvar->fpsCTRL_DisplayMode--;
            if (fpsCTRLvar->fpsCTRL_DisplayMode < 1)
            {
                fpsCTRLvar->fpsCTRL_DisplayMode = 4;
            }
            if (write(STDOUT_FILENO, "\033[2J", 4) < 0)
            {
            }
            break;

        case 's': // (re)scan
            functionparameter_scan_fps(fpsCTRLvar->mode, fpsCTRLvar->fpsnamemask, fps, keywnode,
                                       &fpsCTRLvar->NBkwn, &fpsCTRLvar->NBfps, &fpsCTRLvar->NBindex,
                                       0);
            TUI_clearscreen(NULL, NULL);
            break;

        case ctrl('e'): // stop conf/run, then erase FPS
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstop(&fps[fps_idx]);
            functionparameter_RUNstop(&fps[fps_idx]);
            functionparameter_FPSremove(&fps[fps_idx]);

            functionparameter_scan_fps(fpsCTRLvar->mode, fpsCTRLvar->fpsnamemask, fps, keywnode,
                                       &fpsCTRLvar->NBkwn, &(fpsCTRLvar->NBfps),
                                       &fpsCTRLvar->NBindex, 0);
            TUI_clearscreen(NULL, NULL);
            fpsCTRLvar->run_display      = 0; // skip next display
            fpsCTRLvar->fpsindexSelected = 0; // safeguard
            break;


        case 'T': // initialize tmux session
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_FPS_tmux_init(&fps[fps_idx]);
            break;

        case ctrl('t'): // kill tmux session
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_FPS_tmux_kill(&fps[fps_idx]);
            break;

        case ctrl('a'): // attach to tmux session
        {
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            TUI_exit();
            functionparameter_FPS_tmux_attach(&fps[fps_idx]);
            {
                short unsigned int wrow = 0, wcol = 0;
                TUI_init_terminal(&wrow, &wcol);
            }
        }
        break;


        case 'E': // Stop conf/run, erase FPS, kill tmux
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstop(&fps[fps_idx]);
            functionparameter_RUNstop(&fps[fps_idx]);

            functionparameter_FPSremove(&fps[fps_idx]);
            functionparameter_scan_fps(fpsCTRLvar->mode, fpsCTRLvar->fpsnamemask, fps, keywnode,
                                       &fpsCTRLvar->NBkwn, &fpsCTRLvar->NBfps, &fpsCTRLvar->NBindex,
                                       0);
            TUI_clearscreen(NULL, NULL);
            // safeguard in case current selection disappears
            fpsCTRLvar->fpsindexSelected = 0;
            break;

        case ANSI_KEY_UP:
            fpsCTRLvar->direction = -1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0)
                {
                    l = 0;
                }
                fpsCTRLvar->GUIlineSelected[l]--;
                if (fpsCTRLvar->GUIlineSelected[l] < 0)
                {
                    fpsCTRLvar->GUIlineSelected[l] = 0;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
            {
                fpsCTRLvar->scheduler_wrowstart--;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
            {
                fpsCTRLvar->help_wrowstart--;
            }
            break;


        case ANSI_KEY_DOWN:
            fpsCTRLvar->direction = 1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0)
                {
                    l = 0;
                }
                fpsCTRLvar->GUIlineSelected[l]++;
                if (fpsCTRLvar->GUIlineSelected[l] > fpsCTRLvar->NBindex - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] = fpsCTRLvar->NBindex - 1;
                }
                if (fpsCTRLvar->GUIlineSelected[l] >
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] =
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
            {
                fpsCTRLvar->scheduler_wrowstart++;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
            {
                fpsCTRLvar->help_wrowstart++;
            }
            break;

        case ANSI_KEY_PGUP:
            fpsCTRLvar->direction = -1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0)
                {
                    l = 0;
                }
                fpsCTRLvar->GUIlineSelected[l] -= 10;
                if (fpsCTRLvar->GUIlineSelected[l] < 0)
                {
                    fpsCTRLvar->GUIlineSelected[l] = 0;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
            {
                fpsCTRLvar->scheduler_wrowstart -= 10;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
            {
                fpsCTRLvar->help_wrowstart -= 10;
            }
            break;

        case ANSI_KEY_PGDN:
            fpsCTRLvar->direction = 1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0)
                {
                    l = 0;
                }
                fpsCTRLvar->GUIlineSelected[l] += 10;
                while (fpsCTRLvar->GUIlineSelected[l] > fpsCTRLvar->NBindex - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] = fpsCTRLvar->NBindex - 1;
                }
                while (fpsCTRLvar->GUIlineSelected[l] >
                       keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] =
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
            {
                fpsCTRLvar->scheduler_wrowstart += 10;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
            {
                fpsCTRLvar->help_wrowstart += 10;
            }
            break;


        case ANSI_KEY_LEFT:
            if (fpsCTRLvar->directorynodeSelected != 0) // ROOT has no parent
            {
                fpsCTRLvar->directorynodeSelected =
                    keywnode[fpsCTRLvar->directorynodeSelected].parent_index;
                fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
                fpsCTRLvar->currentlevel = keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;
            }
            break;


        case ANSI_KEY_RIGHT:
            if (keywnode[fpsCTRLvar->nodeSelected].leaf == 0) // this is a directory
            {
                fpsCTRLvar->directorynodeSelected = fpsCTRLvar->nodeSelected;
                fpsCTRLvar->currentlevel = keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;
            }
            break;

        case 13: // enter key (CR in raw mode)
        case 10: // enter key (LF fallback)
            if (keywnode[fpsCTRLvar->nodeSelected].leaf == 1)
            {
                int ei    = fpsCTRLvar->fpsindexSelected;
                int ep    = fpsCTRLvar->pindexSelected;
                int etype = fps[ei].parray[ep].type;

                if (etype == FPTYPE_ONOFF)
                {
                    /* Toggle ON/OFF inline */
                    if (fps[ei].parray[ep].fpflag & FPFLAG_WRITESTATUS)
                    {
                        if (fps[ei].parray[ep].fpflag & FPFLAG_ONOFF)
                        {
                            fps[ei].parray[ep].fpflag &= ~FPFLAG_ONOFF;
                            fps[ei].parray[ep].val.i32[0] = 0;
                        }
                        else
                        {
                            fps[ei].parray[ep].fpflag |= FPFLAG_ONOFF;
                            fps[ei].parray[ep].val.i32[0] = 1;
                        }
                        if (fps[ei].parray[ep].fpflag & FPFLAG_SAVEONCHANGE)
                        {
                            functionparameter_WriteParameterToDisk(&fps[ei], ep, "setval",
                                                                   "UserInputSetParamValue");
                        }
                        fps[ei].parray[ep].cnt0++;
                        fps[ei].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                    }
                }
                else
                {
                    fpsCTRL_inline_edit_param(fps, ei, ep);
                }
            }
            break;

        case ' ':
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            p_idx   = keywnode[fpsCTRLvar->nodeSelected].pindex;

            // toggles ON / OFF
            if (fps[fps_idx].parray[p_idx].fpflag & FPFLAG_WRITESTATUS)
            {
                if (fps[fps_idx].parray[p_idx].type == FPTYPE_ONOFF)
                {
                    if (fps[fps_idx].parray[p_idx].fpflag & FPFLAG_ONOFF) // ON -> OFF
                    {
                        fps[fps_idx].parray[p_idx].fpflag &= ~FPFLAG_ONOFF;
                        fps[fps_idx].parray[p_idx].val.i32[0] = 0;
                    }
                    else // OFF -> ON
                    {
                        fps[fps_idx].parray[p_idx].fpflag |= FPFLAG_ONOFF;
                        fps[fps_idx].parray[p_idx].val.i32[0] = 1;
                    }

                    // Save to disk
                    if (fps[fps_idx].parray[p_idx].fpflag & FPFLAG_SAVEONCHANGE)
                    {
                        functionparameter_WriteParameterToDisk(&fps[fps_idx], p_idx, "setval",
                                                               "UserInputSetParamValue");
                    }
                    fps[fps_idx].parray[p_idx].cnt0++;
                    fps[fps_idx].md->signal |=
                        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
                }
            }

            if (fps[fps_idx].parray[p_idx].type == FPTYPE_EXECFILENAME)
            {
                char cmd[512];
                snprintf(cmd, sizeof(cmd), "tmux send-keys -t %s:run \"cd %s\" C-m",
                         fps[fps_idx].md->name, fps[fps_idx].md->workdir);
                if (system(cmd) != 0)
                {
                } // Corrected escaping for "cd %s"
                snprintf(cmd, sizeof(cmd), "tmux send-keys -t %s:run \"%s %s/%s.fps\" C-m",
                         fps[fps_idx].md->name, fps[fps_idx].parray[p_idx].val.string[0],
                         fps[fps_idx].md->datadir, fps[fps_idx].md->name);
                if (system(cmd) != 0)
                {
                } // Corrected escaping for "%s %s/%s.fps"
            }

            break;


        case 'u': // update conf process
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            fps[fps_idx].md->signal |=
                FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            break;

        case 'R': // start run process
        {
            fps_idx           = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            FPS *selected_fps = &fps[fps_idx];
            functionparameter_FPS_tmux_ensure(selected_fps);
            char progexec[1024];
            if ((strlen(selected_fps->md->execfullpath) > 0) &&
                (strcmp(selected_fps->md->execfullpath, "unknown") != 0))
            {
                strncpy(progexec, selected_fps->md->execfullpath, 1023);
            }
            else
            {
                snprintf(progexec, 1024, "%s-exec", selected_fps->md->callprogname);
            }

            EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run \" cd %s\" C-m",
                                           selected_fps->md->name, selected_fps->md->workdir);
            EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run \" %s %s:runstart\" C-m",
                                           selected_fps->md->name, progexec,
                                           selected_fps->md->name);
        }
        break;

        case ctrl('r'): // stop run process
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_RUNstop(&fps[fps_idx]);
            break;

        case 'r': // legacy stop run process
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_RUNstop(&fps[fps_idx]);
            break;


        case 'C': // legacy start conf process
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstart(&fps[fps_idx]);
            break;

        case 'O': // start conf process
        {
            fps_idx           = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            FPS *selected_fps = &fps[fps_idx];
            functionparameter_FPS_tmux_ensure(selected_fps);
            char progexec[1024];
            if ((strlen(selected_fps->md->execfullpath) > 0) &&
                (strcmp(selected_fps->md->execfullpath, "unknown") != 0))
            {
                strncpy(progexec, selected_fps->md->execfullpath, 1023);
            }
            else
            {
                snprintf(progexec, 1024, "%s-exec", selected_fps->md->callprogname);
            }

            EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" cd %s\" C-m",
                                           selected_fps->md->name, selected_fps->md->workdir);
            EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" %s %s:confstart\" C-m",
                                           selected_fps->md->name, progexec,
                                           selected_fps->md->name);
        }
        break;

        case ctrl('o'): // stop conf process
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstop(&fps[fps_idx]);
            break;

        case 'c': // kill conf process
            fps_idx = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstop(&fps[fps_idx]);
            break;

        case 'l': // list all parameters
            TUI_exit();
            if (system("clear") != 0)
            {
            } // Corrected escaping for "clear"
            printf("FPS entries - Full list \n\n");
            for (int kwn_idx = 0; kwn_idx < fpsCTRLvar->NBkwn; kwn_idx++)
            {
                if (keywnode[kwn_idx].leaf == 1)
                {
                    printf("%4d  %4d  %s\n", keywnode[kwn_idx].fpsindex, keywnode[kwn_idx].pindex,
                           keywnode[kwn_idx].keywordfull);
                }
            }
            printf("  TOTAL :  %d nodes\n\n", fpsCTRLvar->NBkwn);
            printf("Press Enter to Continue\n");
            while (getchar() != '\n')
            {
            }
            {
                short unsigned int wrow = 0, wcol = 0;
                TUI_init_terminal(&wrow, &wcol);
            }
            break;
        }
    }

    return (loopOK);
}

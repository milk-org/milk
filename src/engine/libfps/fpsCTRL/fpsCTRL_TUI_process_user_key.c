/**
 * @file    fpsCTRL_TUI_process_user_key.c
 * @brief   TUI key input processing
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "fps.h"
#include "fps_internal.h"
#include "fpsCTRL_TUIcompat.h"
#include "fpsCTRL_globals.h"
#include "engine/libfpsseq/fpsseq.h"

#include "fps_CONFstart.h"
#include "fps_CONFstop.h"
#include "fps_FPSremove.h"
#include "fps_RUNstart.h"
#include "fps_RUNstop.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_outlog.h"
#include "fps_processcmdline.h"
#include "fps_read_fpsCMD_fifo.h"
#include "fps_save2disk.h"
#include "fps_scan.h"
#include "fps_tmux.h"
#include "fps_userinputsetparamvalue.h"
#include "fps_printparameter_valuestring.h"
#include "fps_GetTypeString.h"

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
 * @fpsindex: index into fps array
 * @pindex:   parameter index within the FPS
 *
 * Return: 0 on success or abort
 */
static int fpsCTRL_inline_edit_param(
    FPS *fps,
    int                        fpsindex,
    int                        pindex
)
{
    char curval[200];
    functionparameter_GetParamValueString(
        &fps[fpsindex].parray[pindex],
        curval,
        200);

    char typestr[STRINGMAXLEN_FPSTYPE];
    functionparameter_GetTypeString(
        fps[fpsindex].parray[pindex].type,
        typestr);

    /* Strip FPS name prefix from keyword */
    const char *display_kw =
        fps[fpsindex].parray[pindex].keywordfull;
    int prefix_len = strlen(fps[fpsindex].md->name);
    if (strncmp(display_kw,
                fps[fpsindex].md->name,
                prefix_len) == 0
        && display_kw[prefix_len] == '.')
    {
        display_kw += prefix_len + 1;
    }

    /* Flush current frame, then draw edit bar */
    sc_frame_flush();

    /* Show cursor */
    if (write(STDOUT_FILENO,
              "\033[?25h", 6) < 0) {}

    /* Position at bottom of terminal */
    {
        char posbuf[32];
        int n = snprintf(posbuf, sizeof(posbuf),
            "\033[%d;1H", sc_term_rows);
        if (n > 0)
        {
            if (write(STDOUT_FILENO,
                      posbuf, (size_t) n) < 0) {}
        }
    }

    /* Clear the bottom line */
    if (write(STDOUT_FILENO,
              "\033[2K", 4) < 0) {}

    /* Print prompt with param context */
    {
        char prompt[512];
        int n;
        if (!(fps[fpsindex].parray[pindex].fpflag
              & FPFLAG_WRITESTATUS))
        {
            n = snprintf(prompt, sizeof(prompt),
                "\033[1;31m"
                " [%s] %s = %s  (read-only)"
                "\033[0m",
                typestr, display_kw, curval);
            if (n > 0)
            {
                if (write(STDOUT_FILENO,
                          prompt, (size_t) n)
                    < 0) {}
            }
            /* Wait for any key, then return */
            usleep(800000);
            /* Drain any buffered keys */
            while (ansi_get_key() != ANSI_KEY_NONE) {}
            /* Hide cursor */
            if (write(STDOUT_FILENO,
                      "\033[?25l", 6) < 0) {}
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
            if (write(STDOUT_FILENO,
                      prompt, (size_t) n) < 0) {}
        }
    }

    /* Read input character-by-character */
    char buf[200];
    int  bufpos = 0;
    int  maxlen = (int) sizeof(buf) - 1;
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
                if (write(STDOUT_FILENO,
                          "\b \b", 3) < 0) {}
            }
            continue;
        }

        /* Ctrl+U — clear input */
        if (key == ctrl('u'))
        {
            while (bufpos > 0)
            {
                bufpos--;
                if (write(STDOUT_FILENO,
                          "\b \b", 3) < 0) {}
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
            char echo = (char) key;
            if (write(STDOUT_FILENO,
                      &echo, 1) < 0) {}
        }
    }
    buf[bufpos] = '\0';

    /* Hide cursor */
    if (write(STDOUT_FILENO,
              "\033[?25l", 6) < 0) {}

    if (!aborted && bufpos > 0)
    {
        if (functionparameter_SetParamValue_fromString(
                &fps[fpsindex], pindex, buf) != 0)
        {
            /* Show error briefly */
            char errmsg[] =
                "\033[1;31m  ERROR: invalid value"
                "\033[0m";
            if (write(STDOUT_FILENO,
                      errmsg, sizeof(errmsg) - 1)
                < 0) {}
            usleep(500000);
        }
        else
        {
            /* processinfo change tracking */
            if (strncmp(
                    fps[fpsindex]
                        .parray[pindex]
                        .keywordfull,
                    ".procinfo.", 10)
                == 0)
            {
                fps[fpsindex]
                    .md->processinfo_change_cnt++;
            }

            /* Notify GUI */
            fps[fpsindex].md->signal |=
                FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

            /* Save to disk if needed */
            if (fps[fpsindex]
                    .parray[pindex]
                    .fpflag
                & FPFLAG_SAVEONCHANGE)
            {
                functionparameter_WriteParameterToDisk(
                    &fps[fpsindex],
                    pindex,
                    "setval",
                    "UserInputSetParamValue");

                functionparameter_SaveFPS2disk(
                    &fps[fpsindex]);
            }
        }
    }

    return 0;
}

int fpsCTRL_TUI_process_user_key(
    int                        ch,
    FPS *fps,
    KEYWORD_TREE_NODE         *keywnode,
    FPSCTRL_TASK_ENTRY        *fpsctrltasklist __attribute__((unused)),
    FPSCTRL_TASK_QUEUE        *fpsctrlqueuelist __attribute__((unused)),
    FPSCTRL_PROCESS_VARS      *fpsCTRLvar
)
{
    int loopOK = 1;
    int fpsindex;
    int pindex;

    if(ch != -1)
    {
        if (fpsCTRLvar->NBfps > 0) {
            int selected_fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            if (fps[selected_fpsindex].md == NULL) {
                // If the selected FPS is invalid (was deleted), ignore commands that operate on it, 
                // but allow navigation and quit commands.
                if (ch == ' ' || ch == 'T' || ch == 't' || ch == 'E' || ch == 'R' ||
                    ch == 'r' || ch == 'C' || ch == 'O' || ch == 'c' || ch == ctrl('e') ||
                    ch == ctrl('r') || ch == ctrl('o')) {
                    return loopOK;
                }
            }
        }

        if (fpsCTRLvar->search_mode > 0) {
            if (ch == 27) { // ESC
                fpsCTRLvar->search_mode = 0;
                fpsCTRLvar->search_string[0] = '\0';
                return loopOK;
            } else if (ch == 10 || ch == 13) { // ENTER
                fpsCTRLvar->search_mode = 2; // Lock in search
                return loopOK;
            } else if (ch == 127 || ch == 8) { // Backspace
                int len = strlen(fpsCTRLvar->search_string);
                if (len > 0) {
                    fpsCTRLvar->search_string[len-1] = '\0';
                }
                return loopOK;
            } else if (ch >= 32 && ch <= 126) {
                int len = strlen(fpsCTRLvar->search_string);
                if (len < sizeof(fpsCTRLvar->search_string) - 1) {
                    fpsCTRLvar->search_string[len] = (char)ch;
                    fpsCTRLvar->search_string[len+1] = '\0';
                }
                return loopOK;
            }
            // If not printable and not enter/esc/backspace, allow fall-through 
            // for navigation keys (UP/DOWN/LEFT/RIGHT/PGUP/PGDN)
            if (ch != ANSI_KEY_UP && ch != ANSI_KEY_DOWN && ch != ANSI_KEY_PGUP && ch != ANSI_KEY_PGDN && ch != ANSI_KEY_LEFT && ch != ANSI_KEY_RIGHT) {
                return loopOK; // Swallow other commands
            }
        }

        /* CTRL+LEFT/RIGHT handled below in the switch */
        if (ch == 'v' || ch == 'V')
        {
            fpsCTRLvar->fpsCTRL_DisplayVerbose =
                !fpsCTRLvar->fpsCTRL_DisplayVerbose;
            if(write(STDOUT_FILENO,
                     "\033[2J", 4) < 0) {}
        }

        switch(ch)
        {

        case 3:       // Ctrl+C
        case 'x':     // Exit control screen
            loopOK = 0;
            break;

        case 27:      // ESC
            fpsCTRLvar->search_mode = 0;
            fpsCTRLvar->search_string[0] = '\0';
            break;

        case '/':     // Enter search mode
            fpsCTRLvar->search_mode = 1;
            fpsCTRLvar->search_string[0] = '\0';
            break;

        case 'S':     // Toggle sort mode (legacy)
            fpsCTRLvar->sort_mode =
                (fpsCTRLvar->sort_mode + 1) % 2;
            break;

        case ']':     // Next sort column
            fpsCTRLvar->sort_mode =
                (fpsCTRLvar->sort_mode + 1) % 3;
            break;

        case '[':     // Toggle sort direction (future)
            break;

        case 'y':     // Yank to tmux buffer
            {
                fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
                pindex = keywnode[fpsCTRLvar->nodeSelected].pindex;
                char yankstr[512] = {0};
                if(keywnode[fpsCTRLvar->nodeSelected].leaf == 1) {
                    char valstr[200] = {0};
                    functionparameter_GetParamValueString(&fps[fpsindex].parray[pindex], valstr, 200);
                    snprintf(yankstr, sizeof(yankstr), "%s = %s", fps[fpsindex].parray[pindex].keywordfull, valstr);
                } else {
                    snprintf(yankstr, sizeof(yankstr), "%s", keywnode[fpsCTRLvar->nodeSelected].keywordfull);
                }
                char cmd[1024];
                snprintf(cmd, sizeof(cmd), "tmux set-buffer \"%s\" 2>/dev/null", yankstr);
                if (system(cmd) != 0) {}
            }
            break;

        case '\t':
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4) {
                char names[20][FPSSEQ_NAME_MAX];
                int count = milkseq_list(names, 20);
                if (count > 0) {
                    int current_idx = -1;
                    if (fpsCTRLvar->milkseq_state != NULL) {
                        for (int i=0; i<count; i++) {
                            if (strcmp(names[i], fpsCTRLvar->milkseq_name) == 0) {
                                current_idx = i;
                                break;
                            }
                        }
                    }
                    int next_idx = (current_idx + 1) % count;
                    if (fpsCTRLvar->milkseq_state != NULL) {
                        milkseq_disconnect((MILKSEQ_STATE *) fpsCTRLvar->milkseq_state);
                        fpsCTRLvar->milkseq_state = NULL;
                    }
                    fpsCTRLvar->milkseq_state = milkseq_connect(names[next_idx]);
                    if (fpsCTRLvar->milkseq_state) {
                        strncpy(
                            fpsCTRLvar->milkseq_name,
                            names[next_idx],
                            sizeof(fpsCTRLvar
                                   ->milkseq_name)
                            - 1);
                        fpsCTRLvar->milkseq_name[
                            sizeof(fpsCTRLvar
                                   ->milkseq_name)
                            - 1] = '\0';
                    } else {
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
            if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
            break;

        case '?': // fps log
            fpsCTRLvar->fpsCTRL_DisplayMode = 2;
            if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
            break;

        case ANSI_KEY_F2: // control
            fpsCTRLvar->fpsCTRL_DisplayMode = 3;
            if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
            break;

        case ANSI_KEY_F3: // scheduler
            fpsCTRLvar->fpsCTRL_DisplayMode = 4;
            if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
            break;

        case ANSI_KEY_CTRL_RIGHT:
            fpsCTRLvar->fpsCTRL_DisplayMode++;
            if(fpsCTRLvar->fpsCTRL_DisplayMode > 4) fpsCTRLvar->fpsCTRL_DisplayMode = 1;
            if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
            break;

        case ANSI_KEY_CTRL_LEFT:
            fpsCTRLvar->fpsCTRL_DisplayMode--;
            if(fpsCTRLvar->fpsCTRL_DisplayMode < 1) fpsCTRLvar->fpsCTRL_DisplayMode = 4;
            if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
            break;

        case 's' : // (re)scan
            functionparameter_scan_fps(
                fpsCTRLvar->mode,
                fpsCTRLvar->fpsnamemask,
                fps,
                keywnode,
                &fpsCTRLvar->NBkwn,
                &fpsCTRLvar->NBfps,
                &fpsCTRLvar->NBindex,
                0);
            TUI_clearscreen(NULL, NULL);
            break;

        case ctrl('e') : // stop conf/run, then erase FPS
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstop(&fps[fpsindex]);
            functionparameter_RUNstop(&fps[fpsindex]);
            functionparameter_FPSremove(&fps[fpsindex]);

            functionparameter_scan_fps(
                fpsCTRLvar->mode,
                fpsCTRLvar->fpsnamemask,
                fps,
                keywnode,
                &fpsCTRLvar->NBkwn,
                &(fpsCTRLvar->NBfps),
                &fpsCTRLvar->NBindex,
                0);
            TUI_clearscreen(NULL, NULL);
            fpsCTRLvar->run_display = 0; // skip next display
            fpsCTRLvar->fpsindexSelected = 0; // safeguard
            break;


        case 'T' : // initialize tmux session
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_FPS_tmux_init(&fps[fpsindex]);
            break;

        case 't' : // kill tmux session
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_FPS_tmux_kill(&fps[fpsindex]);
            break;


        case 'E' : // Stop conf/run, erase FPS, kill tmux
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstop(&fps[fpsindex]);
            functionparameter_RUNstop(&fps[fpsindex]);

            functionparameter_FPSremove(&fps[fpsindex]);
            functionparameter_scan_fps(
                fpsCTRLvar->mode,
                fpsCTRLvar->fpsnamemask,
                fps,
                keywnode,
                &fpsCTRLvar->NBkwn,
                &fpsCTRLvar->NBfps,
                &fpsCTRLvar->NBindex, 0);
            TUI_clearscreen(NULL, NULL);
            // safeguard in case current selection disappears
            fpsCTRLvar->fpsindexSelected = 0; 
            break;

        case ANSI_KEY_UP:
            fpsCTRLvar->direction = -1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0) l = 0;
                fpsCTRLvar->GUIlineSelected[l] --;
                if(fpsCTRLvar->GUIlineSelected[l] < 0)
                {
                    fpsCTRLvar->GUIlineSelected[l] = 0;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
                fpsCTRLvar->scheduler_wrowstart--;
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
                fpsCTRLvar->help_wrowstart--;
            break;


        case ANSI_KEY_DOWN:
            fpsCTRLvar->direction = 1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0) l = 0;
                fpsCTRLvar->GUIlineSelected[l] ++;
                if(fpsCTRLvar->GUIlineSelected[l] > fpsCTRLvar->NBindex - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] = fpsCTRLvar->NBindex - 1;
                }
                if(fpsCTRLvar->GUIlineSelected[l] >
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] =
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
                fpsCTRLvar->scheduler_wrowstart++;
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
                fpsCTRLvar->help_wrowstart++;
            break;

        case ANSI_KEY_PGUP:
            fpsCTRLvar->direction = -1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0) l = 0;
                fpsCTRLvar->GUIlineSelected[l] -= 10;
                if(fpsCTRLvar->GUIlineSelected[l] < 0)
                {
                    fpsCTRLvar->GUIlineSelected[l] = 0;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
                fpsCTRLvar->scheduler_wrowstart -= 10;
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
                fpsCTRLvar->help_wrowstart -= 10;
            break;

        case ANSI_KEY_PGDN:
            fpsCTRLvar->direction = 1;
            {
                int l = fpsCTRLvar->currentlevel;
                if (l < 0) l = 0;
                fpsCTRLvar->GUIlineSelected[l] += 10;
                while(fpsCTRLvar->GUIlineSelected[l] >
                        fpsCTRLvar->NBindex - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] = fpsCTRLvar->NBindex - 1;
                }
                while(fpsCTRLvar->GUIlineSelected[l] >
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
                {
                    fpsCTRLvar->GUIlineSelected[l] =
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
                }
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 4)
                fpsCTRLvar->scheduler_wrowstart += 10;
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 1)
                fpsCTRLvar->help_wrowstart += 10;
            break;


        case ANSI_KEY_LEFT:
            if(fpsCTRLvar->directorynodeSelected != 0)   // ROOT has no parent
            {
                fpsCTRLvar->directorynodeSelected =
                    keywnode[fpsCTRLvar->directorynodeSelected].parent_index;
                fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
                fpsCTRLvar->currentlevel = keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;
            }
            break;


        case ANSI_KEY_RIGHT :
            if(keywnode[fpsCTRLvar->nodeSelected].leaf == 0)   // this is a directory
            {
                fpsCTRLvar->directorynodeSelected = fpsCTRLvar->nodeSelected;
                fpsCTRLvar->currentlevel = keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;
            }
            break;

        case 13 : // enter key (CR in raw mode)
        case 10 : // enter key (LF fallback)
            if(keywnode[fpsCTRLvar->nodeSelected].leaf == 1)
            {
                int ei = fpsCTRLvar->fpsindexSelected;
                int ep = fpsCTRLvar->pindexSelected;
                int etype =
                    fps[ei].parray[ep].type;

                if (etype == FPTYPE_ONOFF)
                {
                    /* Toggle ON/OFF inline */
                    if (fps[ei].parray[ep].fpflag
                        & FPFLAG_WRITESTATUS)
                    {
                        if (fps[ei].parray[ep].fpflag
                            & FPFLAG_ONOFF)
                        {
                            fps[ei].parray[ep].fpflag
                                &= ~FPFLAG_ONOFF;
                            fps[ei].parray[ep]
                                .val.i32[0] = 0;
                        }
                        else
                        {
                            fps[ei].parray[ep].fpflag
                                |= FPFLAG_ONOFF;
                            fps[ei].parray[ep]
                                .val.i32[0] = 1;
                        }
                        if (fps[ei].parray[ep].fpflag
                            & FPFLAG_SAVEONCHANGE)
                        {
                            functionparameter_WriteParameterToDisk(
                                &fps[ei], ep,
                                "setval",
                                "UserInputSetParamValue");
                        }
                        fps[ei].parray[ep].cnt0++;
                        fps[ei].md->signal |=
                            FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                    }
                }
                else
                {
                    fpsCTRL_inline_edit_param(
                        fps, ei, ep);
                }
            }
            break;

        case ' ' :
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            pindex = keywnode[fpsCTRLvar->nodeSelected].pindex;

            // toggles ON / OFF
            if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_WRITESTATUS)
            {
                if(fps[fpsindex].parray[pindex].type == FPTYPE_ONOFF)
                {
                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ONOFF)    // ON -> OFF
                    {
                        fps[fpsindex].parray[pindex].fpflag &= ~FPFLAG_ONOFF;
                        fps[fpsindex].parray[pindex].val.i32[0] = 0;
                    }
                    else     // OFF -> ON
                    {
                        fps[fpsindex].parray[pindex].fpflag |= FPFLAG_ONOFF;
                        fps[fpsindex].parray[pindex].val.i32[0] = 1;
                    }

                    // Save to disk
                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
                    {
                        functionparameter_WriteParameterToDisk(&fps[fpsindex], pindex, "setval",
                                                               "UserInputSetParamValue");
                    }
                    fps[fpsindex].parray[pindex].cnt0 ++;
                    fps[fpsindex].md->signal |= 
                        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
                }
            }

            if(fps[fpsindex].parray[pindex].type == FPTYPE_EXECFILENAME)
            {
                char cmd[512];
                snprintf(cmd, sizeof(cmd), "tmux send-keys -t %s:run \"cd %s\" C-m", fps[fpsindex].md->name, fps[fpsindex].md->workdir);
                if (system(cmd) != 0) { } // Corrected escaping for "cd %s"
                snprintf(cmd, sizeof(cmd), "tmux send-keys -t %s:run \"%s %s/%s.fps\" C-m", fps[fpsindex].md->name, fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].md->datadir, fps[fpsindex].md->name);
                if (system(cmd) != 0) { } // Corrected escaping for "%s %s/%s.fps"
            }

            break;


        case 'u' : // update conf process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            fps[fpsindex].md->signal |= 
                FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            break;

        case 'R' : // start run process
            {
                fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
                FPS *selected_fps = &fps[fpsindex];
                functionparameter_FPS_tmux_ensure(selected_fps);
                char progexec[1024];
                if( (strlen(selected_fps->md->execfullpath) > 0) && (strcmp(selected_fps->md->execfullpath, "unknown") != 0) )
                    strncpy(progexec, selected_fps->md->execfullpath, 1023);
                else
                    snprintf(progexec, 1024, "%s-exec", selected_fps->md->callprogname);
                
                EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run \" cd %s\" C-m", selected_fps->md->name, selected_fps->md->workdir);
                EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run \" %s %s:runstart\" C-m", selected_fps->md->name, progexec, selected_fps->md->name);
            }
            break;

        case ctrl('r') : // stop run process
            fpsindex =
                keywnode[fpsCTRLvar->nodeSelected]
                    .fpsindex;
            functionparameter_RUNstop(&fps[fpsindex]);
            break;

        case 'r' : // legacy stop run process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_RUNstop(&fps[fpsindex]);
            break;


        case 'C' : // legacy start conf process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstart(&fps[fpsindex]);
            break;

        case 'O': // start conf process
            {
                fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
                FPS *selected_fps = &fps[fpsindex];
                functionparameter_FPS_tmux_ensure(selected_fps);
                char progexec[1024];
                if( (strlen(selected_fps->md->execfullpath) > 0) && (strcmp(selected_fps->md->execfullpath, "unknown") != 0) )
                    strncpy(progexec, selected_fps->md->execfullpath, 1023);
                else
                    snprintf(progexec, 1024, "%s-exec", selected_fps->md->callprogname);
                
                EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" cd %s\" C-m", selected_fps->md->name, selected_fps->md->workdir);
                EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:conf \" %s %s:confstart\" C-m", selected_fps->md->name, progexec, selected_fps->md->name);
            }
            break;

        case ctrl('o'): // stop conf process
            fpsindex =
                keywnode[fpsCTRLvar->nodeSelected]
                    .fpsindex;
            functionparameter_CONFstop(&fps[fpsindex]);
            break;

        case 'c': // kill conf process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstop(&fps[fpsindex]);
            break;

        case 'l': // list all parameters
            TUI_exit();
            if(system("clear") != 0) { } // Corrected escaping for "clear"
            printf("FPS entries - Full list \n\n");
            for(int kwnindex = 0; kwnindex < fpsCTRLvar->NBkwn; kwnindex++)
            {
                if(keywnode[kwnindex].leaf == 1)
                {
                    printf("%4d  %4d  %s\n", keywnode[kwnindex].fpsindex, keywnode[kwnindex].pindex,
                           keywnode[kwnindex].keywordfull);
                }
            }
            printf("  TOTAL :  %d nodes\n\n", fpsCTRLvar->NBkwn);
            printf("Press Enter to Continue\n");
            while(getchar() != '\n') {}
            {
                short unsigned int wrow = 0, wcol = 0;
                TUI_init_terminal(&wrow, &wcol);
            }
            break;
        
        }
    }

    return(loopOK);
}

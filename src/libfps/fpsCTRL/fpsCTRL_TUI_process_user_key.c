/**
 * @file    fpsCTRL_TUI_process_user_key.c
 * @brief   TUI key input processing
 */

#include <ncurses.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "fps.h"
#include "fps_internal.h"
#include "TUItools.h"
#include "fpsCTRL_globals.h"

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

#define ctrl(x) ((x) & 0x1f)

int fpsCTRL_TUI_process_user_key(
    int                        ch,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE         *keywnode,
    FPSCTRL_TASK_ENTRY        *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE        *fpsctrlqueuelist,
    FPSCTRL_PROCESS_VARS      *fpsCTRLvar
)
{
    int loopOK = 1;
    int fpsindex;
    int pindex;

    if(ch != -1)
    {
        if(ch == 545 || ch == 560 || ch == 443 || ch == 564 || ch == 554) { // CTRL+LEFT
             fpsCTRLvar->fpsCTRL_DisplayMode--;
             if (fpsCTRLvar->fpsCTRL_DisplayMode < 1) fpsCTRLvar->fpsCTRL_DisplayMode = 3;
        }
        else if (ch == 561 || ch == 566 || ch == 444 || ch == 565 || ch == 569) { // CTRL+RIGHT
             fpsCTRLvar->fpsCTRL_DisplayMode++;
             if (fpsCTRLvar->fpsCTRL_DisplayMode > 3) fpsCTRLvar->fpsCTRL_DisplayMode = 1;
        }

        switch(ch)
        {

        case 'x':     // Exit control screen
            loopOK = 0;
            break;

        // ============ SCREENS

        case 'h': // help
            fpsCTRLvar->fpsCTRL_DisplayMode = 1;
            break;

        case KEY_F(2): // control
            fpsCTRLvar->fpsCTRL_DisplayMode = 2;
            break;

        case KEY_F(3): // scheduler
            fpsCTRLvar->fpsCTRL_DisplayMode = 3;
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
            clear();
            break;

        case 'e' : // erase FPS
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
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
            clear();
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


        case 'E' : // Erase FPS and close tmux sessions
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;

            functionparameter_FPSremove(&fps[fpsindex]);
            functionparameter_scan_fps(
                fpsCTRLvar->mode,
                fpsCTRLvar->fpsnamemask,
                fps,
                keywnode,
                &fpsCTRLvar->NBkwn,
                &fpsCTRLvar->NBfps,
                &fpsCTRLvar->NBindex, 0);
            clear();
            // safeguard in case current selection disappears
            fpsCTRLvar->fpsindexSelected = 0; 
            break;

        case KEY_UP:
            fpsCTRLvar->direction = -1;
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] --;
            if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 3)
                fpsCTRLvar->scheduler_wrowstart--;
            break;


        case KEY_DOWN:
            fpsCTRLvar->direction = 1;
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] ++;
            if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] > fpsCTRLvar->NBindex - 1)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = fpsCTRLvar->NBindex - 1;
            }
            if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] =
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 3)
                fpsCTRLvar->scheduler_wrowstart++;
            break;

        case KEY_PPAGE:
            fpsCTRLvar->direction = -1;
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] -= 10;
            if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 3)
                fpsCTRLvar->scheduler_wrowstart -= 10;
            break;

        case KEY_NPAGE:
            fpsCTRLvar->direction = 1;
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] += 10;
            while(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                    fpsCTRLvar->NBindex - 1)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = fpsCTRLvar->NBindex - 1;
            }
            while(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] =
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
            }
            if (fpsCTRLvar->fpsCTRL_DisplayMode == 3)
                fpsCTRLvar->scheduler_wrowstart += 10;
            break;


        case KEY_LEFT:
            if(fpsCTRLvar->directorynodeSelected != 0)   // ROOT has no parent
            {
                fpsCTRLvar->directorynodeSelected =
                    keywnode[fpsCTRLvar->directorynodeSelected].parent_index;
                fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
                fpsCTRLvar->currentlevel = keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;
            }
            break;


        case KEY_RIGHT :
            if(keywnode[fpsCTRLvar->nodeSelected].leaf == 0)   // this is a directory
            {
                if(keywnode[keywnode[fpsCTRLvar->directorynodeSelected].child[fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]]].leaf
                        == 0)
                {
                    fpsCTRLvar->directorynodeSelected =
                        keywnode[fpsCTRLvar->directorynodeSelected].child[fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]];
                    fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
                    fpsCTRLvar->currentlevel = keywnode[fpsCTRLvar->directorynodeSelected].keywordlevel;
                }
            }
            break;

        case 10 : // enter key
            if(keywnode[fpsCTRLvar->nodeSelected].leaf == 1)   // this is a leaf
            {
                TUI_exit();
                if(system("clear") != 0) { } // Corrected escaping for "clear"
                functionparameter_UserInputSetParamValue(&fps[fpsCTRLvar->fpsindexSelected],
                        fpsCTRLvar->pindexSelected);
                TUI_initncurses();
                TUI_stdio_clear();
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
                    }
                    else     // OFF -> ON
                    {
                        fps[fpsindex].parray[pindex].fpflag |= FPFLAG_ONOFF;
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

        case 'R' : // start run process if possible
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_RUNstart(&fps[fpsindex]);
            break;

        case 'r' : // stop run process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_RUNstop(&fps[fpsindex]);
            break;


        case 'C' : // start conf process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            functionparameter_CONFstart(&fps[fpsindex]);
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
            while(getchar() != '\n');
            TUI_initncurses();
            break;
        
        case 'v':
            fpsCTRLvar->fpsCTRL_DisplayVerbose = 0;
            break;
        case 'V':
            fpsCTRLvar->fpsCTRL_DisplayVerbose = 1;
            break;
        }
    }

    return(loopOK);
}

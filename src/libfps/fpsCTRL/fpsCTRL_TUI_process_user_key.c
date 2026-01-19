/**
 * @file    fpsCTRL_TUI_process_user_key.c
 * @brief   TUI key input processing
 */

#include <ncurses.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_TUI_shim.h"
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

static short unsigned int wrow, wcol;









int fpsCTRL_TUI_process_user_key(
    int                        ch,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE         *keywnode,
    FPSCTRL_TASK_ENTRY        *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE        *fpsctrlqueuelist,
    FPSCTRL_PROCESS_VARS      *fpsCTRLvar
)
{
    DEBUG_TRACE_FSTART();

    int loopOK       = 1;
    int NBindex      = fpsCTRLvar->NBindex;
    int fpsindex     = fpsCTRLvar->fpsindexSelected;
    long pindex      = fpsCTRLvar->pindexSelected;
    int currentlevel = fpsCTRLvar->currentlevel;

    getmaxyx(stdscr, wrow, wcol);

    if(ch != -1)
    {
        if(ch == ctrl('q'))
        {
            loopOK = 0;
        }

        if(ch == 'x')
        {
            loopOK = 0;
        }

        if((ch == 'h') || (ch == KEY_F(1)))
        {
            fpsCTRLvar->fpsCTRL_DisplayMode = 1;
        }

        if(ch == KEY_F(2))
        {
            fpsCTRLvar->fpsCTRL_DisplayMode = 2;
        }

        if(ch == KEY_F(3))
        {
            fpsCTRLvar->fpsCTRL_DisplayMode = 3;
        }

        if(ch == '?')
        {
            fpsCTRLvar->fpsCTRL_DisplayMode = 4;
        }

        if(ch == 'v')
        {
            fpsCTRLvar->fpsCTRL_DisplayVerbose = 0;
        }

        if(ch == 'V')
        {
            fpsCTRLvar->fpsCTRL_DisplayVerbose = 1;
        }

        if(ch == 's')
        {
            functionparameter_scan_fps(fpsCTRLvar->mode,
                                       fpsCTRLvar->fpsnamemask,
                                       fps,
                                       keywnode,
                                       &fpsCTRLvar->NBkwn,
                                       &fpsCTRLvar->NBfps,
                                       &fpsCTRLvar->NBindex,
                                       0 // verbose
                                      );
            fpsCTRLvar->currentlevel = 0;
            fpsCTRLvar->nodeSelected = 1;
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            pindex   = keywnode[fpsCTRLvar->nodeSelected].pindex;
        }

        if((ch == 'T') || (ch == ctrl('t')))
        {
            // start/stop tmux session
            // toggle
            if(strlen(fps[fpsindex].md->tmuxname) > 0)
            {
                functionparameter_FPS_tmux_kill(&fps[fpsindex]);
            }
            else
            {
                functionparameter_FPS_tmux_init(&fps[fpsindex]);
            }
        }

        if(ch == ctrl('a'))
        {
            // attach
            functionparameter_FPS_tmux_attach(&fps[fpsindex]);
        }

        if(ch == ctrl('e'))
        {
            functionparameter_FPSremove(&fps[fpsindex]);
            functionparameter_scan_fps(fpsCTRLvar->mode,
                                       fpsCTRLvar->fpsnamemask,
                                       fps,
                                       keywnode,
                                       &fpsCTRLvar->NBkwn,
                                       &fpsCTRLvar->NBfps,
                                       &fpsCTRLvar->NBindex,
                                       0 // verbose
                                      );
            fpsCTRLvar->currentlevel = 0;
            fpsCTRLvar->nodeSelected = 1;
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            pindex   = keywnode[fpsCTRLvar->nodeSelected].pindex;
        }

        if((ch == 'O') || (ch == ctrl('o')))
        {
            if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
            {
                functionparameter_CONFstop(&fps[fpsindex]);
            }
            else
            {
                functionparameter_CONFstart(&fps[fpsindex]);
            }
        }

        if(ch == 'u')
        {
            fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED;
            fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
        }

        if((ch == 'R') || (ch == ctrl('r')))
        {
            if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
            {
                functionparameter_RUNstop(&fps[fpsindex]);
            }
            else
            {
                functionparameter_RUNstart(&fps[fpsindex]);
            }
        }

        // list entries
        if(ch == 'l')
        {
            endwin(); // Suspend TUI
            printf("List of loaded FPS entries:\n");
            for(int i=0; i<fpsCTRLvar->NBfps; i++) {
                 printf("[%d] %s\n", i, fps[i].md->name);
            }
            printf("Press Enter to return to TUI...");
            while(getchar() != '\n');
            // TUI will restore on next loop iteration/refresh
        }

        // write content to disk
        if(ch == 'f')
        {
            functionparameter_SaveFPS2disk(&fps[fpsindex]);
        }

        // FPS logging
        if(ch == 'G')
        {
            set_FLAG_FPSOUTLOG(1);
        }
        if(ch == 'g')
        {
            set_FLAG_FPSOUTLOG(0);
        }

        if(ch == '>')
        {
            // export values to fpsconfdir
            functionparameter_WriteParameterToDisk(&fps[fpsindex],
                                                   -1,
                                                   "exportconf",
                                                   "FPSCTRL");
        }

        if(ch == '<')
        {
            // import values from fpsconfdir
            // read_parameter_from_disk(fps[fpsindex], pindex);
        }

        if(ch == 'P')
        {
            char scriptfname[200];
            printf("Enter script file name: ");
            if (scanf("%s", scriptfname)) {} // dummy check
            // functionparameter_FPSprocess_cmdfile(scriptfname, fps, keywnode, fpsctrlqueuelist, fpsCTRLvar);
        }

        // Navigation

        if(ch == KEY_UP)
        {
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 2)
            {
                fpsCTRLvar->direction = -1;
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]--;
                if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
                {
                    fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;
                }
            }
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 3)
            {
                fpsCTRLvar->scheduler_wrowstart--;
            }
        }

        if(ch == KEY_DOWN)
        {
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 2)
            {
                fpsCTRLvar->direction = 1;
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]++;
                if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                        fpsCTRLvar->NBindex - 1)
                {
                    fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] =
                        fpsCTRLvar->NBindex - 1;
                }
                if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
                {
                    fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] =
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
                }
            }
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 3)
            {
                fpsCTRLvar->scheduler_wrowstart++;
            }
        }

        if(ch == KEY_PPAGE)
        {
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 2)
            {
                fpsCTRLvar->direction = -1;
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] -= 10;
                if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
                {
                    fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;
                }
            }
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 3)
            {
                fpsCTRLvar->scheduler_wrowstart -= 10;
            }
        }

        if(ch == KEY_NPAGE)
        {
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 2)
            {
                fpsCTRLvar->direction = 1;
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] += 10;
                while(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                        fpsCTRLvar->NBindex - 1)
                {
                    fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] =
                        fpsCTRLvar->NBindex - 1;
                }
                while(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
                {
                    fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] =
                        keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
                }
            }
            if(fpsCTRLvar->fpsCTRL_DisplayMode == 3)
            {
                fpsCTRLvar->scheduler_wrowstart += 10;
            }
        }

        if(ch == KEY_LEFT)
        {
            if(fpsCTRLvar->directorynodeSelected != 0)  // ROOT has no parent
            {
                fpsCTRLvar->currentlevel--;
                fpsCTRLvar->directorynodeSelected =
                    keywnode[fpsCTRLvar->directorynodeSelected].parent_index;
                fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
            }
        }

        if(ch == KEY_RIGHT)
        {
            if(keywnode[fpsCTRLvar->nodeSelected].leaf == 0)  // this is a directory
            {
                if(keywnode[keywnode[fpsCTRLvar->directorynodeSelected]
                            .child[fpsCTRLvar->GUIlineSelected
                                   [fpsCTRLvar->currentlevel]]]
                        .NBchild > 0)
                {
                    fpsCTRLvar->currentlevel++;
                    fpsCTRLvar->directorynodeSelected =
                        keywnode[fpsCTRLvar->directorynodeSelected].child
                        [fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]];
                    fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
                }
            }
        }

        if(ch == 10)  // ENTER
        {
            if(keywnode[fpsCTRLvar->nodeSelected].leaf == 1)  // this is a leaf
            {
                // EDIT VALUE
                functionparameter_UserInputSetParamValue(
                    &fps[fpsCTRLvar->fpsindexSelected],
                    fpsCTRLvar->pindexSelected);
            }
        }

        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        pindex   = keywnode[fpsCTRLvar->nodeSelected].pindex;

        int child_index[MAXNBLEVELS];
        // update selected node
        child_index[0] = fpsCTRLvar->GUIlineSelected[0];
        int knodeindex = keywnode[0].child[child_index[0]];

        for(int level = 0; level < fpsCTRLvar->currentlevel; level++)
        {
            child_index[level + 1] = fpsCTRLvar->GUIlineSelected[level + 1];
            knodeindex             = keywnode[knodeindex].child[child_index[level + 1]];
        }
        fpsCTRLvar->nodeSelected = knodeindex;

        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        pindex   = keywnode[fpsCTRLvar->nodeSelected].pindex;

        fpsCTRLvar->fpsindexSelected = fpsindex;
        fpsCTRLvar->pindexSelected   = pindex;
    }

    return loopOK;
}
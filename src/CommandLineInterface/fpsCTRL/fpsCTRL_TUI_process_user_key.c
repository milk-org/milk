/**
 * @file    fpsCTRL_TUI_process_user_key.c
 * @brief   TUI key input processing
 */

#include <ncurses.h>

#include "CommandLineInterface/CLIcore.h"
#include "TUItools.h"

#include "fps/fps_CONFstart.h"
#include "fps/fps_CONFstop.h"
#include "fps/fps_FPSremove.h"
#include "fps/fps_RUNstart.h"
#include "fps/fps_RUNstop.h"
#include "fps/fps_WriteParameterToDisk.h"
#include "fps/fps_outlog.h"
#include "fps/fps_processcmdline.h"
#include "fps/fps_read_fpsCMD_fifo.h"
#include "fps/fps_save2disk.h"
#include "fps/fps_scan.h"
#include "fps/fps_tmux.h"
#include "fps/fps_userinputsetparamvalue.h"
#include "fps/fps_printparameter_valuestring.h"


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
    int fpsindex;
    int pindex;

    //char msg[stringmaxlen];

    char fname[STRINGMAXLEN_FULLFILENAME];

    FILE *fpin;

    switch(ch)
    {
    case 3: // CTRL-C
        loopOK = 0;
        break;

    case 'x': // Exit control screen
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


    case '?': // fps entry help
        fpsCTRLvar->fpsCTRL_DisplayMode = 4;
        break;


    case 'g':
        set_FLAG_FPSOUTLOG(0);
        break;
    case 'G':
        set_FLAG_FPSOUTLOG(1);
        break;


    case 'v':
        fpsCTRLvar->fpsCTRL_DisplayVerbose = 0;
        break;
    case 'V':
        fpsCTRLvar->fpsCTRL_DisplayVerbose = 1;
        break;




    case 's': // (re)scan
        functionparameter_scan_fps(fpsCTRLvar->mode,
                                   fpsCTRLvar->fpsnamemask,
                                   fps,
                                   keywnode,
                                   &fpsCTRLvar->NBkwn,
                                   &fpsCTRLvar->NBfps,
                                   &fpsCTRLvar->NBindex,
                                   0);
        clear();
        break;

    case ctrl('a'): // tmux attach
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_FPS_tmux_attach(&fps[fpsindex]);
        // Returns upon tmux detach.
        // Need full display refresh and keyboard re-bind?
        TUI_init_terminal(&wrow, &wcol);
        break;

    case 'T': // initialize tmux session
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_FPS_tmux_init(&fps[fpsindex]);
        break;

    case ctrl('t'): // kill tmux session
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_FPS_tmux_kill(&fps[fpsindex]);
        break;

    case ctrl('e'): // Close tmux sessions and erase FPS
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_RUNstop(&fps[fpsindex]);
        functionparameter_CONFstop(&fps[fpsindex]);
        functionparameter_FPS_tmux_kill(&fps[fpsindex]);
        functionparameter_FPSremove(&fps[fpsindex]);
        functionparameter_scan_fps(fpsCTRLvar->mode,
                                   fpsCTRLvar->fpsnamemask,
                                   fps,
                                   keywnode,
                                   &fpsCTRLvar->NBkwn,
                                   &fpsCTRLvar->NBfps,
                                   &fpsCTRLvar->NBindex,
                                   0);
        clear();
        DEBUG_TRACEPOINT(" ");
        // safeguard in case current selection disappears
        fpsCTRLvar->fpsindexSelected = 0;
        break;

    case KEY_UP:
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
        break;

    case KEY_DOWN:
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
        break;

    case KEY_PPAGE:
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
        break;

    case KEY_NPAGE:
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
        break;

    case KEY_LEFT:
        if(fpsCTRLvar->directorynodeSelected != 0)  // ROOT has no parent
        {
            fpsCTRLvar->directorynodeSelected =
                keywnode[fpsCTRLvar->directorynodeSelected].parent_index;
            fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
        }
        break;

    case KEY_RIGHT:
        if(keywnode[fpsCTRLvar->nodeSelected].leaf == 0)  // this is a directory
        {
            if(keywnode[keywnode[fpsCTRLvar->directorynodeSelected]
                        .child[fpsCTRLvar->GUIlineSelected
                               [fpsCTRLvar->currentlevel]]]
                    .leaf == 0)
            {
                fpsCTRLvar->directorynodeSelected =
                    keywnode[fpsCTRLvar->directorynodeSelected].child
                    [fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel]];
                fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
            }
        }
        break;

    case 10: // enter key
        DEBUG_TRACEPOINT_LOG("exiting TUI screen");
        if(keywnode[fpsCTRLvar->nodeSelected].leaf == 1)  // this is a leaf
        {
            DEBUG_TRACEPOINT_LOG("exiting TUI screen");
            TUI_exit();

            if(system("clear") != 0)  // clear screen
            {
                PRINT_ERROR("system() returns non-zero value");
            }
            functionparameter_UserInputSetParamValue(
                &fps[fpsCTRLvar->fpsindexSelected],
                fpsCTRLvar->pindexSelected);

            DEBUG_TRACEPOINT(" ");

            TUI_init_terminal(&wrow, &wcol);
            DEBUG_TRACEPOINT(" ");
        }
        break;

    case ' ':
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        pindex   = keywnode[fpsCTRLvar->nodeSelected].pindex;

        // toggles ON / OFF - this is a special case not using function functionparameter_UserInputSetParamValue
        if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_WRITESTATUS)
        {
            if(fps[fpsindex].parray[pindex].type == FPTYPE_ONOFF)
            {

                if(fps[fpsindex].parray[pindex].fpflag &
                        FPFLAG_ONOFF) // ON -> OFF
                {
                    fps[fpsindex].parray[pindex].fpflag &= ~FPFLAG_ONOFF;
                    fps[fpsindex].parray[pindex].val.i64[0] = 0;
                    functionparameter_outlog("SETVAL", "%s ONOFF OFF", fps[fpsindex].parray[pindex].keywordfull);
                }
                else // OFF -> ON
                {
                    fps[fpsindex].parray[pindex].fpflag |= FPFLAG_ONOFF;
                    fps[fpsindex].parray[pindex].val.i64[0] = 1;
                    functionparameter_outlog("SETVAL", "%s ONOFF ON", fps[fpsindex].parray[pindex].keywordfull);
                }

                // Save to disk
                if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
                {
                    functionparameter_WriteParameterToDisk(
                        &fps[fpsindex],
                        pindex,
                        "setval",
                        "UserInputSetParamValue");
                }
                fps[fpsindex].parray[pindex].cnt0++;
                fps[fpsindex].md->signal |=
                    FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            }
        }

        if(fps[fpsindex].parray[pindex].type == FPTYPE_EXECFILENAME)
        {
            EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \" cd %s\" C-m",
                                   fps[fpsindex].md->name,
                                   fps[fpsindex].md->workdir);
            EXECUTE_SYSTEM_COMMAND(
                "tmux send-keys -t %s:run \" %s %s/%s.fps\" C-m",
                fps[fpsindex].md->name,
                fps[fpsindex].parray[pindex].val.string[0],
                fps[fpsindex].md->datadir,
                fps[fpsindex].md->name);
        }

        break;

    case 'u': // update conf process
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        fps[fpsindex].md->signal |=
            FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
        functionparameter_outlog("FPSCTRL", "UPDATE %s", fps[fpsindex].md->name);
        break;

    case 'R': // start run process if possible
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_outlog("FPSCTRL", "RUNSTART %s", fps[fpsindex].md->name);
        functionparameter_RUNstart(&fps[fpsindex]);
        break;

    case ctrl('r'): // stop run process
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_outlog("FPSCTRL", "RUNSTOP %s", fps[fpsindex].md->name);
        functionparameter_RUNstop(&fps[fpsindex]);
        break;

    case 'O': // start conf process
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_outlog("FPSCTRL", "CONFSTART %s", fps[fpsindex].md->name);
        functionparameter_CONFstart(&fps[fpsindex]);
        break;

    case ctrl('o'): // kill conf process
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_outlog("FPSCTRL", "CONFSTOP %s", fps[fpsindex].md->name);
        functionparameter_CONFstop(&fps[fpsindex]);
        break;

    case 'l': // log conf and run status
        // status of processes
        //
        for(int kwnindex = 0; kwnindex < fpsCTRLvar->NBkwn; kwnindex++)
        {
            int fpsindex = keywnode[kwnindex].fpsindex;
            if(keywnode[kwnindex].leaf == 0)
            {
                if(keywnode[kwnindex].keywordlevel == 1)
                {
                    // at root
                    pid_t pid;

                    pid = data.fpsarray[fpsindex].md->confpid;
                    if((getpgid(pid) >= 0) && (pid > 0))
                    {
                        functionparameter_outlog("STATUS",
                                                 "CONFPID ALIVE %s %ld",
                                                 keywnode[kwnindex].keywordfull,
                                                 pid  );
                    }
                    else // PID not active
                    {
                        if(data.fpsarray[fpsindex].md->status &
                                FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
                        {
                            functionparameter_outlog("STATUS",
                                                     "CONFPID CRASHED %s %ld",
                                                     keywnode[kwnindex].keywordfull,
                                                     pid  );
                        }
                        else
                        {
                            functionparameter_outlog("STATUS",
                                                     "CONFPID STOPPED %s %ld",
                                                     keywnode[kwnindex].keywordfull,
                                                     pid  );
                        }
                    }

                    pid = data.fpsarray[fpsindex].md->runpid;
                    if((getpgid(pid) >= 0) && (pid > 0))
                    {
                        functionparameter_outlog("STATUS",
                                                 "RUNPID ALIVE %s %ld",
                                                 keywnode[kwnindex].keywordfull,
                                                 pid  );
                    }
                    else // PID not active
                    {
                        if(data.fpsarray[fpsindex].md->status &
                                FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
                        {
                            functionparameter_outlog("STATUS",
                                                     "RUNPID CRASHED %s %ld",
                                                     keywnode[kwnindex].keywordfull,
                                                     pid  );
                        }
                        else
                        {
                            functionparameter_outlog("STATUS",
                                                     "RUNPID STOPPED %s %ld",
                                                     keywnode[kwnindex].keywordfull,
                                                     pid  );
                        }
                    }
                }
            }
        }
        /*
        // list all parameters
        TUI_exit();
        if(system("clear") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("FPS entries - Full list \n");
        printf("\n");
        for(int kwnindex = 0; kwnindex < fpsCTRLvar->NBkwn; kwnindex++)
        {
            if(keywnode[kwnindex].leaf == 1)
            {
                printf("%4d  %4d  %s\n",
                       keywnode[kwnindex].fpsindex,
                       keywnode[kwnindex].pindex,
                       keywnode[kwnindex].keywordfull);
            }
        }
        printf("  TOTAL :  %d nodes\n", fpsCTRLvar->NBkwn);
        printf("\n");
        printf("Press Enter to Continue\n");
        getchar();

        TUI_init_terminal(&wrow, &wcol);*/

        break;


    case 'L': // log all parameters
        // status of parameters
        //
        for(int kwnindex = 0; kwnindex < fpsCTRLvar->NBkwn; kwnindex++)
        {
            int fpsindex = keywnode[kwnindex].fpsindex;
            if(keywnode[kwnindex].leaf == 1)
            {
                int pindex = keywnode[kwnindex].pindex;

                char msgstring[STRINGMAXLEN_FPS_LOGMSG];
                functionparameter_PrintParameter_ValueString(
                    &fps[fpsindex].parray[pindex],
                    msgstring,
                    STRINGMAXLEN_FPS_LOGMSG);
                functionparameter_outlog("STATUS",
                                         "%s",
                                         msgstring);
            }
        }
        break;

    case 'f': // export fps content to single file in datadir
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        functionparameter_SaveFPS2disk(&fps[fpsindex]);
        break;

    case '>': // export to confdir
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        fps_datadir_to_confdir(&fps[fpsindex]);
        break;

    case '<': // Load from confdir
        TUI_exit();
        if(system("clear") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
        snprintf(fname,
                 STRINGMAXLEN_FULLFILENAME,
                 "%s/%s.fps",
                 fps[fpsindex].md->confdir,
                 fps[fpsindex].md->name);
        //printf("LOADING FPS FILE %s\n", fname);

        fpin = fopen(fname, "r");
        if(fpin != NULL)
        {
            char   *FPSline = NULL;
            size_t  len     = 0;
            ssize_t read;

            while((read = getline(&FPSline, &len, fpin)) != -1)
            {
                uint64_t taskstatus = 0;

                //printf("READING LINE: %s\n", FPSline);

                char  delimiter[] = " ";
                char *varname, *vartype, *varvalue;
                char *context;

                int inputLength = strlen(FPSline);

                char *inputCopy =
                    (char *) calloc(inputLength + 1, sizeof(char));
                if(inputCopy == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                strncpy(inputCopy, FPSline, inputLength);

                varname = strtok_r(inputCopy, delimiter, &context);
                vartype = strtok_r(NULL, delimiter, &context);
                (void) vartype;
                varvalue = strtok_r(NULL, delimiter, &context);

                //printf("%s [%s] -< %s\n", varname, vartype, varvalue);

                char FPScmdline[200];
                snprintf(FPScmdline, 200, "setval %s %s", varname, varvalue);
                free(inputCopy);
                functionparameter_FPSprocess_cmdline(FPScmdline,
                                                     fpsctrlqueuelist,
                                                     keywnode,
                                                     fpsCTRLvar,
                                                     fps,
                                                     &taskstatus);
            }
            fclose(fpin);
        }
        else
        {
            printf("File not found\n");
        }

        TUI_init_terminal(&wrow, &wcol);
        break;

    case 'F': // process FIFO
        TUI_exit();
        if(system("clear") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("Reading FIFO file \"%s\"  fd=%d\n",
               fpsCTRLvar->fpsCTRLfifoname,
               fpsCTRLvar->fpsCTRLfifofd);

        if(fpsCTRLvar->fpsCTRLfifofd > 0)
        {
            // int verbose = 1;
            functionparameter_read_fpsCMD_fifo(fpsCTRLvar->fpsCTRLfifofd,
                                               fpsctrltasklist,
                                               fpsctrlqueuelist);
        }

        printf("\n");
        printf("Press Enter to Continue\n");
        getchar();
        TUI_init_terminal(&wrow, &wcol);
        break;

    case 'P': // process input command file
        TUI_exit();
        if(system("clear") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("Reading file confscript\n");

        functionparameter_FPSprocess_cmdfile("confscript",
                                             fps,
                                             keywnode,
                                             fpsctrlqueuelist,
                                             fpsCTRLvar);

        printf("\n");
        printf("Press Enter to Continue\n");
        getchar();
        TUI_init_terminal(&wrow, &wcol);
        break;
    }

    DEBUG_TRACE_FEXIT();
    return (loopOK);
}

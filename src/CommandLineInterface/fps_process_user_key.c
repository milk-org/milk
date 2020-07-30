/**
 * @file    fps_process_user_key.c
 * @brief   TUI key input processing
 */


#include <ncurses.h>


#include "CommandLineInterface/CLIcore.h"
#include "TUItools.h"


#include "fps_CONFstart.h"
#include "fps_CONFstop.h"
#include "fps_FPSremove.h"
#include "fps_outlog.h"
#include "fps_processcmdline.h"
#include "fps_read_fpsCMD_fifo.h"
#include "fps_RUNstart.h"
#include "fps_RUNstop.h"
#include "fps_save2disk.h"
#include "fps_scan.h"
#include "fps_tmux.h"
#include "fps_userinputsetparamvalue.h"
#include "fps_WriteParameterToDisk.h"



int fpsCTRLscreen_process_user_key(
    int ch,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_TASK_ENTRY *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
)
{
    int stringmaxlen = 500;
    int loopOK = 1;
    int fpsindex;
    int pindex;
    FILE *fpinputcmd;

    char msg[stringmaxlen];

	char fname[STRINGMAXLEN_FULLFILENAME];

	FILE *fpin;

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
            //DEBUG_TRACEPOINT("fpsCTRLvar->NBfps = %d\n", fpsCTRLvar->NBfps);
            // abort();
            fpsCTRLvar->run_display = 0; // skip next display
            fpsCTRLvar->fpsindexSelected =
                0; // safeguard in case current selection disappears
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
            DEBUG_TRACEPOINT(" ");
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
            break;


        case KEY_DOWN:
            fpsCTRLvar->direction = 1;
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] ++;
            if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] > fpsCTRLvar->NBindex -
                    1)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = fpsCTRLvar->NBindex - 1;
            }
            if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] >
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] =
                    keywnode[fpsCTRLvar->directorynodeSelected].NBchild - 1;
            }
            break;

        case KEY_PPAGE:
            fpsCTRLvar->direction = -1;
            fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] -= 10;
            if(fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] < 0)
            {
                fpsCTRLvar->GUIlineSelected[fpsCTRLvar->currentlevel] = 0;
            }
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
            break;


        case KEY_LEFT:
            if(fpsCTRLvar->directorynodeSelected != 0)   // ROOT has no parent
            {
                fpsCTRLvar->directorynodeSelected =
                    keywnode[fpsCTRLvar->directorynodeSelected].parent_index;
                fpsCTRLvar->nodeSelected = fpsCTRLvar->directorynodeSelected;
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
                }
            }
            break;

        case 10 : // enter key
            if(keywnode[fpsCTRLvar->nodeSelected].leaf == 1)   // this is a leaf
            {
				TUI_exit();
		
                if(system("clear") != 0)   // clear screen
                {
                    PRINT_ERROR("system() returns non-zero value");
                }
                functionparameter_UserInputSetParamValue(&fps[fpsCTRLvar->fpsindexSelected],
                        fpsCTRLvar->pindexSelected);
                

				TUI_initncurses();
				TUI_stdio_clear();
            }
            break;

        case ' ' :
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            pindex = keywnode[fpsCTRLvar->nodeSelected].pindex;

            // toggles ON / OFF - this is a special case not using function functionparameter_UserInputSetParamValue
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
				EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"cd %s\" C-m", fps[fpsindex].md->name, fps[fpsindex].md->fpsdirectory);
                EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"%s %s\" C-m", fps[fpsindex].md->name, fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].md->name);
            }

            break;


        case 'u' : // update conf process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            fps[fpsindex].md->signal |=
                FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            if(snprintf(msg, stringmaxlen, "UPDATE %s", fps[fpsindex].md->name) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            functionparameter_outlog("FPSCTRL", "%s", msg);
            //functionparameter_CONFupdate(fps, fpsindex);
            break;

        case 'R' : // start run process if possible
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            if(snprintf(msg, stringmaxlen, "RUNSTART %s", fps[fpsindex].md->name) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            functionparameter_outlog("FPSCTRL", msg);
            functionparameter_RUNstart(fps, fpsindex);
            break;

        case 'r' : // stop run process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            if(snprintf(msg, stringmaxlen, "RUNSTOP %s", fps[fpsindex].md->name) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            functionparameter_outlog("FPSCTRL", msg);
            functionparameter_RUNstop(fps, fpsindex);
            break;


        case 'C' : // start conf process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            if(snprintf(msg, stringmaxlen, "CONFSTART %s", fps[fpsindex].md->name) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            functionparameter_outlog("FPSCTRL", msg);
            functionparameter_CONFstart(fps, fpsindex);
            break;

        case 'c': // kill conf process
            fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
            if(snprintf(msg, stringmaxlen, "CONFSTOP %s", fps[fpsindex].md->name) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            functionparameter_outlog("FPSCTRL", msg);
            functionparameter_CONFstop(fps, fpsindex);
            break;

        case 'l': // list all parameters
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
                    printf("%4d  %4d  %s\n", keywnode[kwnindex].fpsindex, keywnode[kwnindex].pindex,
                           keywnode[kwnindex].keywordfull);
                }
            }
            printf("  TOTAL :  %d nodes\n", fpsCTRLvar->NBkwn);
            printf("\n");
            printf("Press Any Key to Continue\n");
            getchar();
            
			TUI_initncurses();
			
            break;
        
        
        case '>': // export values to filesystem
			fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
			
			functionparameter_SaveFPS2disk(&fps[fpsindex]);
			break;


        case '<': // import settings from filesystem
			TUI_exit();
            if(system("clear") != 0)
            {
                PRINT_ERROR("system() returns non-zero value");
            }
			fpsindex = keywnode[fpsCTRLvar->nodeSelected].fpsindex;
			sprintf(fname, "%s/fpscmd/fps.%s.cmd", fps[fpsindex].md->fpsdirectory, fps[fpsindex].md->name);		
			printf("READING FILE %s\n", fname);	
			fpin = fopen(fname, "r");
			if(fpin != NULL)
			{				
				char *FPScmdline = NULL;
                size_t len = 0;
                ssize_t read;

                while((read = getline(&FPScmdline, &len, fpin)) != -1)
                {   
					uint64_t taskstatus = 0;
					printf("READING CMD: %s\n", FPScmdline);
                    functionparameter_FPSprocess_cmdline(FPScmdline, fpsctrlqueuelist, keywnode,
                                                         fpsCTRLvar, fps, &taskstatus);
                }				
				fclose(fpin);
			}
			else
			{
				printf("File not found\n");
			}
			sleep(5);
			TUI_initncurses();
			break;
			

        case 'F': // process FIFO
			TUI_exit();
            if(system("clear") != 0)
            {
                PRINT_ERROR("system() returns non-zero value");
            }
            printf("Reading FIFO file \"%s\"  fd=%d\n", fpsCTRLvar->fpsCTRLfifoname,
                   fpsCTRLvar->fpsCTRLfifofd);

            if(fpsCTRLvar->fpsCTRLfifofd > 0)
            {
                // int verbose = 1;
                functionparameter_read_fpsCMD_fifo(fpsCTRLvar->fpsCTRLfifofd, fpsctrltasklist,
                                                   fpsctrlqueuelist);
            }

            printf("\n");
            printf("Press Any Key to Continue\n");
            getchar();
			TUI_initncurses();
            break;


        case 'P': // process input command file
			TUI_exit();
            if(system("clear") != 0)
            {
                PRINT_ERROR("system() returns non-zero value");
            }
            printf("Reading file confscript\n");
            fpinputcmd = fopen("confscript", "r");
            if(fpinputcmd != NULL)
            {
                char *FPScmdline = NULL;
                size_t len = 0;
                ssize_t read;

                while((read = getline(&FPScmdline, &len, fpinputcmd)) != -1)
                {
					uint64_t taskstatus = 0;
                    printf("Processing line : %s\n", FPScmdline);
                    functionparameter_FPSprocess_cmdline(FPScmdline, fpsctrlqueuelist, keywnode,
                                                         fpsCTRLvar, fps, &taskstatus);
                }
                fclose(fpinputcmd);
            }

            printf("\n");
            printf("Press Any Key to Continue\n");
            getchar();
			TUI_initncurses();
            break;
    }


    return(loopOK);
}

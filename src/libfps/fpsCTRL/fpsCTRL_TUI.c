/**
 * @file    fps_CTRLscreen.c
 * @brief   FPS control TUI
 */

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#include <unistd.h>
#include <stdio.h>
#include <ncurses.h>

#include "fps.h"
#include "fps_internal.h"
#include "timeutils.h"

#include "fpsCTRL_TUI_process_user_key.h"
#include "fps_GetTypeString.h"
#include "fps_disconnect.h"
#include "fps_outlog.h"
#include "fps_process_fpsCMDarray.h"
#include "fps_read_fpsCMD_fifo.h"
#include "fps_scan.h"

#include "TUItools.h"
#include "fpsCTRL_globals.h"

#include "fpsCTRL_FPSdisplay.h"
#include "print_nodeinfo.h"
#include "level0node_summary.h"

#include "scheduler_display.h"

static short unsigned int wrow, wcol;

#define DISPLAYMODE_HELP      1
#define DISPLAYMODE_FPSCTRL   2
#define DISPLAYMODE_SEQUENCER 3
#define DISPLAYMODE_FPSHELP   4



inline static void
fpsCTRLscreen_print_DisplayMode_status(
    int fpsCTRL_DisplayMode,
    int NBfps
)
{
    DEBUG_TRACE_FSTART();

    int  stringmaxlen = 500;
    char monstring[stringmaxlen];
    memset(monstring, 0, stringmaxlen * sizeof(
               char)); // Must use memset for a C VLA

    screenprint_setbold();

    if(snprintf(monstring,
                stringmaxlen,
                "[%d x %d] [PID %d] FUNCTION PARAMETER MONITOR: PRESS (x) TO "
                "STOP, (h) FOR HELP [%d FPS]",
                wrow,
                wcol,
                (int) getpid(),
                NBfps) < 0)
    {
        PRINT_ERROR("snprintf error");
    }
    TUI_printfw("%s", monstring); // Simplified header
    screenprint_unsetbold();
    TUI_printfw("\n");




    if(fpsCTRL_DisplayMode == DISPLAYMODE_HELP)
    {
        screenprint_setreverse();
        TUI_printfw("[h] Help");
        screenprint_unsetreverse();
    }
    else
    {
        TUI_printfw("[h] Help");
    }
    TUI_printfw("   ");

    if(fpsCTRL_DisplayMode == DISPLAYMODE_FPSHELP)
    {
        screenprint_setreverse();
        TUI_printfw("[?] FPS help");
        screenprint_unsetreverse();
    }
    else
    {
        TUI_printfw("[?] FPS help");
    }
    TUI_printfw("   ");


    if(fpsCTRL_DisplayMode == DISPLAYMODE_FPSCTRL)
    {
        screenprint_setreverse();
        TUI_printfw("[F2] FPS CTRL");
        screenprint_unsetreverse();
    }
    else
    {
        TUI_printfw("[F2] FPS CTRL");
    }
    TUI_printfw("   ");

    if(fpsCTRL_DisplayMode == DISPLAYMODE_SEQUENCER)
    {
        screenprint_setreverse();
        TUI_printfw("[F3] Sequencer");
        screenprint_unsetreverse();
    }
    else
    {
        TUI_printfw("[F3] Sequencer");
    }
    TUI_printfw("\n");
    DEBUG_TRACE_FEXIT();
}

/**
 * @brief Print help
 * 
 */
inline static void fpsCTRLscreen_print_help()
{
    DEBUG_TRACE_FSTART();
    // int attrval = A_BOLD;

    TUI_printfw("\n");
    print_help_entry("x", "Exit");

    TUI_printfw("\n");
    TUI_printfw("============ SCREENS\n");
    print_help_entry("v/V", "verbose mode on/off");
    print_help_entry("CTRL+L/R", "cycle between tabs");

    TUI_printfw("\n");
    TUI_printfw("============ OTHER\n");
    print_help_entry("s", "rescan");
    print_help_entry("T/t", "initialize/kill tmux session");
    print_help_entry("CTRL+e", "erase FPS");
    print_help_entry("E", "erase FPS and kill tmux sessions");
    print_help_entry("C/c/u", "start/stop/update conf process");
    print_help_entry("R / CTRL+r", "start/stop (R)UN process");
    print_help_entry("l", "list all entries");
    print_help_entry("f", "export fps content to datadir file");
    print_help_entry("g", "FPS log OFF");
    print_help_entry("G", "FPS log ON");
    print_help_entry(">", "export fpsdatadir values to fpsconfdir");
    print_help_entry("<", "import/load values from fpsconfdir to fps");
    print_help_entry("P", "(P)rocess input file \"confscript\"");
    TUI_printfw("        format: setval <paramfulname> <value>\n");

    DEBUG_TRACE_FEXIT();
}


/**
 * @brief Print help
 * 
 */
inline static void fpsCTRLscreen_print_FPShelp(
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
)
{
    DEBUG_TRACE_FSTART();
    // int attrval = A_BOLD;

    TUI_printfw("\n");

    TUI_printfw("FPS entry     : %s\n",
                fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->name);
    TUI_printfw(" call key     : %s\n",
                fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->callfuncname);

    TUI_printfw(" src / line   : %s / %d\n",
                fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->sourcefname,
                fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->sourceline
               );

    TUI_printfw("\n");


    // module load string
    int mloadstring_maxlen = 2000;
    char mloadstring[mloadstring_maxlen];
    memset(mloadstring, 0, sizeof(mloadstring)); // We could macro this eventually
    char mloadstringcp[mloadstring_maxlen];
    memset(mloadstringcp, 0, sizeof(mloadstringcp));
    snprintf(mloadstring, mloadstring_maxlen, " ");
    for(int m = 0;
            m < fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->NBmodule;
            m++)
    {
        snprintf(mloadstringcp,
                 mloadstring_maxlen,
                 "%smload %s;",
                 mloadstring,
                 fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->modulename[m]);
        strcpy(mloadstring, mloadstringcp);
    }

    char helpfunctionstring[2000] = {0};
    snprintf(helpfunctionstring,
             2000,
             "MILK_QUIET=1 MILK_FPSPROCINFO=1 %s-exec -n %s \"%s;%s ?\"\n",
             fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->callprogname,
             fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->name,
             mloadstring,
             fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->callfuncname
            );

    //  TUI_printfw("%s", helpfunctionstring);

    //  TUI_newline();

    // {
    //     FILE *fp = NULL;
    //     //int status = 0;
    //     int LINESLEN = 200;
    //     char line[LINESLEN];
    //     memset(line, 0, sizeof(line));


    //     fp = popen(helpfunctionstring, "r");

    //     {
    //         int printtoggle = 0;

    //         while(fgets(line, LINESLEN, fp) != NULL)
    //         {
    //             if(strncmp(HELPDETAILSSTRINGSTART, line, strlen(HELPDETAILSSTRINGSTART)) == 0)
    //             {
    //                 printtoggle = 1;
    //             }

    //             if(strncmp(HELPDETAILSSTRINGEND, line, strlen(HELPDETAILSSTRINGEND)) == 0)
    //             {
    //                 printtoggle = 0;
    //             }

    //             if(printtoggle == 1)
    //             {
    //                 TUI_printfw("%s", line);
    //             }
    //         }
    //     }

    //     pclose(fp);
    // }


    TUI_printfw("\n");

    DEBUG_TRACE_FEXIT();
}





/** @brief runs fpsCTRL GUI
 *
 * ## Purpose
 *
 * Automatically build simple ASCII GUI from function parameter structure (fps) name mask
 *
 *
 *
 */
errno_t functionparameter_CTRLscreen(
    uint32_t mode,
    char    *fpsnamemask,
    char    *fpsCTRLfifoname
)
{
    DEBUG_TRACE_FSTART();

    FPSCTRL_PROCESS_VARS fpsCTRLvar = {0};

    // keyword tree
    // KEYWORD_TREE_NODE *keywnode = NULL;


    int       loopOK  = 1;
    long long loopcnt = 0;

    long NBtaskLaunchedcnt = 0;


    // What to run ?
    // disable for testing
    int run_display = 1;
    loopOK          = 1;

    {
        struct timespec tnow = {0};
        clock_gettime(CLOCK_REALTIME, &tnow);
        FPS_TIMESTAMP = tnow.tv_sec;
        strcpy(FPS_PROCESS_TYPE, "ctrl");
    }


    {
        char cwd[PATH_MAX];
        if ( getcwd(cwd, sizeof(cwd)) == NULL )
        {
            strcpy(cwd, "ERROR");
        }
        functionparameter_outlog("FPSCTRLSTART", "%s", cwd);
    }


    DEBUG_TRACEPOINT("function start");

    // initialize fpsCTRLvar
    fpsCTRLvar.exitloop              = 0;
    fpsCTRLvar.mode                  = mode;
    fpsCTRLvar.nodeSelected          = 1;
    fpsCTRLvar.run_display           = run_display;
    fpsCTRLvar.fpsindexSelected      = 0;
    fpsCTRLvar.pindexSelected        = 0;
    fpsCTRLvar.directorynodeSelected = 0;
    fpsCTRLvar.currentlevel          = 0;
    fpsCTRLvar.direction             = 1;
    strcpy(fpsCTRLvar.fpsnamemask, fpsnamemask);
    strcpy(fpsCTRLvar.fpsCTRLfifoname, fpsCTRLfifoname);

    fpsCTRLvar.fpsCTRL_DisplayMode = 2;

    INSERT_TUI_SETUP

    fpsCTRLvar.NBindex = 0;

    for(int kn = 0; kn < NB_KEYWNODE_MAX; kn++)
    {
        strcpy(keywnode[kn].keywordfull, "");
        for(int ch = 0; ch < MAX_NB_CHILD; ch++)
        {
            keywnode[kn].child[ch] = 0;
        }
    }

    FPSCTRL_TASK_ENTRY *fpsctrltasklist = calloc(NB_FPSCTRL_TASK_MAX,
                                          sizeof(*fpsctrltasklist));
    if(fpsctrltasklist == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    for(int cmdindex = 0; cmdindex < NB_FPSCTRL_TASK_MAX; cmdindex++)
    {
        fpsctrltasklist[cmdindex].status = 0;
        fpsctrltasklist[cmdindex].queue  = 0;
    }

    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist = calloc(NB_FPSCTRL_TASKQUEUE_MAX,
                                           sizeof(* fpsctrlqueuelist));
    if(fpsctrlqueuelist == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    for(int queueindex = 0; queueindex < NB_FPSCTRL_TASKQUEUE_MAX;
            queueindex++)
    {
        fpsctrlqueuelist[queueindex].priority = 1; // 0 = not active
    }

    if (strlen(fpsCTRLvar.fpsCTRLfifoname) > 0) {
        fpsCTRLvar.fpsCTRLfifofd = open(fpsCTRLvar.fpsCTRLfifoname, O_RDWR | O_NONBLOCK);
    } else {
        fpsCTRLvar.fpsCTRLfifofd = -1;
    }
    long fifocmdcnt = 0;

    for(int level = 0; level < MAXNBLEVELS; level++)
    {
        fpsCTRLvar.GUIlineSelected[level] = 0;
    }

    for(int kindex = 0; kindex < NB_KEYWNODE_MAX; kindex++)
    {
        keywnode[kindex].NBchild = 0;
    }

    {
        long NBpindex = 0;
        functionparameter_scan_fps(fpsCTRLvar.mode,
                                   fpsCTRLvar.fpsnamemask,
                                   fpsarray,
                                   keywnode,
                                   &fpsCTRLvar.NBkwn,
                                   &fpsCTRLvar.NBfps,
                                   &NBpindex,
                                   1 // verbose
                                  );
        TUI_printfw("%d function parameter structure(s) imported, %ld parameters\n",
               fpsCTRLvar.NBfps,
               NBpindex);
    }

    fpsCTRLvar.nodeSelected = 1;

    char shmdname[STRINGMAXLEN_SHMDIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    if(run_display == 0)
    {
        loopOK = 0;
    }

    // how long between getchar probes
    int getchardt_us_ref = 100000;

    // refresh every 0.1 sec without input
    int refreshtimeoutus_ref = 100000;

    int getchardt_us     = getchardt_us_ref;
    int refreshtimeoutus = refreshtimeoutus_ref;

    // if(TUI_get_screenprintmode() == SCREENPRINT_NCURSES)  // ncurses mode
    {
        refreshtimeoutus_ref = 100000; // 10 Hz
    }

    int refresh_screen = 1; // 1 if screen should be refreshed



    while(loopOK == 1)
    {
        int NBtaskLaunched = 0;

        //long icnt = 0;
        int ch = -1;

        int timeoutuscnt = 0;

        while(refresh_screen == 0)  // wait for input
        {
            // put input commands from fifo into the task queue
            if (fpsCTRLvar.fpsCTRLfifofd > 0) {
                int fcnt =
                    functionparameter_read_fpsCMD_fifo(
                        fpsCTRLvar.fpsCTRLfifofd,
                        fpsctrltasklist,
                        fpsctrlqueuelist);
                fifocmdcnt += fcnt;
            }

            DEBUG_TRACEPOINT(" ");

            // execute next command in the queue
            int taskflag =
                function_parameter_process_fpsCMDarray(
                    fpsctrltasklist,
                    fpsctrlqueuelist,
                    keywnode,
                    &fpsCTRLvar,
                    fpsarray);

            if(taskflag > 0)  // task has been performed
            {
                getchardt_us = 1000; // check often
            }
            else
            {
                // gradually slow down
                getchardt_us = (int)(1.01 * getchardt_us);

                if(getchardt_us > getchardt_us_ref)
                {
                    getchardt_us = getchardt_us_ref;
                }
            }
            NBtaskLaunched += taskflag;

            NBtaskLaunchedcnt += NBtaskLaunched;


            usleep(getchardt_us);

            // ==================
            // = GET USER INPUT =
            // ==================
            ch = get_singlechar_nonblock();

            if(ch == -1)
            {
                refresh_screen = 0;
            }
            else
            {
                refresh_screen = 2;
                getchardt_us = 10000; // check often
            }

            timeoutuscnt += getchardt_us;
            if(timeoutuscnt > refreshtimeoutus)
            {
                refresh_screen = 1;
            }

            DEBUG_TRACEPOINT(" ");
        }

        if(refresh_screen > 0)
        {
            refresh_screen--; // will wait next time we enter the loop
        }

        TUI_get_terminal_size(&wrow, &wcol);

        DEBUG_TRACEPOINT(" ");

        loopOK = fpsCTRL_TUI_process_user_key(ch,
                                              fpsarray,
                                              keywnode,
                                              fpsctrltasklist,
                                              fpsctrlqueuelist,
                                              &fpsCTRLvar);

        DEBUG_TRACEPOINT(" ");

        if(fpsCTRLvar.exitloop == 1)
        {
            loopOK = 0;
        }

        if(fpsCTRLvar.run_display == 1)
        {

            TUI_ncurses_erase();

            fpsCTRLscreen_print_DisplayMode_status(
                fpsCTRLvar.fpsCTRL_DisplayMode,
                fpsCTRLvar.NBfps);

            DEBUG_TRACEPOINT(" ");

            if(fpsCTRLvar.fpsCTRL_DisplayVerbose == 1)
            {
                TUI_printfw(
                    "======== FPSCTRL info  ( screen refresh cnt %7lld  "
                    "scan interval %7d us)\n",
                    loopcnt,
                    getchardt_us);
                TUI_printfw(
                    "    INPUT FIFO       :  %s (fd=%d)    fifocmdcnt = "
                    "%ld   NBtaskLaunched = %d -> %ld   [NB FPS = %d]\n",
                    fpsCTRLvar.fpsCTRLfifoname,
                    fpsCTRLvar.fpsCTRLfifofd,
                    fifocmdcnt,
                    NBtaskLaunched,
                    NBtaskLaunchedcnt,
                    fpsCTRLvar.NBfps);

                DEBUG_TRACEPOINT(" ");
                char logfname[STRINGMAXLEN_FULLFILENAME];
                getFPSlogfname(logfname);
                int flagFPSoutlog = get_FLAG_FPSOUTLOG();
                if ( flagFPSoutlog == 0 )
                {
                    TUI_printfw("    OUTPUT LOG   [%2d] : DISABLED (G/g to toggle ON/OFF)", flagFPSoutlog);
                }
                else
                {
                    TUI_printfw("    OUTPUT LOG   [%2d] :  %s", flagFPSoutlog, logfname);
                }
                TUI_printfw("\n");
            }
            DEBUG_TRACEPOINT(" ");




            if(fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_HELP)
            {
                fpsCTRLscreen_print_help();
            }

            if(fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_FPSCTRL)
            {
                fpsCTRL_FPSdisplay(keywnode, &fpsCTRLvar);
            }

            DEBUG_TRACEPOINT(" ");

            if(fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_SEQUENCER)
            {
                fpsCTRL_scheduler_display(fpsctrltasklist,
                                          fpsctrlqueuelist,
                                          wrow,
                                          &fpsCTRLvar.scheduler_wrowstart);
            }

            if(fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_FPSHELP)
            {
                fpsCTRLscreen_print_FPShelp(keywnode, &fpsCTRLvar);
            }



            DEBUG_TRACEPOINT(" ");

            TUI_ncurses_refresh();

            DEBUG_TRACEPOINT(" ");

        } // end run_display

        DEBUG_TRACEPOINT("exit from if( fpsCTRLvar.run_display == 1)");

        fpsCTRLvar.run_display = run_display;

        loopcnt++;

        /*
        if((data.signal_TERM == 1) || (data.signal_INT == 1) ||
                (data.signal_ABRT == 1) || (data.signal_BUS == 1) ||
                (data.signal_SEGV == 1) || (data.signal_HUP == 1) ||
                (data.signal_PIPE == 1))
        {
            printf("Exit condition met\n");
            loopOK = 0;
        }
        */
    }




    if(run_display == 1)
    {
        // TUI_exit();
        endwin();
    }




    {
        char cwd[PATH_MAX];
        if ( getcwd(cwd, sizeof(cwd)) == NULL )
        {
            strcpy(cwd, "ERROR");
        }
        functionparameter_outlog("FPSCTRLSTOP", "%s", cwd);
    }

    DEBUG_TRACEPOINT("Disconnect from FPS entries");
    for(int fpsindex = 0; fpsindex < fpsCTRLvar.NBfps; fpsindex++)
    {
        function_parameter_struct_disconnect(&fpsarray[fpsindex]);
    }


    // free(keywnode);

    free(fpsctrltasklist);
    free(fpsctrlqueuelist);
    functionparameter_outlog("LOGFILECLOSE", "close log file");

    DEBUG_TRACE_FEXIT();

    return RETURN_SUCCESS;
}
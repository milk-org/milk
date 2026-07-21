// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_CTRLscreen.c
 * @brief   FPS control TUI
 */

#include <dirent.h>
#include <sys/stat.h>

#include <poll.h>

#include "fps.h"
#include "fps_isvalid.h"

#include "fpsCTRL_TUI_process_user_key.h"

#include "fpsCTRL_TUIcompat.h"
#include "fpsCTRL_globals.h"

#include "fpsCTRL_FPSdisplay.h"

#include "scheduler_display.h"

static short unsigned int wrow, wcol;

#define DISPLAYMODE_HELP 1
#define DISPLAYMODE_FPSLOG 2
#define DISPLAYMODE_FPSCTRL 3
#define DISPLAYMODE_SEQUENCER 4


inline static void fpsCTRLscreen_print_footer_status(FPSCTRL_PROCESS_VARS *fpsCTRLvar, int NBfps)
{
    DEBUG_TRACE_FSTART();

    int status_row = sc_term_rows;
    if (fpsCTRLvar->search_mode > 0)
    {
        SC_APPEND("\033[%d;1H", status_row - 1);
        SC_APPEND("\033[2K"); // Clear line
        screenprint_setbold();
        if (fpsCTRLvar->search_mode == 1)
        {
            ansi_detect_color_level();
            if (ansi__color_level >= 3)
            {
                SC_APPEND("\033[38;2;255;165;0m"); // Orange
            }
            else if (ansi__color_level == 2)
            {
                SC_APPEND("\033[38;5;214m");
            }
            else
            {
                SC_APPEND("\033[33m");
            }
            SC_APPEND(
                " \xE2\x9C\x94 Search: "); // Fallback magnifying glass equivalent or checkmark. Using '?' for safety in some terms, or just "Search:"
            // Wait, we can use 🔍 if terminal supports it. The user specified "sleek", let's just use "Search:" to be safe with fonts or standard Unicode.
            SC_APPEND("Search: ");
        }
        else
        {
            screenprint_color_flag();
            SC_APPEND(" Filter: ");
        }
        screenprint_unsetbold();
        screenprint_setnormal();
        SC_APPEND("%s", fpsCTRLvar->search_string);
        if (fpsCTRLvar->search_mode == 1)
        {
            SC_APPEND("\033[5m_\033[25m"); // Blinking cursor
        }
    }

    SC_APPEND("\033[%d;1H", status_row);
    screenprint_set_status_bar();
    SC_APPEND("\033[2K"); // clear line with background color

    SC_APPEND(" [MODE: ");
    if (fpsCTRLvar->fpsCTRL_DisplayMode == DISPLAYMODE_HELP)
    {
        SC_APPEND("HELP] ");
    }
    else if (fpsCTRLvar->fpsCTRL_DisplayMode == DISPLAYMODE_FPSLOG)
    {
        SC_APPEND("LOG] ");
    }
    else if (fpsCTRLvar->fpsCTRL_DisplayMode == DISPLAYMODE_FPSCTRL)
    {
        SC_APPEND("CTRL] ");
    }
    else if (fpsCTRLvar->fpsCTRL_DisplayMode == DISPLAYMODE_SEQUENCER)
    {
        SC_APPEND("SEQ] ");
    }

    if (fpsCTRLvar->fpsCTRL_DisplayMode == DISPLAYMODE_FPSCTRL)
    {
        static const char *smode[] = { "Default", "A-Z", "Status" };
        int                sm      = fpsCTRLvar->sort_mode;
        if (sm < 0 || sm > 2)
        {
            sm = 0;
        }
        SC_APPEND("[SORT: %s] ", smode[sm]);
    }

    SC_APPEND("[PID %d] [%d FPS] ", (int) getpid(), NBfps);
    SC_APPEND("| (x) Exit  (h) Help  (?) Log  (F2) CTRL  (F3) SEQ  (/) Search ");

    if (fpsCTRLvar->fpsCTRL_DisplayVerbose)
    {
        SC_APPEND(" [VERBOSE] ");
    }

    // Pad rest of line with background (handled by 2K mostly, but just in case)
    screenprint_setnormal();
    DEBUG_TRACE_FEXIT();
}

/**
 * @brief Print help
 *
 */
inline static void fpsCTRLscreen_print_help(FPSCTRL_PROCESS_VARS *fpsCTRLvar)
{
    DEBUG_TRACE_FSTART();

    char lines[100][256];
    int  line_cnt = 0;

#define ADD_LINE(...)                                      \
    do                                                     \
    {                                                      \
        if (line_cnt < 100)                                \
            snprintf(lines[line_cnt++], 256, __VA_ARGS__); \
    } while (0)
#define ADD_BLANK()                      \
    do                                   \
    {                                    \
        if (line_cnt < 100)              \
            lines[line_cnt++][0] = '\0'; \
    } while (0)

    ADD_BLANK();
    ADD_LINE("  \033[1;38;5;111m============ SCREENS\033[0m");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "v / V", "Verbose mode ON / OFF");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "F2", "Parameter control (Main Screen)");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "F3", "Scheduler / Sequencer");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "?", "FPS log");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "CTRL+L/R", "Cycle between tabs");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "h", "Show this help screen");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "x", "Exit TUI");

    ADD_BLANK();
    ADD_LINE("  \033[1;38;5;111m============ PARAMETER EDITING\033[0m");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "ENTER", "Edit selected parameter value");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "SPACE", "Toggle ON/OFF parameter state");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "UP / DOWN",
             "Navigate up and down between parameters");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "LEFT / RIGHT", "Collapse/expand tree depth");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "/", "Fuzzy search parameter tree");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "S", "Toggle alphabetical sorting");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "]", "Cycle sort: Default -> A-Z -> Status");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "y", "Yank parameter value to tmux buffer");

    ADD_BLANK();
    ADD_LINE("  \033[1;38;5;111m============ PROCESS CONTROL\033[0m");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "O / CTRL+o", "Start/stop conf process");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "R / CTRL+r", "Start/stop run process");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "T / CTRL+t", "Initialize/kill tmux session");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "CTRL+a", "Attach to tmux session");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "CTRL+e", "Stop conf/run, erase FPS");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "E", "Stop conf/run, erase FPS, kill tmux");

    ADD_BLANK();
    ADD_LINE("  \033[1;38;5;111m============ OTHER UTILITIES\033[0m");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "s", "Rescan shared memory directory");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "l", "List all entries");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "f", "Export fps content to datadir file");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "G / g", "FPS log ON / OFF");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", ">", "Export fpsdatadir values to fpsconfdir");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "<", "Import/load values from fpsconfdir to fps");
    ADD_LINE("    \033[1;36m%-20s\033[0m  %s", "P", "Process input file \"confscript\"");

    // Enforce scroll bounds
    int display_lines = sc_term_rows - 4; // leave margin for header/footer
    if (display_lines < 5)
    {
        display_lines = 5;
    }

    int max_scroll = line_cnt - display_lines;
    if (max_scroll < 0)
    {
        max_scroll = 0;
    }
    if (fpsCTRLvar->help_wrowstart > max_scroll)
    {
        fpsCTRLvar->help_wrowstart = max_scroll;
    }
    if (fpsCTRLvar->help_wrowstart < 0)
    {
        fpsCTRLvar->help_wrowstart = 0;
    }

    for (int line_idx = 0; line_idx < display_lines; line_idx++)
    {
        int idx = fpsCTRLvar->help_wrowstart + line_idx;
        if (idx < line_cnt)
        {
            TUI_printfw("%s\n", lines[idx]);
        }
    }

#undef ADD_LINE
#undef ADD_BLANK

    DEBUG_TRACE_FEXIT();
}


/**
 * @brief Print help
 *
 */
inline static void fpsCTRLscreen_print_FPShelp(KEYWORD_TREE_NODE    *keywnode,
                                               FPSCTRL_PROCESS_VARS *fpsCTRLvar)
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
                fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->sourceline);

    TUI_printfw("\n");


    // module load string
    int  mloadstring_maxlen = 2000;
    char mloadstring[mloadstring_maxlen];
    memset(mloadstring, 0, sizeof(mloadstring)); // We could macro this eventually
    char mloadstringcp[mloadstring_maxlen];
    memset(mloadstringcp, 0, sizeof(mloadstringcp));
    snprintf(mloadstring, mloadstring_maxlen, " ");
    for (int mod_idx = 0;
         mod_idx < fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->NBmodule; mod_idx++)
    {
        snprintf(mloadstringcp, mloadstring_maxlen, "%smload %s;", mloadstring,
                 fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->modulename[mod_idx]);
        snprintf(mloadstring, mloadstring_maxlen, "%s", mloadstringcp);
    }

    char helpfunctionstring[2000] = { 0 };
    snprintf(helpfunctionstring, 2000,
             "MILK_QUIET=1 MILK_FPSPROCINFO=1 %s-exec -n %s \"%s;%s ?\"\n",
             fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->callprogname,
             fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->name, mloadstring,
             fpsarray[keywnode[fpsCTRLvar->nodeSelected].fpsindex].md->callfuncname);

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
errno_t functionparameter_CTRLscreen(uint32_t mode,
                                     char    *fpsnamemask,
                                     char    *fpsCTRLfifoname,
                                     double   timeout_sec __attribute__((unused)))
{
    DEBUG_TRACE_FSTART();

    FPSCTRL_PROCESS_VARS fpsCTRLvar = { 0 };

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
        struct timespec tnow = { 0 };
        clock_gettime(CLOCK_REALTIME, &tnow);
        FPS_TIMESTAMP = tnow.tv_sec;
        snprintf(FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "ctrl");
    }


    {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) == NULL)
        {
            snprintf(cwd, sizeof(cwd), "ERROR");
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
    strncpy(fpsCTRLvar.fpsnamemask, fpsnamemask, sizeof(fpsCTRLvar.fpsnamemask) - 1);
    fpsCTRLvar.fpsnamemask[sizeof(fpsCTRLvar.fpsnamemask) - 1] = '\0';
    strncpy(fpsCTRLvar.fpsCTRLfifoname, fpsCTRLfifoname, sizeof(fpsCTRLvar.fpsCTRLfifoname) - 1);
    fpsCTRLvar.fpsCTRLfifoname[sizeof(fpsCTRLvar.fpsCTRLfifoname) - 1] = '\0';

    fpsCTRLvar.fpsCTRL_DisplayMode = DISPLAYMODE_FPSCTRL;


    fpsCTRLvar.NBindex = 0;

    for (int knode_idx = 0; knode_idx < NB_KEYWNODE_MAX; knode_idx++)
    {
        keywnode[knode_idx].keywordfull[0] = '\0';
        for (int child_idx = 0; child_idx < MAX_NB_CHILD; child_idx++)
        {
            keywnode[knode_idx].child[child_idx] = 0;
        }
    }

    fpsCTRLvar.milkseq_state   = NULL;
    fpsCTRLvar.milkseq_name[0] = '\0';

    if (strlen(fpsCTRLvar.fpsCTRLfifoname) > 0)
    {
        // Create FIFO if it does not exist
        if (access(fpsCTRLvar.fpsCTRLfifoname, F_OK) == -1)
        {
            if (mkfifo(fpsCTRLvar.fpsCTRLfifoname, 0666) == -1)
            {
                PRINT_ERROR("mkfifo: %s", strerror(errno));
            }
        }
        fpsCTRLvar.fpsCTRLfifofd = open(fpsCTRLvar.fpsCTRLfifoname, O_RDWR | O_NONBLOCK);
    }
    else
    {
        fpsCTRLvar.fpsCTRLfifofd = -1;
    }
    long fifocmdcnt = 0;

    for (int level_idx = 0; level_idx < MAXNBLEVELS; level_idx++)
    {
        fpsCTRLvar.GUIlineSelected[level_idx] = 0;
    }

    for (int keyw_idx = 0; keyw_idx < NB_KEYWNODE_MAX; keyw_idx++)
    {
        keywnode[keyw_idx].NBchild = 0;
    }

    {
        long NBpindex = 0;
        functionparameter_scan_fps(fpsCTRLvar.mode, fpsCTRLvar.fpsnamemask, fpsarray, keywnode,
                                   &fpsCTRLvar.NBkwn, &fpsCTRLvar.NBfps, &NBpindex,
                                   1 // verbose
        );
        TUI_printfw("%d function parameter structure(s) imported, %ld parameters\n",
                    fpsCTRLvar.NBfps, NBpindex);
    }

    fpsCTRLvar.nodeSelected = 1;

    char shmdname[STRINGMAXLEN_SHMDIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    if (run_display == 0)
    {
        loopOK = 0;
    }

    // poll() timeout: ~10 Hz periodic refresh
    int poll_timeout_ms = 100;

    int refresh_screen = 1; // 1 if screen should be refreshed

    if (run_display == 1)
    {
        TUI_init_terminal(&wcol, &wrow);
    }

    while (loopOK == 1)
    {
        // ==== VALIDATE ALL CONNECTED FPSs ====
        int needs_rescan = 0;
        for (int fps_idx = 0; fps_idx < fpsCTRLvar.NBfps; fps_idx++)
        {
            if (!function_parameter_struct_isvalid(&fpsarray[fps_idx]))
            {
                needs_rescan = 1;
                fps_disconnect(&fpsarray[fps_idx]);
            }
        }

        // ==== CHECK FOR NEW OR REMOVED FPSs BY COUNT ====
        static int last_shm_count    = -1;
        int        current_shm_count = 0;
        DIR       *d                 = opendir(shmdname);
        if (d)
        {
            struct dirent *dir;
            while ((dir = readdir(d)) != NULL)
            {
                if (strstr(dir->d_name, ".fps.shm") != NULL)
                {
                    current_shm_count++;
                }
            }
            closedir(d);
        }

        if (last_shm_count == -1)
        {
            last_shm_count = current_shm_count;
        }
        else if (current_shm_count != last_shm_count)
        {
            needs_rescan   = 1;
            last_shm_count = current_shm_count;
        }

        if (needs_rescan)
        {
            long NBpindex = 0;
            functionparameter_scan_fps(fpsCTRLvar.mode, fpsCTRLvar.fpsnamemask, fpsarray, keywnode,
                                       &fpsCTRLvar.NBkwn, &fpsCTRLvar.NBfps, &NBpindex,
                                       0 // quiet
            );
            refresh_screen = 2; // Force refresh

            // Adjust selected index if it's now out of bounds
            if (fpsCTRLvar.fpsindexSelected >= fpsCTRLvar.NBfps && fpsCTRLvar.NBfps > 0)
            {
                fpsCTRLvar.fpsindexSelected = fpsCTRLvar.NBfps - 1;
            }
            else if (fpsCTRLvar.NBfps == 0)
            {
                fpsCTRLvar.fpsindexSelected = 0;
            }

            // Clamping node indices to the keyword array bounds
            if (fpsCTRLvar.nodeSelected >= fpsCTRLvar.NBkwn && fpsCTRLvar.NBkwn > 0)
            {
                fpsCTRLvar.nodeSelected = fpsCTRLvar.NBkwn - 1;
            }
            else if (fpsCTRLvar.NBkwn == 0)
            {
                fpsCTRLvar.nodeSelected = 0;
            }
            if (fpsCTRLvar.directorynodeSelected >= fpsCTRLvar.NBkwn && fpsCTRLvar.NBkwn > 0)
            {
                fpsCTRLvar.directorynodeSelected = 0; // fallback to ROOT
            }
            else if (fpsCTRLvar.NBkwn == 0)
            {
                fpsCTRLvar.directorynodeSelected = 0;
            }
        }

        int NBtaskLaunched = 0;
        int ch             = ANSI_KEY_NONE;

        // Event-driven input: poll() on stdin
        // Mirrors milkCTRL.c pattern
        {
            struct pollfd pfd;
            pfd.fd     = STDIN_FILENO;
            pfd.events = POLLIN;

            int pr = poll(&pfd, 1, poll_timeout_ms);
            if (pr > 0 && (pfd.revents & POLLIN))
            {
                ch = get_singlechar_nonblock();
                if (ch != ANSI_KEY_NONE)
                {
                    refresh_screen = 2;
                }
            }
            else
            {
                // timeout or error: periodic refresh
                refresh_screen = 1;
            }
        }

        if (refresh_screen > 0)
        {
            refresh_screen--; // will wait next time we enter the loop
        }

        TUI_get_terminal_size(&wrow, &wcol);

        DEBUG_TRACEPOINT(" ");

        loopOK = fpsCTRL_TUI_process_user_key(ch, fpsarray, keywnode, NULL, NULL, &fpsCTRLvar);

        DEBUG_TRACEPOINT(" ");

        if (fpsCTRLvar.exitloop == 1)
        {
            loopOK = 0;
        }

        if (fpsCTRLvar.run_display == 1)
        {
            TUI_stdio_clear();

            DEBUG_TRACEPOINT(" ");

            if (fpsCTRLvar.fpsCTRL_DisplayVerbose == 1)
            {
                TUI_printfw("======== FPSCTRL info  "
                            "(screen refresh cnt %7lld)\n",
                            loopcnt);
                TUI_printfw("    INPUT FIFO       :  %s (fd=%d)    fifocmdcnt = "
                            "%ld   NBtaskLaunched = %d -> %ld   [NB FPS = %d]\n",
                            fpsCTRLvar.fpsCTRLfifoname, fpsCTRLvar.fpsCTRLfifofd, fifocmdcnt,
                            NBtaskLaunched, NBtaskLaunchedcnt, fpsCTRLvar.NBfps);

                DEBUG_TRACEPOINT(" ");
                char logfname[STRINGMAXLEN_FULLFILENAME];
                getFPSlogfname(logfname);
                int flagFPSoutlog = get_FLAG_FPSOUTLOG();
                if (flagFPSoutlog == 0)
                {
                    TUI_printfw("    OUTPUT LOG   [%2d] : DISABLED (G/g to toggle ON/OFF)",
                                flagFPSoutlog);
                }
                else
                {
                    TUI_printfw("    OUTPUT LOG   [%2d] :  %s", flagFPSoutlog, logfname);
                }
                TUI_printfw("\n");

                if (fpsCTRLvar.NBfps > 0 && fpsarray[fpsCTRLvar.fpsindexSelected].md != NULL)
                {
                    TUI_printfw("    FPS keywords     :  %s\n",
                                fpsarray[fpsCTRLvar.fpsindexSelected].md->keywordarray);
                }
            }
            DEBUG_TRACEPOINT(" ");


            if (fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_HELP)
            {
                fpsCTRLscreen_print_help(&fpsCTRLvar);
            }

            if (fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_FPSCTRL)
            {
                fpsCTRL_FPSdisplay(keywnode, &fpsCTRLvar);
            }

            DEBUG_TRACEPOINT(" ");

            if (fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_SEQUENCER)
            {
                fpsCTRL_scheduler_display(&fpsCTRLvar, wrow, &fpsCTRLvar.scheduler_wrowstart);
            }

            if (fpsCTRLvar.fpsCTRL_DisplayMode == DISPLAYMODE_FPSLOG)
            {
                fpsCTRL_FPSlog(keywnode, &fpsCTRLvar);
            }


            DEBUG_TRACEPOINT(" ");


            DEBUG_TRACEPOINT(" ");

            fpsCTRLscreen_print_footer_status(&fpsCTRLvar, fpsCTRLvar.NBfps);

            sc_frame_flush();
        } // end run_display

        DEBUG_TRACEPOINT("exit from if( fpsCTRLvar.run_display == 1)");

        fpsCTRLvar.run_display = run_display;

        loopcnt++;

        if ((processinfo_signal_TERM == 1) || (processinfo_signal_INT == 1) ||
            (processinfo_signal_ABRT == 1) || (processinfo_signal_BUS == 1) ||
            (processinfo_signal_SEGV == 1) || (processinfo_signal_HUP == 1) ||
            (processinfo_signal_PIPE == 1))
        {
            printf("Exit condition met\n");
            loopOK = 0;
        }
    }


    if (run_display == 1)
    {
        // TUI_exit();
        TUI_exit();
    }


    {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) == NULL)
        {
            snprintf(cwd, sizeof(cwd), "ERROR");
        }
        functionparameter_outlog("FPSCTRLSTOP", "%s", cwd);
    }

    DEBUG_TRACEPOINT("Disconnect from FPS entries");
    for (int fps_idx = 0; fps_idx < fpsCTRLvar.NBfps; fps_idx++)
    {
        fps_disconnect(&fpsarray[fps_idx]);
    }


    // free(keywnode);

    // No longer freeing local task queues
    if (strlen(fpsCTRLvar.fpsCTRLfifoname) > 0)
    {
        unlink(fpsCTRLvar.fpsCTRLfifoname);
    }

    // Free persistent scheduler sort buffers
    free(fpsCTRLvar.sched_sort_eval);
    free(fpsCTRLvar.sched_sort_index);

    functionparameter_outlog("LOGFILECLOSE", "close log file");

    DEBUG_TRACE_FEXIT();

    return RETURN_SUCCESS;
}

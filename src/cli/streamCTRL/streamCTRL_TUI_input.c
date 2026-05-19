#include "streamCTRL_TUI_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/**
 * @brief Process keyboard input for streamCTRL TUI.
 *
 * Dispatches key events to navigation, sorting,
 * and filter handlers.
 */
errno_t streamCTRL_keyinput_process(
    int ch,
    streamCTRLarg_struct *streamCTRLdata,
    struct streamCTRL_TUI_state *state
)
{
    char c; // for user input
    int  stringindex;
    time_t  rawtime;
    long sindex;

    switch(ch)
    {
    case ANSI_KEY_CTRL_LEFT: // CTRL+LEFT
        sTUIparam.DisplayMode--;
        if (sTUIparam.DisplayMode < DISPLAY_MODE_HELP)
            sTUIparam.DisplayMode = TAB_COUNT;
        break;

    case ANSI_KEY_CTRL_RIGHT: // CTRL+RIGHT
        sTUIparam.DisplayMode++;
        if (sTUIparam.DisplayMode > TAB_COUNT)
            sTUIparam.DisplayMode = DISPLAY_MODE_HELP;
        break;

    case 3:   // Ctrl+C
    case 'x': // Exit control screen
        sTUIparam.loopOK = 0;
        break;

    case ANSI_KEY_UP:
        sTUIparam.dindexSelected--;
        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        break;

    case ANSI_KEY_DOWN:
        sTUIparam.dindexSelected++;
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }
        break;

    case ANSI_KEY_PGUP:
        sTUIparam.dindexSelected -= 10;
        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        break;

    case ANSI_KEY_LEFT:
        sTUIparam.DisplayDetailLevel = 0;
        break;

    case ANSI_KEY_RIGHT:
        sTUIparam.DisplayDetailLevel = 1;
        break;

    case ANSI_KEY_PGDN:
        sTUIparam.dindexSelected += 10;
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }
        break;

    // ============ SCREENS

    case 'h': // help
        sTUIparam.DisplayMode = DISPLAY_MODE_HELP;
        break;

    case ANSI_KEY_F2: // semvals
        sTUIparam.DisplayMode = DISPLAY_MODE_SUMMARY;
        break;

    case ANSI_KEY_F3: // write PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_WRITE;
        break;

    case ANSI_KEY_F4: // read PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_READ;
        break;

    case ANSI_KEY_F5: // read PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_SPTRACE;
        break;

    case ANSI_KEY_F6: // open files
        if((sTUIparam.DisplayMode == DISPLAY_MODE_FUSER) ||
                (streamCTRLdata->streaminfoproc->fuserUpdate0 == 1))
        {
            streamCTRLdata->streaminfoproc->fuserUpdate = 1;
            time(&rawtime);
            sTUIparam.uttime_lastScan           = gmtime(&rawtime);
            sTUIparam.fuserScan                 = 1;
        }
        sTUIparam.DisplayMode = DISPLAY_MODE_FUSER;
        break;

    // ============ ACTIONS

    case ctrl('e'): // erase stream
        if(sTUIparam.dindexSelected >= 0)
        {
            sindex =
                sTUIparam.ssindex[
                    sTUIparam.dindexSelected];
            // Flag for removal by scan thread
            // Actual destroy happens in scan
            // thread to avoid race condition
            streamCTRLdata->sinfo[sindex].erased = 1;
        }
        break;

    // ============ SCANNING

    case '{': // slower scan update
        streamCTRLdata->streaminfoproc->twaitus = (int)(1.2 *
                streamCTRLdata->streaminfoproc->twaitus);
        if(streamCTRLdata->streaminfoproc->twaitus > 1000000)
        {
            streamCTRLdata->streaminfoproc->twaitus = 1000000;
        }
        break;

    case '}': // faster scan update
        streamCTRLdata->streaminfoproc->twaitus =
            (int)(0.83333333333333333333 * streamCTRLdata->streaminfoproc->twaitus);
        if(streamCTRLdata->streaminfoproc->twaitus < 1000)
        {
            streamCTRLdata->streaminfoproc->twaitus = 1000;
        }
        break;

    case 'o': // output next scan to file
        streamCTRLdata->streaminfoproc->WriteFlistToFile = 1;
        break;

    // ============ DISPLAY

    case '-': // slower display update
        sTUIparam.frequ *= 0.5;
        if(sTUIparam.frequ < 1.0)
        {
            sTUIparam.frequ = 1.0;
        }
        if(sTUIparam.frequ > 64.0)
        {
            sTUIparam.frequ = 64.0;
        }
        break;

    case '+': // faster display update
        sTUIparam.frequ *= 2.0;
        if(sTUIparam.frequ < 1.0)
        {
            sTUIparam.frequ = 1.0;
        }
        if(sTUIparam.frequ > 64.0)
        {
            sTUIparam.frequ = 64.0;
        }
        break;

    case '1': // shortcut: sort by stream name
        sTUIparam.sort_col = STREAM_SORT_NAME;
        sTUIparam.sort_dir = 0;
        sTUIparam.SORTING = 1;
        break;

    case '2': // shortcut: sort by update recency
        sTUIparam.sort_col = STREAM_SORT_NONE;
        sTUIparam.sort_dir = 0;
        sTUIparam.SORTING     = 2;
        sTUIparam.SORT_TOGGLE = 1;
        break;

    case '3': // shortcut: sort by process access
        sTUIparam.sort_col = STREAM_SORT_NONE;
        sTUIparam.sort_dir = 0;
        sTUIparam.SORTING     = 3;
        sTUIparam.SORT_TOGGLE = 1;
        break;

    case '4': // shortcut: sort by frequency
        sTUIparam.sort_col = STREAM_SORT_FREQ;
        sTUIparam.sort_dir = 1; // descending
        sTUIparam.SORTING = 0;
        break;

    case ']': // next sort column
        sTUIparam.sort_col++;
        if (sTUIparam.sort_col > STREAM_NB_SORT_COLS)
            sTUIparam.sort_col = STREAM_SORT_NONE;
        sTUIparam.sort_dir = 0;
        // Disable legacy sort modes
        sTUIparam.SORTING = 0;
        break;

    case '[': // toggle sort direction
        if (sTUIparam.sort_col > STREAM_SORT_NONE)
            sTUIparam.sort_dir = !sTUIparam.sort_dir;
        break;

    case 'f': // stream name filter toggle
        if(streamCTRLdata->streaminfoproc->filter == 0)
        {
            streamCTRLdata->streaminfoproc->filter = 1;
        }
        else
        {
            streamCTRLdata->streaminfoproc->filter = 0;
        }
        break;

    case 'F': // set stream name filter string
        // TUI_exit();
        EXECUTE_SYSTEM_COMMAND_NOCHECK("clear");
        printf("Enter string: ");
        fflush(stdout);
        stringindex = 0;
        while(((c = getchar()) != '\n') &&
                (stringindex < STRINGLENMAX - 2))
        {
            streamCTRLdata->streaminfoproc->namefilter[stringindex] = c;
            if(c == 127)  // delete key
            {
                putchar(0x8);
                putchar(' ');
                putchar(0x8);
                stringindex--;
            }
            else
            {
                //printf("[%d]", (int) c);
                putchar(c); // echo on screen
                stringindex++;
            }
        }
        printf("string entered\n");
        streamCTRLdata->streaminfoproc->namefilter[stringindex] = '\0';
        TUI_init_terminal(&wrow, &wcol);
        break;

    case 's': // toggle all sems / 2 sems
        sTUIparam.DISPLAY_ALL_SEMS = !sTUIparam.DISPLAY_ALL_SEMS;
        break;

    case 'r': // force full screen redraw
        if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
        break;

    // ============ MOUSE

    case ANSI_KEY_MOUSE: // left-click selects entry
    {
        int click_row = ansi__last_mouse.y;
        int body_row  = state->body_start_row;
        if(body_row > 0 && click_row >= body_row)
        {
            int new_sel =
                (int) state->doffsetindex
                + (click_row - body_row);
            if(new_sel >= 0
               && new_sel < sTUIparam.NBsindex)
            {
                sTUIparam.dindexSelected = new_sel;
            }
        }
        break;
    }

    case ANSI_KEY_SCROLL_UP:
        sTUIparam.dindexSelected -= 3;
        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        break;

    case ANSI_KEY_SCROLL_DN:
        sTUIparam.dindexSelected += 3;
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }
        break;
    }
    return EXIT_SUCCESS;
}

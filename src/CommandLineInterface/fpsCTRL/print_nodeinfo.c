#include <limits.h>

#include "CommandLineInterface/CLIcore.h"

#include "fps/fps_GetTypeString.h"
#include "TUItools.h"

/**
 * @brief Print node information
 *
 * @param fps
 * @param keywnode
 * @param nodeSelected
 * @param fpsindexSelected
 * @param pindexSelected
 */
void fpsCTRLscreen_print_nodeinfo(FUNCTION_PARAMETER_STRUCT *fps,
                                  KEYWORD_TREE_NODE         *keywnode,
                                  int                        nodeSelected,
                                  int                        fpsindexSelected,
                                  int                        pindexSelected)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("Selected node %d in fps %d",
                     nodeSelected,
                     keywnode[nodeSelected].fpsindex);

    TUI_printfw("======== FPS info ( # %5d)", keywnode[nodeSelected].fpsindex);
    TUI_newline();

    char teststring[200];
    sprintf(teststring,
            "%s",
            fps[keywnode[nodeSelected].fpsindex].md->sourcefname);
    DEBUG_TRACEPOINT("TEST STRING : %s", teststring);

    DEBUG_TRACEPOINT("TEST LINE : %d",
                     fps[keywnode[nodeSelected].fpsindex].md->sourceline);

    TUI_printfw("    FPS call              : %s -> %s [",
                fps[keywnode[nodeSelected].fpsindex].md->callprogname,
                fps[keywnode[nodeSelected].fpsindex].md->callfuncname);

    for(int i = 0; i < fps[keywnode[nodeSelected].fpsindex].md->NBnameindex;
            i++)
    {
        TUI_printfw(" %s",
                    fps[keywnode[nodeSelected].fpsindex].md->nameindexW[i]);
    }
    TUI_printfw(" ]");
    TUI_newline();

    TUI_printfw("    FPS source            : %s %d",
                fps[keywnode[nodeSelected].fpsindex].md->sourcefname,
                fps[keywnode[nodeSelected].fpsindex].md->sourceline);
    TUI_newline();

    TUI_printfw("   %d libs : ",
                fps[keywnode[nodeSelected].fpsindex].md->NBmodule);
    for(int m = 0; m < fps[keywnode[nodeSelected].fpsindex].md->NBmodule; m++)
    {
        TUI_printfw(" [%s]",
                    fps[keywnode[nodeSelected].fpsindex].md->modulename[m]);
    }
    TUI_newline();

    DEBUG_TRACEPOINT(" ");

    TUI_printfw("    KEYWORDARRAY: %s",
                fps[keywnode[nodeSelected].fpsindex].md->keywordarray);
    TUI_newline();


    TUI_printfw("    FPS work     directory    : %s",
                fps[keywnode[nodeSelected].fpsindex].md->workdir);
    TUI_newline();

    TUI_printfw(
        "    ( FPS output data directory : %s )  ( FPS input conf directory : "
        "%s)",
        fps[keywnode[nodeSelected].fpsindex].md->datadir,
        fps[keywnode[nodeSelected].fpsindex].md->confdir);
    TUI_newline();

    DEBUG_TRACEPOINT(" ");
    TUI_printfw("    FPS tmux sessions     :  ");

    EXECUTE_SYSTEM_COMMAND("tmux has-session -t %s:ctrl 2> /dev/null",
                           fps[keywnode[nodeSelected].fpsindex].md->name);
    if(data.retvalue == 0)
    {
        fps[keywnode[nodeSelected].fpsindex].md->status |=
            FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL;
    }
    else
    {
        fps[keywnode[nodeSelected].fpsindex].md->status &=
            ~FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL;
    }

    EXECUTE_SYSTEM_COMMAND("tmux has-session -t %s:conf 2> /dev/null",
                           fps[keywnode[nodeSelected].fpsindex].md->name);
    if(data.retvalue == 0)
    {
        fps[keywnode[nodeSelected].fpsindex].md->status |=
            FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF;
    }
    else
    {
        fps[keywnode[nodeSelected].fpsindex].md->status &=
            ~FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF;
    }

    EXECUTE_SYSTEM_COMMAND("tmux has-session -t %s:run 2> /dev/null",
                           fps[keywnode[nodeSelected].fpsindex].md->name);
    if(data.retvalue == 0)
    {
        fps[keywnode[nodeSelected].fpsindex].md->status |=
            FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN;
    }
    else
    {
        fps[keywnode[nodeSelected].fpsindex].md->status &=
            ~FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN;
    }

    if(fps[keywnode[nodeSelected].fpsindex].md->status &
            FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("%s:ctrl", fps[keywnode[nodeSelected].fpsindex].md->name);
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(COLOR_ERROR);
        TUI_printfw("%s:ctrl", fps[keywnode[nodeSelected].fpsindex].md->name);
        screenprint_unsetcolor(COLOR_ERROR);
    }
    TUI_printfw(" ");
    if(fps[keywnode[nodeSelected].fpsindex].md->status &
            FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("%s:conf", fps[keywnode[nodeSelected].fpsindex].md->name);
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(COLOR_ERROR);
        TUI_printfw("%s:conf", fps[keywnode[nodeSelected].fpsindex].md->name);
        screenprint_unsetcolor(COLOR_ERROR);
    }
    TUI_printfw(" ");
    if(fps[keywnode[nodeSelected].fpsindex].md->status &
            FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN)
    {
        screenprint_setcolor(COLOR_OK);
        TUI_printfw("%s:run", fps[keywnode[nodeSelected].fpsindex].md->name);
        screenprint_unsetcolor(COLOR_OK);
    }
    else
    {
        screenprint_setcolor(COLOR_ERROR);
        TUI_printfw("%s:run", fps[keywnode[nodeSelected].fpsindex].md->name);
        screenprint_unsetcolor(COLOR_ERROR);
    }
    TUI_newline();

    DEBUG_TRACEPOINT(" ");


    TUI_printfw("======== NODE info ( # %5ld)", nodeSelected);
    TUI_newline();
    TUI_printfw("%-30s ", keywnode[nodeSelected].keywordfull);

    if(keywnode[nodeSelected].leaf > 0)  // If this is not a directory
    {
        char typestring[100];
        functionparameter_GetTypeString(
            fps[fpsindexSelected].parray[pindexSelected].type,
            typestring);
        TUI_printfw("type %s", typestring);
        TUI_newline();

        // print binary flag
        TUI_printfw("FLAG : ");
        uint64_t mask = (uint64_t) 1 << (sizeof(uint64_t) * CHAR_BIT - 1);
        while(mask)
        {
            int digit =
                fps[fpsindexSelected].parray[pindexSelected].fpflag & mask ? 1
                : 0;
            if(digit == 1)
            {
                screenprint_setcolor(2);
                TUI_printfw("%d", digit);
                screenprint_unsetcolor(2);
            }
            else
            {
                TUI_printfw("%d", digit);
            }
            mask >>= 1;
        }
    }
    else
    {
        TUI_printfw("-DIRECTORY-");
        TUI_newline();
    }
    TUI_newline();
    TUI_newline();

    DEBUG_TRACE_FEXIT();
}

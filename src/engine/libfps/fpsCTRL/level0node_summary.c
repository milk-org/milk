/**
 * @file level0node_summary.c
 * @brief Level0node summary module
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "fps.h"
#include "fps_internal.h"
#include "fpsCTRL_TUIcompat.h"
#include "fpsCTRL_globals.h"

#include "level0node_summary.h"

void fpsCTRLscreen_level0node_summary(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex)
{
    pid_t pid;

    pid = fps[fpsindex].md->confpid;
    if((getpgid(pid) >= 0) && (pid > 0))
    {
        screenprint_setcolor(2);
        TUI_printfw("%07d ", (int) pid);
        screenprint_unsetcolor(2);
    }
    else     // PID not active
    {
        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
        {
            // not clean exit
            screenprint_setcolor(4);
            TUI_printfw("%07d ", (int) pid);
            screenprint_unsetcolor(4);
        }
        else
        {
            // All OK
            TUI_printfw("%07d ", (int) pid);
        }
    }


    if(fps[fpsindex].md->conferrcnt > 99)
    {
        screenprint_setcolor(4);
        TUI_printfw("[XX]");
        screenprint_unsetcolor(4);
    }
    else if(fps[fpsindex].md->conferrcnt > 0)
    {
        screenprint_setcolor(4);
        TUI_printfw("[%02d]", (int) fps[fpsindex].md->conferrcnt);
        screenprint_unsetcolor(4);
    }
    else if(fps[fpsindex].md->conferrcnt == 0)
    {
        screenprint_setcolor(2);
        TUI_printfw("[%02d]", (int) fps[fpsindex].md->conferrcnt);
        screenprint_unsetcolor(2);
    }

    pid = fps[fpsindex].md->runpid;
    if((getpgid(pid) >= 0) && (pid > 0))
    {
        screenprint_setcolor(2);
        TUI_printfw("%07d ", (int) pid);
        screenprint_unsetcolor(2);
    }
    else
    {
        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
        {
            // not clean exit
            screenprint_setcolor(4);
            TUI_printfw("%07d ", (int) pid);
            screenprint_unsetcolor(4);
        }
        else
        {
            // All OK
            TUI_printfw("%07d ", (int) pid);
        }
    }
}
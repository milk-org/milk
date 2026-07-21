// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file level0node_summary.c
 * @brief Level0node summary module
 */


#include "fps.h"
#include "fpsCTRL_TUIcompat.h"


/**
 * @brief Render a summary line for a top-level FPS node.
 *
 * Shows FPS name, process status, run/conf state,
 * and loop rate in a single dashboard row.
 */
void fpsCTRLscreen_level0node_summary(FPS *fps, int fps_idx)
{
    pid_t pid;

    pid = fps[fps_idx].md->confpid;
    if ((getpgid(pid) >= 0) && (pid > 0))
    {
        screenprint_setcolor(2);
        TUI_printfw("%07d ", (int) pid);
        screenprint_unsetcolor(2);
    }
    else // PID not active
    {
        if (fps[fps_idx].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
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


    if (fps[fps_idx].md->conferrcnt > 99)
    {
        screenprint_setcolor(4);
        TUI_printfw("[XX]");
        screenprint_unsetcolor(4);
    }
    else if (fps[fps_idx].md->conferrcnt > 0)
    {
        screenprint_setcolor(4);
        TUI_printfw("[%02d]", (int) fps[fps_idx].md->conferrcnt);
        screenprint_unsetcolor(4);
    }
    else if (fps[fps_idx].md->conferrcnt == 0)
    {
        screenprint_setcolor(2);
        TUI_printfw("[%02d]", (int) fps[fps_idx].md->conferrcnt);
        screenprint_unsetcolor(2);
    }

    pid = fps[fps_idx].md->runpid;
    if ((getpgid(pid) >= 0) && (pid > 0))
    {
        screenprint_setcolor(2);
        TUI_printfw("%07d ", (int) pid);
        screenprint_unsetcolor(2);
    }
    else
    {
        if (fps[fps_idx].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
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

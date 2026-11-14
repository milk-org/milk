// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file level0node_summary.c
 * @brief Level0node summary module
 */


#include "fps.h"
#include "fpsCTRL_TUIcompat.h"
#include "fpsCTRL_units.h"


/* The pid of @role's process (the last one if none runs), coloured by the unit state (U10) */
static void level0_role_pid(FPS *fps, FPS_UNIT_ROLE role)
{
    FPS_UNIT_STATUS st;
    fpsctrl_unit_status(fps, role, &st);
    int col = (st.state == FPS_UNIT_RUNNING)   ? 2
              : (st.state == FPS_UNIT_FAILED)  ? 4
              : fps_unit_state_alive(st.state) ? 3
                                               : 0;
    if (col != 0)
    {
        screenprint_setcolor(col);
    }
    TUI_printfw("%07d ", (int) st.pid);
    if (col != 0)
    {
        screenprint_unsetcolor(col);
    }
}

/**
 * @brief Render a summary line for a top-level FPS node: conf pid, conf errors, run pid, coloured by
 * the unit states (green running, yellow starting or stopping, red failed).
 */
void fpsCTRLscreen_level0node_summary(FPS *fps)
{
    level0_role_pid(fps, FPS_UNIT_CONF);

    if (fps->md->conferrcnt > 99)
    {
        screenprint_setcolor(4);
        TUI_printfw("[XX]");
        screenprint_unsetcolor(4);
    }
    else if (fps->md->conferrcnt > 0)
    {
        screenprint_setcolor(4);
        TUI_printfw("[%02d]", (int) fps->md->conferrcnt);
        screenprint_unsetcolor(4);
    }
    else if (fps->md->conferrcnt == 0)
    {
        screenprint_setcolor(2);
        TUI_printfw("[%02d]", (int) fps->md->conferrcnt);
        screenprint_unsetcolor(2);
    }

    level0_role_pid(fps, FPS_UNIT_RUN);
}

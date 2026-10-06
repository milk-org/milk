// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_modal.c
 * @brief Modal dialogs and loop actions (loop renaming, cycle filter, graph switch)
 */

#include "overview_input_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_input__handle_loop_rename - process keyboard events during loop renaming.
 * @key: Pressed key code.
 * @lay: Pointer to overview layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_loop_rename(int key, OV_LAYOUT *lay, OV_MODEL *m)
{
    if (!lay->renaming_loop)
    {
        return 0;
    }

    /* ESC or Ctrl+C — cancel loop rename */
    if (key == 27 || key == 3 || key == ctrl('c'))
    {
        lay->renaming_loop = 0;
        return 1;
    }

    /* ENTER — commit loop rename */
    if (key == '\n' || key == '\r' || key == OV_KEY_ENTER)
    {
        lay->renaming_loop = 0;
        if (lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
        {
            ov_loop_rename(m, lay->sel_loop, lay->rename_buf);
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Loop L%02d renamed to '%s'",
                           m->loops[lay->sel_loop].loop_id, m->loops[lay->sel_loop].name);
        }
        return 1;
    }

    /* Ctrl+U — clear buffer */
    if (key == ctrl('u') || key == 21)
    {
        lay->rename_cursor = 0;
        lay->rename_buf[0] = '\0';
        return 1;
    }

    /* Backspace — delete character */
    if (key == 127 || key == 8)
    {
        if (lay->rename_cursor > 0)
        {
            lay->rename_cursor--;
            lay->rename_buf[lay->rename_cursor] = '\0';
        }
        return 1;
    }

    /* Printable ASCII */
    if (key >= 32 && key < 127 && lay->rename_cursor < (int) sizeof(lay->rename_buf) - 2)
    {
        lay->rename_buf[lay->rename_cursor++] = (char) key;
        lay->rename_buf[lay->rename_cursor]   = '\0';
        return 1;
    }

    return 1;
}

/**
 * ov_input__handle_loop_actions - process loop view action keys (rename, filter, graph).
 * @key: Pressed key code.
 * @lay: Pointer to overview layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_loop_actions(int key, OV_LAYOUT *lay, OV_MODEL *m)
{
    int in_loops =
        (lay->view == OV_VIEW_LOOPS || (lay->view == OV_VIEW_DASHBOARD &&
                                        lay->graph_tab_mode == 1 && lay->focus == OV_FOCUS_GRAPH));
    if (!in_loops)
    {
        return 0;
    }

    /* 'r' or 'R' — rename selected loop */
    if (key == 'r' || key == 'R')
    {
        if (lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
        {
            lay->renaming_loop = 1;
            const OV_LOOP *lp  = &m->loops[lay->sel_loop];
            snprintf(lay->rename_buf, sizeof(lay->rename_buf), "%s",
                     lp->has_custom_name ? lp->custom_name : lp->name);
            lay->rename_cursor = (int) strlen(lay->rename_buf);
            return 1;
        }
    }

    /* 'f' — toggle loop isolation filter */
    if (key == 'f')
    {
        lay->loop_filter_active = !lay->loop_filter_active;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                       lay->loop_filter_active ? "Loop isolation filter: ON"
                                               : "Loop isolation filter: OFF");
        return 1;
    }

    /* 'g' — switch to graph CONNECTIONS tab to view circuit tree */
    if (key == 'g')
    {
        lay->graph_tab_mode = 0;
        if (lay->view == OV_VIEW_LOOPS)
        {
            lay->view = OV_VIEW_GRAPH;
        }
        return 1;
    }

    return 0;
}

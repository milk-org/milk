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
int ov_input__handle_loop_rename(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (!lay->rename.active)
    {
        return 0;
    }

    /* ESC or Ctrl+C — cancel loop rename */
    if (key == 27 || key == 3 || key == ctrl('c'))
    {
        lay->rename.active = 0;
        return 1;
    }

    /* ENTER — commit loop rename */
    if (key == '\n' || key == '\r' || key == OV_KEY_ENTER)
    {
        lay->rename.active = 0;
        if (lay->sub[OV_SUB_LOOPS].sel >= 0 && lay->sub[OV_SUB_LOOPS].sel < m->nb_loops)
        {
            ov_loop_rename(m, lay->sub[OV_SUB_LOOPS].sel, lay->rename.buf);
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Loop L%02d renamed to '%s'",
                           m->loops[lay->sub[OV_SUB_LOOPS].sel].loop_id,
                           ov_loop_name(&m->loops[lay->sub[OV_SUB_LOOPS].sel]));
        }
        return 1;
    }

    /* Ctrl+U — clear buffer */
    if (key == ctrl('u') || key == 21)
    {
        lay->rename.cursor = 0;
        lay->rename.buf[0] = '\0';
        return 1;
    }

    /* Backspace — delete character */
    if (key == 127 || key == 8)
    {
        if (lay->rename.cursor > 0)
        {
            lay->rename.cursor--;
            lay->rename.buf[lay->rename.cursor] = '\0';
        }
        return 1;
    }

    /* Printable ASCII */
    if (key >= 32 && key < 127 && lay->rename.cursor < (int) sizeof(lay->rename.buf) - 2)
    {
        lay->rename.buf[lay->rename.cursor++] = (char) key;
        lay->rename.buf[lay->rename.cursor]   = '\0';
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
int ov_input__handle_loop_actions(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    int in_loops = (lay->view == OV_VIEW_LOOPS ||
                    (lay->view == OV_VIEW_DASHBOARD && lay->subtab == OV_SUB_LOOPS &&
                     lay->focus == OV_FOCUS_GRAPH));
    if (!in_loops)
    {
        return 0;
    }

    /* 'r' or 'R' — rename selected loop */
    if (key == 'r' || key == 'R')
    {
        if (lay->sub[OV_SUB_LOOPS].sel >= 0 && lay->sub[OV_SUB_LOOPS].sel < m->nb_loops)
        {
            lay->rename.active = 1;
            const OV_LOOP *lp  = &m->loops[lay->sub[OV_SUB_LOOPS].sel];
            snprintf(lay->rename.buf, sizeof(lay->rename.buf), "%s", ov_loop_name(lp));
            lay->rename.cursor = (int) strlen(lay->rename.buf);
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
        lay->subtab = OV_SUB_CONN;
        if (lay->view == OV_VIEW_LOOPS)
        {
            ov_set_view(lay, OV_VIEW_GRAPH);
        }
        return 1;
    }

    return 0;
}

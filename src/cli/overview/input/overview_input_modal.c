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

/** loop_rename - r / R: rename the selected loop (the name is saved). */
static int loop_rename(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key;
    int li = lay->sub[OV_SUB_LOOPS].sel;
    if (li < 0 || li >= m->nb_loops)
    {
        return 0;
    }
    lay->rename.active = 1;
    snprintf(lay->rename.buf, sizeof(lay->rename.buf), "%s", ov_loop_name(&m->loops[li]));
    lay->rename.cursor = (int) strlen(lay->rename.buf);
    return 1;
}

/** loop_isolate - f: lists show the selected loop's members only. */
static int loop_isolate(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->loop_filter_active = !lay->loop_filter_active;
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                   lay->loop_filter_active ? "Loop isolation filter: ON"
                                           : "Loop isolation filter: OFF");
    return 1;
}

/** loop_graph - g: the CONNECTIONS tree (F7 opens F6). */
static int loop_graph(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->subtab = OV_SUB_CONN;
    if (lay->view == OV_VIEW_LOOPS)
    {
        ov_set_view(lay, OV_VIEW_GRAPH);
    }
    return 1;
}

/* On a loop list: F7, or the LOOPS sub-tab with focus (DASH and F6) */
static const OV_KEYBIND loop_keys[] = {
    { { 'r', 'R' }, OV_CTX_LOOP_LIST | OV_CTX_MODEL, 0, loop_rename },
    { { 'f' }, OV_CTX_LOOP_LIST, 0, loop_isolate },
    { { 'g' }, OV_CTX_LOOP_LIST, 0, loop_graph },
};

/** ov_input__handle_loop_actions - loop list keys: rename, isolate, graph. Return: 1 consumed. */
int ov_input__handle_loop_actions(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_key_dispatch(loop_keys, OV_KEYBIND_N(loop_keys), key, lay, m);
}

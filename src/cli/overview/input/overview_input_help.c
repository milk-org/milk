// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_help.c
 * @brief Modal keyboard navigation and interactive search for help overlay.
 */

#include "overview_input_internal.h"
#include "overview_help_data.h"

/**
 * ov_input_handle_help_key - handle input when interactive help overlay is open.
 * @key: Keycode or mouse event code.
 * @lay: Mutable layout state.
 *
 * Return: 1 if help overlay handled the key, 0 if help overlay is not active.
 */
int ov_input_handle_help_key(int key, OV_LAYOUT *lay)
{
    if (!lay->help.show)
    {
        return 0;
    }

    /* Mode 1: Full interactive Introduction to milk-CTRL */
    if (lay->help.mode == 1)
    {
        int ph     = (lay->term_rows > 6) ? (lay->term_rows - 3) : lay->term_rows;
        int body_h = ph - 5;
        if (body_h < 4)
        {
            body_h = 4;
        }
        int total_lines = g_intro_total;
        int max_scroll  = (total_lines > body_h) ? (total_lines - body_h) : 0;

        switch (key)
        {
        case '2':
        case '1':
        case 'i':
        case '\t':
            lay->help.mode = 0;
            break;

        case OV_KEY_UP:
        case OV_KEY_MOUSE_UP:
            if (lay->help.intro_scroll > 0)
            {
                lay->help.intro_scroll--;
            }
            break;

        case OV_KEY_DOWN:
        case OV_KEY_MOUSE_DOWN:
            if (lay->help.intro_scroll < max_scroll)
            {
                lay->help.intro_scroll++;
            }
            break;

        case OV_KEY_PGUP:
            lay->help.intro_scroll -= (body_h > 4) ? (body_h - 2) : 4;
            if (lay->help.intro_scroll < 0)
            {
                lay->help.intro_scroll = 0;
            }
            break;

        case OV_KEY_PGDN:
            lay->help.intro_scroll += (body_h > 4) ? (body_h - 2) : 4;
            if (lay->help.intro_scroll > max_scroll)
            {
                lay->help.intro_scroll = max_scroll;
            }
            break;

        case OV_KEY_HOME:
            lay->help.intro_scroll = 0;
            break;

        case OV_KEY_END:
            lay->help.intro_scroll = max_scroll;
            break;

        case OV_KEY_MOUSE_CLICK:
            ov_help_handle_click(lay, ov_mouse_row, ov_mouse_col);
            break;

        case 27: /* ESC — exit help overlay */
        case 3:  /* Ctrl+C */
        case 4:  /* Ctrl+D */
        case 'h':
        case 'q':
        case 'x':
            lay->help.show = 0;
            ov_buf_force_clear();
            break;

        default:
            break;
        }
        return 1;
    }

    /* Mode 0: Controls & Keybindings Reference */
    /* Active search input mode */
    if (lay->help.search_active)
    {
        if (key == 27 || key == 3 || key == ctrl('c')) /* ESC or Ctrl+C */
        {
            if (lay->help.search[0] != '\0')
            {
                lay->help.search[0]     = '\0';
                lay->help.search_cursor = 0;
                lay->help.sel           = 0;
            }
            lay->help.search_active = 0;
            return 1;
        }

        if (key == OV_KEY_ENTER || key == '\r' || key == '\n')
        {
            /* Transfer focus to navigating search results */
            lay->help.search_active = 0;
            return 1;
        }

        if (key == 8 || key == 127 || key == OV_KEY_DEL)
        {
            int len = (int) strlen(lay->help.search);
            if (len > 0)
            {
                lay->help.search[len - 1] = '\0';
                lay->help.search_cursor   = len - 1;
                lay->help.sel             = 0;
            }
            return 1;
        }

        if (key == 21 || key == ctrl('u'))
        {
            lay->help.search[0]     = '\0';
            lay->help.search_cursor = 0;
            lay->help.sel           = 0;
            return 1;
        }

        if (key == OV_KEY_UP || key == OV_KEY_MOUSE_UP)
        {
            if (lay->help.sel > 0)
            {
                lay->help.sel--;
            }
            return 1;
        }

        if (key == OV_KEY_DOWN || key == OV_KEY_MOUSE_DOWN)
        {
            int nvis = ov_help_visible_count(lay);
            if (lay->help.sel < nvis - 1)
            {
                lay->help.sel++;
            }
            return 1;
        }

        if (key == OV_KEY_MOUSE_CLICK)
        {
            ov_help_handle_click(lay, ov_mouse_row, ov_mouse_col);
            return 1;
        }

        if (key >= 32 && key <= 126)
        {
            int len = (int) strlen(lay->help.search);
            if (len + 1 < (int) sizeof(lay->help.search))
            {
                lay->help.search[len]     = (char) key;
                lay->help.search[len + 1] = '\0';
                lay->help.search_cursor   = len + 1;
                lay->help.sel             = 0;
            }
            return 1;
        }

        /* Ignore other keystrokes while typing search query */
        return 1;
    }

    /* Browsing mode (category tree or search results) */
    int nvis = ov_help_visible_count(lay);
    if (nvis < 1)
    {
        nvis = 1;
    }

    switch (key)
    {
    case '1':
    case 'i':
    case '\t':
        lay->help.mode         = 1;
        lay->help.intro_scroll = 0;
        break;

    case '/':
        lay->help.search_active = 1;
        lay->help.search_cursor = (int) strlen(lay->help.search);
        break;

    case OV_KEY_UP:
    case OV_KEY_MOUSE_UP:
        if (lay->help.sel > 0)
        {
            lay->help.sel--;
        }
        break;

    case OV_KEY_DOWN:
    case OV_KEY_MOUSE_DOWN:
        if (lay->help.sel < nvis - 1)
        {
            lay->help.sel++;
        }
        break;

    case OV_KEY_HOME:
        lay->help.sel = 0;
        break;

    case OV_KEY_END:
        lay->help.sel = nvis - 1;
        break;

    case OV_KEY_PGUP:
        lay->help.sel -= 8;
        if (lay->help.sel < 0)
        {
            lay->help.sel = 0;
        }
        break;

    case OV_KEY_PGDN:
        lay->help.sel += 8;
        if (lay->help.sel >= nvis)
        {
            lay->help.sel = nvis - 1;
        }
        break;

    case OV_KEY_RIGHT:
        ov_help_expand_at(lay, lay->help.sel, 1);
        break;

    case OV_KEY_LEFT:
        ov_help_expand_at(lay, lay->help.sel, 0);
        break;

    case OV_KEY_ENTER:
    case '\r':
    {
        if (lay->help.search[0] == '\0' && lay->help.sel == 0)
        {
            /* Landed on Introduction section: toggle into full intro guide */
            lay->help.mode         = 1;
            lay->help.intro_scroll = 0;
            break;
        }
        ov_help_toggle_at(lay, lay->help.sel);
        /* Clamp cursor after expand change */
        int new_nvis = ov_help_visible_count(lay);
        if (lay->help.sel >= new_nvis)
        {
            lay->help.sel = new_nvis - 1;
        }
        break;
    }

    case OV_KEY_MOUSE_CLICK:
        ov_help_handle_click(lay, ov_mouse_row, ov_mouse_col);
        break;

    case 27: /* ESC — clear search query if present, else exit help overlay */
    case 3:  /* Ctrl+C */
    case 4:  /* Ctrl+D */
    case 'h':
    case 'q':
    case 'x':
        if (lay->help.search[0] != '\0')
        {
            lay->help.search[0]     = '\0';
            lay->help.search_cursor = 0;
            lay->help.search_active = 0;
            lay->help.sel           = 0;
        }
        else
        {
            lay->help.show = 0;
            ov_buf_force_clear();
        }
        break;

    default:
        /* All other keystrokes are disabled in help mode:
         * ignore silently so they never affect underlying panels or control */
        break;
    }
    return 1;
}

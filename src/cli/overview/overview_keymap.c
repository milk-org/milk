// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_keymap.c
 * @brief   Key table of milk-CTRL and the text generated from it.
 */

#include <stdio.h>

#include "overview_keymap.h"

/* clang-format off */
const OV_KEYDEF ov_keys[] = {
#include "overview_keymap_nav.inc"
#include "overview_keymap_panels.inc"
};
/* clang-format on */

const int ov_nb_keys = (int) (sizeof(ov_keys) / sizeof(ov_keys[0]));

const char *const ov_keygroup_title[OV_KG_COUNT] = {
    [OV_KG_VIEWS]   = "Views (F2-F7; Ctrl+Left/Right or Ctrl+wheel cycle them)",
    [OV_KG_NAV]     = "Navigation and display",
    [OV_KG_COLUMNS] = "Columns and sorting",
    [OV_KG_LOOPS]   = "Feedback loops (F7, or LOOPS sub-tab)",
    [OV_KG_CTRL]    = "Control mode actions ('c' toggles; y/N asks before destructive ones)",
};

/** ov_key_footer - the status-line hints ("TAB / +/- ...") into @buf. Return: length written. */
size_t ov_key_footer(char *buf, size_t n)
{
    size_t len = 0;
    if (n > 0)
    {
        buf[0] = '\0';
    }
    for (int i = 0; i < ov_nb_keys && len < n; i++)
    {
        if (ov_keys[i].footer != NULL)
        {
            int w = snprintf(buf + len, n - len, "%s%s", (len > 0) ? " " : "", ov_keys[i].footer);
            len += (w > 0) ? (size_t) w : 0;
        }
    }
    return (len < n) ? len : n - 1;
}

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

/**
 * ov_key_hints - status-line hints of the keys in context @ctx (OV_CTX_*): those tied to the context first,
 * then the general ones, in table order. Return: count written to @out.
 */
int ov_key_hints(unsigned ctx, const OV_KEYHINT **out, int max)
{
    int n = 0;
    for (int pass = 0; pass < 2; pass++)
    {
        for (int i = 0; i < ov_nb_keys && n < max; i++)
        {
            const OV_KEYHINT *h = &ov_keys[i].hint;
            if (h->key != NULL && (h->need != 0) == (pass == 0) && (ctx & h->need) == h->need &&
                (ctx & h->forbid) == 0)
            {
                out[n++] = h;
            }
        }
    }
    return n;
}

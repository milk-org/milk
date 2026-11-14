// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_keys.h
 * @brief Key bindings: rows {keys, context needed, context forbidden, handler}, first match wins.
 *
 * Purpose:     each key group says, in a table next to its (static) handlers, which keys it takes
 *              in which state; the documentation of the keys stays in overview_keymap.h.
 * Does:        OV_KEYBIND is one row: a 0-terminated list of key codes (OV_KEY_ANY: every key),
 *              the OV_CTX_* bits it needs and those it forbids, and its handler. ov_key_ctx()
 *              computes the OV_CTX_* bits of the current state; ov_key_dispatch() runs the first
 *              row that matches. A handler returning 0 lets the next matching row, then the next
 *              group, try; a global key returns OV_KEY_QUIT to quit milk-CTRL. The groups run in
 *              the order of ov_handle_key_internal() (input/overview_input.c).
 * Used by:     the key groups of input/ (overview_input_actions.c, overview_input_modal.c,
 *              overview_input_nav.c, overview_input_sort_keys.c, overview_input_toggles.c),
 *              through input/overview_input_internal.h; render/overview_render_status.c
 *              (ov_key_ctx, for the status-line hints).
 * Uses:        overview_keymap.h (the OV_CTX_* bits).
 * See also:    overview_input_keys.c; src/cli/overview/README.md, "Extending milk-CTRL".
 */

#ifndef OVERVIEW_INPUT_KEYS_H
#define OVERVIEW_INPUT_KEYS_H

#include "overview_keymap.h" /* OV_CTX_* */

#define OV_KEY_ANY (-1) /* in keys[]: every key */

#define OV_KEY_QUIT 2 /* handler return: quit milk-CTRL */

typedef struct
{
    int      keys[8];                                      /* key codes, 0-terminated */
    unsigned need;                                         /* OV_CTX_* all required */
    unsigned forbid;                                       /* OV_CTX_* none allowed */
    int (*fn)(int key, OV_LAYOUT *lay, const OV_MODEL *m); /* 0: not consumed */
} OV_KEYBIND;

#define OV_KEYBIND_N(t) ((int) (sizeof(t) / sizeof((t)[0])))

unsigned ov_key_ctx(const OV_LAYOUT *lay, const OV_MODEL *m);
int      ov_key_dispatch(const OV_KEYBIND *t, int n, int key, OV_LAYOUT *lay, const OV_MODEL *m);

#endif /* OVERVIEW_INPUT_KEYS_H */

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_keys.h
 * @brief Key bindings: rows {keys, context needed, context forbidden, handler}, first match wins
 *
 * Each key group keeps its table next to its (static) handlers; the groups run in the order of
 * ov_handle_key(). A handler returning 0 lets the next matching row, then the next group, try.
 * Documentation of the keys: overview_keymap.h.
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

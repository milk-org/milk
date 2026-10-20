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

/* Context bits (ov_key_ctx) */
#define OV_CTX_MODEL (1U << 0)         /* a model is loaded */
#define OV_CTX_CTRL (1U << 1)          /* CONTROL mode */
#define OV_CTX_FROZEN (1U << 2)        /* a selection is frozen */
#define OV_CTX_DASH (1U << 3)          /* view F2 */
#define OV_CTX_F5 (1U << 4)            /* view F5 */
#define OV_CTX_LOOP_LIST (1U << 5)     /* F7, or the LOOPS sub-tab focused */
#define OV_CTX_FOCUS_STREAMS (1U << 6) /* focus on a panel */
#define OV_CTX_FOCUS_PROCS (1U << 7)
#define OV_CTX_FOCUS_FPS (1U << 8)
#define OV_CTX_FOCUS_GRAPH (1U << 9)
#define OV_CTX_PARAM_TREE (1U << 10) /* F5 parameter tree focused */
#define OV_CTX_MSEL (1U << 11)       /* FPS batch marks set */

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

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_input.c
 * @brief Terminal input decoder: bytes plus arrival time in, typed key and mouse events out.
 *
 * Pure (no I/O). A sequence split across reads is decoded once complete; one that stays
 * incomplete for TV_ESC_TIMEOUT_US is resolved (lone ESC) or dropped, never replayed as keys.
 */

#include "termview_internal.h"

#define TV_CSI_MAX 64  /* longer control sequences are skipped */
#define TV_STR_MAX 128 /* longer OSC/DCS/APC/PM/SOS strings are skipped */

enum
{
    TV_SKIP_NONE = 0,
    TV_SKIP_CSI, /* until a final byte */
    TV_SKIP_STR  /* until BEL or ESC */
};

/**
 * tv_input_feed - Append newly read terminal bytes to the input decoder buffer.
 * @in:     Input decoder state structure.
 * @data:   Raw input byte buffer read from the terminal.
 * @n:      Number of bytes available in @data.
 * @now_us: Current monotonic timestamp in microseconds.
 *
 * Copies up to the remaining buffer capacity of @in from @data, marks a
 * lone buffered `ESC` byte as stale if its escape timeout has already
 * elapsed before a non-sequence byte arrived, and updates @in->last_rx_us.
 *
 * Return: Nothing.
 */
void tv_input_feed(tv_input_t *in, const unsigned char *data, size_t n, int64_t now_us)
{
    size_t room = sizeof(in->buf) - in->len;
    if (n > room)
    {
        n = room; /* callers read at most tv_input_room() bytes */
    }
    if (n == 0)
    {
        return;
    }
    /* a lone ESC already past its timeout is a key on its own, however late we read on */
    if (in->len == 1 && in->buf[0] == 0x1B && now_us - in->last_rx_us >= TV_ESC_TIMEOUT_US &&
        data[0] != '[' && data[0] != 'O')
    {
        in->esc_stale = 1;
    }
    memcpy(in->buf + in->len, data, n);
    in->len += n;
    in->last_rx_us = now_us;
}

/**
 * tv_input_deadline - Compute the timeout timestamp for pending incomplete escape sequences.
 * @in: Input decoder state structure.
 *
 * Returns the timestamp at which a partial escape sequence or active skip
 * state expires (`last_rx_us + TV_ESC_TIMEOUT_US`), or -1 when no bytes or
 * skip state are pending.
 *
 * Return: Expiration timestamp in microseconds, or -1 if idle.
 */
int64_t tv_input_deadline(const tv_input_t *in)
{
    return (in->len > 0 || in->skip != TV_SKIP_NONE) ? in->last_rx_us + TV_ESC_TIMEOUT_US : -1;
}

/**
 * tv_ev_key - Populate a decoded key event structure.
 * @ev:   Output event structure to initialize.
 * @key:  Decoded key code (`TV_K_*` or ASCII code).
 * @mods: Active modifier bitmask (`TV_MOD_*`).
 *
 * Clears @ev and sets its type to `TV_EV_KEY` with the given @key and @mods.
 *
 * Return: Nothing.
 */
static void tv_ev_key(tv_event_t *ev, uint32_t key, uint8_t mods)
{
    memset(ev, 0, sizeof(*ev));
    ev->type = TV_EV_KEY;
    ev->key  = key;
    ev->mods = mods;
}

/** tv_mods_from_param() - xterm modifier parameter (1 + bits) to TV_MOD_* bits. */
static uint8_t tv_mods_from_param(int p)
{
    return (p > 1) ? (uint8_t) ((p - 1) & (TV_MOD_SHIFT | TV_MOD_ALT | TV_MOD_CTRL)) : 0;
}

/** tv_parse_params() - Up to 4 numeric parameters from "n;n;..."; returns how many. */
static int tv_parse_params(const unsigned char *p, size_t n, int out[4])
{
    int np = 0;
    out[0] = out[1] = out[2] = out[3] = 0;
    for (size_t k = 0; k < n && np < 4; k++)
    {
        if (p[k] == ';')
        {
            np++;
        }
        else if (p[k] >= '0' && p[k] <= '9' && out[np] < 10000)
        {
            out[np] = out[np] * 10 + (p[k] - '0');
        }
    }
    return (n > 0) ? ((np < 4) ? np + 1 : 4) : 0;
}

/**
 * tv_tilde_key - Map a CSI `~` numeric parameter to a special key code.
 * @p: Numeric parameter preceding `~` in `ESC [ <p> ~`.
 *
 * Translates VT220/xterm tilde-terminated function and navigation key
 * numbers (Home, Insert, Delete, End, PageUp, PageDown, F1-F12) into
 * `TV_K_*` constants.
 *
 * Return: Corresponding `TV_K_*` key code, or 0 if unrecognized.
 */
static uint32_t tv_tilde_key(int p)
{
    switch (p)
    {
    case 1:
    case 7:
        return TV_K_HOME;
    case 2:
        return TV_K_INS;
    case 3:
        return TV_K_DEL;
    case 4:
    case 8:
        return TV_K_END;
    case 5:
        return TV_K_PGUP;
    case 6:
        return TV_K_PGDN;
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
        return TV_K_F1 + (uint32_t) (p - 11);
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
        return TV_K_F1 + 5 + (uint32_t) (p - 17);
    case 23:
    case 24:
        return TV_K_F1 + 10 + (uint32_t) (p - 23);
    default:
        return 0;
    }
}

/**
 * tv_letter_key - Map a CSI or SS3 final letter to a navigation or function key code.
 * @c: Final character byte of the escape sequence.
 *
 * Translates cursor arrow letters (`A`-`D`), `H`/`F` (Home/End), and
 * `P`-`S` (F1-F4) into `TV_K_*` constants.
 *
 * Return: Corresponding `TV_K_*` key code, or 0 if unrecognized.
 */
static uint32_t tv_letter_key(int c)
{
    switch (c)
    {
    case 'A':
        return TV_K_UP;
    case 'B':
        return TV_K_DOWN;
    case 'C':
        return TV_K_RIGHT;
    case 'D':
        return TV_K_LEFT;
    case 'H':
        return TV_K_HOME;
    case 'F':
        return TV_K_END;
    case 'P':
    case 'Q':
    case 'R':
    case 'S':
        return TV_K_F1 + (uint32_t) (c - 'P');
    default:
        return 0;
    }
}

/** tv_ss3_key() - ESC O <c>: application-mode cursor keys, F1-F4 and the keypad. */
static uint32_t tv_ss3_key(int c)
{
    if (c == 'M')
    {
        return '\r';
    }
    if (c >= 'p' && c <= 'y')
    {
        return (uint32_t) ('0' + (c - 'p'));
    }
    switch (c)
    {
    case 'j':
        return '*';
    case 'k':
        return '+';
    case 'l':
        return ',';
    case 'm':
        return '-';
    case 'n':
        return '.';
    case 'o':
        return '/';
    default:
        return tv_letter_key(c);
    }
}

/** tv_sgr_mouse() - "<b;x;y" with final M/m; returns 1 with an event, 0 to drop. */
static int tv_sgr_mouse(const unsigned char *p, size_t n, int fin, tv_event_t *ev)
{
    int v[3] = { 0, 0, 0 };
    int k    = 0;
    for (size_t i = 1; i < n; i++) /* p[0] is '<' */
    {
        if (p[i] == ';')
        {
            if (++k > 2)
            {
                return 0;
            }
        }
        else if (p[i] >= '0' && p[i] <= '9')
        {
            v[k] = (v[k] > 99999) ? v[k] : v[k] * 10 + (p[i] - '0');
        }
        else
        {
            return 0;
        }
    }
    if (k != 2)
    {
        return 0;
    }

    int b = v[0];
    if (b & 128)
    {
        return 0; /* buttons 8-11 */
    }
    if ((b & 64) && fin == 'm')
    {
        return 0; /* wheels have no release */
    }
    memset(ev, 0, sizeof(*ev));
    ev->type = TV_EV_MOUSE;
    ev->mods = (uint8_t) (((b & 4) ? TV_MOD_SHIFT : 0) | ((b & 8) ? TV_MOD_ALT : 0) |
                          ((b & 16) ? TV_MOD_CTRL : 0));
    ev->x    = (int16_t) ((v[1] < 1) ? 1 : (v[1] > 9999) ? 9999 : v[1]);
    ev->y    = (int16_t) ((v[2] < 1) ? 1 : (v[2] > 9999) ? 9999 : v[2]);
    if (b & 64)
    {
        ev->button = (uint8_t) (TV_MB_WHEEL_UP + (b & 3));
        ev->action = TV_MA_PRESS;
        return 1;
    }
    ev->button = (uint8_t) (b & 3);
    if (fin == 'm')
    {
        ev->action = TV_MA_RELEASE;
    }
    else if (b & 32)
    {
        ev->action = (ev->button == TV_MB_NONE) ? TV_MA_MOVE : TV_MA_DRAG;
    }
    else
    {
        ev->action = TV_MA_PRESS;
    }
    return 1;
}

/** tv_skip() - Continue dropping an over-long sequence; 0 when the skip ended at p[0]. */
static size_t tv_skip(tv_input_t *in, const unsigned char *p, size_t n)
{
    size_t i = 0;
    if (in->skip == TV_SKIP_CSI)
    {
        while (i < n && p[i] >= 0x20 && p[i] <= 0x3F)
        {
            i++;
        }
        if (i == n)
        {
            return n;
        }
        in->skip = TV_SKIP_NONE;
        return (p[i] >= 0x40 && p[i] <= 0x7E) ? i + 1 : i;
    }
    while (i < n && p[i] != 0x07 && p[i] != 0x1B)
    {
        i++;
    }
    if (i == n)
    {
        return n;
    }
    in->skip = TV_SKIP_NONE;
    return (p[i] == 0x07) ? i + 1 : i; /* an ESC ends it and is decoded next */
}

/** tv_decode_string() - ESC ] / P / _ / ^ / X ... BEL or ESC \: dropped whole. */
static size_t tv_decode_string(tv_input_t *in, const unsigned char *p, size_t n, int expired)
{
    for (size_t i = 2; i < n; i++)
    {
        if (p[i] == 0x07)
        {
            return i + 1;
        }
        if (p[i] == 0x1B)
        {
            if (i + 1 >= n)
            {
                return expired ? n : 0;
            }
            return (p[i + 1] == '\\') ? i + 2 : i; /* unterminated: drop, re-decode the ESC */
        }
    }
    if (n - 2 > TV_STR_MAX)
    {
        in->skip = TV_SKIP_STR;
        return n;
    }
    return expired ? n : 0;
}

/** tv_decode_ss3() - ESC O [params] final: cursor/F1-F4/keypad, with optional modifier. */
static size_t tv_decode_ss3(const unsigned char *p, size_t n, int expired, tv_event_t *ev)
{
    size_t i = 2;
    while (i < n && p[i] >= 0x30 && p[i] <= 0x3F && i - 2 < 8)
    {
        i++;
    }
    if (i >= n)
    {
        return expired ? n : 0;
    }
    if (p[i] < 0x40 || p[i] > 0x7E)
    {
        return i; /* not SS3 after all: drop the prefix, re-decode the stray byte */
    }
    int      par[4];
    int      np   = tv_parse_params(p + 2, i - 2, par);
    uint32_t k    = tv_ss3_key(p[i]);
    uint8_t  mods = (np > 0) ? tv_mods_from_param(par[np - 1]) : 0;
    if (k != 0)
    {
        tv_ev_key(ev, k, mods);
    }
    return i + 1;
}

/** tv_decode_csi() - ESC [ params intermediates final. */
static size_t tv_decode_csi(tv_input_t          *in,
                            const unsigned char *p,
                            size_t               n,
                            int                  expired,
                            tv_event_t          *ev)
{
    /* parameter bytes 0x30-0x3F, intermediates 0x20-0x2F, final 0x40-0x7E */
    size_t i      = 2;
    int    digits = 1; /* parameters so far are all digits (rxvt "ESC [ n $" keys) */
    int    fin    = -1;
    while (i < n && p[i] >= 0x20 && p[i] <= 0x3F)
    {
        if (p[i] == '$' && digits && i > 2)
        {
            fin = '$';
            break;
        }
        digits &= (p[i] >= '0' && p[i] <= '9');
        i++;
    }
    size_t plen = i - 2;
    if (fin < 0 && i >= n)
    {
        if (plen > TV_CSI_MAX)
        {
            in->skip = TV_SKIP_CSI;
            return n;
        }
        return expired ? n : 0;
    }
    if (fin < 0)
    {
        fin = p[i];
        if (fin < 0x40 || fin > 0x7E)
        {
            return i; /* broken sequence: drop what was read, re-decode the stray byte */
        }
    }
    const unsigned char *par  = p + 2;
    size_t               used = i + 1;
    if (plen > TV_CSI_MAX || fin == '$')
    {
        return used; /* over-long (same outcome as when split) or rxvt shifted key: unbound */
    }

    if (fin == 'M' && plen == 0)
    {
        /* X10 mouse report: three raw bytes follow; consumed, not used */
        if (n < used + 3)
        {
            return expired ? n : 0;
        }
        return used + 3;
    }
    if (fin == '[' && plen == 0)
    {
        /* Linux console F1-F5: ESC [ [ A..E */
        if (n < used + 1)
        {
            return expired ? n : 0;
        }
        if (p[used] < 0x20 || p[used] > 0x7E)
        {
            return used; /* drop the prefix, re-decode the stray byte */
        }
        if (p[used] >= 'A' && p[used] <= 'E')
        {
            tv_ev_key(ev, TV_K_F1 + (uint32_t) (p[used] - 'A'), 0);
        }
        return used + 1;
    }
    if (plen > 0 && par[0] == '<')
    {
        if ((fin != 'M' && fin != 'm') || !tv_sgr_mouse(par, plen, fin, ev))
        {
            ev->type = 0; /* dropped report */
        }
        return used;
    }
    if (plen > 0 && (par[0] < '0' || par[0] > ';'))
    {
        return used; /* private-prefixed replies (?, >, =): not keys */
    }

    int      params[4];
    int      np   = tv_parse_params(par, plen, params);
    uint32_t key  = 0;
    uint8_t  mods = tv_mods_from_param(params[1]);
    if (fin == 'R' && np == 2 && (params[0] >= 2 || params[1] > 16))
    {
        /* a cursor position report (row;col): the viewer asks on row 2 or below and in a column
         * past 16, while modified F3 is CSI 1;<mod> R with mod 2-16 (or CSI <mod> R) */
        ev->type = TV_EV_CPR;
        ev->y    = (int16_t) ((params[0] < INT16_MAX) ? params[0] : INT16_MAX);
        ev->x    = (int16_t) ((params[1] < INT16_MAX) ? params[1] : INT16_MAX);
        return used;
    }
    if (fin == '~')
    {
        key = tv_tilde_key(params[0]);
    }
    else if (fin == 'Z')
    {
        key  = '\t';
        mods = TV_MOD_SHIFT;
    }
    else
    {
        key = tv_letter_key(fin);
        if (np == 1 && fin >= 'P' && fin <= 'S')
        {
            mods =
                tv_mods_from_param(params[0]); /* F1-F4 as CSI <mod> P-S (modifyFunctionKeys:1) */
        }
    }
    if (key != 0)
    {
        tv_ev_key(ev, key, mods);
    }
    return used;
}

/**
 * tv_decode() - Decode one item at @p[0..n).
 * Returns bytes consumed (with ev->type 0 when the item carries no event), or 0 when more bytes
 * are needed. With @expired set, an incomplete item is resolved instead of waiting.
 */
static size_t tv_decode(tv_input_t          *in,
                        const unsigned char *p,
                        size_t               n,
                        int                  expired,
                        tv_event_t          *ev)
{
    memset(ev, 0, sizeof(*ev));

    if (in->skip != TV_SKIP_NONE)
    {
        size_t used = tv_skip(in, p, n);
        if (used > 0)
        {
            return used;
        }
    }

    unsigned char c = p[0];
    if (c >= 0x80)
    {
        /* UTF-8 text is not bound to anything: swallow whole characters */
        size_t need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC2) ? 2 : 1;
        if (c > 0xF4 || need == 1)
        {
            return 1;
        }
        for (size_t i = 1; i < need; i++)
        {
            if (i >= n)
            {
                return expired ? n : 0;
            }
            if ((p[i] & 0xC0) != 0x80)
            {
                return i;
            }
        }
        return need;
    }
    if (c != 0x1B)
    {
        tv_ev_key(ev, c, 0);
        return 1;
    }

    /* ESC ... */
    if (n < 2)
    {
        if (expired)
        {
            tv_ev_key(ev, TV_K_ESC, 0);
            return 1;
        }
        return 0;
    }
    switch (p[1])
    {
    case 0x1B:
        tv_ev_key(ev, TV_K_ESC, 0);
        return 1;
    case 'O':
        return tv_decode_ss3(p, n, expired, ev);
    case '[':
        return tv_decode_csi(in, p, n, expired, ev);
    case ']':
    case 'P':
    case '_':
    case '^':
    case 'X':
        return tv_decode_string(in, p, n, expired);
    default:
        tv_ev_key(ev, p[1], TV_MOD_ALT);
        return 2;
    }
}

/**
 * tv_input_next - Pop and decode the next terminal input event from the buffer.
 * @in:     Input decoder state structure.
 * @now_us: Current monotonic timestamp in microseconds.
 * @ev:     Output structure populated when an event is decoded.
 *
 * Consumes buffered terminal bytes in @in, resolving or dropping incomplete
 * escape sequences if `TV_ESC_TIMEOUT_US` has elapsed since `in->last_rx_us`,
 * and populates @ev with the next key, mouse, or CPR event.
 *
 * Return: 1 if an event was decoded into @ev, or 0 if more bytes are needed.
 */
int tv_input_next(tv_input_t *in, int64_t now_us, tv_event_t *ev)
{
    int expired = (now_us - in->last_rx_us >= TV_ESC_TIMEOUT_US);
    if (expired && in->skip != TV_SKIP_NONE && in->len == 0)
    {
        in->skip = TV_SKIP_NONE; /* the unterminated sequence ended at the timeout */
    }
    if (in->esc_stale)
    {
        in->esc_stale = 0;
        memmove(in->buf, in->buf + 1, in->len - 1);
        in->len--;
        tv_ev_key(ev, TV_K_ESC, 0);
        return 1;
    }
    while (in->len > 0)
    {
        size_t used = tv_decode(in, in->buf, in->len, expired, ev);
        if (used == 0)
        {
            return 0; /* wait for the rest of the sequence */
        }
        memmove(in->buf, in->buf + used, in->len - used);
        in->len -= used;
        if (ev->type != 0)
        {
            return 1;
        }
    }
    return 0;
}

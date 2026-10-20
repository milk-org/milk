// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"

#include <time.h>

int                    ov_mouse_row = 0;
int                    ov_mouse_col = 0;
int                    ov_mouse_btn = 0;
int                    ov_hover_row = 0;
int                    ov_hover_col = 0;
static unsigned char   kbuf[256];
static int             kbuf_len  = 0;
static int             skip_tail = 0; /* drop the rest of a sequence whose head was discarded */
static struct timespec skip_ts;       /* when skip_tail was set; it expires after 100 ms */

/** ov_key_skip_start - start skipping the tail of a dropped sequence. */
static void ov_key_skip_start(void)
{
    skip_tail = 1;
    clock_gettime(CLOCK_MONOTONIC, &skip_ts);
}

/** ov_key_skip_expired - 1 once the dropped sequence's tail can no longer be on its way. */
static int ov_key_skip_expired(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - skip_ts.tv_sec) * 1000 + (now.tv_nsec - skip_ts.tv_nsec) / 1000000 > 100;
}

/**
 * ov_key_fill - poll stdin for up to @timeout_ms and append what arrives to the key buffer.
 * Return: bytes read, 0 if none, -1 on EOF or hangup.
 */
static int ov_key_fill(int timeout_ms)
{
    if (kbuf_len >= (int) sizeof(kbuf))
    {
        return 0;
    }
    struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
    int           pr  = poll(&pfd, 1, timeout_ms);
    int           hup = (pr > 0) && (pfd.revents & (POLLHUP | POLLERR | POLLNVAL));
    if (pr <= 0 || !(pfd.revents & POLLIN))
    {
        return hup ? -1 : 0;
    }
    ssize_t n = read(STDIN_FILENO, kbuf + kbuf_len, sizeof(kbuf) - (size_t) kbuf_len);
    if (n > 0)
    {
        kbuf_len += (int) n;
        return (int) n;
    }
    if (n == 0)
    {
        return hup ? -1 : 0;
    }
    return (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) ? 0 : -1;
}

/** ov_key_consume - drop the first @n bytes of the key buffer. */
static void ov_key_consume(int n)
{
    memmove(kbuf, kbuf + n, (size_t) (kbuf_len - n));
    kbuf_len -= n;
}

/**
 * ov_key_csi_end - index of the final byte of the CSI token at kbuf[0], -1 if incomplete,
 * or -(i + 2) when a byte that cannot belong to a CSI sits at index i.
 */
static int ov_key_csi_end(void)
{
    if (kbuf_len >= 3 && kbuf[2] == '[') /* Linux console F1-F5: ESC [ [ A..E */
    {
        return (kbuf_len >= 4) ? 3 : -1;
    }
    for (int i = 2; i < kbuf_len; i++)
    {
        unsigned char c = kbuf[i];
        if (c >= 0x40 && c <= 0x7E)
        {
            return i;
        }
        if (c < 0x20 || c > 0x3F)
        {
            return -(i + 2);
        }
    }
    return -1;
}

/** ov_key_mouse - decode SGR mouse parameters "<b;c;r" with final M (press) or m (release). */
static int ov_key_mouse(const unsigned char *par, int plen, int press)
{
    int  mb = 0, mc = 0, mr = 0;
    char tmp[64];
    if (plen > 1 && plen < (int) sizeof(tmp))
    {
        memcpy(tmp, par + 1, (size_t) (plen - 1));
        tmp[plen - 1] = '\0';
        sscanf(tmp, "%d;%d;%d", &mb, &mc, &mr);
    }
    ov_mouse_btn = mb;
    ov_mouse_col = mc;
    ov_mouse_row = mr;

    if (mb == 64)
    {
        return OV_KEY_MOUSE_UP;
    }
    if (mb == 65)
    {
        return OV_KEY_MOUSE_DOWN;
    }
    if (mb == 80) /* Ctrl+scroll (#12) */
    {
        return OV_KEY_CTRL_SCROLL_UP;
    }
    if (mb == 81)
    {
        return OV_KEY_CTRL_SCROLL_DOWN;
    }
    if (mb == 35 && press) /* passive move */
    {
        ov_hover_col = mc;
        ov_hover_row = mr;
        return OV_KEY_MOUSE_MOVE;
    }
    if ((mb & 32) && press)
    {
        return OV_KEY_MOUSE_DRAG;
    }
    if (mb == 0 && press)
    {
        return OV_KEY_MOUSE_CLICK;
    }
    return press ? OV_KEY_NONE : OV_KEY_MOUSE_RELEASE;
}

/** ov_key_tilde - key for ESC [ <code> ~ (xterm, rxvt/PuTTY F1-F4 as 11-14). */
static int ov_key_tilde(int code)
{
    switch (code)
    {
    case 1:
        return OV_KEY_HOME;
    case 3:
        return OV_KEY_DEL;
    case 4:
        return OV_KEY_END;
    case 5:
        return OV_KEY_PGUP;
    case 6:
        return OV_KEY_PGDN;
    case 11:
    case 12:
    case 13:
    case 14:
        return OV_KEY_F1 + (code - 11);
    case 15:
        return OV_KEY_F5;
    case 17:
    case 18:
    case 19:
        return OV_KEY_F6 + (code - 17);
    default:
        return OV_KEY_NONE;
    }
}

/**
 * ov_key_csi - decode one complete CSI token, or drop an incomplete/malformed one.
 * Every path consumes the whole token, so parameter bytes never come out as keys.
 */
static int ov_key_csi(void)
{
    int end = ov_key_csi_end();
    if (end == -1 && kbuf_len < (int) sizeof(kbuf))
    {
        ov_key_fill(20);
        end = ov_key_csi_end();
    }
    if (end == -1)
    {
        kbuf_len = 0; /* still incomplete: drop it and skip its tail when it arrives */
        ov_key_skip_start();
        return OV_KEY_NONE;
    }
    if (end < -1)
    {
        ov_key_consume(-end - 2); /* malformed: drop up to the stray byte */
        return OV_KEY_NONE;
    }

    unsigned char        fin  = kbuf[end];
    const unsigned char *par  = kbuf + 2;
    int                  plen = end - 2;
    int                  key  = OV_KEY_NONE;

    if (kbuf[2] == '[' && end == 3)
    {
        key = (fin >= 'A' && fin <= 'E') ? OV_KEY_F1 + (fin - 'A') : OV_KEY_NONE;
    }
    else if (plen == 0)
    {
        switch (fin)
        {
        case 'A':
            key = OV_KEY_UP;
            break;
        case 'B':
            key = OV_KEY_DOWN;
            break;
        case 'C':
            key = OV_KEY_RIGHT;
            break;
        case 'D':
            key = OV_KEY_LEFT;
            break;
        case 'H':
            key = OV_KEY_HOME;
            break;
        case 'F':
            key = OV_KEY_END;
            break;
        case 'Z':
            key = OV_KEY_BTAB;
            break;
        default:
            break;
        }
    }
    else if (fin == '~')
    {
        key = ov_key_tilde(atoi((const char *) par));
    }
    else if (plen == 3 && par[0] == '1' && par[1] == ';' && par[2] == '5')
    {
        key = (fin == 'D') ? OV_KEY_CTRL_LEFT : (fin == 'C') ? OV_KEY_CTRL_RIGHT : OV_KEY_NONE;
    }
    else if (plen == 3 && par[0] == '1' && par[1] == ';' && par[2] == '2')
    {
        key = (fin == 'D')   ? OV_KEY_SHIFT_LEFT
              : (fin == 'C') ? OV_KEY_SHIFT_RIGHT
              : (fin == 'A') ? OV_KEY_SHIFT_UP
              : (fin == 'B') ? OV_KEY_SHIFT_DOWN
                             : OV_KEY_NONE;
    }
    else if (par[0] == '<' && (fin == 'M' || fin == 'm'))
    {
        unsigned char mpar[64];
        int           mlen = (plen < (int) sizeof(mpar)) ? plen : (int) sizeof(mpar) - 1;
        memcpy(mpar, par, (size_t) mlen);
        ov_key_consume(end + 1);
        return ov_key_mouse(mpar, mlen, fin == 'M');
    }
    ov_key_consume(end + 1);
    return key;
}

/**
 * ov_key_ss3 - decode ESC O x (application cursor keys and xterm F1-F4).
 */
static int ov_key_ss3(void)
{
    int key = OV_KEY_NONE;
    switch (kbuf[2])
    {
    case 'A':
        key = OV_KEY_UP;
        break;
    case 'B':
        key = OV_KEY_DOWN;
        break;
    case 'C':
        key = OV_KEY_RIGHT;
        break;
    case 'D':
        key = OV_KEY_LEFT;
        break;
    case 'H':
        key = OV_KEY_HOME;
        break;
    case 'F':
        key = OV_KEY_END;
        break;
    case 'P':
    case 'Q':
    case 'R':
    case 'S':
        key = OV_KEY_F1 + (kbuf[2] - 'P');
        break;
    default:
        break;
    }
    ov_key_consume(3);
    return key;
}

/**
 * ov_get_key - Read one keypress, escape sequence or SGR mouse report from stdin
 *
 * Bytes of a sequence split across reads are kept until it completes (20 ms grace); a sequence
 * that stays incomplete is dropped together with its tail, never replayed as keys.
 *
 * Return: Key code enum (e.g. OV_KEY_UP, OV_KEY_MOUSE), ASCII character code,
 *         OV_KEY_NONE if no key available, or OV_KEY_EOF on EOF.
 */
int ov_get_key(void)
{
    if (kbuf_len == 0 && ov_key_fill(0) < 0)
    {
        return OV_KEY_EOF;
    }
    if (skip_tail && ov_key_skip_expired())
    {
        skip_tail = 0;
    }
    while (skip_tail && kbuf_len > 0)
    {
        unsigned char c = kbuf[0];
        if (c >= 0x20 && c <= 0x3F)
        {
            ov_key_consume(1);
            continue;
        }
        if (c >= 0x40 && c <= 0x7E)
        {
            ov_key_consume(1);
        }
        skip_tail = 0;
    }
    if (kbuf_len == 0)
    {
        return OV_KEY_NONE;
    }

    if (kbuf[0] != 0x1b)
    {
        int key = (int) kbuf[0];
        ov_key_consume(1);
        return key;
    }
    if (kbuf_len == 1)
    {
        ov_key_fill(50);
    }
    if (kbuf_len == 1)
    {
        ov_key_consume(1);
        return OV_KEY_ESC; /* solitary ESC */
    }
    if (kbuf[1] == '[')
    {
        return ov_key_csi();
    }
    if (kbuf[1] == 'O')
    {
        if (kbuf_len == 2)
        {
            ov_key_fill(50);
        }
        if (kbuf_len >= 3)
        {
            return ov_key_ss3();
        }
    }
    ov_key_consume(1); /* ESC followed by an ordinary key */
    return OV_KEY_ESC;
}

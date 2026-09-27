// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"

int ov_mouse_row = 0;
int ov_mouse_col = 0;
int ov_mouse_btn = 0;
int ov_hover_row = 0;
int ov_hover_col = 0;
/**
 * ov_get_key - Read a non-blocking keypress or escape sequence from stdin
 *
 * Parses ANSI escape sequences, arrow keys, function keys, and mouse SGR reports.
 *
 * Return: Key code enum (e.g. OV_KEY_UP, OV_KEY_MOUSE), ASCII character code,
 *         OV_KEY_NONE if no key available, or OV_KEY_EOF on EOF.
 */
int ov_get_key(void)
{
    static unsigned char buf[256];
    static int           buf_len = 0;
    ssize_t              n;

    /* Safety flush if buffer accumulated unexpected volume of bytes */
    if (buf_len > 64)
    {
        buf_len = 0;
    }

    /* If buffer is empty, check if input is actually available before reading */
    if (buf_len == 0)
    {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
        int           pr  = poll(&pfd, 1, 0);
        if (pr <= 0 || !(pfd.revents & POLLIN))
        {
            if (pr > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)))
            {
                return OV_KEY_EOF;
            }
            return OV_KEY_NONE;
        }

        n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0)
        {
            buf_len = (int) n;
        }
        else if (n == 0)
        {
            if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL))
            {
                return OV_KEY_EOF;
            }
            return OV_KEY_NONE;
        }
        else
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
            {
                return OV_KEY_NONE;
            }
            return OV_KEY_EOF;
        }
    }

    if (buf_len == 0)
    {
        return OV_KEY_NONE;
    }

    /* single ASCII byte, not ESC */
    if (buf[0] != 0x1b)
    {
        int key = (int) buf[0];
        memmove(buf, buf + 1, (size_t) (buf_len - 1));
        buf_len--;
        return key;
    }

    /* Check if trailing bytes follow ESC for an escape sequence */
    if (buf_len == 1)
    {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
        if (poll(&pfd, 1, 50) > 0 && (pfd.revents & POLLIN))
        {
            n = read(STDIN_FILENO, buf + buf_len, sizeof(buf) - (size_t) buf_len);
            if (n > 0)
            {
                buf_len += (int) n;
            }
        }
    }

    /* Solitary ESC with no subsequent bytes */
    if (buf_len == 1)
    {
        buf_len = 0;
        return OV_KEY_ESC;
    }

    /* ESC with '[' or 'O' waiting for a 3rd byte */
    if (buf_len == 2 && (buf[1] == '[' || buf[1] == 'O'))
    {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
        if (poll(&pfd, 1, 50) > 0 && (pfd.revents & POLLIN))
        {
            n = read(STDIN_FILENO, buf + buf_len, sizeof(buf) - (size_t) buf_len);
            if (n > 0)
            {
                buf_len += (int) n;
            }
        }
        if (buf_len == 2)
        {
            /* No 3rd byte arrived: treat as solitary ESC followed by '[' or 'O' */
            memmove(buf, buf + 1, (size_t) (buf_len - 1));
            buf_len--;
            return OV_KEY_ESC;
        }
    }

    /* Escape sequence */
    if (buf_len >= 2)
    {
        if (buf[1] == '[')
        {
            if (buf_len >= 3)
            {
                int consumed = 0;
                int key      = 0;
                switch (buf[2])
                {
                case 'A':
                    key      = OV_KEY_UP;
                    consumed = 3;
                    break;
                case 'B':
                    key      = OV_KEY_DOWN;
                    consumed = 3;
                    break;
                case 'C':
                    key      = OV_KEY_RIGHT;
                    consumed = 3;
                    break;
                case 'D':
                    key      = OV_KEY_LEFT;
                    consumed = 3;
                    break;
                case 'H':
                    key      = OV_KEY_HOME;
                    consumed = 3;
                    break;
                case 'F':
                    key      = OV_KEY_END;
                    consumed = 3;
                    break;
                case 'Z':
                    key      = OV_KEY_BTAB;
                    consumed = 3;
                    break;
                default:
                    break;
                }
                if (key)
                {
                    memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                    buf_len -= consumed;
                    return key;
                }

                /* ESC [ <digits> ~ */
                int tilde_idx = -1;
                for (int i = 2; i < buf_len && i < 10; i++)
                {
                    if (buf[i] == '~')
                    {
                        tilde_idx = i;
                        break;
                    }
                    if (buf[i] >= 0x40 && buf[i] <= 0x7E)
                    {
                        break;
                    }
                }

                if (tilde_idx != -1)
                {
                    int code = atoi((char *) buf + 2);
                    consumed = tilde_idx + 1;
                    switch (code)
                    {
                    case 1:
                        key = OV_KEY_HOME;
                        break;
                    case 3:
                        key = OV_KEY_DEL;
                        break;
                    case 4:
                        key = OV_KEY_END;
                        break;
                    case 5:
                        key = OV_KEY_PGUP;
                        break;
                    case 6:
                        key = OV_KEY_PGDN;
                        break;
                    case 15:
                        key = OV_KEY_F5;
                        break;
                    case 17:
                        key = OV_KEY_F6;
                        break;
                    case 18:
                        key = OV_KEY_F7;
                        break;
                    case 19:
                        key = OV_KEY_F8;
                        break;
                    default:
                        break;
                    }
                    if (key)
                    {
                        memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* CTRL+Arrow: ESC [ 1 ; 5 C/D
                 * SHIFT+Arrow: ESC [ 1 ; 2 A/B/C/D */
                if (buf_len >= 6 && buf[2] == '1' && buf[3] == ';')
                {
                    if (buf[4] == '5')
                    {
                        if (buf[5] == 'D')
                        {
                            key      = OV_KEY_CTRL_LEFT;
                            consumed = 6;
                        }
                        else if (buf[5] == 'C')
                        {
                            key      = OV_KEY_CTRL_RIGHT;
                            consumed = 6;
                        }
                    }
                    else if (buf[4] == '2')
                    {
                        if (buf[5] == 'D')
                        {
                            key      = OV_KEY_SHIFT_LEFT;
                            consumed = 6;
                        }
                        else if (buf[5] == 'C')
                        {
                            key      = OV_KEY_SHIFT_RIGHT;
                            consumed = 6;
                        }
                        else if (buf[5] == 'A')
                        {
                            key      = OV_KEY_SHIFT_UP;
                            consumed = 6;
                        }
                        else if (buf[5] == 'B')
                        {
                            key      = OV_KEY_SHIFT_DOWN;
                            consumed = 6;
                        }
                    }
                    if (key)
                    {
                        memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* SGR mouse: ESC [ < btn;col;row M/m */
                if (buf[2] == '<')
                {
                    int end_idx = -1;
                    for (int i = 3; i < buf_len && i < 32; i++)
                    {
                        if (buf[i] == 'M' || buf[i] == 'm')
                        {
                            end_idx = i;
                            break;
                        }
                    }
                    if (end_idx <= 0 && buf_len < 32)
                    {
                        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
                        if (poll(&pfd, 1, 20) > 0 && (pfd.revents & POLLIN))
                        {
                            n = read(STDIN_FILENO, buf + buf_len, sizeof(buf) - (size_t) buf_len);
                            if (n > 0)
                            {
                                buf_len += (int) n;
                                for (int i = 3; i < buf_len && i < 32; i++)
                                {
                                    if (buf[i] == 'M' || buf[i] == 'm')
                                    {
                                        end_idx = i;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (end_idx > 0)
                    {
                        int  mb = 0, mc = 0, mr = 0;
                        char tmp[64];
                        int  tlen = end_idx - 3;
                        if (tlen > 0 && tlen < (int) sizeof(tmp))
                        {
                            memcpy(tmp, buf + 3, (size_t) tlen);
                            tmp[tlen] = '\0';
                            sscanf(tmp, "%d;%d;%d", &mb, &mc, &mr);
                        }
                        int press = (buf[end_idx] == 'M');
                        consumed  = end_idx + 1;
                        memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                        buf_len -= consumed;

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
                        /* Ctrl+scroll (#12) */
                        if (mb == 80)
                        {
                            return OV_KEY_CTRL_SCROLL_UP;
                        }
                        if (mb == 81)
                        {
                            return OV_KEY_CTRL_SCROLL_DOWN;
                        }

                        /* Mouse move (passive) */
                        if (mb == 35 && press)
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
                        if (!press)
                        {
                            return OV_KEY_MOUSE_RELEASE;
                        }
                        return OV_KEY_NONE;
                    }

                    /* Incomplete or malformed mouse sequence: safely discard prefix */
                    int discard = 3;
                    for (int i = 3; i < buf_len; i++)
                    {
                        if ((buf[i] >= 0x40 && buf[i] <= 0x7E) || buf[i] == 0x1b)
                        {
                            discard = (buf[i] == 0x1b) ? i : (i + 1);
                            break;
                        }
                    }
                    memmove(buf, buf + discard, (size_t) (buf_len - discard));
                    buf_len -= discard;
                    return OV_KEY_NONE;
                }

                for (int i = 2; i < buf_len; i++)
                {
                    if (buf[i] >= 0x40 && buf[i] <= 0x7E)
                    {
                        memmove(buf, buf + i + 1, (size_t) (buf_len - (i + 1)));
                        buf_len -= (i + 1);
                        return OV_KEY_NONE;
                    }
                }

                if (buf_len > 16)
                {
                    memmove(buf, buf + 2, (size_t) (buf_len - 2));
                    buf_len -= 2;
                    return OV_KEY_NONE;
                }
            }
            return OV_KEY_NONE;
        }

        /* SS3: ESC O ... (application cursor keys and xterm F1-F4) */
        if (buf[1] == 'O')
        {
            if (buf_len >= 3)
            {
                int key      = 0;
                int consumed = 3;
                switch (buf[2])
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
                    key = OV_KEY_F1;
                    break;
                case 'Q':
                    key = OV_KEY_F2;
                    break;
                case 'R':
                    key = OV_KEY_F3;
                    break;
                case 'S':
                    key = OV_KEY_F4;
                    break;
                default:
                    break;
                }
                memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                buf_len -= consumed;
                return key ? key : OV_KEY_NONE;
            }
            return OV_KEY_NONE;
        }

        /* Consume solitary ESC or unhandled escape prefix */
        memmove(buf, buf + 1, (size_t) (buf_len - 1));
        buf_len--;
        return OV_KEY_ESC;
    }
    return OV_KEY_NONE;
}

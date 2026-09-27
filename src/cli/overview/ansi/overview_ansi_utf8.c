// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"
/**
 * utf8_char_length - Determine the expected byte length of a UTF-8 character
 * @c: Leading byte of UTF-8 sequence
 *
 * Return: Byte count (1 to 4).
 */
int utf8_char_length(unsigned char c)
{
    if ((c & 0x80) == 0)
    {
        return 1;
    }
    if ((c & 0xE0) == 0xC0)
    {
        return 2;
    }
    if ((c & 0xF0) == 0xE0)
    {
        return 3;
    }
    if ((c & 0xF8) == 0xF0)
    {
        return 4;
    }
    return 1;
}

/**
 * ov_utf8_decode - Decode a single UTF-8 codepoint from string
 * @s:   Pointer to byte sequence
 * @len: Available buffer length
 * @cp:  Output decoded Unicode codepoint
 *
 * Return: Number of consumed bytes, or 0 on error.
 */
int ov_utf8_decode(const char *s, int len, uint32_t *cp)
{
    if (len <= 0)
    {
        *cp = 0;
        return 0;
    }
    unsigned char c = (unsigned char) s[0];
    if (c < 0x80)
    {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && len >= 2)
    {
        *cp = (uint32_t) (((c & 0x1F) << 6) | (s[1] & 0x3F));
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && len >= 3)
    {
        *cp = (uint32_t) (((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F));
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && len >= 4)
    {
        *cp = (uint32_t) (((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) |
                          (s[3] & 0x3F));
        return 4;
    }
    *cp = c;
    return 1;
}

/**
 * ov_utf8_next_cluster - Extract next grapheme cluster, handling emoji and variation selectors
 * @s:         Pointer to UTF-8 text
 * @max_len:   Maximum available bytes
 * @bytes_out: Output byte count of grapheme cluster
 * @width_out: Output terminal cell width (1 or 2)
 *
 * Return: 1 if cluster extracted, 0 on end of string or error.
 */
int ov_utf8_next_cluster(const char *s, int max_len, int *bytes_out, int *width_out)
{
    if (max_len <= 0 || s[0] == '\0')
    {
        *bytes_out = 0;
        *width_out = 0;
        return 0;
    }

    uint32_t cp0         = 0;
    int      b0          = ov_utf8_decode(s, max_len, &cp0);
    int      total_bytes = b0;
    int      has_vs16    = 0;

    /* Consume trailing modifiers: VS16 (U+FE0F), Keycap (U+20E3), etc. */
    while (total_bytes < max_len && total_bytes < 15)
    {
        uint32_t next_cp = 0;
        int      nb      = ov_utf8_decode(s + total_bytes, max_len - total_bytes, &next_cp);
        if (next_cp == 0xFE0F)
        {
            has_vs16 = 1;
            total_bytes += nb;
        }
        else if (next_cp == 0x20E3 || next_cp == 0xFE0E)
        {
            total_bytes += nb;
        }
        else
        {
            break;
        }
    }

    int w = 1;
    if (has_vs16)
    {
        w = 2;
    }
    else if (cp0 >= 0x1F000 && cp0 <= 0x1FAFF)
    {
        /* Standard emoji symbols / pictographs */
        w = 2;
    }
    else if (cp0 >= 0x2600 && cp0 <= 0x27BF)
    {
        if (cp0 == 0x2705 || cp0 == 0x274C || cp0 == 0x274E || (cp0 >= 0x2753 && cp0 <= 0x2755) ||
            cp0 == 0x2757 || cp0 == 0x2728 || cp0 == 0x26A0 || cp0 == 0x26A1 || cp0 == 0x26BD ||
            cp0 == 0x26BE || cp0 == 0x26C4 || cp0 == 0x26C5 || cp0 == 0x26D4 || cp0 == 0x26EA ||
            cp0 == 0x26F2 || cp0 == 0x26F3 || cp0 == 0x26F5 || cp0 == 0x26FA || cp0 == 0x26FD)
        {
            w = 2;
        }
        else
        {
            int sys_w = wcwidth((wchar_t) cp0);
            w         = (sys_w == 2) ? 2 : 1;
        }
    }
    else
    {
        int sys_w = wcwidth((wchar_t) cp0);
        w         = (sys_w == 2) ? 2 : 1;
    }

    *bytes_out = total_bytes;
    *width_out = w;
    return 1;
}

/**
 * ov_str_display_width - Calculate total visual column width of UTF-8 string
 * @s: Null-terminated UTF-8 string
 *
 * Return: Total column display width.
 */
int ov_str_display_width(const char *s)
{
    if (s == NULL)
    {
        return 0;
    }
    int len     = (int) strlen(s);
    int total_w = 0;
    int pos     = 0;
    while (pos < len)
    {
        int b = 0, w = 0;
        if (!ov_utf8_next_cluster(s + pos, len - pos, &b, &w))
        {
            break;
        }
        total_w += w;
        pos += b;
    }
    return total_w;
}

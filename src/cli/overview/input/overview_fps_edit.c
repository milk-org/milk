// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_fps_edit.c
 * @brief Inline FPS parameter editing for milk-CTRL
 *
 * Implements an inline parameter edit bar at the bottom
 * of the terminal, modeled on fpsCTRL_inline_edit_param.
 * Uses direct ANSI writes that bypass the double-buffer
 * for the edit prompt (only the prompt row is affected).
 */

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "overview_ansi.h"
#include "overview_layout.h"
#include "overview_data.h"
#include "overview_data_internal.h"
#include "overview_fps_edit.h"

/* libfps types after overview headers */
#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "fps_types.h"

/**
 * ov_fps_type_short_label - return a short type label.
 *
 * @type: FPS parameter type code
 *
 * Return: static string like "INT64", "FLOAT", etc.
 */
static const char *ov_fps_type_short_label(uint32_t type)
{
    if (type == FPTYPE_INT64 || type == FPTYPE_INT32)
    {
        return "INT";
    }
    if (type == FPTYPE_UINT64 || type == FPTYPE_UINT32)
    {
        return "UINT";
    }
    if (type == FPTYPE_FLOAT64 || type == FPTYPE_FLOAT32)
    {
        return "FLT";
    }
    if (type == FPTYPE_ONOFF)
    {
        return "ON/OFF";
    }
    if (type == FPTYPE_STREAMNAME)
    {
        return "STRM";
    }
    if (FPTYPE_IS_STRING(type))
    {
        return "STR";
    }
    if (type == FPTYPE_PID)
    {
        return "PID";
    }
    if (type == FPTYPE_TIMESPEC)
    {
        return "TIME";
    }
    return "???";
}

/**
 * ov_fps_param_rc_msg - command-log text for a refused parameter write (OV_PARAM_* codes).
 */
const char *ov_fps_param_rc_msg(int rc)
{
    switch (rc)
    {
    case OV_PARAM_MOVED:
        return "FPS was re-created: value not written";
    case OV_PARAM_RO_FILE:
        return "FPS shm file is write-protected for you: value not written";
    case OV_PARAM_LOCKED:
        return "parameter is write-protected in the FPS's current state";
    default:
        return "value rejected";
    }
}

/** ov_finite - isfinite() by the exponent bits: the build's -Ofast makes isfinite() always true. */
static int ov_finite(double v)
{
    uint64_t b;
    memcpy(&b, &v, sizeof(b));
    return ((b >> 52) & 0x7ff) != 0x7ff;
}

/**
 * ov_fps_edit_check - parse @s fully for the parameter's numeric type and check its range and
 * MIN/MAX limits. Return: NULL when acceptable, else the reason (in @why).
 */
static const char *ov_fps_edit_check(const ov_fps_param_info_t *pi,
                                     const char                *s,
                                     char                      *why,
                                     size_t                     n)
{
    char  *end = NULL;
    double v   = 0.0;
    errno      = 0;
    switch (pi->type)
    {
    case FPTYPE_INT32:
    case FPTYPE_INT64:
    {
        long long x = strtoll(s, &end, 0);
        if (pi->type == FPTYPE_INT32 && (x < INT32_MIN || x > INT32_MAX))
        {
            errno = ERANGE;
        }
        v = (double) x;
        break;
    }
    case FPTYPE_UINT32:
    case FPTYPE_UINT64:
    {
        unsigned long long x = strtoull(s, &end, 0);
        if (strchr(s, '-') != NULL || (pi->type == FPTYPE_UINT32 && x > UINT32_MAX))
        {
            errno = ERANGE;
        }
        v = (double) x;
        break;
    }
    case FPTYPE_FLOAT32:
    case FPTYPE_FLOAT64:
        v = strtod(s, &end);
        if (!ov_finite(v) || (pi->type == FPTYPE_FLOAT32 && fabs(v) > FLT_MAX))
        {
            errno = ERANGE;
        }
        break;
    default:
        return NULL; /* strings, names: libfps checks them */
    }
    while (end != NULL && isspace((unsigned char) *end))
    {
        end++;
    }
    if (end == s || end == NULL || *end != '\0')
    {
        snprintf(why, n, "not a number: \"%s\"", s);
    }
    else if (errno == ERANGE)
    {
        snprintf(why, n, "out of range for %s", ov_fps_type_short_label(pi->type));
    }
    else if (pi->has_min && v < pi->vmin)
    {
        snprintf(why, n, "below MIN %g", pi->vmin);
    }
    else if (pi->has_max && v > pi->vmax)
    {
        snprintf(why, n, "above MAX %g", pi->vmax);
    }
    else
    {
        return NULL;
    }
    return why;
}

/**
 * ov_fps_inline_edit - edit an FPS parameter inline.
 *
 * Draws an edit prompt on the bottom terminal row,
 * reads character input, and applies the new value
 * to the FPS shared memory on ENTER.
 *
 * The edit bar uses direct ANSI writes (bypassing
 * the shadow buffer) because it is a modal overlay.
 * After editing, the shadow front buffer is cleared
 * to force a full repaint on the next frame.
 *
 * @lay:       layout state
 * @fps_name:  name of the FPS to edit
 * @disp_idx:  display parameter index
 *
 * Return: 0 on success/abort, -1 on error
 */
int ov_fps_inline_edit(OV_LAYOUT *lay, const char *fps_name, int disp_idx)
{
    (void) lay;

    ov_fps_param_info_t pinfo;
    if (ov_fcache_get_param_info(fps_name, disp_idx, &pinfo) != 0)
    {
        return -1;
    }
    ov_str_sanitize(pinfo.display_kw);
    ov_str_sanitize(pinfo.valstr);

    const char *curval     = pinfo.valstr;
    const char *tlabel     = ov_fps_type_short_label(pinfo.type);
    const char *display_kw = pinfo.display_kw;

    /* Get terminal size */
    int trows, tcols;
    ov_get_terminal_size(&trows, &tcols);

    /* Show cursor */
    if (write(ov__tty_fd, "\033[?25h", 6) < 0)
    {
    }

    /* Position at bottom row and clear it */
    {
        char posbuf[32];
        int  n = snprintf(posbuf, sizeof(posbuf), "\033[%d;1H\033[2K", trows);
        if (n > 0)
        {
            if (write(ov__tty_fd, posbuf, (size_t) n) < 0)
            {
            }
        }
    }

    /* Check writability */
    if (!pinfo.is_writable)
    {
        char prompt[512];
        int  n = snprintf(prompt, sizeof(prompt),
                          "\033[1;31m"
                          " [%s] %s = %s  (read-only)"
                          "\033[0m",
                          tlabel, display_kw, curval);
        if (n > 0)
        {
            if (write(ov__tty_fd, prompt, (size_t) n) < 0)
            {
            }
        }
        usleep(800000);
        /* Drain buffered keys */
        while (ov_get_key() > 0)
        {
        }
        /* Hide cursor */
        if (write(ov__tty_fd, "\033[?25l", 6) < 0)
        {
        }
        /* Force full repaint */
        ov_buf_force_clear();
        return 0;
    }

    /* ONOFF: toggle immediately, no text input */
    if (pinfo.type == FPTYPE_ONOFF)
    {
        int  newval = 0;
        char kw[FUNCTION_PARAMETER_STRMAXLEN];
        int  trc = ov_fcache_toggle_param(fps_name, disp_idx, kw, sizeof(kw), &newval);

        /* Brief flash */
        {
            char msg[256];
            int  n = (trc == 0) ? snprintf(msg, sizeof(msg), "\033[1;32m [ON/OFF] %s => %s\033[0m",
                                           display_kw, newval ? "ON" : "OFF")
                                : snprintf(msg, sizeof(msg), "\033[1;31m [ON/OFF] %s: %s\033[0m",
                                           display_kw, ov_fps_param_rc_msg(trc));
            if (n > 0)
            {
                if (write(ov__tty_fd, msg, (size_t) n) < 0)
                {
                }
            }
        }
        usleep(300000);
        while (ov_get_key() > 0)
        {
        }
        if (write(ov__tty_fd, "\033[?25l", 6) < 0)
        {
        }
        ov_buf_force_clear();
        return 0;
    }

    /* Text input mode: show prompt */
    {
        char prompt[512];
        char lim[96] = "";
        if (pinfo.has_min || pinfo.has_max)
        {
            char lo[40] = "", hi[40] = "";
            if (pinfo.has_min)
            {
                snprintf(lo, sizeof(lo), "%g", pinfo.vmin);
            }
            if (pinfo.has_max)
            {
                snprintf(hi, sizeof(hi), "%g", pinfo.vmax);
            }
            snprintf(lim, sizeof(lim), " [%s..%s]", lo, hi);
        }
        int n = snprintf(prompt, sizeof(prompt),
                         "\033[1;36m"
                         " [%s] %s"
                         "\033[0m%s"
                         " (was: "
                         "\033[33m%s\033[0m"
                         ") new value: ",
                         tlabel, display_kw, lim, curval);
        if (n > 0)
        {
            if (write(ov__tty_fd, prompt, (size_t) n) < 0)
            {
            }
        }
    }

    /* Character-by-character input loop */
    char buf[200];
    int  bufpos  = 0;
    int  maxlen  = (int) sizeof(buf) - 1;
    int  aborted = 0;

    for (;;)
    {
        if (OV_SIG_ANY_SET())
        {
            aborted = 1;
            break;
        }
        usleep(10000);
        int key = ov_get_key();
        if (key > 0)
        {
            key &= ~OV_KEY_PASTED; /* the editor is a text prompt: pasted text is welcome */
        }

        if (key == OV_KEY_NONE)
        {
            continue;
        }

        /* ESC or EOF — abort */
        if (key == OV_KEY_ESC || key == OV_KEY_EOF)
        {
            aborted = 1;
            break;
        }

        /* ENTER — confirm */
        if (key == OV_KEY_ENTER || key == 13)
        {
            break;
        }

        /* Backspace (127 or ctrl-h) */
        if (key == 127 || key == 8)
        {
            if (bufpos > 0)
            {
                bufpos--;
                if (write(ov__tty_fd, "\b \b", 3) < 0)
                {
                }
            }
            continue;
        }

        /* Ctrl+U — clear input */
        if (key == ctrl('u'))
        {
            while (bufpos > 0)
            {
                bufpos--;
                if (write(ov__tty_fd, "\b \b", 3) < 0)
                {
                }
            }
            continue;
        }

        /* Ignore non-printable / special keys */
        if (key < 32 || key > 126)
        {
            continue;
        }

        /* Printable character */
        if (bufpos < maxlen)
        {
            buf[bufpos++] = (char) key;
            char echo     = (char) key;
            if (write(ov__tty_fd, &echo, 1) < 0)
            {
            }
        }
    }
    buf[bufpos] = '\0';

    /* Hide cursor */
    if (write(ov__tty_fd, "\033[?25l", 6) < 0)
    {
    }

    if (!aborted && bufpos > 0)
    {
        char        why[96];
        const char *bad = ov_fps_edit_check(&pinfo, buf, why, sizeof(why));
        int         rc =
            (bad != NULL) ? -1 : ov_fcache_set_param_value(fps_name, disp_idx, pinfo.keyword, buf);
        if (rc != 0)
        {
            /* Show the reason briefly; nothing was written */
            char msg[256];
            int  n = snprintf(msg, sizeof(msg), "\033[1;31m  not written: %s\033[0m",
                              (bad != NULL) ? bad : ov_fps_param_rc_msg(rc));
            if (n > 0 && write(ov__tty_fd, msg, (size_t) n) < 0)
            {
            }
            if (rc == OV_PARAM_MOVED)
            {
                ov_scan_force_update();
            }
            usleep(800000);
        }
    }

    /* Force full repaint */
    ov_buf_force_clear();
    return 0;
}

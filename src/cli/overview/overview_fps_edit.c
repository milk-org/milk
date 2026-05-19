/**
 * @file overview_fps_edit.c
 * @brief Inline FPS parameter editing for milk-CTRL
 *
 * Implements an inline parameter edit bar at the bottom
 * of the terminal, modeled on fpsCTRL_inline_edit_param.
 * Uses direct ANSI writes that bypass the double-buffer
 * for the edit prompt (only the prompt row is affected).
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "overview_ansi.h"
#include "overview_layout.h"
#include "overview_data.h"
#include "overview_data_internal.h"

/* libfps headers after overview headers to
 * avoid macro redefinition warnings */
#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "fps_types.h"
#include "fps_paramvalue.h"
#include "fps_printparameter_valuestring.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_save2disk.h"

/**
 * ov_fps_type_short_label - return a short type label.
 *
 * @type: FPS parameter type code
 *
 * Return: static string like "INT64", "FLOAT", etc.
 */
static const char *ov_fps_type_short_label(
    uint32_t type)
{
    if(type == FPTYPE_INT64 || type == FPTYPE_INT32)
    {
        return "INT";
    }
    if(type == FPTYPE_UINT64 || type == FPTYPE_UINT32)
    {
        return "UINT";
    }
    if(type == FPTYPE_FLOAT64 || type == FPTYPE_FLOAT32)
    {
        return "FLT";
    }
    if(type == FPTYPE_ONOFF)
    {
        return "ON/OFF";
    }
    if(type == FPTYPE_STREAMNAME)
    {
        return "STRM";
    }
    if(FPTYPE_IS_STRING(type))
    {
        return "STR";
    }
    if(type == FPTYPE_PID)
    {
        return "PID";
    }
    if(type == FPTYPE_TIMESPEC)
    {
        return "TIME";
    }
    return "???";
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
int ov_fps_inline_edit(
    OV_LAYOUT  *lay,
    const char *fps_name,
    int        disp_idx)
{
    (void) lay;

    FPS *fps = ov_fcache_get_fps(fps_name);
    if(fps == NULL || fps->md == NULL)
    {
        return -1;
    }

    int pindex = ov_fcache_get_param_index(
                     fps_name, disp_idx);
    if(pindex < 0)
    {
        return -1;
    }

    FPS_PARAM *fp = &fps->parray[pindex];

    /* Get current value as string */
    char curval[200];
    functionparameter_GetParamValueString(
        fp, curval, (int) sizeof(curval));

    /* Short type label */
    const char *tlabel =
        ov_fps_type_short_label(fp->type);

    /* Strip FPS name prefix from keyword */
    const char *display_kw = fp->keywordfull;
    int prefix_len = (int) strlen(fps->md->name);
    if(strncmp(display_kw,
               fps->md->name,
               (size_t) prefix_len) == 0
            && display_kw[prefix_len] == '.')
    {
        display_kw += prefix_len + 1;
    }

    /* Get terminal size */
    int trows, tcols;
    ov_get_terminal_size(&trows, &tcols);

    /* Show cursor */
    if(write(STDOUT_FILENO,
             "\033[?25h", 6) < 0) {}

    /* Position at bottom row and clear it */
    {
        char posbuf[32];
        int n = snprintf(posbuf, sizeof(posbuf),
                         "\033[%d;1H\033[2K", trows);
        if(n > 0)
        {
            if(write(STDOUT_FILENO,
                     posbuf, (size_t) n) < 0) {}
        }
    }

    /* Check writability */
    if(!(fp->fpflag & FPFLAG_WRITESTATUS))
    {
        char prompt[512];
        int n = snprintf(prompt, sizeof(prompt),
                         "\033[1;31m"
                         " [%s] %s = %s  (read-only)"
                         "\033[0m",
                         tlabel, display_kw, curval);
        if(n > 0)
        {
            if(write(STDOUT_FILENO,
                     prompt, (size_t) n) < 0) {}
        }
        usleep(800000);
        /* Drain buffered keys */
        while(ov_get_key() != OV_KEY_NONE) {}
        /* Hide cursor */
        if(write(STDOUT_FILENO,
                 "\033[?25l", 6) < 0) {}
        /* Force full repaint */
        ov_buf_force_clear();
        return 0;
    }

    /* ONOFF: toggle immediately, no text input */
    if(fp->type == FPTYPE_ONOFF)
    {
        int64_t newval = fp->val.i64[0] ? 0 : 1;
        fp->val.i64[0] = newval;

        fps->md->signal |=
            FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

        if(fp->fpflag & FPFLAG_SAVEONCHANGE)
        {
            functionparameter_WriteParameterToDisk(
                fps, pindex,
                "setval", "milk-CTRL_toggle");
            functionparameter_SaveFPS2disk(fps);
        }

        /* Brief flash */
        {
            char msg[256];
            int n = snprintf(msg, sizeof(msg),
                             "\033[1;32m"
                             " [ON/OFF] %s => %s"
                             "\033[0m",
                             display_kw,
                             newval ? "ON" : "OFF");
            if(n > 0)
            {
                if(write(STDOUT_FILENO,
                         msg, (size_t) n)
                        < 0) {}
            }
        }
        usleep(300000);
        while(ov_get_key() != OV_KEY_NONE) {}
        if(write(STDOUT_FILENO,
                 "\033[?25l", 6) < 0) {}
        ov_buf_force_clear();
        return 0;
    }

    /* Text input mode: show prompt */
    {
        char prompt[512];
        int n = snprintf(prompt, sizeof(prompt),
                         "\033[1;36m"
                         " [%s] %s"
                         "\033[0m"
                         " (was: "
                         "\033[33m%s\033[0m"
                         ") new value: ",
                         tlabel, display_kw, curval);
        if(n > 0)
        {
            if(write(STDOUT_FILENO,
                     prompt, (size_t) n) < 0) {}
        }
    }

    /* Character-by-character input loop */
    char buf[200];
    int  bufpos  = 0;
    int  maxlen  = (int) sizeof(buf) - 1;
    int  aborted = 0;

    for(;;)
    {
        usleep(10000);
        int key = ov_get_key();

        if(key == OV_KEY_NONE)
        {
            continue;
        }

        /* ESC — abort */
        if(key == OV_KEY_ESC)
        {
            aborted = 1;
            break;
        }

        /* ENTER — confirm */
        if(key == OV_KEY_ENTER
                || key == 13)
        {
            break;
        }

        /* Backspace (127 or ctrl-h) */
        if(key == 127 || key == 8)
        {
            if(bufpos > 0)
            {
                bufpos--;
                if(write(STDOUT_FILENO,
                         "\b \b", 3) < 0) {}
            }
            continue;
        }

        /* Ctrl+U — clear input */
        if(key == ctrl('u'))
        {
            while(bufpos > 0)
            {
                bufpos--;
                if(write(STDOUT_FILENO,
                         "\b \b", 3) < 0) {}
            }
            continue;
        }

        /* Ignore non-printable / special keys */
        if(key < 32 || key > 126)
        {
            continue;
        }

        /* Printable character */
        if(bufpos < maxlen)
        {
            buf[bufpos++] = (char) key;
            char echo = (char) key;
            if(write(STDOUT_FILENO,
                     &echo, 1) < 0) {}
        }
    }
    buf[bufpos] = '\0';

    /* Hide cursor */
    if(write(STDOUT_FILENO,
             "\033[?25l", 6) < 0) {}

    if(!aborted && bufpos > 0)
    {
        if(functionparameter_SetParamValue_fromString(
                    fps, pindex, buf) != 0)
        {
            /* Show error briefly */
            const char errmsg[] =
                "\033[1;31m  ERROR: invalid value"
                "\033[0m";
            if(write(STDOUT_FILENO,
                     errmsg, sizeof(errmsg) - 1)
                    < 0) {}
            usleep(500000);
        }
        else
        {
            /* Signal update */
            fps->md->signal |=
                FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

            /* Processinfo change tracking */
            if(strncmp(
                        fp->keywordfull,
                        ".procinfo.", 10) == 0)
            {
                fps->md->processinfo_change_cnt++;
            }

            /* Save to disk if flagged */
            if(fp->fpflag & FPFLAG_SAVEONCHANGE)
            {
                functionparameter_WriteParameterToDisk(
                    fps, pindex,
                    "setval",
                    "milk-CTRL_SetParamValue");
                functionparameter_SaveFPS2disk(fps);
            }
        }
    }

    /* Force full repaint */
    ov_buf_force_clear();
    return 0;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script_cmd_io.c
 *
 * @brief CLI builtin I/O and parameter commands.
 *
 * Implements the milk-cli builtin commands that
 * handle text I/O, variable management, and FPS
 * parameter writing:
 *
 *   cli_fps_set_param() — FPS write helper (shared)
 *   cli_cmd_fpsset()    — fpsset command
 *   cli_cmd_echo()      — echo command (fallback)
 *   cli_cmd_read()      — read command
 *   cli_cmd_export()    — export command
 *   cli_cmd_shift()     — shift command
 *   cli_cmd_printf()    — printf command
 *
 * All public symbols declared in CLIcore_script.h.
 */

#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"

#include "fps_GetParamIndex.h"
#include "fps_connect.h"


/* ============================================================
 *  FPS write helper
 * ============================================================
 */

/**
 * @brief Set an FPS parameter by name
 *
 * Connects to the named FPS, locates the parameter,
 * writes the value string, then disconnects. Used by
 * both the fpsset command and the @fpsname.param=value
 * expansion syntax.
 *
 * Supports FPS parameter types: INT32, UINT32,
 * INT64, UINT64, FLOAT32, FLOAT64, ONOFF, PID, and
 * all string-like types (STRING, FILENAME, DIRNAME,
 * STREAMNAME, FPSNAME, etc.).
 *
 * @param fpsname  FPS shared-memory name
 * @param pname    Parameter key inside FPS
 * @param valstr   Value to write (as string)
 * @return 0 on success, -1 on error
 */
int cli_fps_set_param(const char *fpsname, const char *pname, const char *valstr)
{
    FPS fps;
    int fpsconn = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);

    if (fpsconn == -1 || fps.parray == NULL)
    {
        printf("Error: cannot connect to "
               "FPS '%s'\n",
               fpsname);
        return -1;
    }

    int pindex = functionparameter_GetParamIndex(&fps, pname);

    if (pindex < 0)
    {
        char dotname[512];
        snprintf(dotname, sizeof(dotname), ".%s", pname);
        pindex = functionparameter_GetParamIndex(&fps, dotname);
    }

    if (pindex < 0)
    {
        printf("Error: parameter '%s' not "
               "found in FPS '%s'\n",
               pname, fpsname);
        fps_disconnect(&fps);
        return -1;
    }

    uint32_t ptype = fps.parray[pindex].type;

    if (ptype & FPTYPE_INT32)
    {
        fps.parray[pindex].val.i32[0] = (int32_t) strtol(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_UINT32)
    {
        fps.parray[pindex].val.ui32[0] = (uint32_t) strtoul(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_INT64)
    {
        fps.parray[pindex].val.i64[0] = (int64_t) strtoll(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_UINT64)
    {
        fps.parray[pindex].val.ui64[0] = (uint64_t) strtoull(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_FLOAT64)
    {
        fps.parray[pindex].val.f64[0] = strtod(valstr, NULL);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_FLOAT32)
    {
        fps.parray[pindex].val.f32[0] = (float) strtod(valstr, NULL);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_ONOFF)
    {
        if (strcmp(valstr, "ON") == 0 || strcmp(valstr, "on") == 0 || strcmp(valstr, "1") == 0)
        {
            fps.parray[pindex].val.i64[0] = 1;
        }
        else
        {
            fps.parray[pindex].val.i64[0] = 0;
        }
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_PID)
    {
        fps.parray[pindex].val.pid[0] = (pid_t) strtol(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (FPTYPE_IS_STRING(ptype))
    {
        strncpy(fps.parray[pindex].val.string[0], valstr, FUNCTION_PARAMETER_STRMAXLEN - 1);
        fps.parray[pindex].val.string[0][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
        fps.parray[pindex].cnt0++;
    }
    else
    {
        printf("Error: unsupported param "
               "type 0x%x for '%s' in "
               "FPS '%s'\n",
               ptype, pname, fpsname);
        fps_disconnect(&fps);
        return -1;
    }

    fps_disconnect(&fps);
    return 0;
}


/* ============================================================
 *  fpsset command
 * ============================================================
 */

/**
 * @brief fpsset command — write FPS parameter
 *
 * Delegates to cli_fps_set_param() after
 * splitting the "fpsname.param" argument.
 *
 * Usage: fpsset <fpsname.param> <value>
 */
errno_t cli_cmd_fpsset(void)
{
    if (data.cmdNBarg < 3)
    {
        printf("Usage: fpsset "
               "<fpsname.param> <value>\n");
        return RETURN_FAILURE;
    }

    const char *fullname = data.cmdargtoken[1].val.string;
    const char *valstr   = data.cmdargtoken[2].val.string;

    char fpsname[256];
    strncpy(fpsname, fullname, sizeof(fpsname) - 1);
    fpsname[sizeof(fpsname) - 1] = '\0';

    char *dot = strchr(fpsname, '.');
    if (dot == NULL)
    {
        printf("Error: fpsset requires "
               "fpsname.paramname\n");
        return RETURN_FAILURE;
    }
    *dot              = '\0';
    const char *pname = dot + 1;

    if (cli_fps_set_param(fpsname, pname, valstr) != 0)
    {
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
}


/* ============================================================
 *  echo command
 * ============================================================
 */

/**
 * @brief echo command — print arguments
 *
 * Note: echo is intercepted before tokenization
 * in CLIcore_UI_execute.c. This registered version
 * is a fallback that handles the tokenized form.
 *
 * Supports -n to suppress trailing newline.
 */
errno_t cli_cmd_echo(void)
{
    int start   = 1;
    int newline = 1;

    if (data.cmdNBarg >= 2 && strcmp(data.cmdargtoken[1].val.string, "-n") == 0)
    {
        newline = 0;
        start   = 2;
    }

    for (int i = start; i < data.cmdNBarg; i++)
    {
        if (i > start)
        {
            printf(" ");
        }
        printf("%s", data.cmdargtoken[i].val.string);
    }
    if (newline)
    {
        printf("\n");
    }
    return RETURN_SUCCESS;
}


/* ============================================================
 *  read command
 * ============================================================
 */

/**
 * @brief read command — read a line into a variable
 *
 * Reads one line from stdin, optionally with a prompt,
 * timeout, character limit, or array-split mode.
 *
 * Flags:
 *   -p prompt   Display prompt string before reading
 *   -t N        Timeout after N seconds (sets $?=1)
 *   -a arr      Split input into array elements
 *   -n N        Read exactly N characters (raw mode)
 *
 * Usage: read [-p prompt] [-t N] [-n N] [-a arr] <var>
 */
errno_t cli_cmd_read(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: read [-p prompt]"
               " [-t N] [-n N]"
               " [-a arr] <var>\n");
        return RETURN_FAILURE;
    }

    const char *prompt      = "";
    int         timeout_sec = -1;
    int         nchars      = -1;
    const char *arrayname   = NULL;
    const char *varname     = NULL;

    /* ---- Parse flags ---- */
    {
        int i = 1;
        while (i < data.cmdNBarg)
        {
            const char *tok = data.cmdargtoken[i].val.string;

            if (strcmp(tok, "-p") == 0)
            {
                i++;
                if (i < data.cmdNBarg)
                {
                    prompt = data.cmdargtoken[i].val.string;
                }
            }
            else if (strcmp(tok, "-t") == 0)
            {
                i++;
                if (i < data.cmdNBarg)
                {
                    timeout_sec = (int) strtol(data.cmdargtoken[i].val.string, NULL, 10);
                }
            }
            else if (strcmp(tok, "-n") == 0)
            {
                i++;
                if (i < data.cmdNBarg)
                {
                    nchars = (int) strtol(data.cmdargtoken[i].val.string, NULL, 10);
                }
            }
            else if (strcmp(tok, "-a") == 0)
            {
                i++;
                if (i < data.cmdNBarg)
                {
                    arrayname = data.cmdargtoken[i].val.string;
                }
            }
            else
            {
                varname = tok;
            }
            i++;
        }
    }

    if (varname == NULL && arrayname == NULL)
    {
        printf("Usage: read [-p prompt]"
               " [-t N] [-n N]"
               " [-a arr] <var>\n");
        return RETURN_FAILURE;
    }

    /* ---- Display prompt ---- */
    if (prompt[0] != '\0')
    {
        printf("%s", prompt);
        fflush(stdout);
    }

    char buf[CLI_VAR_VALLEN];
    buf[0] = '\0';

    /* ---- Timed read with poll() ---- */
    if (timeout_sec >= 0)
    {
        struct pollfd pfd;
        pfd.fd     = STDIN_FILENO;
        pfd.events = POLLIN;
        int rc     = poll(&pfd, 1, timeout_sec * 1000);
        if (rc <= 0)
        {
            /* Timeout or error */
            cli_var_set("?", "1");
            return RETURN_SUCCESS;
        }
    }

    /* ---- Raw N-char read ---- */
    if (nchars > 0)
    {
        struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= (tcflag_t) ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        int nc = nchars;
        if (nc > (int) sizeof(buf) - 1)
        {
            nc = (int) sizeof(buf) - 1;
        }
        for (int ci = 0; ci < nc; ci++)
        {
            int ch = getchar();
            if (ch == EOF)
            {
                break;
            }
            buf[ci]     = (char) ch;
            buf[ci + 1] = '\0';
        }

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

        if (varname != NULL)
        {
            cli_var_set(varname, buf);
        }
        cli_var_set("?", "0");
        return RETURN_SUCCESS;
    }

    /* ---- Normal line read ---- */
    if (fgets(buf, (int) sizeof(buf), stdin) == NULL)
    {
        cli_var_set("?", "1");
        return RETURN_SUCCESS;
    }
    {
        size_t bl = strlen(buf);
        while (bl > 0 && (buf[bl - 1] == '\n' || buf[bl - 1] == '\r'))
        {
            buf[--bl] = '\0';
        }
    }

    /* ---- Array split mode (-a) ---- */
    if (arrayname != NULL)
    {
        int   ai  = 0;
        char *tok = strtok(buf, " \t");
        while (tok != NULL && ai < CLI_ARRAY_MAXELEM)
        {
            char aelem[CLI_VAR_NAMELEN];
            snprintf(aelem, sizeof(aelem), "%s[%d]", arrayname, ai);
            cli_var_set(aelem, tok);
            ai++;
            tok = strtok(NULL, " \t");
        }
        char aelem[CLI_VAR_NAMELEN];
        snprintf(aelem, sizeof(aelem), "%s[#]", arrayname);
        char acnt[16];
        snprintf(acnt, sizeof(acnt), "%d", ai);
        cli_var_set(aelem, acnt);
    }
    else
    {
        cli_var_set(varname, buf);
    }

    cli_var_set("?", "0");
    return RETURN_SUCCESS;
}


/* ============================================================
 *  export command
 * ============================================================
 */

/**
 * @brief export command — copy CLI var to environ
 *
 * Marks a variable for export to the environment of
 * subsequently spawned processes. Supports inline
 * assignment syntax (export VAR=value).
 *
 * Usage: export <varname>[=value]
 */
errno_t cli_cmd_export(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: export <varname>"
               "[=value]\n");
        return RETURN_FAILURE;
    }

    const char *arg = data.cmdargtoken[1].val.string;

    char vname[CLI_VAR_NAMELEN];
    strncpy(vname, arg, CLI_VAR_NAMELEN - 1);
    vname[CLI_VAR_NAMELEN - 1] = '\0';
    char *eq                   = strchr(vname, '=');

    if (eq != NULL)
    {
        *eq             = '\0';
        const char *val = eq + 1;
        cli_var_set(vname, val);
        setenv(vname, val, 1);
    }
    else
    {
        const char *val = cli_var_get(vname);
        if (val != NULL)
        {
            setenv(vname, val, 1);
        }
        else
        {
            printf("export: variable '%s'"
                   " not set\n",
                   vname);
            return RETURN_FAILURE;
        }
    }

    return RETURN_SUCCESS;
}


/* ============================================================
 *  shift command
 * ============================================================
 */

/**
 * @brief shift command — rotate positional params
 *
 * Shifts $1..$9 left by N positions (default 1),
 * discarding the leftmost N values and making
 * the remainder accessible from $1 onward.
 *
 * Usage: shift [N]
 */
errno_t cli_cmd_shift(void)
{
    int n = 1;
    if (data.cmdNBarg >= 2)
    {
        n = (int) data.cmdargtoken[1].val.numl;
    }
    if (n < 1)
    {
        n = 1;
    }

    /* Read current $1..$N */
    char vals[CLI_FUNC_MAXARGS][CLI_VAR_VALLEN];
    for (int i = 0; i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname), "%d", i + 1);
        const char *v = cli_var_get(aname);
        if (v != NULL)
        {
            strncpy(vals[i], v, CLI_VAR_VALLEN - 1);
            vals[i][CLI_VAR_VALLEN - 1] = '\0';
        }
        else
        {
            vals[i][0] = '\0';
        }
    }

    /* Shift left by n */
    for (int i = 0; i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname), "%d", i + 1);
        int src = i + n;
        if (src < CLI_FUNC_MAXARGS && vals[src][0] != '\0')
        {
            cli_var_set(aname, vals[src]);
        }
        else
        {
            cli_var_unset(aname);
        }
    }

    return RETURN_SUCCESS;
}


/* ============================================================
 *  printf command
 * ============================================================
 */

/**
 * @brief printf command — formatted output
 *
 * Processes a format string token-by-token, expanding
 * backslash sequences (\n, \t, \\) and printf-style
 * format specifiers (%s, %d, %f, etc.) against the
 * remaining argument tokens.
 *
 * Usage: printf <format> [args...]
 */
errno_t cli_cmd_printf(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: printf <format>"
               " [args...]\n");
        return RETURN_FAILURE;
    }

    const char *fmt = data.cmdargtoken[1].val.string;
    int         ai  = 2;

    for (const char *p = fmt; *p != '\0'; p++)
    {
        /* Backslash escapes */
        if (*p == '\\' && *(p + 1) != '\0')
        {
            p++;
            switch (*p)
            {
            case 'n':
                putchar('\n');
                break;
            case 't':
                putchar('\t');
                break;
            case '\\':
                putchar('\\');
                break;
            default:
                putchar('\\');
                putchar(*p);
                break;
            }
            continue;
        }

        /* Format specifiers */
        if (*p == '%')
        {
            const char *q = p + 1;

            if (*q == '%')
            {
                putchar('%');
                p = q;
                continue;
            }
            if (*q == '\0')
            {
                break;
            }

            /* Accept only: flags, numeric width, numeric precision,
             * and conversions in diouxXfFeEgGsc.
             * Reject '*', length modifiers, and unknown conversions.
             */
            char spec[32];
            int  si = 0;
            char fc;
            int  valid = 1;

            spec[si++] = '%';

            while (*q != '\0' && strchr("-+ #0", *q) != NULL && si < (int) sizeof(spec) - 2)
            {
                spec[si++] = *q++;
            }

            if (*q == '*')
            {
                valid = 0;
            }

            while (valid && *q != '\0' && *q >= '0' && *q <= '9' && si < (int) sizeof(spec) - 2)
            {
                spec[si++] = *q++;
            }

            if (valid && *q == '.')
            {
                if (si >= (int) sizeof(spec) - 2)
                {
                    valid = 0;
                }
                else
                {
                    spec[si++] = *q++;
                }

                if (valid && *q == '*')
                {
                    valid = 0;
                }

                while (valid && *q != '\0' && *q >= '0' && *q <= '9' && si < (int) sizeof(spec) - 2)
                {
                    spec[si++] = *q++;
                }
            }

            if (valid && *q != '\0' && strchr("hlLjzt", *q) != NULL)
            {
                valid = 0;
            }

            if (!valid || *q == '\0')
            {
                putchar('%');
                continue;
            }

            fc = *q;
            if (strchr("diouxXfFeEgGsc", fc) == NULL)
            {
                putchar('%');
                continue;
            }

            spec[si++] = fc;
            spec[si]   = '\0';

            const char *aval = "";
            if (ai < data.cmdNBarg)
            {
                aval = data.cmdargtoken[ai].val.string;
                ai++;
            }

            {
                char outbuf[256];
                int  nw = -1;

                if (fc == 's')
                {
                    nw = snprintf(outbuf, sizeof(outbuf), spec, aval);
                }
                else if (fc == 'c')
                {
                    nw = snprintf(outbuf, sizeof(outbuf), spec, (unsigned char) aval[0]);
                }
                else if (fc == 'd' || fc == 'i' || fc == 'o' || fc == 'u' || fc == 'x' || fc == 'X')
                {
                    long lv = strtol(aval, NULL, 0);
                    nw      = snprintf(outbuf, sizeof(outbuf), spec, lv);
                }
                else
                {
                    double dv = strtod(aval, NULL);
                    nw        = snprintf(outbuf, sizeof(outbuf), spec, dv);
                }

                if (nw >= 0)
                {
                    fputs(outbuf, stdout);
                }
            }

            p = q;
            continue;
        }

        putchar(*p);
    }

    fflush(stdout);
    return RETURN_SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"


/**
 * @brief Execute trap handlers for signal
 */
void cli_trap_run(int signum)
{
    for(int i = 0;
        i < CLI_TRAP_MAXSIGS; i++)
    {
        if(cli_traps[i].used
           && cli_traps[i].signum
           == signum)
        {
            strncpy(
                data.CLIcmdline,
                cli_traps[i].cmd,
                STRINGMAXLEN_CLICMDLINE
                - 1);
            CLI_execute_line();
        }
    }
}

/**
 * @brief Run EXIT traps (signal 0)
 */
void cli_trap_run_exit(void)
{
    cli_trap_run(0);
}

/**
 * @brief echo command — print arguments
 *
 * Note: echo is intercepted before tokenization
 * in CLIcore_UI.c. This registered version is a
 * fallback that should rarely be called.
 */
errno_t cli_cmd_echo(void)
{
    int start = 1;
    int newline = 1;

    if(data.cmdNBarg >= 2
            && strcmp(
                data.cmdargtoken[1].val.string,
                "-n") == 0)
    {
        newline = 0;
        start = 2;
    }

    for(int i = start; i < data.cmdNBarg; i++)
    {
        if(i > start)
        {
            printf(" ");
        }
        printf("%s",
               data.cmdargtoken[i].val.string);
    }
    if(newline)
    {
        printf("\n");
    }
    return RETURN_SUCCESS;
}


/* ============================================================
 *  FPS Write — reusable helper and fpsset command
 * ============================================================
 */

#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_printparameter_valuestring.h"
#include "fps_connect.h"

/**
 * @brief Set an FPS parameter by name
 *
 * Connects to the named FPS, locates the
 * parameter, writes the value string, and
 * disconnects.  Reused by both the fpsset
 * command and the @fpsname.param=value
 * expansion syntax.
 *
 * @param fpsname  FPS shared-memory name
 * @param pname    Parameter key inside FPS
 * @param valstr   Value to write (as string)
 * @return 0 on success, -1 on error
 */
int cli_fps_set_param(
    const char *fpsname,
    const char *pname,
    const char *valstr
)
{
    FUNCTION_PARAMETER_STRUCT fps;
    int fpsconn =
        function_parameter_struct_connect(
            fpsname, &fps,
            FPSCONNECT_SIMPLE);

    if(fpsconn == -1
       || fps.parray == NULL)
    {
        printf("Error: cannot connect to "
               "FPS '%s'\n", fpsname);
        return -1;
    }

    int pindex =
        functionparameter_GetParamIndex(
            &fps, pname);

    if(pindex < 0)
    {
        char dotname[512];
        snprintf(dotname, sizeof(dotname),
                 ".%s", pname);
        pindex =
            functionparameter_GetParamIndex(
                &fps, dotname);
    }

    if(pindex < 0)
    {
        printf("Error: parameter '%s' not "
               "found in FPS '%s'\n",
               pname, fpsname);
        function_parameter_struct_disconnect(
            &fps);
        return -1;
    }

    uint32_t ptype =
        fps.parray[pindex].type;

    if(ptype & FPTYPE_INT64)
    {
        fps.parray[pindex].val.i64[0] =
            (int64_t) strtol(
                valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if(ptype & FPTYPE_FLOAT64)
    {
        fps.parray[pindex].val.f64[0] =
            strtod(valstr, NULL);
        fps.parray[pindex].cnt0++;
    }
    else if(ptype & FPTYPE_FLOAT32)
    {
        fps.parray[pindex].val.f32[0] =
            (float) strtod(valstr, NULL);
        fps.parray[pindex].cnt0++;
    }
    else if(ptype & FPTYPE_ONOFF)
    {
        if(strcmp(valstr, "ON") == 0
           || strcmp(valstr, "on") == 0
           || strcmp(valstr, "1") == 0)
        {
            fps.parray[pindex]
                .val.i64[0] = 1;
        }
        else
        {
            fps.parray[pindex]
                .val.i64[0] = 0;
        }
        fps.parray[pindex].cnt0++;
    }
    else if(ptype & FPTYPE_STRING)
    {
        strncpy(
            fps.parray[pindex]
                .val.string[0],
            valstr,
            FUNCTION_PARAMETER_STRMAXLEN
            - 1);
        fps.parray[pindex]
            .val.string[0][
                FUNCTION_PARAMETER_STRMAXLEN
                - 1] = '\0';
        fps.parray[pindex].cnt0++;
    }
    else
    {
        printf("Warning: unsupported param "
               "type 0x%x\n", ptype);
    }

    function_parameter_struct_disconnect(
        &fps);
    return 0;
}

/**
 * @brief fpsset command — write FPS parameter
 *
 * Usage: fpsset fpsname.param value
 */
errno_t cli_cmd_fpsset(void)
{
    if(data.cmdNBarg < 3)
    {
        printf("Usage: fpsset "
               "<fpsname.param> <value>\n");
        return RETURN_FAILURE;
    }

    const char *fullname =
        data.cmdargtoken[1].val.string;
    const char *valstr =
        data.cmdargtoken[2].val.string;

    char fpsname[256];
    strncpy(fpsname, fullname,
            sizeof(fpsname) - 1);
    fpsname[sizeof(fpsname) - 1] = '\0';

    char *dot = strchr(fpsname, '.');
    if(dot == NULL)
    {
        printf("Error: fpsset requires "
               "fpsname.paramname\n");
        return RETURN_FAILURE;
    }
    *dot = '\0';
    const char *pname = dot + 1;

    if(cli_fps_set_param(
           fpsname, pname, valstr) != 0)
    {
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  read — interactive variable input
 * ============================================================
 */

/**
 * @brief read command — read a line into a variable
 *
 * Usage: read [-p "prompt"] <varname>
 */
errno_t cli_cmd_read(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: read [-p \"prompt\"]"
               " <varname>\n");
        return RETURN_FAILURE;
    }

    const char *prompt = "";
    int varidx = 1;

    /* Parse -p flag */
    if(data.cmdNBarg >= 4
       && strcmp(
           data.cmdargtoken[1].val.string,
           "-p") == 0)
    {
        prompt =
            data.cmdargtoken[2].val.string;
        varidx = 3;
    }

    const char *varname =
        data.cmdargtoken[varidx].val.string;

    /* Display prompt and read */
    if(prompt[0] != '\0')
    {
        printf("%s", prompt);
        fflush(stdout);
    }
    char buf[CLI_VAR_VALLEN];
    if(fgets(buf, sizeof(buf),
             stdin) == NULL)
    {
        buf[0] = '\0';
    }
    {
        size_t len = strlen(buf);
        if(len > 0 && buf[len - 1] == '\n')
        {
            buf[len - 1] = '\0';
        }
    }

    cli_var_set(varname, buf);
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  export — push CLI variable to environment
 * ============================================================
 */

/**
 * @brief export command — copy CLI var to environ
 *
 * Usage: export <varname>
 *        export <varname>=<value>
 */
errno_t cli_cmd_export(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: export <varname>"
               "[=value]\n");
        return RETURN_FAILURE;
    }

    const char *arg =
        data.cmdargtoken[1].val.string;

    /* Check for inline assignment */
    char vname[CLI_VAR_NAMELEN];
    strncpy(vname, arg,
            CLI_VAR_NAMELEN - 1);
    vname[CLI_VAR_NAMELEN - 1] = '\0';
    char *eq = strchr(vname, '=');

    if(eq != NULL)
    {
        *eq = '\0';
        const char *val = eq + 1;
        cli_var_set(vname, val);
        setenv(vname, val, 1);
    }
    else
    {
        const char *val =
            cli_var_get(vname);
        if(val != NULL)
        {
            setenv(vname, val, 1);
        }
        else
        {
            printf("export: variable '%s'"
                   " not set\n", vname);
            return RETURN_FAILURE;
        }
    }

    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  shift — rotate positional parameters
 * ============================================================
 */

/**
 * @brief shift command — rotate $1..$9 left
 *
 * Usage: shift [N]
 * Shifts positional parameters left by N
 * (default 1).
 */
errno_t cli_cmd_shift(void)
{
    int n = 1;
    if(data.cmdNBarg >= 2)
    {
        n = (int) data.cmdargtoken[1]
                .val.numl;
    }
    if(n < 1)
    {
        n = 1;
    }

    /* Read current $1..$9 */
    char vals[CLI_FUNC_MAXARGS][
        CLI_VAR_VALLEN];
    for(int i = 0;
        i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname),
                 "%d", i + 1);
        const char *v = cli_var_get(aname);
        if(v != NULL)
        {
            strncpy(vals[i], v,
                    CLI_VAR_VALLEN - 1);
            vals[i][CLI_VAR_VALLEN - 1] =
                '\0';
        }
        else
        {
            vals[i][0] = '\0';
        }
    }

    /* Shift left by n */
    for(int i = 0;
        i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname),
                 "%d", i + 1);
        int src = i + n;
        if(src < CLI_FUNC_MAXARGS
           && vals[src][0] != '\0')
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


/*
 * ============================================================
 *  printf — native format string output
 * ============================================================
 */

/**
 * @brief printf command — formatted output
 *
 * Supports: %s, %d, %ld, %f, %g, %e,
 * %06d, %.3f, %%, \n, \t, \\.
 *
 * Usage: printf <format> [args...]
 */
errno_t cli_cmd_printf(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: printf <format>"
               " [args...]\n");
        return RETURN_FAILURE;
    }

    const char *fmt =
        data.cmdargtoken[1].val.string;
    int ai = 2; /* argument index */

    for(const char *p = fmt; *p != '\0'; p++)
    {
        /* Backslash escapes */
        if(*p == '\\' && *(p + 1) != '\0')
        {
            p++;
            switch(*p)
            {
                case 'n':
                    putchar('\n'); break;
                case 't':
                    putchar('\t'); break;
                case '\\':
                    putchar('\\'); break;
                default:
                    putchar('\\');
                    putchar(*p);
                    break;
            }
            continue;
        }

        /* Format specifiers */
        if(*p == '%')
        {
            p++;
            if(*p == '%')
            {
                putchar('%');
                continue;
            }
            if(*p == '\0')
            {
                break;
            }

            /* Collect full specifier */
            char spec[32];
            int si = 0;
            spec[si++] = '%';
            while(*p != '\0' && si < 30
                  && strchr(
                      "diouxXeEfFgGscpl",
                      *p) == NULL)
            {
                spec[si++] = *p++;
            }
            if(*p == '\0')
            {
                break;
            }
            spec[si++] = *p;
            spec[si] = '\0';

            const char *aval = "";
            if(ai < data.cmdNBarg)
            {
                aval = data.cmdargtoken[ai]
                           .val.string;
                ai++;
            }

            /* Dispatch by final char */
            char fc = spec[si - 1];
            if(fc == 's' || fc == 'c')
            {
                printf(spec, aval);
            }
            else if(fc == 'd' || fc == 'i'
                    || fc == 'o' || fc == 'u'
                    || fc == 'x' || fc == 'X')
            {
                long lv = strtol(aval,
                                 NULL, 0);
                printf(spec, lv);
            }
            else if(fc == 'f' || fc == 'F'
                    || fc == 'e' || fc == 'E'
                    || fc == 'g' || fc == 'G')
            {
                double dv = strtod(aval,
                                   NULL);
                printf(spec, dv);
            }
            else if(fc == 'l')
            {
                /* %ld -> parse as long */
                long lv = strtol(aval,
                                 NULL, 0);
                printf(spec, lv);
            }
            else
            {
                printf("%s", aval);
            }
            continue;
        }

        putchar(*p);
    }

    fflush(stdout);
    return RETURN_SUCCESS;
}
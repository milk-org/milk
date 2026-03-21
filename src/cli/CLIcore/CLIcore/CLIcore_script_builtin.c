#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "CLIcore.h"
#include "CLIcore_UI.h"
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
 *  FPS Write — fpsset command
 * ============================================================
 */

#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_printparameter_valuestring.h"
#include "fps_connect.h"

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

    /* Split fpsname.param at first dot */
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

    /* Connect to FPS */
    FUNCTION_PARAMETER_STRUCT fps;
    int fpsconn =
        function_parameter_struct_connect(
            fpsname, &fps, FPSCONNECT_SIMPLE);

    if(fpsconn == -1 || fps.parray == NULL)
    {
        printf("Error: cannot connect to "
               "FPS '%s'\n", fpsname);
        return RETURN_FAILURE;
    }

    /* Find parameter index */
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
        return RETURN_FAILURE;
    }

    /* Set value based on parameter type */
    uint32_t ptype =
        fps.parray[pindex].type;

    if(ptype & FPTYPE_INT64)
    {
        fps.parray[pindex].val.i64[0] =
            (int64_t) strtol(valstr, NULL, 0);
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
            fps.parray[pindex].val.i64[0] = 1;
        }
        else
        {
            fps.parray[pindex].val.i64[0] = 0;
        }
        fps.parray[pindex].cnt0++;
    }
    else if(ptype & FPTYPE_STRING)
    {
        strncpy(
            fps.parray[pindex].val.string[0],
            valstr,
            FUNCTION_PARAMETER_STRMAXLEN - 1);
        fps.parray[pindex].val.string[0][
            FUNCTION_PARAMETER_STRMAXLEN - 1]
            = '\0';
        fps.parray[pindex].cnt0++;
    }
    else
    {
        printf("Warning: unsupported param "
               "type 0x%x\n", ptype);
    }

    function_parameter_struct_disconnect(&fps);
    return RETURN_SUCCESS;
}
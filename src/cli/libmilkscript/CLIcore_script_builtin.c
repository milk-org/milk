#include <dirent.h>
#include <fnmatch.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"
#include "ImageStreamIO/ImageStreamIO.h"

/* processinfo functions — linked
 * via milkprocessinfo */
extern PROCESSINFO *processinfo_shm_link(
    const char *pname, int *fd);
extern errno_t processinfo_procdirname(
    char *procdname);


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
    cli_defer_run();
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
 * Usage: read [-p "prompt"] [-t N] [-n N]
 *             [-a arr] <varname>
 *
 * Flags:
 *   -p prompt   Display prompt string
 *   -t N        Timeout after N seconds
 *   -a arr      Split input into array
 *   -n N        Read exactly N characters
 *               (raw mode, no Enter needed)
 *
 * On timeout, $? is set to 1 and the variable
 * is not modified.
 */
errno_t cli_cmd_read(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: read [-p prompt]"
               " [-t N] [-n N]"
               " [-a arr] <var>\n");
        return RETURN_FAILURE;
    }

    const char *prompt = "";
    int timeout_sec = -1;  /* -1 = no timeout */
    int nchars = -1;       /* -1 = full line  */
    const char *arrayname = NULL;
    const char *varname = NULL;

    /* ---- Parse flags ---- */
    {
        int i = 1;
        while(i < data.cmdNBarg)
        {
            const char *tok =
                data.cmdargtoken[i]
                    .val.string;

            if(strcmp(tok, "-p") == 0)
            {
                i++;
                if(i < data.cmdNBarg)
                {
                    prompt =
                        data.cmdargtoken[i]
                            .val.string;
                }
            }
            else if(strcmp(tok, "-t") == 0)
            {
                i++;
                if(i < data.cmdNBarg)
                {
                    timeout_sec = (int)
                        strtol(
                            data.cmdargtoken[i]
                                .val.string,
                            NULL, 10);
                }
            }
            else if(strcmp(tok, "-n") == 0)
            {
                i++;
                if(i < data.cmdNBarg)
                {
                    nchars = (int)
                        strtol(
                            data.cmdargtoken[i]
                                .val.string,
                            NULL, 10);
                }
            }
            else if(strcmp(tok, "-a") == 0)
            {
                i++;
                if(i < data.cmdNBarg)
                {
                    arrayname =
                        data.cmdargtoken[i]
                            .val.string;
                }
            }
            else
            {
                /* First non-flag is varname */
                varname = tok;
            }
            i++;
        }
    }

    /* Need either a var or an array target */
    if(varname == NULL && arrayname == NULL)
    {
        printf("Usage: read [-p prompt]"
               " [-t N] [-n N]"
               " [-a arr] <var>\n");
        return RETURN_FAILURE;
    }

    /* ---- Display prompt ---- */
    if(prompt[0] != '\0')
    {
        printf("%s", prompt);
        fflush(stdout);
    }

    char buf[CLI_VAR_VALLEN];
    buf[0] = '\0';

    /* ---- Timed read with poll() ---- */
    if(timeout_sec >= 0)
    {
        struct pollfd pfd;
        pfd.fd = STDIN_FILENO;
        pfd.events = POLLIN;

        int tmout_ms = timeout_sec * 1000;
        int ret = poll(&pfd, 1, tmout_ms);

        if(ret <= 0)
        {
            /* Timeout or error */
            cli_last_retval = 1;
            return RETURN_SUCCESS;
        }
    }

    /* ---- Raw-mode read (-n nchars) ---- */
    if(nchars > 0)
    {
        struct termios oldt;
        struct termios newt;
        int tty = isatty(STDIN_FILENO);

        if(tty)
        {
            tcgetattr(STDIN_FILENO, &oldt);
            newt = oldt;
            newt.c_lflag &=
                (tcflag_t) ~(ICANON | ECHO);
            newt.c_cc[VMIN] = 1;
            newt.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO,
                      TCSANOW, &newt);
        }

        int cap = nchars;
        if(cap >= (int) sizeof(buf))
        {
            cap = (int) sizeof(buf) - 1;
        }

        ssize_t nr = read(
            STDIN_FILENO, buf,
            (size_t) cap);

        if(nr < 0)
        {
            nr = 0;
        }
        buf[nr] = '\0';

        if(tty)
        {
            tcsetattr(STDIN_FILENO,
                      TCSANOW, &oldt);
            /* Print newline after raw read
             * for clean terminal output */
            printf("\n");
        }
    }
    else
    {
        /* ---- Normal line read ---- */
        if(fgets(buf, sizeof(buf),
                 stdin) == NULL)
        {
            buf[0] = '\0';
        }
        {
            size_t len = strlen(buf);
            if(len > 0
               && buf[len - 1] == '\n')
            {
                buf[len - 1] = '\0';
            }
        }
    }

    /* ---- Store result ---- */
    if(arrayname != NULL)
    {
        /* Split buf by whitespace into array */
        int slot = -1;
        for(int i = 0;
            i < CLI_MAX_ARRAYS; i++)
        {
            if(cli_arrays[i].used
               && strcmp(cli_arrays[i].name,
                         arrayname) == 0)
            {
                slot = i;
                break;
            }
        }
        if(slot < 0)
        {
            for(int i = 0;
                i < CLI_MAX_ARRAYS; i++)
            {
                if(!cli_arrays[i].used)
                {
                    slot = i;
                    break;
                }
            }
        }
        if(slot < 0)
        {
            printf("Error: array table"
                   " full\n");
            return RETURN_FAILURE;
        }

        strncpy(cli_arrays[slot].name,
                arrayname,
                CLI_VAR_NAMELEN - 1);
        cli_arrays[slot].name[
            CLI_VAR_NAMELEN - 1] = '\0';
        cli_arrays[slot].used = 1;
        cli_arrays[slot].nelem = 0;

        char *saveptr = NULL;
        char *word = strtok_r(
            buf, " \t", &saveptr);

        while(word != NULL)
        {
            int idx =
                cli_arrays[slot].nelem;
            if(idx >= CLI_ARRAY_MAXELEM)
            {
                break;
            }
            strncpy(
                cli_arrays[slot]
                    .elem[idx],
                word,
                CLI_VAR_VALLEN - 1);
            cli_arrays[slot]
                .elem[idx][
                    CLI_VAR_VALLEN - 1]
                = '\0';
            cli_arrays[slot].nelem++;
            word = strtok_r(
                NULL, " \t", &saveptr);
        }
    }
    else
    {
        cli_var_set(varname, buf);
    }

    cli_last_retval = 0;
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

/*
 * ============================================================
 *  fpslist — list live FPS instances
 * ============================================================
 */

#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_shmdirname.h"

/**
 * @brief fpslist command — list live FPS instances
 *
 * Scans the SHM directory for *.fps.shm files,
 * connects to each, and prints a summary table
 * showing name, status and description.
 *
 * Usage: fpslist [pattern]
 *
 * An optional glob pattern (e.g. "dm*") limits
 * output to matching FPS names.
 */
errno_t cli_cmd_fpslist(void)
{
    char shmdname[STRINGMAXLEN_SHMDIRNAME];
    function_parameter_struct_shmdirname(
        shmdname);

    /* Optional glob pattern from $1
     * (cmdNBarg includes the command itself,
     *  so >= 2 means one argument was given) */
    const char *pat = NULL;
    if(data.cmdNBarg >= 2)
    {
        pat =
            data.cmdargtoken[1].val.string;
    }

    /* Header */
    printf("%-24s %-12s %s\n",
           "FPS NAME", "STATUS",
           "DESCRIPTION");
    printf("%-24s %-12s %s\n",
           "------------------------",
           "------------",
           "--------------------"
           "--------------------");

    DIR           *d;
    struct dirent *de;
    d = opendir(shmdname);
    if(d == NULL)
    {
        printf("Cannot open SHM dir: %s\n",
               shmdname);
        return RETURN_FAILURE;
    }

    while((de = readdir(d)) != NULL)
    {
        /* Only process *.fps.shm files */
        char *sfx = strstr(de->d_name,
                           ".fps.shm");
        if(sfx == NULL)
        {
            continue;
        }

        /* Extract FPS name */
        char fpsname[STRINGMAXLEN_FPS_NAME];
        size_t nlen = (size_t)(sfx
                               - de->d_name);
        if(nlen >= sizeof(fpsname))
        {
            continue;
        }
        strncpy(fpsname, de->d_name, nlen);
        fpsname[nlen] = '\0';

        /* Apply glob filter when provided */
        if(pat != NULL
           && fnmatch(pat, fpsname, 0) != 0)
        {
            continue;
        }

        /* Connect to FPS */
        FUNCTION_PARAMETER_STRUCT fps;
        fps.SMfd = -1;
        int rc =
            function_parameter_struct_connect(
                fpsname, &fps,
                FPSCONNECT_SIMPLE);
        if(rc == -1 || fps.md == NULL)
        {
            printf("%-24s %-12s %s\n",
                   fpsname,
                   "UNAVAIL", "");
            continue;
        }

        /* Build status string */
        uint32_t st = fps.md->status;
        /* longest value: "CONF_ON\0" = 8 */
        char ststr[16];
        if(st & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
        {
            strncpy(ststr, "RUN",
                    sizeof(ststr) - 1);
        }
        else if(st
                & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
        {
            strncpy(ststr, "CONF_ON",
                    sizeof(ststr) - 1);
        }
        else
        {
            strncpy(ststr, "IDLE",
                    sizeof(ststr) - 1);
        }
        ststr[sizeof(ststr) - 1] = '\0';

        printf("%-24s %-12s %s\n",
               fpsname,
               ststr,
               fps.md->description);

        function_parameter_struct_disconnect(
            &fps);
    }
    closedir(d);

    return RETURN_SUCCESS;
}


/* ============================================================
 *  fpsdump — dump all params of an FPS as key=value
 * ============================================================
 */

/**
 * @brief fpsdump command — dump all parameters
 *
 * Connects to the named FPS and prints every
 * active parameter as key=value lines.
 *
 * Usage: fpsdump [-t|--json] <fpsname>
 *   -t      tab-separated: key\tTYPE\tvalue
 *   --json  JSON object with raw typed values
 */
errno_t cli_cmd_fpsdump(void)
{
    int tabmode = 0;
    int jsonmode = 0;
    int arg_idx = 1;

    if(data.cmdNBarg < 2)
    {
        printf("Usage: fpsdump [-t|--json] "
               "<fpsname>\n");
        return RETURN_FAILURE;
    }

    if(data.cmdNBarg >= 3)
    {
        if(strcmp(data.cmdargtoken[1].val.string, "-t") == 0)
        {
            tabmode = 1;
            arg_idx = 2;
        }
        else if(strcmp(data.cmdargtoken[1].val.string, "--json") == 0)
        {
            jsonmode = 1;
            arg_idx = 2;
        }
    }

    const char *fpsname =
        data.cmdargtoken[arg_idx].val.string;

    FUNCTION_PARAMETER_STRUCT fps;
    fps.SMfd = -1;
    int rc =
        function_parameter_struct_connect(
            fpsname, &fps,
            FPSCONNECT_SIMPLE);
    if(rc == -1 || fps.md == NULL
       || fps.parray == NULL)
    {
        printf("fpsdump: cannot connect "
               "to FPS '%s'\n", fpsname);
        return RETURN_FAILURE;
    }

    if(jsonmode) { printf("{\n"); }
    int first_json_item = 1;

    for(int pi = 0;
        pi < fps.md->NBparamMAX; pi++)
    {
        if(!(fps.parray[pi].fpflag
             & FPFLAG_ACTIVE))
        {
            continue;
        }
        char vstr[512];
        functionparameter_GetParamValueString(
            &fps.parray[pi],
            vstr,
            (int) sizeof(vstr));

        if(tabmode)
        {
            /* Type name from type enum */
            const char *tname = "UNKNOWN";
            switch(fps.parray[pi].type)
            {
            case FPTYPE_INT64:
                tname = "INT64";
                break;
            case FPTYPE_FLOAT64:
                tname = "FLOAT64";
                break;
            case FPTYPE_FLOAT32:
                tname = "FLOAT32";
                break;
            case FPTYPE_STRING:
                tname = "STRING";
                break;
            case FPTYPE_ONOFF:
                tname = "ONOFF";
                break;
            case FPTYPE_FILENAME:
                tname = "FILENAME";
                break;
            case FPTYPE_FITSFILENAME:
                tname = "FITSFILENAME";
                break;
            case FPTYPE_EXECFILENAME:
                tname = "EXECFILENAME";
                break;
            case FPTYPE_DIRNAME:
                tname = "DIRNAME";
                break;
            case FPTYPE_STREAMNAME:
                tname = "STREAMNAME";
                break;
            case FPTYPE_FPSNAME:
                tname = "FPSNAME";
                break;
            case FPTYPE_TIMESPEC:
                tname = "TIMESPEC";
                break;
            case FPTYPE_PID:
                tname = "PID";
                break;
            default:
                break;
            }
            printf("%s\t%s\t%s\n",
                   fps.parray[pi].keyword[0],
                   tname, vstr);
        }
        else if(jsonmode)
        {
            if(!first_json_item) { printf(",\n"); }
            first_json_item = 0;
            switch(fps.parray[pi].type)
            {
            case FPTYPE_INT64:
                printf("  \"%s\": %lld", fps.parray[pi].keyword[0], (long long)fps.parray[pi].val.i64[0]);
                break;
            case FPTYPE_FLOAT64:
                printf("  \"%s\": %g", fps.parray[pi].keyword[0], fps.parray[pi].val.f64[0]);
                break;
            case FPTYPE_FLOAT32:
                printf("  \"%s\": %g", fps.parray[pi].keyword[0], fps.parray[pi].val.f32[0]);
                break;
            case FPTYPE_ONOFF:
                printf("  \"%s\": %d", fps.parray[pi].keyword[0], (int)fps.parray[pi].val.i64[0]);
                break;
            default:
                printf("  \"%s\": \"%s\"", fps.parray[pi].keyword[0], vstr);
                break;
            }
        }
        else
        {
            printf("%s=%s\n",
                   fps.parray[pi].keyword[0],
                   vstr);
        }
    }

    if(jsonmode) { printf("\n}\n"); }

    function_parameter_struct_disconnect(&fps);

    return RETURN_SUCCESS;
}


/* ============================================================
 *  streamlist — enumerate live SHM streams
 * ============================================================
 */

/**
 * @brief streamlist command — list live streams
 *
 * Scans SHM directory for *.im.shm files.
 *
 * Usage: streamlist [-l|--json] [pattern]
 *   -l       long format: name WxH type cnt0
 *   --json   JSON array of stream metadata
 *   pattern  glob filter (e.g. "dm*")
 */
errno_t cli_cmd_streamlist(void)
{
    int longmode = 0;
    int jsonmode = 0;
    const char *pat = NULL;
    int argpos = 1;

    /* Parse flags */
    for(int a = 1; a < data.cmdNBarg; a++)
    {
        const char *tok =
            data.cmdargtoken[a].val.string;
        if(strcmp(tok, "-l") == 0)
        {
            longmode = 1;
            argpos = a + 1;
        }
        else if(strcmp(tok, "--json") == 0)
        {
            jsonmode = 1;
            argpos = a + 1;
        }
    }
    if(argpos < data.cmdNBarg)
    {
        pat =
            data.cmdargtoken[argpos]
                .val.string;
    }

    const char *shmdname = dcshmdir;

    DIR           *d;
    struct dirent *de;
    d = opendir(shmdname);
    if(d == NULL)
    {
        printf("Cannot open SHM dir: %s\n",
               shmdname);
        return RETURN_FAILURE;
    }

    if(jsonmode) { printf("[\n"); }
    int first_json_item = 1;

    while((de = readdir(d)) != NULL)
    {
        char *sfx = strstr(de->d_name,
                           ".im.shm");
        if(sfx == NULL)
        {
            continue;
        }
        /* Skip FPS files */
        if(strstr(de->d_name,
                  ".fps.shm") != NULL)
        {
            continue;
        }

        char sname[256];
        size_t nlen = (size_t)(sfx
                               - de->d_name);
        if(nlen >= sizeof(sname))
        {
            continue;
        }
        strncpy(sname, de->d_name, nlen);
        sname[nlen] = '\0';

        if(pat != NULL
           && fnmatch(pat, sname, 0) != 0)
        {
            continue;
        }

        if(!longmode && !jsonmode)
        {
            printf("%s\n", sname);
        }
        else
        {
            IMAGE img;
            memset(&img, 0, sizeof(IMAGE));
            errno_t sret =
                ImageStreamIO_openIm(
                    &img, sname);
            if(sret == IMAGESTREAMIO_SUCCESS
               && img.md != NULL)
            {
                if(jsonmode)
                {
                    if(!first_json_item) { printf(",\n"); }
                    first_json_item = 0;
                    
                    printf("  {\n");
                    printf("    \"name\": \"%s\",\n", sname);
                    printf("    \"naxis\": %u,\n", img.md->naxis);
                    
                    printf("    \"size\": [");
                    for(int ax = 0; ax < img.md->naxis; ax++)
                    {
                        if(ax > 0) printf(", ");
                        printf("%u", img.md->size[ax]);
                    }
                    printf("],\n");
                    
                    printf("    \"type\": \"%s\",\n", ImageStreamIO_typename(img.md->datatype));
                    printf("    \"cnt0\": %lu\n", (unsigned long)img.md->cnt0);
                    printf("  }");
                }
                else
                {
                    /* Build size string */
                char szstr[64];
                if(img.md->naxis == 1)
                {
                    snprintf(szstr,
                        sizeof(szstr),
                        "%u",
                        img.md->size[0]);
                }
                else if(img.md->naxis == 2)
                {
                    snprintf(szstr,
                        sizeof(szstr),
                        "%ux%u",
                        img.md->size[0],
                        img.md->size[1]);
                }
                else
                {
                    snprintf(szstr,
                        sizeof(szstr),
                        "%ux%ux%u",
                        img.md->size[0],
                        img.md->size[1],
                        img.md->size[2]);
                }
                printf("%-24s %-12s %-7s "
                       "cnt0=%lu\n",
                    sname, szstr,
                    ImageStreamIO_typename(
                        img.md->datatype),
                    (unsigned long)
                        img.md->cnt0);
                }
                ImageStreamIO_closeIm(&img);
            }
            else
            {
                if(!jsonmode) {
                    printf("%-24s UNAVAIL\n",
                           sname);
                }
            }
        }
    }
    
    if(jsonmode) { printf("\n]\n"); }
    closedir(d);

    return RETURN_SUCCESS;
}


/* ============================================================
 *  proclist — enumerate active processes
 * ============================================================
 */

/**
 * @brief proclist command — list active procs
 *
 * Iterates the processinfo list and prints
 * active process names, one per line.
 *
 * Usage: proclist [-l] [--json]
 *   -l      long format: name  state  freq
 *   --json  JSON array of process metadata
 */
errno_t cli_cmd_proclist(void)
{
    int longmode = 0;
    int jsonmode = 0;
    for(int a = 1; a < data.cmdNBarg; a++)
    {
        if(strcmp(data.cmdargtoken[a].val.string, "-l") == 0)
        {
            longmode = 1;
        }
        else if(strcmp(data.cmdargtoken[a].val.string, "--json") == 0)
        {
            jsonmode = 1;
        }
    }

    if(pinfolist == NULL)
    {
        printf("proclist: processinfo "
               "not available\n");
        return RETURN_FAILURE;
    }

    if(jsonmode) { printf("[\n"); }
    int first_json_item = 1;

    for(int pi = 0;
        pi < PROCESSINFOLISTSIZE; pi++)
    {
        if(!pinfolist->active[pi])
        {
            continue;
        }

        if(!longmode && !jsonmode)
        {
            printf("%s\n",
                   pinfolist
                       ->pnamearray[pi]);
        }
        else
        {
            /* Connect to get CTRLval and
             * loop frequency */
            pid_t fpid =
                pinfolist->PIDarray[pi];
            const char *state = "UNKNOWN";
            double freq = 0.0;

            if(fpid > 0)
            {
                char pfn[512];
                char pdname[256];
                processinfo_procdirname(
                    pdname);
                snprintf(pfn, sizeof(pfn),
                         "%s/proc.%d.shm",
                         pdname,
                         (int) fpid);
                int pfd = -1;
                PROCESSINFO *pi_shm =
                    processinfo_shm_link(
                        pfn, &pfd);
                if(pi_shm != MAP_FAILED
                   && pi_shm != NULL)
                {
                    switch(pi_shm->CTRLval)
                    {
                    case PROCESSINFO_CTRLVAL_RUN:
                        state = "ACTIVE";
                        break;
                    case PROCESSINFO_CTRLVAL_PAUSE:
                        state = "PAUSED";
                        break;
                    case PROCESSINFO_CTRLVAL_EXIT:
                        state = "STOPPED";
                        break;
                    default:
                        state = "OTHER";
                        break;
                    }
                    /* Frequency from median
                     * iteration timing */
                    if(pi_shm
                       ->dtmedian_iter_ns
                       > 0)
                    {
                        freq =
                            1.0e9
                            / (double)
                            pi_shm
                            ->dtmedian_iter_ns;
                    }
                    munmap(pi_shm,
                           sizeof(
                               PROCESSINFO));
                    close(pfd);
                }
                else if(pfd >= 0)
                {
                    close(pfd);
                }
            }

            if(jsonmode)
            {
                if(!first_json_item) { printf(",\n"); }
                first_json_item = 0;
                printf("  {\n");
                printf("    \"name\": \"%s\",\n", pinfolist->pnamearray[pi]);
                printf("    \"state\": \"%s\",\n", state);
                printf("    \"pid\": %d,\n", (int)fpid);
                printf("    \"freq_hz\": %f\n", freq);
                printf("  }");
            }
            else
            {
                printf("%-24s %-8s %8.1f Hz\n",
                       pinfolist
                           ->pnamearray[pi],
                       state, freq);
            }
        }
    }

    if(jsonmode) { printf("\n]\n"); }

    return RETURN_SUCCESS;
}


/* ============================================================
 *  defer — register LIFO cleanup command
 * ============================================================
 */

#define CLI_DEFER_MAX 32

static char cli_defer_stack
    [CLI_DEFER_MAX][STRINGMAXLEN_CLICMDLINE];
static int  cli_defer_count = 0;

/**
 * @brief defer command — register cleanup
 *
 * Pushes a command onto a LIFO stack that is
 * executed in reverse order when the script
 * exits (integrated with trap EXIT).
 *
 * Usage: defer <command ...>
 */
errno_t cli_cmd_defer(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: defer <command>\n");
        return RETURN_FAILURE;
    }

    if(cli_defer_count >= CLI_DEFER_MAX)
    {
        printf("defer: stack full "
               "(max %d)\n", CLI_DEFER_MAX);
        return RETURN_FAILURE;
    }

    /* Capture the deferred command from the original
     * command line after the "defer" keyword, so that
     * quoting/escaping are preserved. */
    char cmd[STRINGMAXLEN_CLICMDLINE];
    cmd[0] = '\0';

    const char *line = data.CLIcmdline;
    const char *p    = line;

    /* Skip leading whitespace */
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }

    /* Expect "defer" as the first token */
    const char keyword[] = "defer";
    size_t      klen     = sizeof(keyword) - 1;

    if(strncmp(p, keyword, klen) == 0)
    {
        p += klen;

        /* Skip whitespace between "defer" and the deferred command */
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
    }

    /* p now points to the start of the deferred command
     * as typed by the user (may be empty if no command). */
    strncpy(cmd, p, sizeof(cmd) - 1);
    cmd[sizeof(cmd) - 1] = '\0';
    strncpy(
        cli_defer_stack[cli_defer_count],
        cmd,
        STRINGMAXLEN_CLICMDLINE - 1);
    cli_defer_stack[cli_defer_count]
        [STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    cli_defer_count++;

    return RETURN_SUCCESS;
}

/**
 * @brief Execute deferred cleanup commands
 *
 * Called from cli_trap_run_exit() to run
 * all deferred commands in LIFO order.
 */
void cli_defer_run(void)
{
    static int running = 0;

    if(running)
    {
        return;
    }
    running = 1;

    /* Pop-and-execute in LIFO order.
     * New defers pushed by a deferred command
     * are picked up because we re-check
     * cli_defer_count each iteration. */
    while(cli_defer_count > 0)
    {
        cli_defer_count--;
        char cmd[STRINGMAXLEN_CLICMDLINE];
        strncpy(cmd,
                cli_defer_stack[cli_defer_count],
                STRINGMAXLEN_CLICMDLINE - 1);
        cmd[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        CLI_execute_string(cmd);
    }

    running = 0;
}

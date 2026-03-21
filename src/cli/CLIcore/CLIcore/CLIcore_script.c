/**
 * @file CLIcore_script.c
 * @brief CLI scripting engine — variables, FPS access,
 *        arithmetic, flow control, user functions
 *
 * Implements bash-style scripting constructs for the
 * milk CLI:
 * - Variable assignment (VAR=val), expansion ($VAR)
 * - FPS parameter read (@fpsname.param)
 * - FPS parameter write (fpsset)
 * - Arithmetic $(( expr ))
 * - Flow control: if/then/else/fi,
 *   while/do/done, for/do/done
 * - User-defined functions:
 *   function name { body }
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/wait.h>

#include "CLIcore.h"
#include "CLIcore_script.h"

/* ============================================================
 *  CLI Variable Storage
 * ============================================================
 */

CLI_VAR cli_vars[CLI_MAX_VARS];
int     cli_last_retval = 0;

/* ---- set flags ---- */
int     cli_flag_errexit = 0;  /* set -e */
int     cli_flag_xtrace  = 0;  /* set -x */

/* ---- Trap Handlers ---- */
CLI_TRAP cli_traps[CLI_TRAP_MAXSIGS];

/**
 * @brief Map signal name to number
 *
 * @param name  Signal name (EXIT, INT, etc.)
 * @return signal number, or 0 for EXIT
 */
static int cli_trap_signum(const char *name)
{
    if(strcasecmp(name, "EXIT") == 0)
    {
        return 0;
    }
    if(strcasecmp(name, "INT") == 0)
    {
        return SIGINT;
    }
    if(strcasecmp(name, "TERM") == 0)
    {
        return SIGTERM;
    }
    if(strcasecmp(name, "HUP") == 0)
    {
        return SIGHUP;
    }
    if(strcasecmp(name, "USR1") == 0)
    {
        return SIGUSR1;
    }
    if(strcasecmp(name, "USR2") == 0)
    {
        return SIGUSR2;
    }
    return (int) strtol(name, NULL, 0);
}

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

/* ---- Array Storage ---- */
CLI_ARRAY cli_arrays[CLI_MAX_ARRAYS];

/* ---- Associative Array Storage ---- */
CLI_ASSOC_ARRAY cli_assoc[CLI_MAX_ASSOC];

/* Local variable scoping stack for functions */
#define CLI_MAX_LOCAL_DEPTH 32
#define CLI_MAX_LOCALS_PER_FUNC 64
typedef struct {
    char name[CLI_VAR_NAMELEN];
    char val[CLI_VAR_VALLEN];
    int  was_used;
} CLI_LOCAL_SHADOW;

static CLI_LOCAL_SHADOW cli_local_shadows[CLI_MAX_LOCAL_DEPTH][CLI_MAX_LOCALS_PER_FUNC];
static int cli_local_shadow_count[CLI_MAX_LOCAL_DEPTH];
static int cli_local_depth = 0;

/**
 * @brief Look up a CLI variable by name
 *
 * @param name  Variable name
 * @return pointer to value string, or NULL
 */
const char *cli_var_get(const char *name)
{
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            return cli_vars[i].val;
        }
    }
    return NULL;
}

/**
 * @brief Set a CLI variable (create or update)
 *
 * @param name  Variable name
 * @param val   Value string
 */
void cli_var_set(
    const char *name,
    const char *val
)
{
    /* Update existing */
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            strncpy(cli_vars[i].val, val,
                    CLI_VAR_VALLEN - 1);
            cli_vars[i].val[
                CLI_VAR_VALLEN - 1] = '\0';
            return;
        }
    }
    /* Find empty slot */
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(!cli_vars[i].used)
        {
            strncpy(cli_vars[i].name, name,
                    CLI_VAR_NAMELEN - 1);
            cli_vars[i].name[
                CLI_VAR_NAMELEN - 1] = '\0';
            strncpy(cli_vars[i].val, val,
                    CLI_VAR_VALLEN - 1);
            cli_vars[i].val[
                CLI_VAR_VALLEN - 1] = '\0';
            cli_vars[i].used = 1;
            return;
        }
    }
    printf("Error: variable table full "
           "(max %d)\n", CLI_MAX_VARS);
}

/**
 * @brief Remove a CLI variable
 *
 * @param name  Variable name
 */
void cli_var_unset(const char *name)
{
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            cli_vars[i].used = 0;
            cli_vars[i].name[0] = '\0';
            cli_vars[i].val[0] = '\0';
            return;
        }
    }
}

/**
 * @brief Unified variable lookup: CLI vars,
 *        then special vars ($?), then env vars
 *
 * @param name  Variable name
 * @return pointer to value string, or NULL
 */
const char *cli_var_lookup(const char *name)
{
    static char retbuf[32];

    /* $? — last return value */
    if(strcmp(name, "?") == 0)
    {
        snprintf(retbuf, sizeof(retbuf),
                 "%d", cli_last_retval);
        return retbuf;
    }

    /* CLI variable */
    const char *v = cli_var_get(name);
    if(v != NULL)
    {
        return v;
    }

    /* stream.prop — shared memory
     * stream metadata */
    {
        const char *dot =
            strchr(name, '.');
        if(dot != NULL)
        {
            char sname[128];
            int sn = (int)(dot - name);
            if(sn >= (int) sizeof(sname))
            {
                sn =
                    (int) sizeof(sname)
                    - 1;
            }
            memcpy(sname, name,
                   (size_t) sn);
            sname[sn] = '\0';
            const char *prop = dot + 1;

            /* Try stream SHM */
            char spath[256];
            snprintf(spath,
                     sizeof(spath),
                     "/dev/shm/%s"
                     ".im.shm",
                     sname);
            if(access(spath,
                      F_OK) == 0)
            {
                IMAGE img;
                if(ImageStreamIO_read_sharedmem_image_toIMAGE(
                    sname, &img) == 0)
                {
                    if(strcmp(prop,
                        "xsize") == 0
                       && img.md != NULL
                       && img.md->naxis
                       >= 1)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%u",
                            img.md
                            ->size[0]);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "ysize") == 0
                       && img.md != NULL
                       && img.md->naxis
                       >= 2)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%u",
                            img.md
                            ->size[1]);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "zsize") == 0
                       && img.md != NULL
                       && img.md->naxis
                       >= 3)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%u",
                            img.md
                            ->size[2]);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "type") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%d",
                            (int)
                            img.md
                            ->datatype);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "cnt0") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%lu",
                            (unsigned
                            long)
                            img.md
                            ->cnt0);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "cnt1") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%lu",
                            (unsigned
                            long)
                            img.md
                            ->cnt1);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "naxis") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%d",
                            (int)
                            img.md
                            ->naxis);
                        return retbuf;
                    }
                }
            }

            /* Try FPS SHM */
            char fpath[256];
            snprintf(fpath,
                     sizeof(fpath),
                     "/dev/shm/"
                     "fps.%s.shm",
                     sname);
            if(access(fpath,
                      F_OK) == 0)
            {
                if(strcmp(prop,
                    "status") == 0)
                {
                    snprintf(
                        retbuf,
                        sizeof(retbuf),
                        "1");
                    return retbuf;
                }
            }
        }
    }

    /* Fall through to environment */
    return getenv(name);
}


/* ============================================================
 *  Variable Assignment Detection
 * ============================================================
 */

/**
 * @brief Check if line is VAR=val assignment
 *
 * @param line  Command line string
 * @return 1 if handled as assignment, 0 otherwise
 */
int cli_try_var_assign(const char *line)
{
    const char *p = line;

    /* Skip leading whitespace */
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }

    /* Must start with alpha or underscore */
    if(!isalpha((unsigned char) *p)
            && *p != '_')
    {
        return 0;
    }

    /* Scan variable name */
    const char *name_start = p;
    while(isalnum((unsigned char) *p)
            || *p == '_')
    {
        p++;
    }

    /* Must hit '=' immediately */
    if(*p != '=')
    {
        return 0;
    }

    {
        int namelen = (int)(p - name_start);
        char tmpname[CLI_VAR_NAMELEN];
        if(namelen >= CLI_VAR_NAMELEN)
        {
            namelen = CLI_VAR_NAMELEN - 1;
        }
        memcpy(tmpname, name_start,
               (size_t) namelen);
        tmpname[namelen] = '\0';

        /* Extract value (everything after '=') */
        const char *val = p + 1;

        /* Strip trailing whitespace/newline */
        char valbuf[CLI_VAR_VALLEN];
        strncpy(valbuf, val,
                CLI_VAR_VALLEN - 1);
        valbuf[CLI_VAR_VALLEN - 1] = '\0';
        {
            size_t vl = strlen(valbuf);
            while(vl > 0
                    && (valbuf[vl - 1] == ' '
                        || valbuf[vl - 1] == '\t'
                        || valbuf[vl - 1] == '\n'))
            {
                valbuf[--vl] = '\0';
            }
        }

        cli_var_set(tmpname, valbuf);
        return 1;
    }
}

/**
 * @brief Check if line is array assignment
 *
 * Syntax: arr=(val1 val2 val3)
 *
 * @param line  Command line string
 * @return 1 if handled, 0 otherwise
 */
int cli_try_array_assign(const char *line)
{
    const char *p = line;
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }
    if(!isalpha((unsigned char) *p)
       && *p != '_')
    {
        return 0;
    }
    const char *ns = p;
    while(isalnum((unsigned char) *p)
          || *p == '_')
    {
        p++;
    }
    if(*p != '=')
    {
        return 0;
    }
    if(*(p + 1) != '(')
    {
        return 0;
    }

    int nlen = (int)(p - ns);
    char aname[CLI_VAR_NAMELEN];
    if(nlen >= CLI_VAR_NAMELEN)
    {
        nlen = CLI_VAR_NAMELEN - 1;
    }
    memcpy(aname, ns, (size_t) nlen);
    aname[nlen] = '\0';

    p += 2; /* skip =( */

    /* Find or create array slot */
    int slot = -1;
    for(int i = 0;
        i < CLI_MAX_ARRAYS; i++)
    {
        if(cli_arrays[i].used
           && strcmp(cli_arrays[i].name,
                    aname) == 0)
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
        printf("Error: array table full\n");
        return 1;
    }

    strncpy(cli_arrays[slot].name,
            aname,
            CLI_VAR_NAMELEN - 1);
    cli_arrays[slot].used = 1;
    cli_arrays[slot].nelem = 0;

    /* Parse elements */
    while(*p != '\0' && *p != ')')
    {
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        if(*p == ')' || *p == '\0')
        {
            break;
        }
        int ei = 0;
        int idx =
            cli_arrays[slot].nelem;
        if(idx >= CLI_ARRAY_MAXELEM)
        {
            break;
        }
        while(*p != '\0'
              && *p != ' '
              && *p != '\t'
              && *p != ')'
              && ei < CLI_VAR_VALLEN - 1)
        {
            cli_arrays[slot]
                .elem[idx][ei++] = *p++;
        }
        cli_arrays[slot]
            .elem[idx][ei] = '\0';
        cli_arrays[slot].nelem++;
    }
    return 1;
}


/* ============================================================
 *  CLI Commands: unset, vars, echo, fpsset
 * ============================================================
 */

/**
 * @brief unset command — remove a variable
 */
errno_t cli_cmd_unset(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: unset <varname>\n");
        return RETURN_FAILURE;
    }
    cli_var_unset(
        data.cmdargtoken[1].val.string);
    return RETURN_SUCCESS;
}

/**
 * @brief vars command — list all CLI variables
 */
errno_t cli_cmd_vars(void)
{
    int count = 0;
    printf("\n  CLI Variables:\n");
    printf("  %-20s  %s\n",
           "NAME", "VALUE");
    printf("  %-20s  %s\n",
           "----", "-----");
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used)
        {
            printf("  %-20s  %s\n",
                   cli_vars[i].name,
                   cli_vars[i].val);
            count++;
        }
    }
    if(count == 0)
    {
        printf("  (none)\n");
    }
    printf("  $? = %d\n\n", cli_last_retval);
    return RETURN_SUCCESS;
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


/* ============================================================
 *  FPS Variable Expansion — @fpsname.param
 * ============================================================
 */

/**
 * @brief Expand @fpsname.param tokens in place
 *
 * @param line   Command line buffer
 * @param maxlen Buffer size
 */
void cli_expand_fpsvar(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0'
            && opos < maxlen - 1)
    {
        if(line[i] == '@')
        {
            i++; /* skip @ */
            char token[512];
            int  tlen = 0;
            while(line[i] != '\0'
                    && tlen < 511
                    && (isalnum(
                            (unsigned char)
                            line[i])
                        || line[i] == '_'
                        || line[i] == '.'
                        || line[i] == '-'))
            {
                token[tlen++] = line[i++];
            }
            token[tlen] = '\0';

            char *dot = strchr(token, '.');
            if(dot == NULL)
            {
                if(opos < maxlen - 1)
                {
                    out[opos++] = '@';
                }
                int clen = tlen;
                if(opos + clen > maxlen - 1)
                {
                    clen = maxlen - 1 - opos;
                }
                memcpy(out + opos, token,
                       (size_t) clen);
                opos += clen;
                continue;
            }

            *dot = '\0';
            const char *fpsname = token;
            const char *pname = dot + 1;

            FUNCTION_PARAMETER_STRUCT fps;
            int fpsconn =
                function_parameter_struct_connect(
                    fpsname, &fps,
                    FPSCONNECT_SIMPLE);

            if(fpsconn == -1
                    || fps.parray == NULL)
            {
                continue;
            }

            int pindex =
                functionparameter_GetParamIndex(
                    &fps, pname);

            if(pindex < 0)
            {
                char dotname[512];
                snprintf(dotname,
                         sizeof(dotname),
                         ".%s", pname);
                pindex =
                    functionparameter_GetParamIndex(
                        &fps, dotname);
            }

            if(pindex >= 0)
            {
                char vstr[512];
                functionparameter_GetParamValueString(
                    &fps.parray[pindex],
                    vstr,
                    (int) sizeof(vstr));

                int vlen = (int) strlen(vstr);
                int avail = maxlen - 1 - opos;
                int clen = vlen < avail
                           ? vlen : avail;
                memcpy(out + opos, vstr,
                       (size_t) clen);
                opos += clen;
            }

            function_parameter_struct_disconnect(
                &fps);
        }
        else
        {
            out[opos++] = line[i++];
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}


/* ============================================================
 *  Arithmetic Expansion — $(( expr ))
 * ============================================================
 */

typedef struct
{
    const char *s;
    int         pos;
} ArithParser;

static double arith_expr(ArithParser *p);

static void arith_skip_ws(ArithParser *p)
{
    while(p->s[p->pos] == ' '
            || p->s[p->pos] == '\t')
    {
        p->pos++;
    }
}

static double arith_atom(ArithParser *p)
{
    arith_skip_ws(p);

    /* Unary minus */
    if(p->s[p->pos] == '-')
    {
        p->pos++;
        return -arith_atom(p);
    }

    /* Parenthesized sub-expression */
    if(p->s[p->pos] == '(')
    {
        p->pos++;
        double v = arith_expr(p);
        arith_skip_ws(p);
        if(p->s[p->pos] == ')')
        {
            p->pos++;
        }
        return v;
    }

    /* Variable name (bare identifier) */
    if(isalpha((unsigned char) p->s[p->pos])
       || p->s[p->pos] == '_')
    {
        char vname[256];
        int vn = 0;
        while(vn < 255
              && (isalnum(
                      (unsigned char)
                      p->s[p->pos])
                  || p->s[p->pos] == '_'))
        {
            vname[vn++] = p->s[p->pos++];
        }
        vname[vn] = '\0';
        const char *vv = cli_var_lookup(vname);
        if(vv != NULL)
        {
            return strtod(vv, NULL);
        }
        return 0.0;
    }

    /* Number */
    arith_skip_ws(p);
    const char *start = p->s + p->pos;
    char *end = NULL;
    double v = strtod(start, &end);
    if(end > start)
    {
        p->pos += (int)(end - start);
        return v;
    }

    return 0.0;
}

static double arith_factor(ArithParser *p)
{
    double left = arith_atom(p);
    arith_skip_ws(p);

    while(p->s[p->pos] == '*'
            || p->s[p->pos] == '/'
            || p->s[p->pos] == '%')
    {
        char op = p->s[p->pos];
        p->pos++;
        double right = arith_atom(p);
        arith_skip_ws(p);
        if(op == '*')
        {
            left *= right;
        }
        else if(op == '/')
        {
            if(right != 0.0)
            {
                left /= right;
            }
        }
        else if(op == '%')
        {
            if(right != 0.0)
            {
                left = fmod(left, right);
            }
        }
    }
    return left;
}

static double arith_term(ArithParser *p)
{
    double left = arith_factor(p);
    arith_skip_ws(p);

    while(p->s[p->pos] == '+'
            || p->s[p->pos] == '-')
    {
        char op = p->s[p->pos];
        p->pos++;
        double right = arith_factor(p);
        arith_skip_ws(p);
        if(op == '+')
        {
            left += right;
        }
        else
        {
            left -= right;
        }
    }
    return left;
}

static double arith_compare(ArithParser *p)
{
    double left = arith_term(p);
    arith_skip_ws(p);

    if(p->s[p->pos] == '<'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left <= right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '>'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left >= right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '<')
    {
        p->pos++;
        double right = arith_term(p);
        return (left < right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '>')
    {
        p->pos++;
        double right = arith_term(p);
        return (left > right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '='
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left == right) ? 1.0 : 0.0;
    }
    if(p->s[p->pos] == '!'
            && p->s[p->pos + 1] == '=')
    {
        p->pos += 2;
        double right = arith_term(p);
        return (left != right) ? 1.0 : 0.0;
    }
    return left;
}

static double arith_expr(ArithParser *p)
{
    return arith_compare(p);
}


/**
 * @brief Expand $(( expr )) in place
 */
void cli_expand_arith(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0'
            && opos < maxlen - 1)
    {
        if(line[i] == '$'
                && line[i + 1] == '('
                && line[i + 2] == '(')
        {
            i += 3;

            char expr[512];
            int  elen = 0;
            int  depth = 1;
            while(line[i] != '\0'
                    && elen < 511)
            {
                if(line[i] == '('
                        && line[i + 1] == '(')
                {
                    depth++;
                    expr[elen++] = line[i++];
                    expr[elen++] = line[i++];
                    continue;
                }
                if(line[i] == ')'
                        && line[i + 1] == ')')
                {
                    depth--;
                    if(depth == 0)
                    {
                        i += 2;
                        break;
                    }
                    expr[elen++] = line[i++];
                    expr[elen++] = line[i++];
                    continue;
                }
                expr[elen++] = line[i++];
            }
            expr[elen] = '\0';

            ArithParser parser;
            parser.s = expr;
            parser.pos = 0;
            double result = arith_expr(&parser);

            char rbuf[64];
            if(result == floor(result)
                    && fabs(result) < 1e15)
            {
                snprintf(rbuf, sizeof(rbuf),
                         "%ld", (long) result);
            }
            else
            {
                snprintf(rbuf, sizeof(rbuf),
                         "%g", result);
            }

            int rlen = (int) strlen(rbuf);
            int avail = maxlen - 1 - opos;
            int clen = rlen < avail
                       ? rlen : avail;
            memcpy(out + opos, rbuf,
                   (size_t) clen);
            opos += clen;
        }
        else
        {
            out[opos++] = line[i++];
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}


/* ============================================================
 *  Test Condition Evaluator — [ expr ]
 * ============================================================
 *
 * Evaluates bash-style test expressions:
 *   [ val1 -eq val2 ]   numeric equal
 *   [ val1 -ne val2 ]   numeric not equal
 *   [ val1 -lt val2 ]   numeric less than
 *   [ val1 -gt val2 ]   numeric greater than
 *   [ val1 -le val2 ]   numeric less or equal
 *   [ val1 -ge val2 ]   numeric greater or equal
 *   [ str1 == str2 ]    string equal
 *   [ str1 != str2 ]    string not equal
 *   [ -n str ]          string not empty
 *   [ -z str ]          string empty
 *   [ -f path ]         regular file exists
 *   [ -d path ]         directory exists
 *   [ -e path ]         path exists
 *   [ -s path ]         file exists, non-empty
 *   [ ! expr ]          logical NOT
 */

/**
 * @brief Evaluate [ ... ] test expression
 *
 * @param expr   The content between [ and ]
 * @return 1 = true, 0 = false
 */
int cli_eval_test(const char *expr)
{
    /* Tokenize expression */
    char buf[512];
    strncpy(buf, expr, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    /* Strip leading/trailing whitespace */
    char *p = buf;
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }
    {
        size_t len = strlen(p);
        while(len > 0
              && (p[len - 1] == ' '
                  || p[len - 1] == '\t'))
        {
            p[--len] = '\0';
        }
    }

    /* Tokenize into words */
    char *tokens[16];
    int ntok = 0;
    {
        char *saveptr = NULL;
        char *tok = strtok_r(p, " \t", &saveptr);
        while(tok != NULL && ntok < 16)
        {
            tokens[ntok++] = tok;
            tok = strtok_r(NULL, " \t", &saveptr);
        }
    }

    if(ntok == 0)
    {
        return 0;
    }

    /* Unary: -n str, -z str */
    if(ntok == 2
       && strcmp(tokens[0], "-n") == 0)
    {
        return strlen(tokens[1]) > 0 ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-z") == 0)
    {
        return strlen(tokens[1]) == 0 ? 1 : 0;
    }

    /* File tests: -f, -d, -e, -s */
    if(ntok == 2
       && strcmp(tokens[0], "-f") == 0)
    {
        struct stat sb;
        return (stat(tokens[1], &sb) == 0
                && S_ISREG(sb.st_mode))
               ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-d") == 0)
    {
        struct stat sb;
        return (stat(tokens[1], &sb) == 0
                && S_ISDIR(sb.st_mode))
               ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-e") == 0)
    {
        struct stat sb;
        return stat(tokens[1], &sb) == 0
               ? 1 : 0;
    }
    if(ntok == 2
       && strcmp(tokens[0], "-s") == 0)
    {
        struct stat sb;
        return (stat(tokens[1], &sb) == 0
                && sb.st_size > 0)
               ? 1 : 0;
    }

    /* Variable test: -v VAR */
    if(ntok == 2
       && strcmp(tokens[0], "-v") == 0)
    {
        const char *vv =
            cli_var_get(tokens[1]);
        if(vv != NULL)
        {
            return 1;
        }
        const char *ev =
            getenv(tokens[1]);
        return ev != NULL ? 1 : 0;
    }

    /* Logical NOT: ! expr */
    if(ntok >= 2
       && strcmp(tokens[0], "!") == 0)
    {
        /* Rebuild sub-expression */
        char subexpr[512];
        subexpr[0] = '\0';
        for(int i = 1; i < ntok; i++)
        {
            if(i > 1)
            {
                strncat(subexpr, " ",
                        sizeof(subexpr)
                        - strlen(subexpr)
                        - 1);
            }
            strncat(subexpr, tokens[i],
                    sizeof(subexpr)
                    - strlen(subexpr) - 1);
        }
        return cli_eval_test(subexpr)
               ? 0 : 1;
    }

    /* Single value: true if non-empty */
    if(ntok == 1)
    {
        return strlen(tokens[0]) > 0 ? 1 : 0;
    }

    /* Binary: val1 op val2 */
    if(ntok >= 3)
    {
        const char *lhs = tokens[0];
        const char *op = tokens[1];
        const char *rhs = tokens[2];

        double lv = strtod(lhs, NULL);
        double rv = strtod(rhs, NULL);

        if(strcmp(op, "-eq") == 0)
        {
            return (lv == rv) ? 1 : 0;
        }
        if(strcmp(op, "-ne") == 0)
        {
            return (lv != rv) ? 1 : 0;
        }
        if(strcmp(op, "-lt") == 0)
        {
            return (lv < rv) ? 1 : 0;
        }
        if(strcmp(op, "-gt") == 0)
        {
            return (lv > rv) ? 1 : 0;
        }
        if(strcmp(op, "-le") == 0)
        {
            return (lv <= rv) ? 1 : 0;
        }
        if(strcmp(op, "-ge") == 0)
        {
            return (lv >= rv) ? 1 : 0;
        }
        if(strcmp(op, "==") == 0)
        {
            return strcmp(lhs, rhs) == 0
                   ? 1 : 0;
        }
        if(strcmp(op, "!=") == 0)
        {
            return strcmp(lhs, rhs) != 0
                   ? 1 : 0;
        }
    }

    printf("Error: invalid test expression\n");
    return 0;
}


/* ============================================================
 *  Block Accumulator — flow control engine
 * ============================================================
 *
 * Multi-line constructs (if/while/for/function)
 * are accumulated in a block buffer until the
 * closing keyword is seen, then the complete
 * block is evaluated.
 */

CLI_BLOCK cli_block_stack[CLI_BLOCK_MAXDEPTH];
int       cli_block_level = 0;

/* Break/continue/return flags */
static int cli_break_flag = 0;
static int cli_continue_flag = 0;
int        cli_return_flag = 0;

/* Forward declaration */
static void cli_exec_block_if(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
static void cli_exec_block_while(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
static void cli_exec_block_for(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);


/* ---- Helper: strip whitespace ---- */

static const char *strip_ws(const char *s)
{
    while(*s == ' ' || *s == '\t')
    {
        s++;
    }
    return s;
}

static int starts_with(
    const char *line,
    const char *prefix
)
{
    return strncmp(line, prefix,
                   strlen(prefix)) == 0;
}

/**
 * @brief Execute lines through CLI_execute_line
 */
void cli_exec_lines(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int  nlines
)
{
    for(int i = 0; i < nlines; i++)
    {
        if(cli_break_flag
           || cli_continue_flag
           || cli_return_flag)
        {
            break;
        }

        /* Copy to cmdline and execute */
        strncpy(data.CLIcmdline, lines[i],
                STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[
            STRINGMAXLEN_CLICMDLINE - 1]
            = '\0';
        CLI_execute_line();
    }
}


/* ---- Parse if/then/elif/else/fi block ---- */

/**
 * @brief Evaluate a condition line
 *
 * Handles "if [ cond ]", "elif [ cond ]",
 * or bare "if val" forms. The keyword
 * (if/elif) is skipped before evaluation.
 *
 * @param raw   Raw condition line
 * @param skip  Chars to skip ("if"=2, "elif"=4)
 * @return 1 = true, 0 = false
 */
static int eval_cond_line(
    const char *raw,
    int skip
)
{
    char cl[STRINGMAXLEN_CLICMDLINE];
    strncpy(cl, raw,
            STRINGMAXLEN_CLICMDLINE - 1);
    cl[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    cli_expand_fpsvar(
        cl, STRINGMAXLEN_CLICMDLINE);
    cli_expand_env(
        cl, STRINGMAXLEN_CLICMDLINE);
    cli_expand_arith(
        cl, STRINGMAXLEN_CLICMDLINE);

    const char *p = strip_ws(cl);
    p += skip;
    p = strip_ws(p);

    if(*p == '[')
    {
        p++;
        const char *end = strrchr(p, ']');
        if(end != NULL)
        {
            char cs[512];
            int clen = (int)(end - p);
            if(clen >= (int) sizeof(cs))
            {
                clen = (int) sizeof(cs) - 1;
            }
            memcpy(cs, p, (size_t) clen);
            cs[clen] = '\0';
            return cli_eval_test(cs);
        }
        return 0;
    }
    return (strtod(p, NULL) != 0.0) ? 1 : 0;
}

/**
 * @brief Execute an if/then/elif/else/fi block
 *
 * Supports cascading elif:
 *   if [ cond1 ]; then
 *       body1
 *   elif [ cond2 ]; then
 *       body2
 *   else
 *       body3
 *   fi
 */
static void cli_exec_block_if(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
)
{
    if(nlines < 2)
    {
        return;
    }

    /* Build a list of branches:
     * Each branch has a condition line index
     * and body range [start, end).
     * The final else has cond_idx = -1. */

    typedef struct
    {
        int cond_idx;
        int body_start;
        int body_end;
    } Branch;

    Branch branches[64];
    int nbranch = 0;

    /* First branch: the if line */
    int body_s = 1;
    /* Skip standalone "then" */
    if(body_s < nlines)
    {
        const char *ts =
            strip_ws(lines[body_s]);
        if(strcmp(ts, "then") == 0)
        {
            body_s++;
        }
    }

    branches[0].cond_idx = 0;
    branches[0].body_start = body_s;
    nbranch = 1;

    /* Scan for elif/else at depth 0 */
    int depth = 0;
    for(int i = body_s; i < nlines; i++)
    {
        const char *ln = strip_ws(lines[i]);
        if(starts_with(ln, "if ")
           || starts_with(ln, "if\t"))
        {
            depth++;
            continue;
        }
        if(strcmp(ln, "fi") == 0)
        {
            if(depth > 0)
            {
                depth--;
                continue;
            }
            /* Close current branch */
            branches[nbranch - 1].body_end
                = i;
            break;
        }
        if(depth > 0)
        {
            continue;
        }
        if(starts_with(ln, "elif ")
           || starts_with(ln, "elif\t"))
        {
            branches[nbranch - 1].body_end
                = i;
            int bs = i + 1;
            if(bs < nlines)
            {
                const char *t2 =
                    strip_ws(lines[bs]);
                if(strcmp(t2, "then") == 0)
                {
                    bs++;
                }
            }
            if(nbranch < 64)
            {
                branches[nbranch].cond_idx
                    = i;
                branches[nbranch].body_start
                    = bs;
                nbranch++;
            }
        }
        else if(strcmp(ln, "else") == 0)
        {
            branches[nbranch - 1].body_end
                = i;
            if(nbranch < 64)
            {
                branches[nbranch].cond_idx
                    = -1; /* else */
                branches[nbranch].body_start
                    = i + 1;
                branches[nbranch].body_end
                    = nlines;
                nbranch++;
            }
            /* Find fi to close else */
            for(int j = i + 1;
                j < nlines; j++)
            {
                const char *l2 =
                    strip_ws(lines[j]);
                if(strcmp(l2, "fi") == 0)
                {
                    branches[
                        nbranch - 1]
                        .body_end = j;
                    break;
                }
            }
            break;
        }
    }

    /* Evaluate branches in order */
    for(int b = 0; b < nbranch; b++)
    {
        int run = 0;
        if(branches[b].cond_idx < 0)
        {
            /* else — always true */
            run = 1;
        }
        else
        {
            const char *cl2 = strip_ws(
                lines[branches[b].cond_idx]);
            int skip = 2; /* "if" */
            if(starts_with(cl2, "elif"))
            {
                skip = 4;
            }
            run = eval_cond_line(
                lines[branches[b].cond_idx],
                skip);
        }
        if(run)
        {
            int bs = branches[b].body_start;
            int be = branches[b].body_end;
            if(be > bs)
            {
                cli_exec_lines(
                    lines + bs, be - bs);
            }
            break;
        }
    }
}


/* ---- Parse while/do/done block ---- */

/**
 * @brief Execute a while/do/done block
 *
 * Expected format:
 *   lines[0]: "while [ condition ]; do"
 *   ...body...
 *   "done"
 */
static void cli_exec_block_while(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
)
{
    if(nlines < 2)
    {
        return;
    }

    /* Find body start (after "do") */
    int body_start = 1;
    int body_end = nlines;
    int max_iter = 100000;

    /* Skip standalone 'do' line from
     * semicolon-split */
    if(body_start < body_end)
    {
        const char *ds =
            strip_ws(lines[body_start]);
        if(strcmp(ds, "do") == 0)
        {
            body_start++;
        }
    }

    for(int iter = 0; iter < max_iter; iter++)
    {
        /* Re-expand condition each iteration */
        char condline[STRINGMAXLEN_CLICMDLINE];
        strncpy(condline, lines[0],
                STRINGMAXLEN_CLICMDLINE - 1);
        condline[
            STRINGMAXLEN_CLICMDLINE - 1] = '\0';

        /* Run expansion on condition */
        cli_expand_fpsvar(
            condline,
            STRINGMAXLEN_CLICMDLINE);
        cli_expand_env(
            condline,
            STRINGMAXLEN_CLICMDLINE);
        cli_expand_arith(
            condline,
            STRINGMAXLEN_CLICMDLINE);

        /* Parse condition */
        const char *cl = strip_ws(condline);
        cl += 5; /* skip "while" */
        cl = strip_ws(cl);

        int cond_result = 0;
        if(*cl == '[')
        {
            cl++;
            const char *end = strrchr(cl, ']');
            if(end != NULL)
            {
                char cs[512];
                int clen = (int)(end - cl);
                if(clen >= (int) sizeof(cs))
                {
                    clen =
                        (int) sizeof(cs) - 1;
                }
                memcpy(cs, cl, (size_t) clen);
                cs[clen] = '\0';
                cond_result =
                    cli_eval_test(cs);
            }
        }

        if(!cond_result)
        {
            break;
        }

        /* Execute body */
        cli_continue_flag = 0;
        cli_exec_lines(
            lines + body_start,
            body_end - body_start);

        if(cli_break_flag)
        {
            cli_break_flag = 0;
            break;
        }
    }
}


/* ---- Parse for/do/done block ---- */

/**
 * @brief Execute select/do/done block
 *
 * Syntax:
 *   select VAR in v1 v2 ...; do
 *     body
 *   done
 */
static void cli_exec_block_select(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int  nlines
)
{
    if(nlines < 2)
    {
        return;
    }
    /* Parse: select VAR in v1 v2 ... */
    const char *hdr = strip_ws(lines[0]);
    hdr += 7; /* skip 'select ' */
    while(*hdr == ' '
          || *hdr == '\t')
    {
        hdr++;
    }
    char vn[CLI_VAR_NAMELEN];
    {
        int vi = 0;
        while(*hdr != '\0'
              && *hdr != ' '
              && *hdr != '\t'
              && vi
              < CLI_VAR_NAMELEN - 1)
        {
            vn[vi++] = *hdr++;
        }
        vn[vi] = '\0';
    }
    while(*hdr == ' '
          || *hdr == '\t')
    {
        hdr++;
    }
    if(starts_with(hdr, "in "))
    {
        hdr += 3;
    }
    /* Collect values */
    char sv[256][CLI_VAR_VALLEN];
    int nsv = 0;
    while(*hdr != '\0'
          && nsv < 256)
    {
        while(*hdr == ' '
              || *hdr == '\t')
        {
            hdr++;
        }
        if(*hdr == ';'
           || *hdr == '\0')
        {
            break;
        }
        int vi = 0;
        while(*hdr != '\0'
              && *hdr != ' '
              && *hdr != '\t'
              && *hdr != ';'
              && vi
              < CLI_VAR_VALLEN - 1)
        {
            sv[nsv][vi++] = *hdr++;
        }
        sv[nsv][vi] = '\0';
        nsv++;
    }
    if(nsv == 0)
    {
        return;
    }
    /* Loop: print menu, read, exec */
    for(;;)
    {
        for(int i = 0;
            i < nsv; i++)
        {
            printf("%d) %s\n",
                   i + 1, sv[i]);
        }
        printf("#? ");
        fflush(stdout);
        char rb[64];
        if(fgets(rb, sizeof(rb),
                 stdin) == NULL)
        {
            break;
        }
        int ch =
            (int) strtol(rb,
                         NULL, 10);
        if(ch >= 1 && ch <= nsv)
        {
            cli_var_set(
                vn, sv[ch - 1]);
        }
        else
        {
            cli_var_set(vn, "");
        }
        cli_exec_lines(
            lines + 1,
            nlines - 1);
    }
}

/**
 * @brief Execute a for/do/done block
 *
 * Expected format:
 *   lines[0]: "for VAR in val1 val2 ...; do"
 *   ...body...
 *   "done"
 */
static void cli_exec_block_for(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
)
{
    if(nlines < 2)
    {
        return;
    }

    /* Check for arithmetic for:
     * for ((init; cond; step)); do */
    {
        const char *af =
            strip_ws(lines[0]);
        af += 3; /* skip "for" */
        af = strip_ws(af);
        if(af[0] == '(' && af[1] == '(')
        {
            af += 2; /* skip "((" */
            /* Find closing )) */
            const char *ce =
                strstr(af, "))");
            if(ce != NULL)
            {
                char abuf[
                    STRINGMAXLEN_CLICMDLINE
                ];
                int alen =
                    (int)(ce - af);
                memcpy(abuf, af,
                       (size_t) alen);
                abuf[alen] = '\0';
                /* Split on ; */
                char ainit[256] = "";
                char acond[256] = "";
                char astep[256] = "";
                char *s1 =
                    strchr(abuf, ';');
                if(s1 != NULL)
                {
                    *s1 = '\0';
                    strncpy(ainit,
                            abuf, 255);
                    char *s2 =
                        strchr(
                            s1 + 1,
                            ';');
                    if(s2 != NULL)
                    {
                        *s2 = '\0';
                        strncpy(
                            acond,
                            s1 + 1,
                            255);
                        strncpy(
                            astep,
                            s2 + 1,
                            255);
                    }
                }
                /* Execute init */
                {
                    char einit[
                        STRINGMAXLEN_CLICMDLINE
                    ];
                    snprintf(
                        einit,
                        sizeof(einit),
                        "$(( %s ))",
                        ainit);
                    cli_expand_arith(
                        einit,
                        STRINGMAXLEN_CLICMDLINE
                    );
                }
                /* Loop: eval cond,
                 * exec body, eval step */
                for(;;)
                {
                    char econd[
                        STRINGMAXLEN_CLICMDLINE
                    ];
                    snprintf(
                        econd,
                        sizeof(econd),
                        "$(( %s ))",
                        acond);
                    cli_expand_arith(
                        econd,
                        STRINGMAXLEN_CLICMDLINE
                    );
                    long cv =
                        strtol(econd,
                               NULL,
                               10);
                    if(cv == 0)
                    {
                        break;
                    }
                    cli_exec_lines(
                        lines + 1,
                        nlines - 1);
                    /* step */
                    {
                        char estep[
                            STRINGMAXLEN_CLICMDLINE
                        ];
                        snprintf(
                            estep,
                            sizeof(estep),
                            "$(( %s ))",
                            astep);
                        cli_expand_arith(
                            estep,
                            STRINGMAXLEN_CLICMDLINE
                        );
                    }
                }
                return;
            }
        }
    }

    /* Parse: for VAR in val1 val2 ... */
    const char *fl = strip_ws(lines[0]);
    fl += 3; /* skip "for" */
    fl = strip_ws(fl);

    /* Get variable name */
    char varname[CLI_VAR_NAMELEN];
    {
        int vn = 0;
        while(*fl != '\0' && *fl != ' '
              && *fl != '\t'
              && vn < CLI_VAR_NAMELEN - 1)
        {
            varname[vn++] = *fl++;
        }
        varname[vn] = '\0';
    }
    fl = strip_ws(fl);

    /* Skip "in" */
    if(strncmp(fl, "in ", 3) == 0
       || strncmp(fl, "in\t", 3) == 0)
    {
        fl += 2;
        fl = strip_ws(fl);
    }

    /* Collect values (strip trailing ;do) */
    char vallist[STRINGMAXLEN_CLICMDLINE];
    strncpy(vallist, fl,
            STRINGMAXLEN_CLICMDLINE - 1);
    vallist[
        STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    /* Remove trailing "; do" or ";do" */
    {
        char *semi = strstr(vallist, ";");
        if(semi != NULL)
        {
            *semi = '\0';
        }
    }

    /* Strip trailing whitespace */
    {
        size_t vl = strlen(vallist);
        while(vl > 0
              && (vallist[vl - 1] == ' '
                  || vallist[vl - 1] == '\t'
                  || vallist[vl - 1] == '\n'))
        {
            vallist[--vl] = '\0';
        }
    }

    int body_start = 1;
    int body_end = nlines;

    /* Skip standalone 'do' line */
    if(body_start < body_end)
    {
        const char *ds =
            strip_ws(lines[body_start]);
        if(strcmp(ds, "do") == 0)
        {
            body_start++;
        }
    }

    /* Iterate over values */
    char *saveptr = NULL;
    char *val = strtok_r(vallist, " \t",
                         &saveptr);
    while(val != NULL)
    {
        cli_var_set(varname, val);

        cli_continue_flag = 0;
        cli_exec_lines(
            lines + body_start,
            body_end - body_start);

        if(cli_break_flag)
        {
            cli_break_flag = 0;
            break;
        }

        val = strtok_r(NULL, " \t", &saveptr);
    }
}


/* ============================================================
 *  User-Defined Functions
 * ============================================================
 */

CLI_FUNC cli_funcs[CLI_MAX_FUNCS];

/**
 * @brief Find a user-defined function by name
 */
CLI_FUNC *cli_func_find(const char *name)
{
    for(int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if(cli_funcs[i].used
           && strcmp(cli_funcs[i].name, name)
              == 0)
        {
            return &cli_funcs[i];
        }
    }
    return NULL;
}

/**
 * @brief Register a new user function
 */
static void cli_func_define(
    const char *name,
    char body[][STRINGMAXLEN_CLICMDLINE],
    int nbody
)
{
    /* Update existing */
    for(int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if(cli_funcs[i].used
           && strcmp(cli_funcs[i].name, name)
              == 0)
        {
            cli_funcs[i].nbody = nbody;
            for(int j = 0; j < nbody; j++)
            {
                strncpy(
                    cli_funcs[i].body[j],
                    body[j],
                    STRINGMAXLEN_CLICMDLINE - 1
                );
            }
            return;
        }
    }
    /* Find empty slot */
    for(int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if(!cli_funcs[i].used)
        {
            strncpy(cli_funcs[i].name, name,
                    CLI_FUNC_NAMELEN - 1);
            cli_funcs[i].name[
                CLI_FUNC_NAMELEN - 1] = '\0';
            cli_funcs[i].nbody = nbody;
            cli_funcs[i].used = 1;
            for(int j = 0; j < nbody; j++)
            {
                strncpy(
                    cli_funcs[i].body[j],
                    body[j],
                    STRINGMAXLEN_CLICMDLINE - 1
                );
            }
            return;
        }
    }
    printf("Error: function table full "
           "(max %d)\n", CLI_MAX_FUNCS);
}


/**
 * @brief Try to call a user-defined function
 *
 * Syntax: funcname arg1 arg2 ...
 * Inside the function body, $1..$9 are args.
 *
 * @return 1 if matched, 0 if not
 */
int cli_try_func_call(const char *line)
{
    const char *p = strip_ws(line);

    /* Extract first word (function name) */
    char fname[CLI_FUNC_NAMELEN];
    {
        int fn = 0;
        while(*p != '\0' && *p != ' '
              && *p != '\t'
              && fn < CLI_FUNC_NAMELEN - 1)
        {
            fname[fn++] = *p++;
        }
        fname[fn] = '\0';
    }

    CLI_FUNC *func = cli_func_find(fname);
    if(func == NULL)
    {
        return 0;
    }

    /* Parse arguments */
    p = strip_ws(p);
    char *args[CLI_FUNC_MAXARGS];
    char argbuf[CLI_FUNC_MAXARGS][
        CLI_VAR_VALLEN];
    int nargs = 0;

    while(*p != '\0'
          && nargs < CLI_FUNC_MAXARGS)
    {
        int ai = 0;
        while(*p != '\0' && *p != ' '
              && *p != '\t'
              && ai < CLI_VAR_VALLEN - 1)
        {
            argbuf[nargs][ai++] = *p++;
        }
        argbuf[nargs][ai] = '\0';
        args[nargs] = argbuf[nargs];
        nargs++;
        p = strip_ws(p);
    }

    /* Save old $1..$9, set new ones */
    char old_args[CLI_FUNC_MAXARGS][
        CLI_VAR_VALLEN];
    int old_used[CLI_FUNC_MAXARGS];
    for(int i = 0; i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname),
                 "%d", i + 1);
        const char *ov = cli_var_get(aname);
        old_used[i] = (ov != NULL) ? 1 : 0;
        if(ov != NULL)
        {
            strncpy(old_args[i], ov,
                    CLI_VAR_VALLEN - 1);
            old_args[i][
                CLI_VAR_VALLEN - 1] = '\0';
        }
        if(i < nargs)
        {
            cli_var_set(aname, args[i]);
        }
        else
        {
            cli_var_unset(aname);
        }
    }

    /* Push local variable scope */
    if(cli_local_depth < CLI_MAX_LOCAL_DEPTH - 1)
    {
        cli_local_depth++;
        cli_local_shadow_count[cli_local_depth] = 0;
    }

    /* Execute body lines */
    cli_return_flag = 0;
    cli_exec_lines(func->body, func->nbody);
    cli_return_flag = 0;

    /* Restore old $1..$9 */
    for(int i = 0; i < CLI_FUNC_MAXARGS; i++)
    {
        char aname[4];
        snprintf(aname, sizeof(aname),
                 "%d", i + 1);
        if(old_used[i])
        {
            cli_var_set(aname, old_args[i]);
        }
        else
        {
            cli_var_unset(aname);
        }
    }

    /* Restore variables shadowed by 'local' */
    if(cli_local_depth > 0)
    {
        int scount = cli_local_shadow_count[cli_local_depth];
        for(int i = 0; i < scount; i++)
        {
            CLI_LOCAL_SHADOW *sh = &cli_local_shadows[cli_local_depth][i];
            if(sh->was_used)
            {
                cli_var_set(sh->name, sh->val);
            }
            else
            {
                cli_var_unset(sh->name);
            }
        }
        cli_local_depth--;
    }

    return 1;
}


/* ============================================================
 *  Block Intercept — main entry point
 * ============================================================
 *
 * Called from CLI_execute_line() before any other
 * processing. Returns 1 if the line was consumed
 * (buffered or block completed).
 */

/**
 * @brief Intercept line for flow control
 *
 * @param line  The raw command line
 * @return 1 if consumed, 0 if not
 */
/* ============================================================
 *  Case/esac Evaluator
 * ============================================================
 *
 * Syntax:
 *   case <word> in
 *     pattern1) cmd1 ;;
 *     pat2|pat3) cmd2 ;;
 *     *) default ;;
 *   esac
 */
static void cli_exec_block_case(
    char (*lines)[STRINGMAXLEN_CLICMDLINE],
    int    nlines
)
{
    /* Line 0 = "case <word> in" */
    const char *hdr = strip_ws(lines[0]);
    hdr += 4; /* skip "case" */
    hdr = strip_ws(hdr);
    char word[256];
    {
        int wi = 0;
        while(*hdr != '\0'
              && *hdr != ' '
              && *hdr != '\t'
              && wi < 255)
        {
            word[wi++] = *hdr++;
        }
        word[wi] = '\0';
    }
    /* Expand word */
    cli_expand_env(word, 256);

    /* Scan patterns: "pat) body ;;" */
    for(int i = 1; i < nlines; i++)
    {
        const char *lp =
            strip_ws(lines[i]);
        /* Find closing ')' */
        const char *cp = strchr(lp, ')');
        if(cp == NULL)
        {
            continue;
        }
        /* Extract pattern(s) */
        char pat[256];
        int plen = (int)(cp - lp);
        if(plen >= 256)
        {
            plen = 255;
        }
        memcpy(pat, lp, (size_t) plen);
        pat[plen] = '\0';

        /* Check match (supports pat1|pat2
         * and * wildcard) */
        int matched = 0;
        {
            char ptmp[256];
            strncpy(ptmp, pat,
                    sizeof(ptmp) - 1);
            ptmp[sizeof(ptmp) - 1] = '\0';
            char *psave = NULL;
            char *pp =
                strtok_r(ptmp, "|",
                         &psave);
            while(pp != NULL)
            {
                /* strip ws */
                while(*pp == ' '
                      || *pp == '\t')
                {
                    pp++;
                }
                if(strcmp(pp, "*") == 0
                   || strcmp(pp, word)
                   == 0)
                {
                    matched = 1;
                    break;
                }
                pp = strtok_r(NULL,
                              "|",
                              &psave);
            }
        }
        if(!matched)
        {
            continue;
        }

        /* Collect body lines until ;; */
        const char *body_start = cp + 1;
        while(*body_start == ' '
              || *body_start == '\t')
        {
            body_start++;
        }
        /* If body is on same line */
        if(*body_start != '\0')
        {
            /* Strip ;; from end */
            char cmdline[STRINGMAXLEN_CLICMDLINE];
            strncpy(cmdline, body_start,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
            cmdline[STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
            {
                int cl =
                    (int) strlen(cmdline);
                while(cl > 1
                      && cmdline[cl - 1]
                      == ';'
                      && cmdline[cl - 2]
                      == ';')
                {
                    cmdline[cl - 2] = '\0';
                    cl -= 2;
                }
                /* Trim trailing ws */
                while(cl > 0
                      && (cmdline[cl - 1]
                          == ' '
                          || cmdline[cl - 1]
                          == '\t'))
                {
                    cmdline[--cl] = '\0';
                }
            }
            if(strlen(cmdline) > 0)
            {
                strncpy(
                    data.CLIcmdline,
                    cmdline,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
                CLI_execute_line();
            }
        }
        else
        {
            /* Multi-line body */
            for(int j = i + 1;
                j < nlines; j++)
            {
                const char *bl =
                    strip_ws(lines[j]);
                if(strcmp(bl, ";;") == 0)
                {
                    break;
                }
                /* Strip trailing ;; */
                char cmd2[
                    STRINGMAXLEN_CLICMDLINE];
                strncpy(cmd2, bl,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                cmd2[
                    STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
                {
                    int c2l =
                        (int) strlen(cmd2);
                    int ends_dsemi = 0;
                    while(c2l > 1
                          && cmd2[c2l - 1]
                          == ';'
                          && cmd2[c2l - 2]
                          == ';')
                    {
                        cmd2[c2l - 2] =
                            '\0';
                        c2l -= 2;
                        ends_dsemi = 1;
                    }
                    while(c2l > 0
                          && (cmd2[c2l - 1]
                              == ' '
                              || cmd2[
                                  c2l - 1]
                              == '\t'))
                    {
                        cmd2[--c2l] = '\0';
                    }
                    if(strlen(cmd2) > 0)
                    {
                        strncpy(
                            data.CLIcmdline,
                            cmd2,
                            STRINGMAXLEN_CLICMDLINE
                            - 1);
                        CLI_execute_line();
                    }
                    if(ends_dsemi)
                    {
                        break;
                    }
                }
            }
        }
        return; /* first match only */
    }
}


int cli_script_intercept(const char *line)
{
    const char *p = strip_ws(line);

    /* ---- Heredoc accumulation state ---- */
    static int  heredoc_active = 0;
    static char heredoc_var[CLI_VAR_NAMELEN];
    static char heredoc_delim[64];
    static char heredoc_buf[16384];
    static int  heredoc_pos = 0;

    if(heredoc_active)
    {
        if(strcmp(p, heredoc_delim) == 0)
        {
            /* End of heredoc — assign */
            heredoc_buf[heredoc_pos] = '\0';
            cli_var_set(heredoc_var,
                        heredoc_buf);
            heredoc_active = 0;
        }
        else
        {
            /* Append line + newline */
            int llen = (int) strlen(p);
            if(heredoc_pos + llen + 1
               < (int) sizeof(heredoc_buf))
            {
                memcpy(
                    heredoc_buf + heredoc_pos,
                    p, (size_t) llen);
                heredoc_pos += llen;
                heredoc_buf[
                    heredoc_pos++] = '\n';
            }
        }
        return 1;
    }

    /* Check if this line starts a heredoc:
     *   VAR=<<DELIM */
    if(strchr(p, '=') != NULL)
    {
        const char *eq = strchr(p, '=');
        if(eq[1] == '<' && eq[2] == '<')
        {
            int nlen = (int)(eq - p);
            if(nlen > 0
               && nlen < CLI_VAR_NAMELEN)
            {
                memcpy(heredoc_var, p,
                       (size_t) nlen);
                heredoc_var[nlen] = '\0';
                const char *d = eq + 3;
                while(*d == ' '
                      || *d == '\t')
                {
                    d++;
                }
                int dlen = (int) strlen(d);
                if(dlen > 0 && dlen < 64)
                {
                    strncpy(heredoc_delim,
                            d, 63);
                    heredoc_delim[63] = '\0';
                    heredoc_active = 1;
                    heredoc_pos = 0;
                    heredoc_buf[0] = '\0';
                    return 1;
                }
            }
        }
    }

    /* If we're already accumulating a block,
     * buffer the line */
    if(cli_block_level > 0)
    {
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level - 1];

        /* Check for nested openers */
        if(starts_with(p, "if ")
           || starts_with(p, "if\t")
           || starts_with(p, "while ")
           || starts_with(p, "while\t")
           || starts_with(p, "for ")
           || starts_with(p, "for\t")
           || starts_with(p, "select ")
           || starts_with(p,
                          "select\t")
           || starts_with(p, "function ")
           || starts_with(p, "function\t")
           || starts_with(p, "case ")
           || starts_with(p, "case\t"))
        {
            blk->depth++;
        }

        /* Check for closers.
         * When depth > 0 (nested block),
         * ANY closer keyword decrements
         * the depth. Only at depth 0 does
         * the closer need to match the
         * outer block type. */
        int is_close = 0;
        int is_any_close =
            (strcmp(p, "fi") == 0
             || strcmp(p, "done") == 0
             || strcmp(p, "}") == 0
             || strcmp(p, "esac") == 0);

        if(is_any_close && blk->depth > 0)
        {
            /* Nested closer — decrement
             * depth and buffer */
            blk->depth--;
            if(blk->nlines
               < CLI_BLOCK_MAXLINES)
            {
                strncpy(
                    blk->lines[
                        blk->nlines],
                    p,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
                blk->nlines++;
            }
            return 1;
        }

        /* Check outer block closer */
        if(blk->type == CLI_BLOCK_IF
           && strcmp(p, "fi") == 0)
        {
            is_close = 1;
        }
        if((blk->type == CLI_BLOCK_WHILE
            || blk->type == CLI_BLOCK_FOR)
           && strcmp(p, "done") == 0)
        {
            is_close = 1;
        }
        if(blk->type == CLI_BLOCK_FUNC
           && strcmp(p, "}") == 0)
        {
            is_close = 1;
        }
        if(blk->type == CLI_BLOCK_CASE
           && strcmp(p, "esac") == 0)
        {
            is_close = 1;
        }

        if(is_close)
        {
            /* Outer block complete — save
             * data locally because the stack
             * slot may be reused by nested
             * blocks during execution.
             * Use malloc to avoid stack
             * overflow on deep nesting. */
            int saved_type = blk->type;
            int saved_nlines = blk->nlines;
            char (*saved_lines)[
                STRINGMAXLEN_CLICMDLINE] =
                malloc(
                    (size_t) saved_nlines
                    * STRINGMAXLEN_CLICMDLINE);
            if(saved_lines == NULL)
            {
                printf("Error: malloc failed "
                       "for block lines\n");
                cli_block_level--;
                return 1;
            }
            for(int si = 0;
                si < saved_nlines; si++)
            {
                strncpy(
                    saved_lines[si],
                    blk->lines[si],
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
                saved_lines[si][
                    STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
            }
            cli_block_level--;

            if(saved_type == CLI_BLOCK_IF)
            {
                cli_exec_block_if(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type == CLI_BLOCK_WHILE)
            {
                cli_exec_block_while(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type == CLI_BLOCK_FOR)
            {
                cli_exec_block_for(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type
                ==
                CLI_BLOCK_SELECT)
            {
                cli_exec_block_select(
                    saved_lines,
                    saved_nlines);
            }
            else if(
                saved_type == CLI_BLOCK_FUNC)
            {
                /* Define function from
                 * buffered lines */
                const char *fl =
                    strip_ws(
                        saved_lines[0]);
                fl += 8; /* "function" */
                fl = strip_ws(fl);
                char fname[CLI_FUNC_NAMELEN];
                {
                    int fn = 0;
                    while(*fl != '\0'
                          && *fl != ' '
                          && *fl != '\t'
                          && *fl != '{'
                          && fn
                             < CLI_FUNC_NAMELEN
                               - 1)
                    {
                        fname[fn++] = *fl++;
                    }
                    fname[fn] = '\0';
                }
                /* Body starts at line 1
                 * (skip function header) */
                cli_func_define(
                    fname,
                    saved_lines + 1,
                    saved_nlines - 1);
            }
            else if(
                saved_type == CLI_BLOCK_CASE)
            {
                cli_exec_block_case(
                    saved_lines,
                    saved_nlines);
            }

            free(saved_lines);
            return 1;
        }

        /* Buffer normal line */
        if(blk->nlines < CLI_BLOCK_MAXLINES)
        {
            strncpy(
                blk->lines[blk->nlines],
                p,
                STRINGMAXLEN_CLICMDLINE - 1);
            blk->nlines++;
        }
        return 1;
    }

    /* ---- Not in a block: check openers ---- */

    /* break / continue / return */
    if(strcmp(p, "break") == 0)
    {
        cli_break_flag = 1;
        return 1;
    }
    if(strcmp(p, "continue") == 0)
    {
        cli_continue_flag = 1;
        return 1;
    }
    if(strcmp(p, "return") == 0
       || starts_with(p, "return ")
       || starts_with(p, "return\t"))
    {
        const char *rv = p + 6;
        while(*rv == ' ' || *rv == '\t')
        {
            rv++;
        }
        if(*rv != '\0')
        {
            cli_last_retval =
                (int) strtol(rv, NULL, 0);
        }
        cli_return_flag = 1;
        return 1;
    }

    /* exit [N] — exit CLI entirely */
    if(strcmp(p, "exit") == 0
       || starts_with(p, "exit ")
       || starts_with(p, "exit\t"))
    {
        int exitcode = 0;
        if(strlen(p) > 4)
        {
            const char *ev = p + 4;
            while(*ev == ' ' || *ev == '\t')
            {
                ev++;
            }
            if(*ev != '\0')
            {
                exitcode =
                    (int) strtol(ev,
                                 NULL, 0);
            }
        }
        exit(exitcode);
    }

    /* shift [N] — shift positional params */
    if(strcmp(p, "shift") == 0
       || starts_with(p, "shift ")
       || starts_with(p, "shift\t"))
    {
        int n = 1;
        if(strlen(p) > 5)
        {
            const char *sv = p + 5;
            while(*sv == ' ' || *sv == '\t')
            {
                sv++;
            }
            if(*sv != '\0')
            {
                n = (int) strtol(sv,
                                 NULL, 0);
            }
        }
        if(n < 1)
        {
            n = 1;
        }
        /* Shift $1..$9 by n positions */
        for(int i = 1;
            i < CLI_FUNC_MAXARGS; i++)
        {
            char dst[4], src[4];
            snprintf(dst, sizeof(dst),
                     "%d", i);
            snprintf(src, sizeof(src),
                     "%d", i + n);
            if(i + n < CLI_FUNC_MAXARGS)
            {
                const char *sv2 =
                    cli_var_get(src);
                if(sv2 != NULL)
                {
                    cli_var_set(dst, sv2);
                }
                else
                {
                    cli_var_unset(dst);
                }
            }
            else
            {
                cli_var_unset(dst);
            }
        }
        return 1;
    }

    /* trap 'cmd' SIGNAL [SIGNAL...] */
    if(starts_with(p, "trap ")
       || starts_with(p, "trap\t"))
    {
        p += 4;
        p = strip_ws(p);
        /* Extract quoted command */
        char tcmd[CLI_TRAP_CMDLEN];
        tcmd[0] = '\0';
        if(*p == '\'' || *p == '"')
        {
            char q = *p++;
            int ti = 0;
            while(*p != '\0' && *p != q
                  && ti
                  < CLI_TRAP_CMDLEN - 1)
            {
                tcmd[ti++] = *p++;
            }
            tcmd[ti] = '\0';
            if(*p == q)
            {
                p++;
            }
        }
        p = strip_ws(p);
        /* Parse signal names */
        while(*p != '\0')
        {
            char sname[32];
            int si = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && si < 31)
            {
                sname[si++] = *p++;
            }
            sname[si] = '\0';
            p = strip_ws(p);
            if(si == 0)
            {
                break;
            }
            int sn =
                cli_trap_signum(sname);
            /* Find or alloc slot */
            int slot = -1;
            for(int i = 0;
                i < CLI_TRAP_MAXSIGS;
                i++)
            {
                if(cli_traps[i].used
                   && cli_traps[i].signum
                   == sn)
                {
                    slot = i;
                    break;
                }
            }
            if(slot < 0)
            {
                for(int i = 0;
                    i < CLI_TRAP_MAXSIGS;
                    i++)
                {
                    if(!cli_traps[i]
                        .used)
                    {
                        slot = i;
                        break;
                    }
                }
            }
            if(slot >= 0)
            {
                cli_traps[slot].signum =
                    sn;
                strncpy(
                    cli_traps[slot].cmd,
                    tcmd,
                    CLI_TRAP_CMDLEN - 1);
                cli_traps[slot].used = 1;
            }
        }
        return 1;
    }

    /* set -e / set -x / set +e / set +x */
    if(starts_with(p, "set ")
       || starts_with(p, "set\t"))
    {
        p += 3;
        p = strip_ws(p);
        while(*p != '\0')
        {
            if(*p == '-' || *p == '+')
            {
                int on = (*p == '-');
                p++;
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t')
                {
                    if(*p == 'e')
                    {
                        cli_flag_errexit =
                            on;
                    }
                    else if(*p == 'x')
                    {
                        cli_flag_xtrace =
                            on;
                    }
                    p++;
                }
            }
            else
            {
                p++;
            }
            p = strip_ws(p);
        }
        return 1;
    }

    /* export VAR=val — set env variable */
    if(starts_with(p, "export ")
       || starts_with(p, "export\t"))
    {
        p += 6;
        p = strip_ws(p);
        const char *eq = strchr(p, '=');
        if(eq != NULL)
        {
            char ename[CLI_VAR_NAMELEN];
            int elen = (int)(eq - p);
            if(elen >= CLI_VAR_NAMELEN)
            {
                elen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(ename, p,
                   (size_t) elen);
            ename[elen] = '\0';
            const char *eval = eq + 1;
            /* Strip quotes */
            int evlen =
                (int) strlen(eval);
            if(evlen >= 2
               && ((eval[0] == '"'
                    && eval[evlen - 1]
                    == '"')
                   || (eval[0] == '\''
                       && eval[
                           evlen - 1]
                       == '\'')))
            {
                char ebuf[
                    CLI_VAR_VALLEN];
                memcpy(ebuf,
                       eval + 1,
                       (size_t)
                       (evlen - 2));
                ebuf[evlen - 2] = '\0';
                setenv(ename,
                       ebuf, 1);
                cli_var_set(ename,
                            ebuf);
            }
            else
            {
                setenv(ename,
                       eval, 1);
                cli_var_set(ename,
                            eval);
            }
        }
        else
        {
            /* export VAR (no =val):
             * push current value */
            const char *eval =
                cli_var_get(p);
            if(eval != NULL)
            {
                setenv(p, eval, 1);
            }
        }
        return 1;
    }

    /* source file  or  . file */
    if(starts_with(p, "source ")
       || starts_with(p, "source\t")
       || (p[0] == '.'
           && (p[1] == ' '
               || p[1] == '\t')))
    {
        const char *fn = p;
        if(p[0] == '.')
        {
            fn = p + 1;
        }
        else
        {
            fn = p + 6;
        }
        fn = strip_ws(fn);
        FILE *sf = fopen(fn, "r");
        if(sf == NULL)
        {
            fprintf(stderr,
                    "source: %s: "
                    "No such file\n",
                    fn);
        }
        else
        {
            char sline[
                STRINGMAXLEN_CLICMDLINE];
            while(fgets(
                      sline,
                      (int) sizeof(
                          sline),
                      sf) != NULL)
            {
                /* Strip newline */
                int sl =
                    (int) strlen(sline);
                if(sl > 0
                   && sline[sl - 1]
                   == '\n')
                {
                    sline[sl - 1] =
                        '\0';
                }
                CLI_execute_line(sline);
            }
            fclose(sf);
        }
        return 1;
    }

    /* readonly VAR=val */
    if(starts_with(p, "readonly ")
       || starts_with(p,
                      "readonly\t"))
    {
        p += 8;
        p = strip_ws(p);
        const char *eq =
            strchr(p, '=');
        if(eq != NULL)
        {
            char rn[CLI_VAR_NAMELEN];
            int rl = (int)(eq - p);
            if(rl >= CLI_VAR_NAMELEN)
            {
                rl =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(rn, p,
                   (size_t) rl);
            rn[rl] = '\0';
            cli_var_set(rn, eq + 1);
        }
        /* Mark as readonly via env */
        return 1;
    }

    /* break [N] */
    if(starts_with(p, "break")
       && (p[5] == '\0'
           || p[5] == ' '
           || p[5] == '\t'))
    {
        /* Set break level */
        int n = 1;
        if(p[5] != '\0')
        {
            n = (int) strtol(
                p + 5, NULL, 10);
            if(n < 1)
            {
                n = 1;
            }
        }
        cli_last_retval = n;
        return 1;
    }

    /* continue [N] */
    if(starts_with(p, "continue")
       && (p[8] == '\0'
           || p[8] == ' '
           || p[8] == '\t'))
    {
        int n = 1;
        if(p[8] != '\0')
        {
            n = (int) strtol(
                p + 8, NULL, 10);
            if(n < 1)
            {
                n = 1;
            }
        }
        cli_last_retval = n;
        return 1;
    }

    /* printf "fmt" args... */
    if(starts_with(p, "printf ")
       || starts_with(p, "printf\t"))
    {
        p += 6;
        p = strip_ws(p);
        /* Parse format string */
        char fmt[
            STRINGMAXLEN_CLICMDLINE];
        int fi = 0;
        char delim = ' ';
        if(*p == '"' || *p == '\'')
        {
            delim = *p;
            p++;
        }
        while(*p != '\0'
              && *p != delim
              && fi
              < STRINGMAXLEN_CLICMDLINE
              - 1)
        {
            if(*p == '\\'
               && p[1] != '\0')
            {
                switch(p[1])
                {
                case 'n':
                    fmt[fi++] = '\n';
                    break;
                case 't':
                    fmt[fi++] = '\t';
                    break;
                case '\\':
                    fmt[fi++] = '\\';
                    break;
                default:
                    fmt[fi++] = p[1];
                    break;
                }
                p += 2;
            }
            else
            {
                fmt[fi++] = *p++;
            }
        }
        fmt[fi] = '\0';
        if(*p == delim)
        {
            p++;
        }
        /* Collect remaining args */
        char args[32][256];
        int nargs = 0;
        p = strip_ws(p);
        while(*p != '\0'
              && nargs < 32)
        {
            int ai = 0;
            if(*p == '"'
               || *p == '\'')
            {
                char qc = *p++;
                while(*p != '\0'
                      && *p != qc
                      && ai < 255)
                {
                    args[nargs][ai++] =
                        *p++;
                }
                if(*p == qc)
                {
                    p++;
                }
            }
            else
            {
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t'
                      && ai < 255)
                {
                    args[nargs][ai++] =
                        *p++;
                }
            }
            args[nargs][ai] = '\0';
            nargs++;
            p = strip_ws(p);
        }
        /* Simple printf: scan fmt for %s/%d */
        int ai = 0;
        const char *f = fmt;
        while(*f != '\0')
        {
            if(*f == '%'
               && f[1] != '\0')
            {
                if(f[1] == 's')
                {
                    if(ai < nargs)
                    {
                        printf("%s",
                               args[
                                   ai++]);
                    }
                    f += 2;
                }
                else if(f[1] == 'd')
                {
                    if(ai < nargs)
                    {
                        printf(
                            "%d",
                            (int) strtol(
                                args[
                                    ai++],
                                NULL,
                                10));
                    }
                    f += 2;
                }
                else if(f[1] == 'f')
                {
                    if(ai < nargs)
                    {
                        printf(
                            "%f",
                            strtod(
                                args[
                                    ai++],
                                NULL));
                    }
                    f += 2;
                }
                else if(f[1] == '%')
                {
                    putchar('%');
                    f += 2;
                }
                else
                {
                    putchar(*f);
                    f++;
                }
            }
            else
            {
                putchar(*f);
                f++;
            }
        }
        fflush(stdout);
        return 1;
    }

    /* getopts optstring var */
    if(starts_with(p, "getopts ")
       || starts_with(p,
                      "getopts\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* Parse optstring */
        char optstr[128];
        {
            int oi = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && oi < 127)
            {
                optstr[oi++] = *p++;
            }
            optstr[oi] = '\0';
        }
        p = strip_ws(p);
        /* Parse varname */
        char gvar[CLI_VAR_NAMELEN];
        {
            int gi = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && gi
                  < CLI_VAR_NAMELEN - 1)
            {
                gvar[gi++] = *p++;
            }
            gvar[gi] = '\0';
        }
        /* Get OPTIND */
        const char *oidx =
            cli_var_get("OPTIND");
        int optind_val =
            oidx ? (int) strtol(
                       oidx, NULL, 10)
            : 1;
        /* Get current positional arg */
        char pname[32];
        snprintf(pname, sizeof(pname),
                 "%d", optind_val);
        const char *arg =
            cli_var_get(pname);
        if(arg == NULL
           || arg[0] != '-'
           || arg[1] == '\0')
        {
            cli_var_set(gvar, "?");
            cli_last_retval = 1;
            return 1;
        }
        char optch = arg[1];
        /* Check if valid */
        const char *found =
            strchr(optstr, optch);
        if(found == NULL)
        {
            cli_var_set(gvar, "?");
        }
        else
        {
            char ov[2];
            ov[0] = optch;
            ov[1] = '\0';
            cli_var_set(gvar, ov);
            if(found[1] == ':')
            {
                /* Next arg is OPTARG */
                optind_val++;
                char pn2[32];
                snprintf(
                    pn2, sizeof(pn2),
                    "%d", optind_val);
                const char *oa =
                    cli_var_get(pn2);
                if(oa != NULL)
                {
                    cli_var_set(
                        "OPTARG", oa);
                }
            }
        }
        optind_val++;
        {
            char oib[32];
            snprintf(oib,
                     sizeof(oib),
                     "%d",
                     optind_val);
            cli_var_set("OPTIND", oib);
        }
        cli_last_retval = 0;
        return 1;
    }

    /* local VAR=val — set variable in
     * current scope (true shadowing) */
    if(starts_with(p, "local ")
       || starts_with(p, "local\t"))
    {
        p += 5;
        p = strip_ws(p);
        
        char vn[CLI_VAR_NAMELEN];
        const char *eq = strchr(p, '=');
        if(eq != NULL)
        {
            int nl = (int)(eq - p);
            if(nl >= CLI_VAR_NAMELEN) nl = CLI_VAR_NAMELEN - 1;
            memcpy(vn, p, (size_t) nl);
            vn[nl] = '\0';
        }
        else
        {
            strncpy(vn, p, CLI_VAR_NAMELEN - 1);
            vn[CLI_VAR_NAMELEN - 1] = '\0';
        }
        
        /* Save shadow if in function scope and not already shadowed */
        if(cli_local_depth > 0)
        {
            int scount = cli_local_shadow_count[cli_local_depth];
            int already_shadowed = 0;
            for(int i = 0; i < scount; i++)
            {
                if(strcmp(cli_local_shadows[cli_local_depth][i].name, vn) == 0)
                {
                    already_shadowed = 1;
                    break;
                }
            }
            if(!already_shadowed && scount < CLI_MAX_LOCALS_PER_FUNC)
            {
                CLI_LOCAL_SHADOW *sh = &cli_local_shadows[cli_local_depth][scount];
                strncpy(sh->name, vn, CLI_VAR_NAMELEN - 1);
                sh->name[CLI_VAR_NAMELEN - 1] = '\0';
                const char *ov = cli_var_get(vn);
                sh->was_used = (ov != NULL) ? 1 : 0;
                if(ov != NULL)
                {
                    strncpy(sh->val, ov, CLI_VAR_VALLEN - 1);
                    sh->val[CLI_VAR_VALLEN - 1] = '\0';
                }
                cli_local_shadow_count[cli_local_depth]++;
            }
        }

        if(eq != NULL)
        {
            cli_var_set(vn, eq + 1);
        }
        else
        {
            if(cli_var_get(vn) == NULL)
            {
                cli_var_set(vn, "");
            }
        }
        return 1;
    }

    /* declare [-i|-a|-r|-x] VAR[=val] */
    if(starts_with(p, "declare ")
       || starts_with(p,
                      "declare\t")
       || starts_with(p, "typeset ")
       || starts_with(p,
                      "typeset\t"))
    {
        p += 7;
        if(p[0] == ' ' || p[0] == '\t')
        {
            p++;
        }
        p = strip_ws(p);
        /* Parse flags */
        int fl_int = 0;
        int fl_arr = 0;
        int fl_ro = 0;
        int fl_exp = 0;
        while(p[0] == '-')
        {
            p++;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t')
            {
                if(*p == 'i')
                {
                    fl_int = 1;
                }
                else if(*p == 'a')
                {
                    fl_arr = 1;
                }
                else if(*p == 'r')
                {
                    fl_ro = 1;
                }
                else if(*p == 'x')
                {
                    fl_exp = 1;
                }
                p++;
            }
            p = strip_ws(p);
        }
        /* Parse VAR=val */
        const char *eq =
            strchr(p, '=');
        char vn[CLI_VAR_NAMELEN];
        if(eq != NULL)
        {
            int nl = (int)(eq - p);
            if(nl >= CLI_VAR_NAMELEN)
            {
                nl =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, p,
                   (size_t) nl);
            vn[nl] = '\0';
            if(fl_arr)
            {
                /* declare -a arr */
                for(int k = 0;
                    k < CLI_MAX_ARRAYS;
                    k++)
                {
                    if(!cli_arrays[k]
                        .used)
                    {
                        cli_arrays[k]
                            .used = 1;
                        strncpy(
                            cli_arrays[k]
                            .name,
                            vn,
                            CLI_VAR_NAMELEN
                            - 1);
                        cli_arrays[k]
                            .nelem = 0;
                        break;
                    }
                }
            }
            else if(fl_int)
            {
                /* Integer eval */
                long iv = strtol(
                    eq + 1, NULL, 0);
                char ib[32];
                snprintf(ib,
                         sizeof(ib),
                         "%ld", iv);
                cli_var_set(vn, ib);
            }
            else
            {
                cli_var_set(
                    vn, eq + 1);
            }
            if(fl_exp)
            {
                const char *v =
                    cli_var_get(vn);
                if(v != NULL)
                {
                    setenv(vn, v, 1);
                }
            }
        }
        else
        {
            strncpy(vn, p,
                    CLI_VAR_NAMELEN
                    - 1);
            vn[CLI_VAR_NAMELEN - 1] =
                '\0';
            if(cli_var_get(vn) == NULL)
            {
                cli_var_set(vn, "");
            }
        }
        (void) fl_ro; /* TODO: track */
        return 1;
    }

    /* let "expr" or let expr */
    if(starts_with(p, "let ")
       || starts_with(p, "let\t"))
    {
        p += 3;
        p = strip_ws(p);
        /* Strip optional quotes */
        char lexpr[
            STRINGMAXLEN_CLICMDLINE];
        strncpy(lexpr, p,
                STRINGMAXLEN_CLICMDLINE
                - 1);
        lexpr[STRINGMAXLEN_CLICMDLINE
              - 1] = '\0';
        int ll = (int) strlen(lexpr);
        if(ll >= 2
           && ((lexpr[0] == '"'
                && lexpr[ll - 1]
                == '"')
               || (lexpr[0] == '\''
                   && lexpr[ll - 1]
                   == '\'')))
        {
            lexpr[ll - 1] = '\0';
            memmove(lexpr,
                    lexpr + 1,
                    (size_t)(ll - 1));
        }
        /* Build $(( )) expression */
        char ecmd[
            STRINGMAXLEN_CLICMDLINE];
        snprintf(ecmd, sizeof(ecmd),
                 "$((%s))", lexpr);
        /* Find assignment target */
        char *aeq =
            strchr(lexpr, '=');
        if(aeq != NULL
           && aeq != lexpr
           && aeq[-1] != '!'
           && aeq[-1] != '<'
           && aeq[-1] != '>')
        {
            /* Has assignment, e.g.
             * let "x = 1 + 2" */
            *aeq = '\0';
            /* Trim target var */
            char tvar[
                CLI_VAR_NAMELEN];
            {
                const char *ts =
                    lexpr;
                while(*ts == ' '
                      || *ts == '\t')
                {
                    ts++;
                }
                int ti = 0;
                while(*ts != '\0'
                      && *ts != ' '
                      && *ts != '\t'
                      && ti
                      < CLI_VAR_NAMELEN
                      - 1)
                {
                    tvar[ti++] =
                        *ts++;
                }
                tvar[ti] = '\0';
            }
            /* Eval RHS */
            const char *rhs =
                aeq + 1;
            while(*rhs == ' '
                  || *rhs == '\t')
            {
                rhs++;
            }
            char arith[
                STRINGMAXLEN_CLICMDLINE
            ];
            snprintf(arith,
                     sizeof(arith),
                     "$((%s))", rhs);
            cli_expand_env(
                arith,
                STRINGMAXLEN_CLICMDLINE
            );
            cli_var_set(
                tvar, arith);
        }
        else
        {
            /* No assignment, just
             * evaluate */
            cli_expand_env(
                ecmd,
                STRINGMAXLEN_CLICMDLINE
            );
            cli_last_retval =
                (strtol(ecmd, NULL,
                        10) == 0) ? 1
                : 0;
        }
        return 1;
    }

    /* eval "cmd" — execute string */
    if(starts_with(p, "eval ")
       || starts_with(p, "eval\t"))
    {
        p += 4;
        p = strip_ws(p);
        /* Strip outer quotes */
        char ecmd[
            STRINGMAXLEN_CLICMDLINE];
        strncpy(ecmd, p,
                STRINGMAXLEN_CLICMDLINE
                - 1);
        ecmd[STRINGMAXLEN_CLICMDLINE
             - 1] = '\0';
        int el = (int) strlen(ecmd);
        if(el >= 2
           && ((ecmd[0] == '"'
                && ecmd[el - 1]
                == '"')
               || (ecmd[0] == '\''
                   && ecmd[el - 1]
                   == '\'')))
        {
            ecmd[el - 1] = '\0';
            memmove(ecmd, ecmd + 1,
                    (size_t)(el - 1));
        }
        CLI_execute_line(ecmd);
        return 1;
    }

    /* type / command -v — check cmd */
    if(starts_with(p, "type ")
       || starts_with(p, "type\t"))
    {
        p += 4;
        p = strip_ws(p);
        /* Search registered commands */
        int found = 0;
        for(int ci = 0;
            ci < data.NBcmd; ci++)
        {
            if(strcmp(
                   data.cmd[ci].key,
                   p) == 0)
            {
                printf("%s is a "
                       "CLI command\n",
                       p);
                found = 1;
                break;
            }
        }
        if(!found)
        {
            printf("%s: not found\n",
                   p);
            cli_last_retval = 1;
        }
        else
        {
            cli_last_retval = 0;
        }
        return 1;
    }
    if(starts_with(p, "command ")
       || starts_with(p,
                      "command\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* command -v cmd */
        if(starts_with(p, "-v "))
        {
            p += 3;
            p = strip_ws(p);
            int found = 0;
            for(int ci = 0;
                ci < data.NBcmd; ci++)
            {
                if(strcmp(
                       data.cmd[ci]
                       .key,
                       p) == 0)
                {
                    printf("%s\n", p);
                    found = 1;
                    break;
                }
            }
            cli_last_retval =
                found ? 0 : 1;
            return 1;
        }
        /* command cmd — run directly */
        CLI_execute_line((char *) p);
        return 1;
    }

    /* timeout N cmd */
    if(starts_with(p, "timeout ")
       || starts_with(p,
                      "timeout\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* Parse timeout seconds */
        char *endp;
        double tsec =
            strtod(p, &endp);
        if(endp == p)
        {
            fprintf(stderr,
                    "timeout: "
                    "invalid time\n");
            cli_last_retval = 1;
            return 1;
        }
        const char *cmd_start =
            endp;
        while(*cmd_start == ' '
              || *cmd_start == '\t')
        {
            cmd_start++;
        }
        pid_t tpid = fork();
        if(tpid == 0)
        {
            /* Child: run cmd */
            CLI_execute_line(
                (char *) cmd_start);
            _exit(cli_last_retval);
        }
        else if(tpid > 0)
        {
            /* Parent: wait with
             * timeout */
            struct timespec ts;
            ts.tv_sec =
                (time_t) tsec;
            ts.tv_nsec =
                (long)((tsec
                        - (double)
                        ts.tv_sec)
                       * 1e9);
            int wst = 0;
            struct timespec start;
            clock_gettime(
                CLOCK_MONOTONIC,
                &start);
            while(1)
            {
                int wr =
                    waitpid(tpid,
                            &wst,
                            WNOHANG);
                if(wr > 0)
                {
                    cli_last_retval =
                        WEXITSTATUS(
                            wst);
                    break;
                }
                struct timespec now;
                clock_gettime(
                    CLOCK_MONOTONIC,
                    &now);
                double elapsed =
                    (double)(
                        now.tv_sec
                        - start
                        .tv_sec)
                    + (double)(
                        now.tv_nsec
                        - start
                        .tv_nsec)
                    / 1e9;
                if(elapsed >= tsec)
                {
                    kill(tpid,
                         SIGTERM);
                    usleep(100000);
                    kill(tpid,
                         SIGKILL);
                    waitpid(tpid,
                            &wst, 0);
                    cli_last_retval =
                        124;
                    break;
                }
                usleep(10000);
            }
        }
        return 1;
    }

    /* mapfile / readarray -t arr < file */
    if(starts_with(p, "mapfile ")
       || starts_with(p, "mapfile\t")
       || starts_with(p,
                      "readarray ")
       || starts_with(p,
                      "readarray\t"))
    {
        /* Skip command name */
        if(p[0] == 'm')
        {
            p += 7;
        }
        else
        {
            p += 9;
        }
        p = strip_ws(p);
        /* Parse optional -t flag */
        int strip_nl = 0;
        if(p[0] == '-'
           && p[1] == 't')
        {
            strip_nl = 1;
            p += 2;
            p = strip_ws(p);
        }
        /* Array name */
        char aname[CLI_VAR_NAMELEN];
        {
            int ai = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && *p != '<'
                  && ai
                  < CLI_VAR_NAMELEN - 1)
            {
                aname[ai++] = *p++;
            }
            aname[ai] = '\0';
        }
        p = strip_ws(p);
        /* Check for < file */
        FILE *mf = stdin;
        int should_close = 0;
        if(*p == '<')
        {
            p++;
            p = strip_ws(p);
            mf = fopen(p, "r");
            if(mf == NULL)
            {
                fprintf(stderr,
                        "mapfile: "
                        "%s: "
                        "cannot open\n",
                        p);
                return 1;
            }
            should_close = 1;
        }
        /* Find or create array */
        int slot = -1;
        for(int k = 0;
            k < CLI_MAX_ARRAYS; k++)
        {
            if(cli_arrays[k].used
               && strcmp(
                      cli_arrays[k]
                      .name,
                      aname) == 0)
            {
                slot = k;
                cli_arrays[k].nelem =
                    0;
                break;
            }
        }
        if(slot < 0)
        {
            for(int k = 0;
                k < CLI_MAX_ARRAYS;
                k++)
            {
                if(!cli_arrays[k].used)
                {
                    slot = k;
                    cli_arrays[k]
                        .used = 1;
                    strncpy(
                        cli_arrays[k]
                        .name,
                        aname,
                        CLI_VAR_NAMELEN
                        - 1);
                    cli_arrays[k]
                        .nelem = 0;
                    break;
                }
            }
        }
        if(slot >= 0)
        {
            char mline[
                CLI_VAR_VALLEN];
            while(
                fgets(
                    mline,
                    CLI_VAR_VALLEN,
                    mf) != NULL
                && cli_arrays[slot]
                   .nelem
                < CLI_ARRAY_MAXELEM)
            {
                if(strip_nl)
                {
                    int ml =
                        (int) strlen(
                            mline);
                    if(ml > 0
                       && mline[ml - 1]
                       == '\n')
                    {
                        mline[ml - 1] =
                            '\0';
                    }
                }
                strncpy(
                    cli_arrays[slot]
                    .elem[
                        cli_arrays[slot]
                            .nelem],
                    mline,
                    CLI_VAR_VALLEN
                    - 1);
                cli_arrays[slot]
                    .nelem++;
            }
        }
        if(should_close)
        {
            fclose(mf);
        }
        return 1;
    }

    /* wait — wait for bg children */
    if(strcmp(p, "wait") == 0
       || starts_with(p, "wait ")
       || starts_with(p, "wait\t"))
    {
        int wstatus;
        while(waitpid(-1, &wstatus,
                      0) > 0)
        {
            /* reap all children */
        }
        cli_last_retval = 0;
        return 1;
    }

    /* [[ expr ]] — extended test */
    if(starts_with(p, "[[ "))
    {
        int plen = (int) strlen(p);
        if(plen >= 5
           && p[plen - 1] == ']'
           && p[plen - 2] == ']')
        {
            /* Extract inner expr */
            char texpr[
                STRINGMAXLEN_CLICMDLINE];
            memcpy(texpr, p + 3,
                   (size_t)(plen - 5));
            texpr[plen - 5] = '\0';
            /* Trim whitespace */
            int tlen =
                (int) strlen(texpr);
            while(tlen > 0
                  && (texpr[tlen - 1]
                      == ' '
                      || texpr[tlen - 1]
                      == '\t'))
            {
                texpr[--tlen] = '\0';
            }
            int result =
                cli_eval_test(texpr);
            cli_last_retval =
                result ? 0 : 1;
        }
        return 1;
    }

    /* local VAR=val — set variable (only
     * meaningful inside function, but works
     * anywhere) */
    if(starts_with(p, "local ")
       || starts_with(p, "local\t"))
    {
        p += 5;
        p = strip_ws(p);
        cli_try_var_assign(p);
        return 1;
    }

    /* if ... */
    if(starts_with(p, "if ")
       || starts_with(p, "if\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_IF;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* while ... */
    if(starts_with(p, "while ")
       || starts_with(p, "while\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_WHILE;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* for ... */
    if(starts_with(p, "for ")
       || starts_with(p, "for\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_FOR;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* select VAR in val1 val2; do ... */
    if(starts_with(p, "select ")
       || starts_with(p, "select\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type =
            CLI_BLOCK_SELECT;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE
                - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* function name { ... } */
    if(starts_with(p, "function ")
       || starts_with(p, "function\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_FUNC;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* case <word> in ... esac */
    if(starts_with(p, "case ")
       || starts_with(p, "case\t"))
    {
        if(cli_block_level
           >= CLI_BLOCK_MAXDEPTH)
        {
            printf("Error: max block "
                   "nesting exceeded\n");
            return 1;
        }
        CLI_BLOCK *blk =
            &cli_block_stack[
                cli_block_level];
        memset(blk, 0, sizeof(*blk));
        blk->type = CLI_BLOCK_CASE;
        blk->active = 1;
        strncpy(blk->lines[0], p,
                STRINGMAXLEN_CLICMDLINE - 1);
        blk->nlines = 1;
        cli_block_level++;
        return 1;
    }

    /* ==============================
     * Tier 9: true / false
     * ============================== */

    if(strcmp(p, "true") == 0)
    {
        cli_last_retval = 0;
        return 1;
    }
    if(strcmp(p, "false") == 0)
    {
        cli_last_retval = 1;
        return 1;
    }

    /* ==============================
     * Tier 9: (( expr )) conditional
     * ============================== */

    if(starts_with(p, "((")
       && strlen(p) >= 5)
    {
        int plen = (int) strlen(p);
        if(p[plen - 1] == ')'
           && p[plen - 2] == ')')
        {
            char aexpr[
                STRINGMAXLEN_CLICMDLINE
            ];
            int elen = plen - 4;
            if(elen
               >= STRINGMAXLEN_CLICMDLINE)
            {
                elen =
                    STRINGMAXLEN_CLICMDLINE
                    - 1;
            }
            memcpy(aexpr, p + 2,
                   (size_t) elen);
            aexpr[elen] = '\0';
            /* Wrap in $(( )) and
             * expand */
            char wrap[
                STRINGMAXLEN_CLICMDLINE
            ];
            snprintf(wrap,
                     sizeof(wrap),
                     "$((%s))",
                     aexpr);
            cli_expand_env(
                wrap,
                STRINGMAXLEN_CLICMDLINE
            );
            long val =
                strtol(wrap, NULL, 10);
            cli_last_retval =
                (val != 0) ? 0 : 1;
            return 1;
        }
    }

    /* ==============================
     * Tier 9: alias / unalias
     * ============================== */

    if(starts_with(p, "alias ")
       || starts_with(p, "alias\t")
       || strcmp(p, "alias") == 0)
    {
        p += 5;
        p = strip_ws(p);
        if(*p == '\0')
        {
            /* List all aliases */
            for(int k = 0;
                k < data.NBalias;
                k++)
            {
                printf("alias %s="
                       "'%s'\n",
                       data.alias[
                           k].name,
                       data.alias[
                           k].cmd);
            }
        }
        else
        {
            /* alias name='cmd' or
             * alias name=cmd */
            char *eq = strchr(p, '=');
            if(eq != NULL)
            {
                char aname[
                    CLI_ALIAS_NAMELEN
                ];
                int nl =
                    (int)(eq - p);
                if(nl
                   >= CLI_ALIAS_NAMELEN)
                {
                    nl =
                        CLI_ALIAS_NAMELEN
                        - 1;
                }
                memcpy(aname, p,
                       (size_t) nl);
                aname[nl] = '\0';
                const char *av =
                    eq + 1;
                /* Strip quotes */
                int avl =
                    (int) strlen(av);
                if(avl >= 2
                   && ((av[0] == '\''
                        && av[avl - 1]
                        == '\'')
                       || (av[0] == '"'
                           && av[
                               avl - 1]
                           == '"')))
                {
                    av++;
                    avl -= 2;
                }
                /* Update existing? */
                int slot = -1;
                for(int k = 0;
                    k < data.NBalias;
                    k++)
                {
                    if(strcmp(
                        data.alias[k]
                        .name,
                        aname) == 0)
                    {
                        slot = k;
                        break;
                    }
                }
                if(slot < 0
                   && data.NBalias
                   < CLI_MAX_ALIASES)
                {
                    slot =
                        data.NBalias++;
                }
                if(slot >= 0)
                {
                    strncpy(
                        data.alias[
                            slot].name,
                        aname,
                        CLI_ALIAS_NAMELEN
                        - 1);
                    data.alias[slot]
                        .name[
                        CLI_ALIAS_NAMELEN
                        - 1] = '\0';
                    int cl =
                        avl
                        < CLI_ALIAS_CMDLEN
                        - 1
                        ? avl
                        : CLI_ALIAS_CMDLEN
                        - 1;
                    memcpy(
                        data.alias[
                            slot].cmd,
                        av,
                        (size_t) cl);
                    data.alias[slot]
                        .cmd[cl] = '\0';
                }
            }
        }
        return 1;
    }

    if(starts_with(p, "unalias ")
       || starts_with(p, "unalias\t"))
    {
        p += 7;
        p = strip_ws(p);
        for(int k = 0;
            k < data.NBalias; k++)
        {
            if(strcmp(
                data.alias[k].name,
                p) == 0)
            {
                /* Shift remaining */
                for(int j = k;
                    j < data.NBalias
                    - 1; j++)
                {
                    data.alias[j] =
                        data.alias[
                            j + 1];
                }
                data.NBalias--;
                break;
            }
        }
        return 1;
    }

    /* ==============================
     * Tier 9: assoc array map[k]=v
     * ============================== */

    {
        const char *br =
            strchr(p, '[');
        if(br != NULL)
        {
            const char *brend =
                strchr(br, ']');
            if(brend != NULL
               && *(brend + 1) == '=')
            {
                char aname[
                    CLI_VAR_NAMELEN];
                int nl =
                    (int)(br - p);
                if(nl
                   >= CLI_VAR_NAMELEN)
                {
                    nl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(aname, p,
                       (size_t) nl);
                aname[nl] = '\0';
                char key[
                    CLI_VAR_NAMELEN];
                int kl =
                    (int)(brend
                          - br - 1);
                if(kl
                   >= CLI_VAR_NAMELEN)
                {
                    kl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(key, br + 1,
                       (size_t) kl);
                key[kl] = '\0';
                const char *val =
                    brend + 2;
                /* Find or create
                 * assoc array */
                int slot = -1;
                for(int k = 0;
                    k < CLI_MAX_ASSOC;
                    k++)
                {
                    if(cli_assoc[k]
                        .used
                       && strcmp(
                           cli_assoc[k]
                           .name,
                           aname)
                       == 0)
                    {
                        slot = k;
                        break;
                    }
                }
                if(slot < 0)
                {
                    for(int k = 0;
                        k
                        < CLI_MAX_ASSOC;
                        k++)
                    {
                        if(!cli_assoc[
                            k].used)
                        {
                            slot = k;
                            cli_assoc[k]
                                .used
                                = 1;
                            strncpy(
                                cli_assoc[
                                    k]
                                .name,
                                aname,
                                CLI_VAR_NAMELEN
                                - 1);
                            cli_assoc[k]
                                .nelem
                                = 0;
                            break;
                        }
                    }
                }
                if(slot >= 0)
                {
                    /* Find existing
                     * key or add */
                    int ki = -1;
                    for(int k = 0;
                        k
                        < cli_assoc[
                            slot]
                        .nelem;
                        k++)
                    {
                        if(strcmp(
                            cli_assoc[
                                slot]
                            .keys[k],
                            key) == 0)
                        {
                            ki = k;
                            break;
                        }
                    }
                    if(ki < 0
                       && cli_assoc[
                           slot]
                       .nelem
                       < CLI_ASSOC_MAXELEM)
                    {
                        ki =
                            cli_assoc[
                                slot]
                            .nelem++;
                        strncpy(
                            cli_assoc[
                                slot]
                            .keys[ki],
                            key,
                            CLI_VAR_NAMELEN
                            - 1);
                    }
                    if(ki >= 0)
                    {
                        strncpy(
                            cli_assoc[
                                slot]
                            .vals[ki],
                            val,
                            CLI_VAR_VALLEN
                            - 1);
                    }
                }
                return 1;
            }
        }
    }

    /* ==============================
     * Tier 10: basename / dirname
     * ============================== */

    if(starts_with(p, "basename "))
    {
        p += 8;
        p = strip_ws(p);
        /* Find last / */
        const char *sl =
            strrchr(p, '/');
        if(sl != NULL)
        {
            printf("%s\n", sl + 1);
        }
        else
        {
            printf("%s\n", p);
        }
        cli_last_retval = 0;
        return 1;
    }

    if(starts_with(p, "dirname "))
    {
        p += 7;
        p = strip_ws(p);
        const char *sl =
            strrchr(p, '/');
        if(sl != NULL && sl != p)
        {
            printf("%.*s\n",
                   (int)(sl - p), p);
        }
        else if(sl == p)
        {
            printf("/\n");
        }
        else
        {
            printf(".\n");
        }
        cli_last_retval = 0;
        return 1;
    }

    /* ==============================
     * Tier 10: pushd / popd / dirs
     * ============================== */

    if(starts_with(p, "pushd ")
       || starts_with(p, "pushd\t"))
    {
        p += 5;
        p = strip_ws(p);
        char cwd[1024];
        if(getcwd(cwd, sizeof(cwd))
           != NULL)
        {
            /* Push current dir as
             * cli var */
            char idx[32];
            /* Count existing
             * _dirstack entries */
            int dcnt = 0;
            for(int k = 0;
                k < CLI_MAX_VARS; k++)
            {
                if(cli_vars[k].used
                   && strncmp(
                       cli_vars[k]
                       .name,
                       "_ds_",
                       4) == 0)
                {
                    dcnt++;
                }
            }
            snprintf(idx,
                     sizeof(idx),
                     "_ds_%d", dcnt);
            cli_var_set(idx, cwd);
        }
        if(chdir(p) != 0)
        {
            printf("pushd: %s: %s\n",
                   p,
                   strerror(errno));
            cli_last_retval = 1;
        }
        else
        {
            cli_last_retval = 0;
        }
        return 1;
    }

    if(strcmp(p, "popd") == 0)
    {
        /* Find highest _ds_N */
        int maxn = -1;
        int maxk = -1;
        for(int k = 0;
            k < CLI_MAX_VARS; k++)
        {
            if(cli_vars[k].used
               && strncmp(
                   cli_vars[k].name,
                   "_ds_", 4) == 0)
            {
                int n = atoi(
                    cli_vars[k].name
                    + 4);
                if(n > maxn)
                {
                    maxn = n;
                    maxk = k;
                }
            }
        }
        if(maxk >= 0)
        {
            if(chdir(
                cli_vars[maxk].val)
               != 0)
            {
                printf("popd: %s\n",
                       strerror(
                           errno));
            }
            cli_vars[maxk].used = 0;
        }
        else
        {
            printf("popd: directory "
                   "stack empty\n");
        }
        cli_last_retval = 0;
        return 1;
    }

    if(strcmp(p, "dirs") == 0)
    {
        char cwd[1024];
        if(getcwd(cwd, sizeof(cwd))
           != NULL)
        {
            printf("%s", cwd);
        }
        for(int n = 0;
            n < CLI_MAX_VARS; n++)
        {
            char idx[32];
            snprintf(idx,
                     sizeof(idx),
                     "_ds_%d", n);
            const char *dv =
                cli_var_get(idx);
            if(dv == NULL)
            {
                break;
            }
            printf(" %s", dv);
        }
        printf("\n");
        cli_last_retval = 0;
        return 1;
    }

    /* ==============================
     * Tier 10: seq START [STEP] END
     * ============================== */

    if(starts_with(p, "seq "))
    {
        p += 3;
        p = strip_ws(p);
        double s1 = 0.0;
        double step = 1.0;
        double s2 = 0.0;
        /* Parse up to 3 numbers */
        char *end1 = NULL;
        s1 = strtod(p, &end1);
        if(end1 != NULL
           && *end1 != '\0')
        {
            const char *p2 =
                strip_ws(end1);
            char *end2 = NULL;
            double v2 =
                strtod(p2, &end2);
            if(end2 != NULL
               && *end2 != '\0')
            {
                const char *p3 =
                    strip_ws(end2);
                double v3 =
                    strtod(p3, NULL);
                /* 3-arg: s1 step s2 */
                step = v2;
                s2 = v3;
            }
            else
            {
                /* 2-arg: s1 s2 */
                s2 = v2;
            }
        }
        else
        {
            /* 1-arg: 1..s1 */
            s2 = s1;
            s1 = 1.0;
        }
        if(step > 0.0)
        {
            for(double v = s1;
                v <= s2 + 1e-12;
                v += step)
            {
                printf("%g\n", v);
            }
        }
        else if(step < 0.0)
        {
            for(double v = s1;
                v >= s2 - 1e-12;
                v += step)
            {
                printf("%g\n", v);
            }
        }
        cli_last_retval = 0;
        return 1;
    }

    /* ==============================
     * Tier 11: waitfor_stream
     * ============================== */

    if(starts_with(p, "waitfor_stream "))
    {
        p += 14;
        p = strip_ws(p);
        char sname[CLI_VAR_NAMELEN];
        int si = 0;
        while(*p != '\0'
              && *p != ' '
              && *p != '\t'
              && si
              < CLI_VAR_NAMELEN - 1)
        {
            sname[si++] = *p++;
        }
        sname[si] = '\0';
        p = strip_ws(p);
        double tout = 10.0;
        if(*p != '\0')
        {
            tout = strtod(p, NULL);
        }
        struct timespec wstart;
        clock_gettime(
            CLOCK_MONOTONIC,
            &wstart);
        int found = 0;
        while(1)
        {
            /* Check if SHM exists */
            char shmpath[256];
            snprintf(shmpath,
                     sizeof(shmpath),
                     "/dev/shm/%s"
                     ".im.shm",
                     sname);
            if(access(shmpath,
                      F_OK) == 0)
            {
                found = 1;
                break;
            }
            struct timespec wnow;
            clock_gettime(
                CLOCK_MONOTONIC,
                &wnow);
            double elapsed =
                (double)(wnow.tv_sec
                    - wstart.tv_sec)
                + (double)(
                    wnow.tv_nsec
                    - wstart.tv_nsec)
                / 1e9;
            if(elapsed >= tout)
            {
                break;
            }
            usleep(50000);
        }
        cli_last_retval =
            found ? 0 : 1;
        return 1;
    }

    /* ==============================
     * Tier 11: waitfor_fps
     * ============================== */

    if(starts_with(p, "waitfor_fps "))
    {
        p += 11;
        p = strip_ws(p);
        char fname[CLI_VAR_NAMELEN];
        int fi = 0;
        while(*p != '\0'
              && *p != ' '
              && *p != '\t'
              && fi
              < CLI_VAR_NAMELEN - 1)
        {
            fname[fi++] = *p++;
        }
        fname[fi] = '\0';
        p = strip_ws(p);
        double tout = 10.0;
        if(*p != '\0')
        {
            tout = strtod(p, NULL);
        }
        struct timespec wstart;
        clock_gettime(
            CLOCK_MONOTONIC,
            &wstart);
        int found = 0;
        while(1)
        {
            char fpath[256];
            snprintf(fpath,
                     sizeof(fpath),
                     "/dev/shm/"
                     "fps.%s.shm",
                     fname);
            if(access(fpath,
                      F_OK) == 0)
            {
                found = 1;
                break;
            }
            struct timespec wnow;
            clock_gettime(
                CLOCK_MONOTONIC,
                &wnow);
            double elapsed =
                (double)(wnow.tv_sec
                    - wstart.tv_sec)
                + (double)(
                    wnow.tv_nsec
                    - wstart.tv_nsec)
                / 1e9;
            if(elapsed >= tout)
            {
                break;
            }
            usleep(50000);
        }
        cli_last_retval =
            found ? 0 : 1;
        return 1;
    }

    /* Try alias expansion before
     * user-defined function call */
    {
        char firstword[
            CLI_FUNC_NAMELEN];
        int fw = 0;
        const char *pp = p;
        while(*pp != '\0'
              && *pp != ' '
              && *pp != '\t'
              && fw
              < CLI_FUNC_NAMELEN - 1)
        {
            firstword[fw++] = *pp++;
        }
        firstword[fw] = '\0';

        /* Check aliases first */
        for(int k = 0;
            k < data.NBalias;
            k++)
        {
            if(strcmp(
                   data.alias[k]
                   .name,
                   firstword) == 0)
            {
                /* Build expanded
                 * command */
                char expanded[
                    STRINGMAXLEN_CLICMDLINE
                ];
                snprintf(
                    expanded,
                    sizeof(expanded),
                    "%s%s",
                    data.alias[k].cmd,
                    pp);
                CLI_execute_line(
                    expanded);
                return 1;
            }
        }

        /* Then try user function */
        if(cli_func_find(firstword)
           != NULL)
        {
            p = strip_ws(line);
            cli_try_func_call(p);
            return 1;
        }
    }

    return 0;
}

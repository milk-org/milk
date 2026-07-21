// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stddef.h>
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "milkscript.h"
#include "CLIcore_utils.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_checkargs.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>

extern int cli_block_level;
extern int cli_break_flag;
extern int CLI_trap_enable;
extern int cli_cmd_delay_us;


/**
 * @brief Handler: set a shell variable.
 */
int cli_intercept_cmd_set(const char *p)
{
    if (starts_with(p, "set ") || starts_with(p, "set\t"))
    {
        p += 3;
        p = strip_ws(p);
        while (*p != '\0')
        {
            if (*p == '-' || *p == '+')
            {
                int on = (*p == '-');
                p++;
                while (*p != '\0' && *p != ' ' && *p != '\t')
                {
                    if (*p == 'e')
                    {
                        cli_flag_errexit = on;
                    }
                    else if (*p == 'x')
                    {
                        cli_flag_xtrace = on;
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
    return 0;
}

/**
 * @brief Handler: export a variable to child processes.
 */
int cli_intercept_cmd_export(const char *p)
{
    if (starts_with(p, "export ") || starts_with(p, "export\t"))
    {
        p += 6;
        p              = strip_ws(p);
        const char *eq = strchr(p, '=');
        if (eq != NULL)
        {
            char ename[CLI_VAR_NAMELEN];
            int  elen = (int) (eq - p);
            if (elen >= CLI_VAR_NAMELEN)
            {
                elen = CLI_VAR_NAMELEN - 1;
            }
            memcpy(ename, p, (size_t) elen);
            ename[elen]      = '\0';
            const char *eval = eq + 1;
            /* Strip quotes */
            int evlen = (int) strlen(eval);
            if (evlen >= 2 && ((eval[0] == '"' && eval[evlen - 1] == '"') ||
                               (eval[0] == '\'' && eval[evlen - 1] == '\'')))
            {
                char ebuf[CLI_VAR_VALLEN];
                memcpy(ebuf, eval + 1, (size_t) (evlen - 2));
                ebuf[evlen - 2] = '\0';
                setenv(ename, ebuf, 1);
                cli_var_set(ename, ebuf);
            }
            else
            {
                setenv(ename, eval, 1);
                cli_var_set(ename, eval);
            }
        }
        else
        {
            /* export VAR (no =val):
             * push current value */
            const char *eval = cli_var_get(p);
            if (eval != NULL)
            {
                setenv(p, eval, 1);
            }
        }
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: source (execute) a script file.
 */
int cli_intercept_cmd_source(const char *p)
{
    if (starts_with(p, "source ") || starts_with(p, "source\t") ||
        (p[0] == '.' && (p[1] == ' ' || p[1] == '\t')))
    {
        const char *fn = p;
        if (p[0] == '.')
        {
            fn = p + 1;
        }
        else
        {
            fn = p + 6;
        }
        fn       = strip_ws(fn);
        FILE *sf = fopen(fn, "r");
        if (sf == NULL)
        {
            fprintf(stderr,
                    "source: %s: "
                    "No such file\n",
                    fn);
        }
        else
        {
            char sline[STRINGMAXLEN_CLICMDLINE];
            while (fgets(sline, (int) sizeof(sline), sf) != NULL)
            {
                /* Strip newline */
                int sl = (int) strlen(sline);
                if (sl > 0 && sline[sl - 1] == '\n')
                {
                    sline[sl - 1] = '\0';
                }
                CLI_execute_string(sline);
            }
            fclose(sf);
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_readonly(const char *p)
{
    if (starts_with(p, "readonly ") || starts_with(p, "readonly\t"))
    {
        p += 8;
        p              = strip_ws(p);
        const char *eq = strchr(p, '=');
        if (eq != NULL)
        {
            char rn[CLI_VAR_NAMELEN];
            int  rl = (int) (eq - p);
            if (rl >= CLI_VAR_NAMELEN)
            {
                rl = CLI_VAR_NAMELEN - 1;
            }
            memcpy(rn, p, (size_t) rl);
            rn[rl] = '\0';
            cli_var_set(rn, eq + 1);
        }
        /* Mark as readonly via env */
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_break(const char *p)
{
    if (starts_with(p, "break") && (p[5] == '\0' || p[5] == ' ' || p[5] == '\t'))
    {
        /* Set break level */
        int n = 1;
        if (p[5] != '\0')
        {
            n = (int) strtol(p + 5, NULL, 10);
            if (n < 1)
            {
                n = 1;
            }
        }
        cli_last_retval = n;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_continue(const char *p)
{
    if (starts_with(p, "continue") && (p[8] == '\0' || p[8] == ' ' || p[8] == '\t'))
    {
        int n = 1;
        if (p[8] != '\0')
        {
            n = (int) strtol(p + 8, NULL, 10);
            if (n < 1)
            {
                n = 1;
            }
        }
        cli_last_retval = n;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_printf(const char *p)
{
    if (starts_with(p, "printf ") || starts_with(p, "printf\t"))
    {
        p += 6;
        p = strip_ws(p);
        /* Parse format string */
        char fmt[STRINGMAXLEN_CLICMDLINE];
        int  fi    = 0;
        char delim = ' ';
        if (*p == '"' || *p == '\'')
        {
            delim = *p;
            p++;
        }
        while (*p != '\0' && *p != delim && fi < STRINGMAXLEN_CLICMDLINE - 1)
        {
            if (*p == '\\' && p[1] != '\0')
            {
                switch (p[1])
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
        if (*p == delim)
        {
            p++;
        }
        /* Collect remaining args */
        char args[32][256];
        int  nargs = 0;
        p          = strip_ws(p);
        while (*p != '\0' && nargs < 32)
        {
            int ai = 0;
            if (*p == '"' || *p == '\'')
            {
                char qc = *p++;
                while (*p != '\0' && *p != qc && ai < 255)
                {
                    args[nargs][ai++] = *p++;
                }
                if (*p == qc)
                {
                    p++;
                }
            }
            else
            {
                while (*p != '\0' && *p != ' ' && *p != '\t' && ai < 255)
                {
                    args[nargs][ai++] = *p++;
                }
            }
            args[nargs][ai] = '\0';
            nargs++;
            p = strip_ws(p);
        }
        /* Simple printf: scan fmt for %s/%d */
        int         ai = 0;
        const char *f  = fmt;
        while (*f != '\0')
        {
            if (*f == '%' && f[1] != '\0')
            {
                if (f[1] == 's')
                {
                    if (ai < nargs)
                    {
                        printf("%s", args[ai++]);
                    }
                    f += 2;
                }
                else if (f[1] == 'd')
                {
                    if (ai < nargs)
                    {
                        printf("%d", (int) strtol(args[ai++], NULL, 10));
                    }
                    f += 2;
                }
                else if (f[1] == 'f')
                {
                    if (ai < nargs)
                    {
                        printf("%f", strtod(args[ai++], NULL));
                    }
                    f += 2;
                }
                else if (f[1] == '%')
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
    return 0;
}

int cli_intercept_cmd_getopts(const char *p)
{
    if (starts_with(p, "getopts ") || starts_with(p, "getopts\t"))
    {
        p += 7;
        p = strip_ws(p);
        /* Parse optstring */
        char optstr[128];
        {
            int oi = 0;
            while (*p != '\0' && *p != ' ' && *p != '\t' && oi < 127)
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
            while (*p != '\0' && *p != ' ' && *p != '\t' && gi < CLI_VAR_NAMELEN - 1)
            {
                gvar[gi++] = *p++;
            }
            gvar[gi] = '\0';
        }
        /* Get OPTIND */
        const char *oidx       = cli_var_get("OPTIND");
        int         optind_val = oidx ? (int) strtol(oidx, NULL, 10) : 1;
        /* Get current positional arg */
        char pname[32];
        snprintf(pname, sizeof(pname), "%d", optind_val);
        const char *arg = cli_var_get(pname);
        if (arg == NULL || arg[0] != '-' || arg[1] == '\0')
        {
            cli_var_set(gvar, "?");
            cli_last_retval = 1;
            return 1;
        }
        char optch = arg[1];
        /* Check if valid */
        const char *found = strchr(optstr, optch);
        if (found == NULL)
        {
            cli_var_set(gvar, "?");
        }
        else
        {
            char ov[2];
            ov[0] = optch;
            ov[1] = '\0';
            cli_var_set(gvar, ov);
            if (found[1] == ':')
            {
                /* Next arg is OPTARG */
                optind_val++;
                char pn2[32];
                snprintf(pn2, sizeof(pn2), "%d", optind_val);
                const char *oa = cli_var_get(pn2);
                if (oa != NULL)
                {
                    cli_var_set("OPTARG", oa);
                }
            }
        }
        optind_val++;
        {
            char oib[32];
            snprintf(oib, sizeof(oib), "%d", optind_val);
            cli_var_set("OPTIND", oib);
        }
        cli_last_retval = 0;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_local(const char *p)
{
    if (starts_with(p, "local ") || starts_with(p, "local\t"))
    {
        p += 5;
        p = strip_ws(p);

        char        vn[CLI_VAR_NAMELEN];
        const char *eq = strchr(p, '=');
        if (eq != NULL)
        {
            int nl = (int) (eq - p);
            if (nl >= CLI_VAR_NAMELEN)
            {
                nl = CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, p, (size_t) nl);
            vn[nl] = '\0';
        }
        else
        {
            strncpy(vn, p, CLI_VAR_NAMELEN - 1);
            vn[CLI_VAR_NAMELEN - 1] = '\0';
        }

        /* Save shadow if in function scope and not already shadowed */
        if (cli_local_depth > 0)
        {
            int scount           = cli_local_shadow_count[cli_local_depth];
            int already_shadowed = 0;
            for (int i = 0; i < scount; i++)
            {
                if (strcmp(cli_local_shadows[cli_local_depth][i].name, vn) == 0)
                {
                    already_shadowed = 1;
                    break;
                }
            }
            if (!already_shadowed && scount < CLI_MAX_LOCALS_PER_FUNC)
            {
                CLI_LOCAL_SHADOW *sh = &cli_local_shadows[cli_local_depth][scount];
                strncpy(sh->name, vn, CLI_VAR_NAMELEN - 1);
                sh->name[CLI_VAR_NAMELEN - 1] = '\0';
                const char *ov                = cli_var_get(vn);
                sh->was_used                  = (ov != NULL) ? 1 : 0;
                if (ov != NULL)
                {
                    strncpy(sh->val, ov, CLI_VAR_VALLEN - 1);
                    sh->val[CLI_VAR_VALLEN - 1] = '\0';
                }
                cli_local_shadow_count[cli_local_depth]++;
            }
        }

        if (eq != NULL)
        {
            cli_var_set(vn, eq + 1);
        }
        else
        {
            if (cli_var_get(vn) == NULL)
            {
                cli_var_set(vn, "");
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_declare(const char *p)
{
    if (starts_with(p, "declare ") || starts_with(p, "declare\t") || starts_with(p, "typeset ") ||
        starts_with(p, "typeset\t"))
    {
        p += 7;
        if (p[0] == ' ' || p[0] == '\t')
        {
            p++;
        }
        p = strip_ws(p);
        /* Parse flags */
        int fl_int = 0;
        int fl_arr = 0;
        int fl_ro  = 0;
        int fl_exp = 0;
        while (p[0] == '-')
        {
            p++;
            while (*p != '\0' && *p != ' ' && *p != '\t')
            {
                if (*p == 'i')
                {
                    fl_int = 1;
                }
                else if (*p == 'a')
                {
                    fl_arr = 1;
                }
                else if (*p == 'r')
                {
                    fl_ro = 1;
                }
                else if (*p == 'x')
                {
                    fl_exp = 1;
                }
                p++;
            }
            p = strip_ws(p);
        }
        /* Parse VAR=val */
        const char *eq = strchr(p, '=');
        char        vn[CLI_VAR_NAMELEN];
        if (eq != NULL)
        {
            int nl = (int) (eq - p);
            if (nl >= CLI_VAR_NAMELEN)
            {
                nl = CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, p, (size_t) nl);
            vn[nl] = '\0';
            if (fl_arr)
            {
                /* declare -a arr */
                for (int k = 0; k < CLI_MAX_ARRAYS; k++)
                {
                    if (!cli_arrays[k].used)
                    {
                        cli_arrays[k].used = 1;
                        strncpy(cli_arrays[k].name, vn, CLI_VAR_NAMELEN - 1);
                        cli_arrays[k].nelem = 0;
                        break;
                    }
                }
            }
            else if (fl_int)
            {
                /* Integer eval */
                long iv = strtol(eq + 1, NULL, 0);
                char ib[32];
                snprintf(ib, sizeof(ib), "%ld", iv);
                cli_var_set(vn, ib);
            }
            else
            {
                cli_var_set(vn, eq + 1);
            }
            if (fl_exp)
            {
                const char *v = cli_var_get(vn);
                if (v != NULL)
                {
                    setenv(vn, v, 1);
                }
            }
        }
        else
        {
            strncpy(vn, p, CLI_VAR_NAMELEN - 1);
            vn[CLI_VAR_NAMELEN - 1] = '\0';
            if (cli_var_get(vn) == NULL)
            {
                cli_var_set(vn, "");
            }
        }
        (void) fl_ro; /* TODO: track */
        return 1;
    }
    return 0;
}

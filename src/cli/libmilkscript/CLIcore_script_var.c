#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <wordexp.h>
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"

#include "COREMOD_memory/COREMOD_memory.h"

/**
 * @brief Set a CLI variable (create or update)
 *
 * @param name  Variable name
 * @param val   Value string
 */
void cli_var_set(
    const char *name,
    const char *val)
{
    int type = 2; // default string
    long numl = 0;
    double numf = 0.0;
    char valbuf[CLI_VAR_VALLEN];

    if (val != NULL && *val != '\0') {
        char *endptr = NULL;
        strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
        valbuf[CLI_VAR_VALLEN - 1] = '\0';
        size_t len = strlen(valbuf);

        // Try parsing as integer
        numl = strtol(valbuf, &endptr, 10);
        if (endptr != valbuf && *endptr == '\0') {
            type = 1; // long
            strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
            valbuf[CLI_VAR_VALLEN - 1] = '\0';
        } else {
            // Try parsing as float
            if (len > 0 && (valbuf[len-1] == 'f' || valbuf[len-1] == 'F')) {
                valbuf[len-1] = '\0';
            }
            numf = strtod(valbuf, &endptr);
            if (endptr != valbuf && *endptr == '\0' && len > 0) {
                type = 0; // double
                strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
                valbuf[CLI_VAR_VALLEN - 1] = '\0';
            } else {
                int mtype;
                long mlval;
                double mdval;
                if (cli_calc_eval_math_to_val(valbuf, &mtype, &mlval, &mdval)) {
                    if (mtype == 1) {
                        type = 1; // long
                        numl = mlval;
                        snprintf(valbuf, CLI_VAR_VALLEN, "%ld", numl);
                    } else if (mtype == 2) {
                        type = 0; // double
                        numf = mdval;
                        snprintf(valbuf, CLI_VAR_VALLEN, "%.*g", cli_float_digits, numf);
                    } else {
                        type = 2; // string
                    }
                } else {
                    type = 2; // string
                    // Restore original value if we stripped trailing f from it
                    strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
                    valbuf[CLI_VAR_VALLEN - 1] = '\0';
                }
            }
        }
    } else {
        // empty val
        valbuf[0] = '\0';
    }

    /* Update existing */
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            strncpy(cli_vars[i].val, valbuf, CLI_VAR_VALLEN - 1);
            cli_vars[i].val[CLI_VAR_VALLEN - 1] = '\0';
            cli_vars[i].type = type;
            if (type == 1) { 
                cli_vars[i].num.l = numl;
                create_variable_long_ID(name, numl);
            }
            if (type == 0) {
                cli_vars[i].num.f = numf;
                create_variable_ID(name, numf);
            }
            return;
        }
    }
    /* Find empty slot */
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(!cli_vars[i].used)
        {
            strncpy(cli_vars[i].name, name, CLI_VAR_NAMELEN - 1);
            cli_vars[i].name[CLI_VAR_NAMELEN - 1] = '\0';
            strncpy(cli_vars[i].val, valbuf, CLI_VAR_VALLEN - 1);
            cli_vars[i].val[CLI_VAR_VALLEN - 1] = '\0';
            cli_vars[i].type = type;
            if (type == 1) {
                cli_vars[i].num.l = numl;
                create_variable_long_ID(name, numl);
            }
            if (type == 0) {
                cli_vars[i].num.f = numf;
                create_variable_ID(name, numf);
            }
            cli_vars[i].used = 1;
            return;
        }
    }
    printf("Error: variable table full " "(max %d)\n", CLI_MAX_VARS);
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

    int namelen = (int)(p - name_start);

    /* Skip spaces before '=' */
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }

    /* Must hit '=' */
    if(*p != '=')
    {
        return 0;
    }

    {
        char tmpname[CLI_VAR_NAMELEN];
        if(namelen >= CLI_VAR_NAMELEN)
        {
            namelen = CLI_VAR_NAMELEN - 1;
        }
        memcpy(tmpname, name_start, (size_t) namelen);
        tmpname[namelen] = '\0';

        /* Extract value (everything after '=') */
        const char *val = p + 1;

        /* Strip trailing whitespace/newline */
        char valbuf[CLI_VAR_VALLEN];
        strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
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

        /* Use wordexp to evaluate command substitution, variables, and quotes natively. */
        wordexp_t p;
        cli_export_vars_to_env();
        if(wordexp(valbuf, &p, 0) == 0)
        {
            char expanded_val[CLI_VAR_VALLEN] = "";
            for(size_t i = 0; i < p.we_wordc; i++)
            {
                if(i > 0)
                {
                    strncat(expanded_val, " ", CLI_VAR_VALLEN - strlen(expanded_val) - 1);
                }
                strncat(expanded_val, p.we_wordv[i], CLI_VAR_VALLEN - strlen(expanded_val) - 1);
            }
            wordfree(&p);
            cli_var_set(tmpname, expanded_val);
        }
        else
        {
            cli_var_set(tmpname, valbuf);
        }

        /* Print the assigned value if Debug is enabled */
        if (data.core.Debug > 0)
        {
            for (int i = 0; i < CLI_MAX_VARS; i++)
            {
                if (cli_vars[i].used
                    && strcmp(cli_vars[i].name,
                              tmpname) == 0)
                {
                    if (cli_vars[i].type == 1)
                    {
                        printf("    %s long: %ld\n", cli_vars[i].name, cli_vars[i].num.l);
                    }
                    else if (cli_vars[i].type == 0)
                    {
                        printf("    %s double: %.*g\n",
                               cli_vars[i].name, cli_float_digits, cli_vars[i].num.f);
                    }
                    else
                    {
                        printf("    %s string: %s\n", cli_vars[i].name, cli_vars[i].val);
                    }
                    break;
                }
            }
        }
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

    /* Pre-scan: if the closing ')' is followed by
     * non-whitespace this is a math expression like
     * a=(1+2)/3, not an array assignment.  Do this
     * check BEFORE mutating cli_arrays. */
    {
        const char *scan = p;
        while(*scan != '\0' && *scan != ')')
        {
            scan++;
        }
        if(*scan == ')')
        {
            const char *q = scan + 1;
            while(*q == ' ' || *q == '\t')
            {
                q++;
            }
            if(*q != '\0' && *q != '\n')
            {
                return 0;
            }
        }
    }

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

    strncpy(cli_arrays[slot].name, aname, CLI_VAR_NAMELEN - 1);
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
        int idx = cli_arrays[slot].nelem;
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
            cli_arrays[slot] .elem[idx][ei++] = *p++;
        }
        cli_arrays[slot] .elem[idx][ei] = '\0';
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
    cli_var_unset(data.cmdargtoken[1].val.string);
    return RETURN_SUCCESS;
}

/**
 * @brief vars command — list all CLI variables
 */
errno_t cli_cmd_vars(void)
{
    int count = 0;
    printf("\n  CLI Variables:\n");
    printf("  %-20s  %-6s  %s\n", "NAME", "TYPE", "VALUE");
    printf("  %-20s  %-6s  %s\n", "----", "----", "-----");
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used)
        {
            const char *typestr = "STR";
            if(cli_vars[i].type == 0) typestr = "FLT";
            else if(cli_vars[i].type == 1) typestr = "INT";
            
            printf("  %-20s  [%-3s]  %s\n", cli_vars[i].name, typestr, cli_vars[i].val);
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

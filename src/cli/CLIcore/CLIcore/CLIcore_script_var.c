#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include "CLIcore.h"
#include "CLIcore_UI.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"


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
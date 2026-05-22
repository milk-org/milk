#include <stddef.h>
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "CLIcore_script.h"
#include "CLIcore.h"
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
 * @brief List all active trap handlers.
 */
static int cli_trap_list_active(void)
{
    printf("POSIX traps:\n");
    for (int i = 0; i < CLI_TRAP_MAXSIGS; i++)
    {
        if (cli_traps[i].used)
        {
            printf("  sig=%d "
                   "cmd='%s'\n",
                   cli_traps[i].signum, cli_traps[i].cmd);
        }
    }
    printf("Engine traps:\n");
    for (int i = 0; i < CLI_ENGINE_TRAP_MAX; i++)
    {
        CLI_ENGINE_TRAP *et = &cli_engine_traps[i];
        if (!et->used)
        {
            continue;
        }
        const char *tstr = "?";
        if (et->type == CLI_ETRAP_STREAM)
        {
            tstr = "STREAM";
        }
        else if (et->type == CLI_ETRAP_FPS)
        {
            tstr = "FPS";
        }
        else if (et->type == CLI_ETRAP_PROC)
        {
            tstr = "PROC";
        }
        printf("  %s:%s", tstr, et->target);
        if (et->type == CLI_ETRAP_FPS)
        {
            printf(".%s", et->param);
        }
        printf(" ival=%ldms"
               " n=%d/%d"
               " cmd='%s'\n",
               et->min_interval_ms, et->fire_count, et->max_fires, et->cmd);
    }
    return 1;
}

/**
 * @brief Process stream-based trap events.
 */
static void cli_trap_process_stream(const char *nm,
                                    const char *tcmd,
                                    long        opt_interval_ms,
                                    int         opt_max_fires)
{
    /* Find or alloc slot */
    int slot = -1;
    for (int i = 0; i < CLI_ENGINE_TRAP_MAX; i++)
    {
        if (cli_engine_traps[i].used && cli_engine_traps[i].type == CLI_ETRAP_STREAM &&
            strcmp(cli_engine_traps[i].target, nm) == 0)
        {
            slot = i;
            break;
        }
    }
    if (slot < 0)
    {
        for (int i = 0; i < CLI_ENGINE_TRAP_MAX; i++)
        {
            if (!cli_engine_traps[i].used)
            {
                slot = i;
                break;
            }
        }
    }
    if (slot >= 0)
    {
        CLI_ENGINE_TRAP *et = &cli_engine_traps[slot];
        if (tcmd[0] == '\0')
        {
            /* Clear trap */
            et->used      = 0;
            et->connected = 0;
        }
        else
        {
            memset(et, 0, sizeof(*et));
            et->type = CLI_ETRAP_STREAM;
            strncpy(et->target, nm, sizeof(et->target) - 1);
            strncpy(et->cmd, tcmd, CLI_TRAP_CMDLEN - 1);
            et->min_interval_ms = opt_interval_ms;
            et->max_fires       = opt_max_fires;
            et->used            = 1;
        }
    }
}

/**
 * @brief Process FPS-based trap events.
 */
static void cli_trap_process_fps(const char *fp,
                                 const char *tcmd,
                                 long        opt_interval_ms,
                                 int         opt_max_fires)
{
    /* Split fpsname.param
     * and optional op+val */
    char   fpsn[128];
    char   parn[64];
    int    eop     = CLI_ETRAP_OP_EQ;
    double eval    = 0.0;
    int    has_cmp = 0;
    {
        char tmp[128];
        strncpy(tmp, fp, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';

        /* Find operator */
        char *opp  = NULL;
        char *p_ne = strstr(tmp, "!=");
        char *p_ge = strstr(tmp, ">=");
        char *p_le = strstr(tmp, "<=");
        char *p_eq = strchr(tmp, '=');

        if (p_ne)
        {
            opp     = p_ne;
            eop     = CLI_ETRAP_OP_NE;
            *opp    = '\0';
            eval    = strtod(opp + 2, NULL);
            has_cmp = 1;
        }
        else if (p_ge)
        {
            opp     = p_ge;
            eop     = CLI_ETRAP_OP_GE;
            *opp    = '\0';
            eval    = strtod(opp + 2, NULL);
            has_cmp = 1;
        }
        else if (p_le)
        {
            opp     = p_le;
            eop     = CLI_ETRAP_OP_LE;
            *opp    = '\0';
            eval    = strtod(opp + 2, NULL);
            has_cmp = 1;
        }
        else if (p_eq)
        {
            opp     = p_eq;
            eop     = CLI_ETRAP_OP_EQ;
            *opp    = '\0';
            eval    = strtod(opp + 1, NULL);
            has_cmp = 1;
        }

        /* Split at dot */
        char *dot = strchr(tmp, '.');
        if (dot)
        {
            *dot = '\0';
            strncpy(fpsn, tmp, sizeof(fpsn) - 1);
            fpsn[sizeof(fpsn) - 1] = '\0';
            strncpy(parn, dot + 1, sizeof(parn) - 1);
            parn[sizeof(parn) - 1] = '\0';
        }
        else
        {
            strncpy(fpsn, tmp, sizeof(fpsn) - 1);
            fpsn[sizeof(fpsn) - 1] = '\0';
            parn[0]                = '\0';
        }
    }

    int slot = -1;
    for (int i = 0; i < CLI_ENGINE_TRAP_MAX; i++)
    {
        if (!cli_engine_traps[i].used)
        {
            slot = i;
            break;
        }
    }
    if (slot >= 0)
    {
        CLI_ENGINE_TRAP *et = &cli_engine_traps[slot];
        if (tcmd[0] == '\0')
        {
            et->used      = 0;
            et->connected = 0;
        }
        else
        {
            memset(et, 0, sizeof(*et));
            et->type = CLI_ETRAP_FPS;
            strncpy(et->target, fpsn, sizeof(et->target) - 1);
            strncpy(et->param, parn, sizeof(et->param) - 1);
            et->op      = eop;
            et->has_cmp = has_cmp;
            et->cmpval  = eval;
            strncpy(et->cmd, tcmd, CLI_TRAP_CMDLEN - 1);
            et->min_interval_ms = opt_interval_ms;
            et->max_fires       = opt_max_fires;
            et->used            = 1;
        }
    }
}

static void cli_trap_process_proc(const char *pp,
                                  const char *tcmd,
                                  long        opt_interval_ms,
                                  int         opt_max_fires)
{
    char pname[128];
    int  pstate = 0;
    {
        char *col = strchr(pp, ':');
        if (col)
        {
            size_t len = (size_t) (col - pp);
            if (len >= sizeof(pname))
            {
                len = sizeof(pname) - 1;
            }
            strncpy(pname, pp, len);
            pname[len]     = '\0';
            const char *ss = col + 1;
            if (strcasecmp(ss, "ACTIVE") == 0)
            {
                pstate = PROCESSINFO_LOOPSTAT_ACTIVE;
            }
            else if (strcasecmp(ss, "STOP") == 0)
            {
                pstate = PROCESSINFO_LOOPSTAT_STOP;
            }
            else if (strcasecmp(ss, "PAUSE") == 0)
            {
                pstate = PROCESSINFO_LOOPSTAT_PAUSE;
            }
            else if (strcasecmp(ss, "CRASHED") == 0)
            {
                pstate = PROCESSINFO_LOOPSTAT_CRASHED;
            }
            else if (strcasecmp(ss, "ERROR") == 0)
            {
                pstate = PROCESSINFO_LOOPSTAT_ERROR;
            }
        }
        else
        {
            strncpy(pname, pp, sizeof(pname) - 1);
            pname[sizeof(pname) - 1] = '\0';
        }
    }

    int slot = -1;
    for (int i = 0; i < CLI_ENGINE_TRAP_MAX; i++)
    {
        if (!cli_engine_traps[i].used)
        {
            slot = i;
            break;
        }
    }
    if (slot >= 0)
    {
        CLI_ENGINE_TRAP *et = &cli_engine_traps[slot];
        if (tcmd[0] == '\0')
        {
            et->used      = 0;
            et->connected = 0;
        }
        else
        {
            memset(et, 0, sizeof(*et));
            et->type = CLI_ETRAP_PROC;
            strncpy(et->target, pname, sizeof(et->target) - 1);
            et->proc_state = pstate;
            strncpy(et->cmd, tcmd, CLI_TRAP_CMDLEN - 1);
            et->min_interval_ms = opt_interval_ms;
            et->max_fires       = opt_max_fires;
            et->used            = 1;
        }
    }
}

static void cli_trap_process_posix(const char *sname, const char *tcmd)
{
    /* POSIX signal name */
    int sn   = cli_trap_signum(sname);
    int slot = -1;
    for (int i = 0; i < CLI_TRAP_MAXSIGS; i++)
    {
        if (cli_traps[i].used && cli_traps[i].signum == sn)
        {
            slot = i;
            break;
        }
    }
    if (slot < 0)
    {
        for (int i = 0; i < CLI_TRAP_MAXSIGS; i++)
        {
            if (!cli_traps[i].used)
            {
                slot = i;
                break;
            }
        }
    }
    if (slot >= 0)
    {
        cli_traps[slot].signum = sn;
        strncpy(cli_traps[slot].cmd, tcmd, CLI_TRAP_CMDLEN - 1);
        cli_traps[slot].used = 1;
    }
}

int cli_intercept_cmd_trap(const char *p)
{
    if (starts_with(p, "trap ") || starts_with(p, "trap\t"))
    {
        p += 4;
        p = strip_ws(p);

        if (strncmp(p, "-l", 2) == 0 && (p[2] == '\0' || p[2] == ' ' || p[2] == '\t'))
        {
            return cli_trap_list_active();
        }

        /* Parse optional flags before
         * the quoted command */
        long opt_interval_ms = CLI_ETRAP_DEFAULT_MS;
        int  opt_max_fires   = -1;

        while (*p == '-')
        {
            if (strncmp(p, "-n", 2) == 0)
            {
                p += 2;
                while (*p == ' ' || *p == '\t')
                {
                    p++;
                }
                char *endp = NULL;
                long  nv   = strtol(p, &endp, 10);
                if (endp != p && nv > 0)
                {
                    opt_max_fires = (int) nv;
                    p             = endp;
                }
            }
            else if (strncmp(p, "-i", 2) == 0)
            {
                p += 2;
                while (*p == ' ' || *p == '\t')
                {
                    p++;
                }
                char *endp = NULL;
                long  iv   = strtol(p, &endp, 10);
                if (endp != p && iv >= 0)
                {
                    opt_interval_ms = iv;
                    p               = endp;
                }
            }
            else
            {
                break;
            }
            while (*p == ' ' || *p == '\t')
            {
                p++;
            }
        }

        /* Extract quoted command */
        char tcmd[CLI_TRAP_CMDLEN];
        tcmd[0] = '\0';
        if (*p == '\'' || *p == '"')
        {
            char q  = *p++;
            int  ti = 0;
            while (*p != '\0' && *p != q && ti < CLI_TRAP_CMDLEN - 1)
            {
                tcmd[ti++] = *p++;
            }
            tcmd[ti] = '\0';
            if (*p == q)
            {
                p++;
            }
        }
        p = strip_ws(p);

        /* Parse signal / event names */
        while (*p != '\0')
        {
            char sname[128];
            int  si = 0;
            while (*p != '\0' && *p != ' ' && *p != '\t' && si < 127)
            {
                sname[si++] = *p++;
            }
            sname[si] = '\0';
            p         = strip_ws(p);
            if (si == 0)
            {
                break;
            }

            if (strncmp(sname, "STREAM:", 7) == 0)
            {
                cli_trap_process_stream(sname + 7, tcmd, opt_interval_ms, opt_max_fires);
                continue;
            }

            if (strncmp(sname, "FPS:", 4) == 0)
            {
                cli_trap_process_fps(sname + 4, tcmd, opt_interval_ms, opt_max_fires);
                continue;
            }

            if (strncmp(sname, "PROC:", 5) == 0)
            {
                cli_trap_process_proc(sname + 5, tcmd, opt_interval_ms, opt_max_fires);
                continue;
            }

            cli_trap_process_posix(sname, tcmd);
        }
        return 1;
    }
    return 0;
}

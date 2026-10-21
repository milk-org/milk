// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    seq_cli.c
 * @brief   CLI commands for milk-seq Sequencer
 *
 * Provides seq.list, seq.submit, seq.status, seq.tasks, seq.start, seq.stop
 */

#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "CLIcore.h"
#include "milk_runnable.h"
#include "fpsseq.h"
#include "seq_cli.h"

/* Linked to the first CLI argument of every command taking a sequencer name */
static char *seq_name;

/* Mandatory second argument of submit; the full command is rebuilt from data.cmdargtoken */
static char *seq_extra;

#define SEQ_NAME_FARG                            \
    { CLIARG_STR,                                \
      ".name",                                   \
      "sequencer name",                          \
      "loop01",                                  \
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
      (void **) &seq_name,                       \
      NULL }

/**
 * @brief Resolve SHM directory for PID/log files
 *
 * Mirrors the logic in milk-seq.c.
 */
static const char *get_shm_dir(void)
{
    const char *dir = getenv("MILK_SHM_DIR");
    if (dir && dir[0] != '\0')
    {
        return dir;
    }
    struct stat st;
    if (stat("/milk/shm", &st) == 0 && S_ISDIR(st.st_mode))
    {
        return "/milk/shm";
    }
    return "/tmp";
}

/**
 * @brief List all running sequencer instances
 */
static errno_t cli_seq_list(void)
{
    char names[100][FPSSEQ_NAME_MAX];
    int  count = milkseq_list(names, 100);

    if (count == 0)
    {
        printf("No active sequencers found.\n");
        return RETURN_SUCCESS;
    }

    printf("%-20s %-10s %-10s %-10s %-10s %s\n", "NAME", "PID", "STATUS", "TASKS", "ERRORS",
           "FIFO PATH");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < count; i++)
    {
        MILKSEQ_STATE *state = milkseq_connect(names[i]);
        if (state)
        {
            const char *status_str = "UNKNOWN";
            if (state->status & MILKSEQ_STATUS_IDLE)
            {
                status_str = "IDLE";
            }
            if (state->status & MILKSEQ_STATUS_RUNNING)
            {
                status_str = "RUNNING";
            }
            if (state->status & MILKSEQ_STATUS_ERROR)
            {
                status_str = "ERROR";
            }
            if (state->status & MILKSEQ_STATUS_STOPPING)
            {
                status_str = "STOPPING";
            }

            printf("%-20s %-10d %-10s %-10u %-10u %s\n", state->name, state->pid, status_str,
                   state->NBtasks_active, state->error_count, state->fifo_path);
            milkseq_disconnect(state);
        }
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Submit a command to a sequencer's FIFO
 */
static errno_t cli_seq_submit(void)
{
    const char    *seqname = seq_name;
    MILKSEQ_STATE *state   = milkseq_connect(seqname);
    if (!state)
    {
        printf("ERROR: Sequencer '%s' not found.\n", seqname);
        return RETURN_SUCCESS;
    }

    int fd = open(state->fifo_path, O_WRONLY | O_NONBLOCK);
    if (fd < 0)
    {
        printf("ERROR: Cannot open FIFO %s\n", state->fifo_path);
        milkseq_disconnect(state);
        return RETURN_SUCCESS;
    }

    /* Reconstruct command string from remaining tokens */
    char cmdstr[2048] = { 0 };
    for (int i = 2; i < NB_ARG_MAX; i++)
    {
        if (data.cmdargtoken[i].type == 0)
        {
            break;
        }
        if (i > 2)
        {
            strncat(cmdstr, " ", sizeof(cmdstr) - strlen(cmdstr) - 1);
        }
        strncat(cmdstr, data.cmdargtoken[i].val.string, sizeof(cmdstr) - strlen(cmdstr) - 1);
    }
    strncat(cmdstr, "\n", sizeof(cmdstr) - strlen(cmdstr) - 1);

    if (write(fd, cmdstr, strlen(cmdstr)) < 0)
    {
        printf("ERROR: Failed to write to FIFO.\n");
    }
    else
    {
        printf("Submitted to %s: %s", seqname, cmdstr);
    }

    close(fd);
    milkseq_disconnect(state);
    return RETURN_SUCCESS;
}

/**
 * @brief Show sequencer status details
 */
static errno_t cli_seq_status(void)
{
    const char    *seqname = seq_name;
    MILKSEQ_STATE *state   = milkseq_connect(seqname);
    if (!state)
    {
        printf("ERROR: Sequencer '%s' not found.\n", seqname);
        return RETURN_SUCCESS;
    }

    printf("Sequencer: %s\n", state->name);
    printf("  PID:       %d\n", state->pid);

    printf("  Status:    ");
    if (state->status & MILKSEQ_STATUS_IDLE)
    {
        printf("IDLE ");
    }
    if (state->status & MILKSEQ_STATUS_RUNNING)
    {
        printf("RUNNING ");
    }
    if (state->status & MILKSEQ_STATUS_ERROR)
    {
        printf("ERROR ");
    }
    if (state->status & MILKSEQ_STATUS_STOPPING)
    {
        printf("STOPPING ");
    }
    printf("\n");

    printf("  Tasks max: %u\n", state->NBtasks_max);
    printf("  Tasks act: %u\n", state->NBtasks_active);
    printf("  Tasks cmp: %u\n", state->NBtasks_completed);
    printf("  Inputs:    %lu\n", (unsigned long) state->task_input_counter);
    printf("  Errors:    %u\n", state->error_count);
    if (state->error_count > 0)
    {
        printf("  Last Err:  %s\n", state->last_error);
    }
    printf("  Script:    %s\n", state->script_path[0] ? state->script_path : "(none)");
    printf("  FIFO:      %s\n", state->fifo_path);

    milkseq_disconnect(state);
    return RETURN_SUCCESS;
}

/**
 * @brief Start a sequencer as a POSIX daemon
 */
static errno_t cli_seq_start(void)
{
    const char *seqname = seq_name;
    char        cmd[512];

    if (data.cmdargtoken[2].type == 4 && strcmp(data.cmdargtoken[2].val.string, "-f") == 0)
    {
        if (data.cmdargtoken[3].type == 0)
        {
            printf("ERROR: Missing script file "
                   "after -f\n");
            return RETURN_SUCCESS;
        }
        snprintf(cmd, sizeof(cmd), "milk-seq -n %s -f %s --daemon", seqname,
                 data.cmdargtoken[3].val.string);
    }
    else
    {
        snprintf(cmd, sizeof(cmd), "milk-seq -n %s --daemon", seqname);
    }

    printf("Starting sequencer: %s\n", cmd);
    if (system(cmd) == -1)
    {
        printf("ERROR: Failed to launch "
               "sequencer.\n");
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Stop a sequencer via SIGTERM (PID file)
 *
 * Falls back to FIFO "exit" if PID file is absent.
 */
static errno_t cli_seq_stop(void)
{
    const char *seqname = seq_name;

    /* Try PID file first */
    char pidpath[256];
    snprintf(pidpath, sizeof(pidpath), "%s/milkseq.%s.pid", get_shm_dir(), seqname);

    FILE *fp = fopen(pidpath, "r");
    if (fp)
    {
        int pid = 0;
        if (fscanf(fp, "%d", &pid) == 1 && pid > 0)
        {
            if (kill((pid_t) pid, SIGTERM) == 0)
            {
                printf("SIGTERM sent to '%s' "
                       "(PID %d).\n",
                       seqname, pid);
                fclose(fp);
                return RETURN_SUCCESS;
            }
            printf("WARNING: kill(%d) failed, "
                   "trying FIFO fallback.\n",
                   pid);
        }
        fclose(fp);
    }

    /* Fallback: FIFO "exit" command */
    MILKSEQ_STATE *state = milkseq_connect(seqname);
    if (!state)
    {
        printf("ERROR: Sequencer '%s' not "
               "found.\n",
               seqname);
        return RETURN_SUCCESS;
    }

    int fd = open(state->fifo_path, O_WRONLY | O_NONBLOCK);
    if (fd >= 0)
    {
        if (write(fd, "exit\n", 5) < 0)
        {
            printf("ERROR: Failed to write "
                   "to FIFO.\n");
        }
        else
        {
            printf("Stop command sent to "
                   "'%s' via FIFO.\n",
                   seqname);
        }
        close(fd);
    }
    else
    {
        printf("ERROR: Could not open FIFO "
               "for '%s'.\n",
               seqname);
    }

    milkseq_disconnect(state);
    return RETURN_SUCCESS;
}

/**
 * @brief Print tail of sequencer log file
 */
static errno_t cli_seq_log(void)
{
    const char *seqname = seq_name;

    char logpath[256];
    snprintf(logpath, sizeof(logpath), "%s/milkseq.%s.log", get_shm_dir(), seqname);

    FILE *fp = fopen(logpath, "r");
    if (!fp)
    {
        printf("No log file found: %s\n", logpath);
        return RETURN_SUCCESS;
    }

    /* Count total lines in the log */
    int nlines = 0;
    {
        int ch;
        while ((ch = fgetc(fp)) != EOF)
        {
            if (ch == '\n')
            {
                nlines++;
            }
        }
    }

    /* Seek past all but the last 50 lines */
    int skip = (nlines > 50) ? nlines - 50 : 0;
    rewind(fp);
    {
        int cur = 0;
        while (cur < skip)
        {
            int ch = fgetc(fp);
            if (ch == EOF)
            {
                break;
            }
            if (ch == '\n')
            {
                cur++;
            }
        }
    }

    printf("--- %s (last %d lines) ---\n", logpath, nlines - skip);
    {
        int ch;
        while ((ch = fgetc(fp)) != EOF)
        {
            putchar(ch);
        }
    }
    printf("--- end ---\n");

    fclose(fp);
    return RETURN_SUCCESS;
}

/* Descriptor of a CLI-only command; the module registry is in sequencer.c */
#define SEQ_RUNNABLE(sym, key, descr, farray, nb, func)                              \
    MILK_RUNNABLE_RT_DEFINE(seq_##sym);                                              \
    const MILK_RUNNABLE runnable_sequencer_##sym = {                                 \
        .kind     = MILK_RUNNABLE_CLI,                                               \
        .app      = { .fps_name = "seq" #sym, .cmdkey = key, .description = descr }, \
        .srcfile  = __FILE__,                                                        \
        .farg     = farray,                                                          \
        .nbarg    = nb,                                                              \
        .cli_func = func,                                                            \
        .rt       = &seq_##sym##_rt,                                                 \
    }

#define SEQ_NBARG(farray) ((int) (sizeof(farray) / sizeof((farray)[0])))

static CLICMDARGDEF farg_name[] = { SEQ_NAME_FARG };

static CLICMDARGDEF farg_submit[] = {
    SEQ_NAME_FARG,
    { CLIARG_STR, ".cmd", "command (remaining tokens are appended)", "sleep",
      (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), (void **) &seq_extra, NULL },
};

SEQ_RUNNABLE(list, "list", "List active sequencer instances", NULL, 0, cli_seq_list);
SEQ_RUNNABLE(submit,
             "submit",
             "Submit a command to a sequencer",
             farg_submit,
             SEQ_NBARG(farg_submit),
             cli_seq_submit);
SEQ_RUNNABLE(status,
             "status",
             "Show status of a sequencer",
             farg_name,
             SEQ_NBARG(farg_name),
             cli_seq_status);
SEQ_RUNNABLE(start,
             "start",
             "Start a new sequencer instance (optional: -f script.seq)",
             farg_name,
             SEQ_NBARG(farg_name),
             cli_seq_start);
SEQ_RUNNABLE(stop,
             "stop",
             "Stop a sequencer instance safely",
             farg_name,
             SEQ_NBARG(farg_name),
             cli_seq_stop);
SEQ_RUNNABLE(log,
             "log",
             "Show tail of sequencer log",
             farg_name,
             SEQ_NBARG(farg_name),
             cli_seq_log);

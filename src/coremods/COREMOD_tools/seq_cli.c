/**
 * @file    seq_cli.c
 * @brief   CLI commands for milk-seq Sequencer
 *
 * Provides seq.list, seq.submit, seq.status, seq.tasks, seq.start, seq.stop
 */

#include "seq_cli.h"

#ifndef MILK_NO_CLI

#include "CLIcore.h"
#include "fpsseq.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

/**
 * @brief List all running sequencer instances
 */
static errno_t cli_seq_list(void)
{
    char names[100][FPSSEQ_NAME_MAX];
    int count = milkseq_list(names, 100);

    if (count == 0) {
        printf("No active sequencers found.\n");
        return RETURN_SUCCESS;
    }

    printf("%-20s %-10s %-10s %-10s %-10s %s\n",
           "NAME", "PID", "STATUS", "TASKS", "ERRORS", "FIFO PATH");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < count; i++) {
        MILKSEQ_STATE *state = milkseq_connect(names[i]);
        if (state) {
            const char *status_str = "UNKNOWN";
            if (state->status & MILKSEQ_STATUS_IDLE) status_str = "IDLE";
            if (state->status & MILKSEQ_STATUS_RUNNING) status_str = "RUNNING";
            if (state->status & MILKSEQ_STATUS_ERROR) status_str = "ERROR";
            if (state->status & MILKSEQ_STATUS_STOPPING) status_str = "STOPPING";

            printf("%-20s %-10d %-10s %-10u %-10u %s\n",
                   state->name,
                   state->pid,
                   status_str,
                   state->NBtasks_active,
                   state->error_count,
                   state->fifo_path);
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
    if (CLI_checkarg(1, 4) == 0) {
        return RETURN_SUCCESS;
    }

    const char *seqname = data.cmdargtoken[1].val.string;
    MILKSEQ_STATE *state = milkseq_connect(seqname);
    if (!state) {
        printf("ERROR: Sequencer '%s' not found.\n", seqname);
        return RETURN_SUCCESS;
    }

    int fd = open(state->fifo_path, O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        printf("ERROR: Cannot open FIFO %s\n", state->fifo_path);
        milkseq_disconnect(state);
        return RETURN_SUCCESS;
    }

    /* Reconstruct command string from remaining tokens */
    char cmdstr[2048] = {0};
    for (int i = 2; i < NB_ARG_MAX; i++) {
        if (data.cmdargtoken[i].type == 0) break;
        if (i > 2) strncat(cmdstr, " ", sizeof(cmdstr) - strlen(cmdstr) - 1);
        strncat(cmdstr, data.cmdargtoken[i].val.string, sizeof(cmdstr) - strlen(cmdstr) - 1);
    }
    strncat(cmdstr, "\n", sizeof(cmdstr) - strlen(cmdstr) - 1);

    if (write(fd, cmdstr, strlen(cmdstr)) < 0) {
        printf("ERROR: Failed to write to FIFO.\n");
    } else {
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
    if (CLI_checkarg(1, 4) == 0) {
        return RETURN_SUCCESS;
    }

    const char *seqname = data.cmdargtoken[1].val.string;
    MILKSEQ_STATE *state = milkseq_connect(seqname);
    if (!state) {
        printf("ERROR: Sequencer '%s' not found.\n", seqname);
        return RETURN_SUCCESS;
    }

    printf("Sequencer: %s\n", state->name);
    printf("  PID:       %d\n", state->pid);
    
    printf("  Status:    ");
    if (state->status & MILKSEQ_STATUS_IDLE) printf("IDLE ");
    if (state->status & MILKSEQ_STATUS_RUNNING) printf("RUNNING ");
    if (state->status & MILKSEQ_STATUS_ERROR) printf("ERROR ");
    if (state->status & MILKSEQ_STATUS_STOPPING) printf("STOPPING ");
    printf("\n");

    printf("  Tasks max: %u\n", state->NBtasks_max);
    printf("  Tasks act: %u\n", state->NBtasks_active);
    printf("  Tasks cmp: %u\n", state->NBtasks_completed);
    printf("  Inputs:    %lu\n", (unsigned long)state->task_input_counter);
    printf("  Errors:    %u\n", state->error_count);
    if (state->error_count > 0) {
        printf("  Last Err:  %s\n", state->last_error);
    }
    printf("  Script:    %s\n", state->script_path[0] ? state->script_path : "(none)");
    printf("  FIFO:      %s\n", state->fifo_path);

    milkseq_disconnect(state);
    return RETURN_SUCCESS;
}

/**
 * @brief Start a sequencer process
 */
static errno_t cli_seq_start(void)
{
    if (CLI_checkarg(1, 4) == 0) {
        return RETURN_SUCCESS;
    }

    const char *seqname = data.cmdargtoken[1].val.string;
    char cmd[512];
    
    // Check if script file provided via -f
    if (data.cmdargtoken[2].type == 4 && strcmp(data.cmdargtoken[2].val.string, "-f") == 0) {
        if (data.cmdargtoken[3].type == 0) {
            printf("ERROR: Missing script file after -f\n");
            return RETURN_SUCCESS;
        }
        snprintf(cmd, sizeof(cmd), "milk-seq -n %s -f %s --headless &", 
                 seqname, data.cmdargtoken[3].val.string);
    } else {
        snprintf(cmd, sizeof(cmd), "milk-seq -n %s --headless &", seqname);
    }

    printf("Starting sequencer: %s\n", cmd);
    if (system(cmd) == -1) {
        printf("ERROR: Failed to launch sequencer.\n");
    }
    
    return RETURN_SUCCESS;
}

/**
 * @brief Stop a sequencer process
 */
static errno_t cli_seq_stop(void)
{
    if (CLI_checkarg(1, 4) == 0) {
        return RETURN_SUCCESS;
    }

    const char *seqname = data.cmdargtoken[1].val.string;
    MILKSEQ_STATE *state = milkseq_connect(seqname);
    if (!state) {
        printf("ERROR: Sequencer '%s' not found.\n", seqname);
        return RETURN_SUCCESS;
    }

    int fd = open(state->fifo_path, O_WRONLY | O_NONBLOCK);
    if (fd >= 0) {
        if (write(fd, "exit\n", 5) < 0) {
            printf("ERROR: Failed to write to FIFO.\n");
        } else {
            printf("Stop command sent to '%s'.\n", seqname);
        }
        close(fd);
    } else {
        printf("ERROR: Could not open FIFO for '%s'.\n", seqname);
    }

    milkseq_disconnect(state);
    return RETURN_SUCCESS;
}

/**
 * @brief Register all commands
 */
errno_t CLIADDCMD_COREMOD_tools__seq_cli(void)
{
    RegisterCLIcommand("seq.list", __func__, cli_seq_list,
                       "List active sequencer instances",
                       "seq.list", "seq.list", "");
                       
    RegisterCLIcommand("seq.submit", __func__, cli_seq_submit,
                       "Submit a command to a sequencer",
                       "seq.submit <name> <cmd...>", 
                       "seq.submit loop01 sleep 1.5", "");

    RegisterCLIcommand("seq.status", __func__, cli_seq_status,
                       "Show status of a sequencer",
                       "seq.status <name>", "seq.status loop01", "");

    RegisterCLIcommand("seq.start", __func__, cli_seq_start,
                       "Start a new sequencer instance",
                       "seq.start <name> [-f script.seq]", 
                       "seq.start calib -f test.seq", "");

    RegisterCLIcommand("seq.stop", __func__, cli_seq_stop,
                       "Stop a sequencer instance safely",
                       "seq.stop <name>", "seq.stop calib", "");

    return RETURN_SUCCESS;
}

#endif /* MILK_NO_CLI */

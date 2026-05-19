/**
 * @file fpsseq.h
 * @brief Public API for the milk-seq FPS Sequencer library
 *
 * Provides functions for shared memory lifecycle, task scheduling,
 * FIFO command reading, and script execution.
 */

#ifndef FPSSEQ_H
#define FPSSEQ_H

#include "fpsseq_types.h"
#include "fps_types.h"



/* =========================================================================
 * SHM Lifecycle (fpsseq_shm.c)
 * ========================================================================= */

/**
 * @brief Create a new sequencer instance in shared memory
 * @param name Sequencer name
 * @return Pointer to mapped state, or NULL on error
 */
MILKSEQ_STATE *milkseq_create(const char *name);

/**
 * @brief Connect to an existing sequencer instance
 * @param name Sequencer name
 * @return Pointer to mapped state (read/write access), or NULL if not found
 */
MILKSEQ_STATE *milkseq_connect(const char *name);

/**
 * @brief Disconnect from a sequencer instance
 * @param state Mapped pointer to unmap
 * @return 0 on success
 */
int milkseq_disconnect(MILKSEQ_STATE *state);

/**
 * @brief Destroy a sequencer instance (unlink SHM and FIFO)
 * @param name Sequencer name
 * @return 0 on success
 */
int milkseq_destroy(const char *name);

/**
 * @brief List all active sequencer instances in the system
 * @param names Array of buffers to hold names
 * @param maxcount Maximum number of names to return
 * @return Number of sequencers found
 */
int milkseq_list(char names[][FPSSEQ_NAME_MAX], int maxcount);


/* =========================================================================
 * Scheduler (fpsseq_scheduler.c)
 * ========================================================================= */

/**
 * @brief Run one iteration of the sequencer task scheduler
 *
 * @param state Sequencer state mapped from SHM
 * @param fps Array of all FPS entries
 * @param keywnode Keyword tree root
 * @param vars TUI-level variables (for backwards compat logging)
 * @return Task index executed, or -1 if idle/no task
 */
int milkseq_scheduler_step(
    MILKSEQ_STATE *state,
    FPS *fps,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_PROCESS_VARS *vars);


/* =========================================================================
 * FIFO Input (fpsseq_fifo.c)
 * ========================================================================= */

/**
 * @brief Read and enqueue commands from the named FIFO
 *
 * Continues reading until EAGAIN (non-blocking).
 * @param state Sequence state mapped from SHM
 * @param fifo_fd File descriptor of the opened FIFO (O_NONBLOCK)
 * @return Number of commands read
 */
int milkseq_fifo_read(MILKSEQ_STATE *state, int fifo_fd);


/* =========================================================================
 * Command Execution (fpsseq_cmdexec.c)
 * ========================================================================= */

/**
 * @brief Parse and execute one sequencer command string
 *
 * @param cmdindex Task index in the state's tasklist
 * @param state Sequencer state mapped from SHM
 * @param fps Array of all FPS entries
 * @param keywnode Keyword tree root
 * @param vars TUI-level variables
 * @param taskstatus Output value OR-ed with the final status of the task
 * @return The 1D index of the functional parameter accessed, or -1 if none
 */
int milkseq_exec_cmd(
    uint32_t             cmdindex,
    MILKSEQ_STATE        *state,
    FPS                  *fps,
    KEYWORD_TREE_NODE    *keywnode,
    FPSCTRL_PROCESS_VARS *vars,
    uint64_t             *taskstatus);


/* =========================================================================
 * Script Loading (fpsseq_script.c - Phase 2)
 * ========================================================================= */

/**
 * @brief Load and compile a .seq script into the task array
 *
 * @param state Sequencer state
 * @param filename Path to script file
 * @param fps Array of all FPS entries
 * @param keywnode Keyword tree root
 * @return 0 on success, or an errno value on failure
 */
errno_t milkseq_load_script(
    MILKSEQ_STATE     *state,
    const char        *filename,
    FPS               *fps,
    KEYWORD_TREE_NODE *keywnode);

#endif // FPSSEQ_H

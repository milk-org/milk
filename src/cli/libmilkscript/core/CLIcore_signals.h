// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_signals.h
 *
 * @brief signals and debugging
 *
 */

#ifndef CLICORE_SIGNALS_H

#define CLICORE_SIGNALS_H

#include <setjmp.h>

errno_t write_process_log();

errno_t set_signal_catch();

errno_t write_process_exit_report(const char *__restrict errortypestring);

void sig_handler(int signo);

void        cli_set_interactive_mode(int mode);
int         cli_is_interactive_mode(void);
sigjmp_buf *cli_get_repl_env(void);
void        cli_fault_isolation_arm(void);
void        cli_fault_isolation_disarm(void);
int         cli_is_fault_isolation_armed(void);

#endif

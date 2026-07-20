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

errno_t write_process_log();

errno_t set_signal_catch();

errno_t write_process_exit_report(const char *__restrict errortypestring);

void sig_handler(int signo);

#endif

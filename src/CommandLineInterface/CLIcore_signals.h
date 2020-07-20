/**
 * @file CLIcore_signals.h
 * 
 * @brief signals and debugging
 *
 */


#ifndef CLICORE_SIGNALS_H

#define CLICORE_SIGNALS_H



errno_t set_signal_catch();


errno_t write_process_exit_report(
    const char *restrict errortypestring
);


void sig_handler(
    int signo
);


#endif

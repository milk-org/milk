/**
 * @file CLIcore_UI.h
 * 
 * @brief User input
 *
 *
 */

#ifndef CLICORE_UI_H

#define CLICORE_UI_H



void rl_cb_linehandler(char *linein);


errno_t runCLI_prompt(
    char *promptstring,
    char *prompt
);


char **CLI_completion(const char *, int, int);


errno_t CLI_execute_line();

errno_t write_tracedebugfile();

#endif

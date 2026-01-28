/**
 * @file CLIcore_UI.h
 *
 * @brief User input
 *
 *
 */

#ifndef CLICORE_UI_H

#define CLICORE_UI_H

#ifdef USE_READLINE
void rl_cb_linehandler(char *linein);
#endif

errno_t runCLI_prompt(char *promptstring, char *prompt);

#ifdef USE_READLINE
char **CLI_completion(const char *, int, int);
#endif

errno_t CLI_execute_line();

errno_t write_tracedebugfile();

void CLI_configure_readline();

#endif

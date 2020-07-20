/**
 * @file CLIcore_help.h
 * 
 * @brief help functions
 *
 */


#ifndef CLICORE_HELP_H

#define CLICORE_HELP_H


errno_t printInfo();

errno_t list_commands();

errno_t list_commands_module(
    const char *restrict modulename
);



errno_t help_command(
    const char *restrict cmdkey
);


#endif

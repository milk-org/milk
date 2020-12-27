/**
 * @file CLIcore_help.h
 * 
 * @brief help functions
 *
 */


#ifndef CLICORE_HELP_H
#define CLICORE_HELP_H



errno_t help();

errno_t helpreadline();


int CLIhelp_make_argstring(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    char *outargstring
);

int CLIhelp_make_cmdexamplestring(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    char *shortname,
	char *outcmdexstring
);


errno_t help_cmd();

errno_t help_module();



errno_t printInfo();

errno_t list_commands();

errno_t list_commands_module(
    const char *restrict modulename
);



errno_t help_command(
    const char *restrict cmdkey
);


#endif

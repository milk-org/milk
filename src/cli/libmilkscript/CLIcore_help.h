// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_help.h
 *
 * @brief help functions
 *
 */

#ifndef CLICORE_HELP_H
#define CLICORE_HELP_H

typedef const char *__restrict CONST_WORD;

errno_t help();

errno_t helpreadline();

int CLIhelp_make_argstring(CLICMDARGDEF fpscliarg[], int nbarg, char *outargstring);

int CLIhelp_make_cmdexamplestring(CLICMDARGDEF fpscliarg[],
                                  int          nbarg,
                                  char        *shortname,
                                  char        *outcmdexstring);

errno_t help_cmd();

errno_t cmdinfosearch();

errno_t help_module();

errno_t printInfo();

/** @brief Print milk framework overview help */
void print_milk_framework_help(void);

/** @brief Print milk CLI-specific help */
extern int help_format_mode;
void       print_milk_cli_help(void);

/** @brief Print only the available-topics list */
void print_help_topic_list(void);

/* Per-topic help functions */
void help_topic_cmdopts(void);
void help_topic_syntax(void);
void help_topic_commands(void);
void help_topic_variables(void);
void help_topic_flowcontrol(void);
void help_topic_scripting(void);
void help_topic_milk(void);

/**
 * @brief Dispatch help to a named topic.
 * @return 0 on success, 1 if topic unknown.
 */
int help_topic_dispatch(const char *topic);


errno_t list_commands();


errno_t list_commands_module(CONST_WORD modulename);

errno_t help_command(CONST_WORD cmdkey);

int cli_fhelp(void);

int cli_fhist(void);

int cli_fparam(void);

#endif

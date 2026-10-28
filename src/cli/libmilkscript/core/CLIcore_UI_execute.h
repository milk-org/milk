// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_execute.h
 *
 * @brief User input and interactive UI declarations
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
void    handle_did_you_mean(const char *input_cmd);

errno_t write_tracedebugfile();

void CLI_configure_readline();
void CLI_setup_hint_area(void);
void CLI_cleanup_scroll_region(void);

/* CLIcore_UI_completion.c */
void        cli_set_active_prompt(const char *prompt);
const char *cli_get_active_prompt(void);
void       *xmalloc(int size);
char       *dupstr(const char *s);
extern int  ghost_chars_on_line;

#ifdef USE_READLINE
int   cli_accept_line(int count, int key);
char *CLI_generator(const char *text, int state);
int   levenshtein_distance(const char *s1, const char *s2);
#endif

/* CLIcore_UI_highlight.c */
int cli_is_command(const char *word);
#ifdef USE_READLINE
void cli_highlight_redisplay(void);
#endif

/* CLIcore_UI_hintarea.c */
#ifdef USE_READLINE
int  find_command_match(const char *firstword);
void set_pending_suggestion(const char *text, int replace_len);
#endif

/* Startup script */
void cli_milkrc_load(void);

/* Command statistics */
errno_t cli_cmdstats(void);

/* Command timing */
errno_t cli_timing_toggle(void);

/* Syntax highlighting toggle */
#ifdef USE_READLINE
errno_t cli_syntax_highlight_toggle(void);
errno_t cli_syntax_diag_toggle(void);
errno_t cli_auto_indent_toggle(void);
errno_t cli_showmatch_toggle(void);
#endif

/* Script formatting and block folding */
/** cli_format_file() - print @fname re-indented by its syntax tree (@in_place: rewrite it); 0 or 1. */
int cli_format_file(const char *fname, int in_place, int width);

/** cli_outline_file() - print the multi-line block outline of @fname; 0 or 1. */
int cli_outline_file(const char *fname);

/* Persistent history */
void    cli_history_load(void);
void    cli_history_save(void);
errno_t cli_savehistory(void);

/* Configurable prompt */
errno_t cli_setprompt(void);

/* Session logging */
errno_t cli_sessionlog(void);
void    cli_session_log_cmd(const char *cmd);

/* History display */
errno_t cli_history_show(void);

/* Fuzzy history search */
errno_t cli_searchhist(void);
errno_t cli_fhist(void);

/* Structured history log */
void    cli_history_log_init(void);
errno_t cli_ghistory(void);
errno_t cli_lhistory(void);

void        cli_history_log_cmd(const char *cmd);
void        cli_history_log_prompt(const char *prompt);
void        cli_history_log_prompt_lines(const char *text); /* one P entry per line, one flush */
void        cli_history_log_shell(const char *cmd);
const char *CLI_history_file(void);
const char *cli_history_session(void); /* the session the history log records lines under */

#endif /* CLICORE_UI_H */

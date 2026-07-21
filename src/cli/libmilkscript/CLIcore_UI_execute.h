// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_execute.h
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

errno_t CLI_execute_string(const char *cmd);
errno_t CLI_execute_line();

errno_t write_tracedebugfile();

void CLI_configure_readline();
void CLI_setup_hint_area(void);
void CLI_cleanup_scroll_region(void);

/* -- Cross-file helpers (CLIcore_UI_*.c) -- */

/* CLIcore_UI_completion.c */
void      *xmalloc(int size);
char      *dupstr(char *s);
extern int ghost_chars_on_line;

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

/* Command aliases */
void    cli_alias_load(void);
errno_t cli_alias_add(void);
errno_t cli_alias_remove(void);
errno_t cli_alias_list(void);

/* Watch command */
errno_t cli_watch(void);

/* List active shared memory structures */
errno_t cli_list_streams(void);
errno_t cli_list_fps(void);


/* Startup script */
void cli_milkrc_load(void);

/* Command timing */
errno_t cli_time(void);

/* Command statistics */
errno_t cli_cmdstats(void);

/* Command timing */
errno_t cli_timing_toggle(void);

/* Syntax highlighting toggle */
#ifdef USE_READLINE
errno_t cli_syntax_highlight_toggle(void);
#endif

/* Persistent history */
void cli_history_load(void);
void cli_history_save(void);

/* Script execution */
errno_t cli_source(void);

/* Script save */
errno_t cli_savescript(void);

/* History save */
errno_t cli_savehistory(void);

/* Configurable prompt */
errno_t cli_setprompt(void);
void    cli_build_prompt(const char *fmt, char *out, int maxlen);

/* Command bookmarks */
errno_t cli_bookmark(void);
void    cli_bookmark_load(void);

/* Session logging */
errno_t cli_sessionlog(void);

/* History display */
errno_t cli_history_show(void);

/* Fuzzy history search */
errno_t cli_searchhist(void);
errno_t cli_fhist(void);

/* Built-in cd and pwd */
errno_t cli_cd(void);
errno_t cli_pwd(void);

/* Structured history log */
void    cli_history_log_init(void);
errno_t cli_ghistory(void);
errno_t cli_lhistory(void);

#define BOOKMARK_MAX 64
#define BOOKMARK_NAMELEN 200
#define BOOKMARK_CMDLEN 2000

void        cli_history_expand(void);
void        cli_alias_expand(void);
void        cli_expand_braces(char *line, int maxlen);
void        cli_session_log_cmd(const char *cmd);
void        cli_history_log_cmd(const char *cmd);
void        cli_history_log_prompt(const char *prompt);
void        cli_history_log_shell(const char *cmd);
const char *CLI_history_file(void);
void        cli_save_last_argument(void);
const char *strip_ws(const char *s);

#endif /* CLICORE_UI_H */

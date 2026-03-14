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
void CLI_setup_hint_area(void);
void CLI_cleanup_scroll_region(void);

/* Command aliases */
void    cli_alias_load(void);
errno_t cli_alias_add(void);
errno_t cli_alias_remove(void);
errno_t cli_alias_list(void);

/* Watch command */
errno_t cli_watch(void);

/* Startup script */
void cli_milkrc_load(void);

/* Command timing */
errno_t cli_time(void);

/* Command statistics */
errno_t cli_cmdstats(void);

/* Syntax highlighting toggle */
#ifdef USE_READLINE
errno_t cli_syntax_highlight_toggle(void);
#endif

/* Persistent history */
void cli_history_load(void);
void cli_history_save(void);

/* Script execution */
errno_t cli_source(void);

/* Configurable prompt */
errno_t cli_setprompt(void);
void cli_build_prompt(
    const char *fmt,
    char       *out,
    int         maxlen
);

/* Command bookmarks */
errno_t cli_bookmark(void);
void cli_bookmark_load(void);

/* Session logging */
errno_t cli_sessionlog(void);

/* History display */
errno_t cli_history_show(void);

/* Fuzzy history search */
errno_t cli_searchhist(void);

/* Built-in cd and pwd */
errno_t cli_cd(void);
errno_t cli_pwd(void);

#endif

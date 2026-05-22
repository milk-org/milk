/**
 * @file procCTRL_TUI.h
 * @brief Procctrl tui module
 */

#ifndef _PROCCTRL_TUI_H
#define _PROCCTRL_TUI_H

#define PROCCTRL_DISPLAYMODE_HELP 1
#define PROCCTRL_DISPLAYMODE_CTRL 2
#define PROCCTRL_DISPLAYMODE_RESOURCES 3
#define PROCCTRL_DISPLAYMODE_TRIGGER 4
#define PROCCTRL_DISPLAYMODE_TIMING 5
#define PROCCTRL_DISPLAYMODE_PROCINFO 6

extern int  procCTRL_debug_mode;
extern char procCTRL_logfile[1024];

#endif
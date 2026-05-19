/**
 * @file CLIcore_help_tui.c
 *
 * @brief Interactive TUI help and fuzzy search
 *
 * Provides ncurses-free interactive help screens
 * that run inside the terminal using raw ANSI
 * escape codes and termios raw mode:
 *
 * - **fhelp**: Interactive fuzzy command search.
 *   Type to filter commands, arrow keys to navigate,
 *   Enter to view full help for the selected command.
 *
 * - **fhist**: Interactive fuzzy history search.
 *   Type to filter past commands, Enter to re-execute.
 *
 * - **fparam**: Interactive FPS parameter browser.
 *   Type to filter FPS names, Enter to view params.
 *
 * ## Design approach
 *
 * These functions use termios to switch stdin to raw
 * mode, then render a scrollable list with ANSI escape
 * sequences. The fuzzy scoring function ranks matches
 * by consecutive-character bonus and position penalty,
 * giving a "fish-shell-like" feel.
 */

// -----------------------------------------------------------------------------
// Interactive Fuzzy Help (fhelp)
// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <ctype.h>
#ifdef USE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#include <sys/select.h>

#include "CLIcore.h"
#include "CLIcore_help.h"
#include "fps.h"
#include "fps_connect.h"

/**
 * @brief Read one byte from stdin using read(), bypassing stdio buffering.
 *
 * All TUI input loops must use this instead of getchar() so that
 * select() and read() operate on the same file-descriptor buffer.
 * When getchar() is used, stdio pre-reads multiple bytes into its
 * internal buffer, leaving the kernel fd empty; select() then sees
 * no data and incorrectly treats an arrow-key ESC sequence as a
 * bare ESC press.
 *
 * Returns the byte read (0-255 as unsigned char cast to int),
 * or -1 on error / EOF.
 */
static int tui_readchar(void)
{
    unsigned char ch;
    ssize_t n = read(STDIN_FILENO, &ch, 1);
    return (n == 1) ? (int)ch : -1;
}

/**
 * @brief Wait up to ms milliseconds for stdin fd to have data.
 *
 * Returns 1 if data is ready, 0 on timeout.
 * Must be paired with tui_readchar() (not getchar()) so that
 * select() and read() both observe the same kernel buffer.
 */
static int tui_stdin_wait_ms(int ms)
{
    fd_set fds;
    struct timeval tv;

    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    tv.tv_sec  = 0;
    tv.tv_usec = ms * 1000;
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

/**
 * @brief Compute a fuzzy subsequence match score.
 *
 * Walks through @query and @target simultaneously.
 * Each character in @query that matches a character
 * in @target scores +10. A length penalty (-len)
 * favors shorter targets. Returns -1000 if @query
 * was not fully consumed (incomplete match).
 *
 * @param query   Search string from user input
 * @param target  Candidate string to match against
 * @return Score (higher = better), -1000 if no match
 */
static int fuzzy_match_score(
    const char *query,
    const char *target)
{
    if (!query || !query[0]) return 10000; // Empty query matches perfectly
    
    int score = 0;
    const char *q = query;
    const char *t = target;
    
    while (*q && *t) {
        if (tolower(*q) == tolower(*t)) {
            score += 10;
            q++;
        }
        t++;
    }
    
    // Penalty for target length (prefer shorter exact matches)
    score -= strlen(target);
    
    if (*q) return -1000; // Didn't match all characters in query
    return score;
}

// Structure for sorting matches
typedef struct {
    int index;
    int score;
} MatchScore;

/**
 * @brief qsort comparator — sort matches by
 *        descending score.
 */
static int compare_matches(
    const void *a,
    const void *b)
{
    return ((MatchScore*)b)->score - ((MatchScore*)a)->score;
}

/**
 * @brief Interactive fuzzy command search (fhelp).
 *
 * Provides a TUI where the user types a partial
 * command name and sees a live-filtered, ranked
 * list of matching CLI commands. Arrow keys
 * navigate; Enter selects and inserts the command
 * into the readline buffer.
 */
int cli_fhelp(void)
{
    struct termios oldt, newt;
    char query[128] = {0};
    int query_len = 0;
    int selected = 0;
    int num_matches = 0;
    MatchScore matches[1024];

    // Setup raw terminal mode
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO | ISIG);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while (1) {
        // Compute matches
        num_matches = 0;
        for (long i = 0; i < data.NBcmd; i++) {
            if (num_matches >= 1024) break;
            
            int s1 = fuzzy_match_score(query, data.cmd[i].key);
            int s2 = fuzzy_match_score(query, data.cmd[i].info);
            int best_score = s1 > s2 ? s1 : s2;
            
            if (best_score > -500) {
                matches[num_matches].index = i;
                matches[num_matches].score = best_score;
                num_matches++;
            }
        }
        
        qsort(matches, num_matches, sizeof(MatchScore), compare_matches);

        // Clamp selection
        if (selected >= num_matches) selected = num_matches - 1;
        if (selected < 0) selected = 0;

        // Render UI
        printf("\033[2J\033[H"); // Clear screen
        printf("\033[1;36m>> Interactive Fuzzy Help <<\033[0m\n");
        printf("Search: \033[1m%s\033[0m_\n\n", query);
        
        int display_count = num_matches > 20 ? 20 : num_matches;
        
        for (int i = 0; i < display_count; i++) {
            int cmd_idx = matches[i].index;
            if (i == selected) {
                printf("\033[1;33m> %-20s : %s\033[0m\n", data.cmd[cmd_idx].key, data.cmd[cmd_idx].info);
            } else {
                printf("  %-20s : %s\n", data.cmd[cmd_idx].key, data.cmd[cmd_idx].info);
            }
        }
        
        printf("\n\033[2m[Up/Down/PgUp/PgDn] Navigate  [Enter] Select  [Esc/Ctrl+C] Cancel\033[0m\n");

        // Input loop
        int c = tui_readchar();
        if (c == 27) { // Escape seq
            if (tui_stdin_wait_ms(50)) {
                int b1 = tui_readchar();
                int b2 = tui_readchar();
                if (b1 == '[') {
                    if (b2 == 'A') selected--; // Up
                    else if (b2 == 'B') selected++; // Down
                    else if (b2 == '5') { tui_readchar(); selected -= 10; } // PgUp
                    else if (b2 == '6') { tui_readchar(); selected += 10; } // PgDn
                }
            } else {
                selected = -1;
                break; // Bare ESC — cancel
            }
        } else if (c == 10 || c == 13) { // Enter
            break; // Select
        } else if (c == 3 || c == 4) { // Ctrl+C or Ctrl+D
            selected = -1;
            break;
        } else if (c == 127 || c == 8) { // Backspace
            if (query_len > 0) {
                query_len--;
                query[query_len] = '\0';
                selected = 0;
            }
        } else if (c >= 32 && c <= 126 && query_len < 127) { // Printable
            query[query_len++] = (char)c;
            query[query_len] = '\0';
            selected = 0;
        }
    }

    // Restore terminal
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    printf("\033[2J\033[H"); // Clear screen on exit

    if (selected >= 0 && selected < num_matches) {
        int cmd_idx = matches[selected].index;
        printf("Selected command: \033[1m%s\033[0m\n", data.cmd[cmd_idx].key);
        
#ifdef USE_READLINE
        // Stuff the selected command into the readline input stream. 
        // This is safer than modifying the rl_line_buffer directly while inside the handler
        // because readline will process these stuffed characters on its very next read cycle
        // and echo them correctly as if the user is typing them at the prompt.
        for (size_t i = 0; i < strlen(data.cmd[cmd_idx].key); i++) {
            rl_stuff_char(data.cmd[cmd_idx].key[i]);
        }
        rl_stuff_char(' ');
#else
        // If not using readline, just print it and they can copy-paste.
#endif
    }

    return RETURN_SUCCESS;
}


// -----------------------------------------------------------------------------
// Interactive Fuzzy History (fhist)
// -----------------------------------------------------------------------------

/**
 * @brief Interactive fuzzy history search (fhist).
 *
 * TUI interface for searching through readline
 * command history. The user types a substring;
 * matching history entries are shown ranked by
 * fuzzy score. Selecting an entry inserts it into
 * the readline buffer for immediate re-execution.
 */
int cli_fhist(void)
{
#ifdef USE_READLINE
    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL || history_length == 0)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    struct termios oldt, newt;
    char query[128] = {0};
    int query_len = 0;
    int selected = 0;
    int num_matches = 0;
    MatchScore matches[1024];

    // Setup raw terminal mode
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO | ISIG);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while (1) {
        // Compute matches
        num_matches = 0;
        // Search backwards through history
        for (int i = history_length - 1; i >= 0; i--) {
            if (num_matches >= 1024) break;
            
            // Skip duplicates (simple lookback to recent matching entries)
            // It's common to have the same command consecutively in history
            int is_dup = 0;
            for (int j = 0; j < num_matches; j++) {
                if (strcmp(hlist[i]->line, hlist[matches[j].index]->line) == 0) {
                    is_dup = 1;
                    break;
                }
            }
            if (is_dup) continue;
            
            int score = fuzzy_match_score(query, hlist[i]->line);
            
            if (score > -500) {
                matches[num_matches].index = i;
                // Penalize older entries so recent ones stay at top if scores match
                matches[num_matches].score = score - (history_length - i);
                num_matches++;
            }
        }
        
        // Sort if we have a query. If not, keep chronological (reversed).
        if (query_len > 0) {
            qsort(matches, num_matches, sizeof(MatchScore), compare_matches);
        }

        // Clamp selection
        if (selected >= num_matches) selected = num_matches - 1;
        if (selected < 0) selected = 0;

        // Render UI
        printf("\033[2J\033[H"); // Clear screen
        printf("\033[1;36m>> Interactive Fuzzy History Search <<\033[0m\n");
        printf("Search: \033[1m%s\033[0m_\n\n", query);
        
        int display_count = num_matches > 20 ? 20 : num_matches;
        
        for (int i = 0; i < display_count; i++) {
            int hist_idx = matches[i].index;
            if (i == selected) {
                printf("\033[1;33m> %s\033[0m\n", hlist[hist_idx]->line);
            } else {
                printf("  %s\n", hlist[hist_idx]->line);
            }
        }
        
        printf("\n\033[2m[Up/Down/PgUp/PgDn] Navigate  [Enter] Select  [Esc/Ctrl+C] Cancel\033[0m\n");

        // Input loop
        int c = tui_readchar();
        if (c == 27) { // Escape seq
            if (tui_stdin_wait_ms(50)) {
                int b1 = tui_readchar();
                int b2 = tui_readchar();
                if (b1 == '[') {
                    if (b2 == 'A') selected--; // Up
                    else if (b2 == 'B') selected++; // Down
                    else if (b2 == '5') { tui_readchar(); selected -= 10; } // PgUp
                    else if (b2 == '6') { tui_readchar(); selected += 10; } // PgDn
                }
            } else {
                selected = -1;
                break; // Bare ESC — cancel
            }
        } else if (c == 10 || c == 13) { // Enter
            break; // Select
        } else if (c == 3 || c == 4) { // Ctrl+C or Ctrl+D
            selected = -1;
            break;
        } else if (c == 127 || c == 8) { // Backspace
            if (query_len > 0) {
                query_len--;
                query[query_len] = '\0';
                selected = 0;
            }
        } else if (c >= 32 && c <= 126 && query_len < 127) { // Printable
            query[query_len++] = (char)c;
            query[query_len] = '\0';
            selected = 0;
        }
    }

    // Restore terminal
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    printf("\033[2J\033[H"); // Clear screen on exit

    if (selected >= 0 && selected < num_matches) {
        int hist_idx = matches[selected].index;
        printf("Selected history: \033[1m%s\033[0m\n", hlist[hist_idx]->line);
        
        for (size_t i = 0; i < strlen(hlist[hist_idx]->line); i++) {
            rl_stuff_char(hlist[hist_idx]->line[i]);
        }
    }
#else
    printf("Readline not available. History fuzzy search requires readline.\n");
#endif
    return RETURN_SUCCESS;
}


// -----------------------------------------------------------------------------
// Interactive FPS Parameter Edit (fparam)
// -----------------------------------------------------------------------------

/**
 * @brief Interactive FPS parameter editor (fparam).
 *
 * Connects to a live FPS by name and presents a
 * TUI list of its parameters. Arrow keys navigate;
 * Enter opens inline editing of the selected
 * parameter value. Changes are written directly
 * to the FPS shared memory.
 *
 * This provides a lightweight alternative to
 * milk-fpsCTRL for quick parameter edits.
 */
int cli_fparam(void)
{
    if (data.cmdargtoken[1].type != CMDARGTOKEN_TYPE_STRING) {
        printf("Usage: fparam <fpsname>\n");
        return RETURN_SUCCESS;
    }
    
    char *fpsname = data.cmdargtoken[1].val.string;
    
    FPS fps;
    fps.SMfd = -1;

    if (fps_connect(fpsname, &fps, 0) == -1) {
        printf("Error: cannot connect to FPS '%s'.\n", fpsname);
        return RETURN_SUCCESS;
    }

    struct termios oldt, newt;
    int selected = 0;
    
    // collect active params
    int active_pindices[1024];
    int num_params = 0;
    
    for (int pindex = 0; pindex < fps.md->NBparamMAX; pindex++) {
        if (fps.parray[pindex].fpflag & FPFLAG_USED) {
            if (num_params < 1024) {
               active_pindices[num_params++] = pindex;
            }
        }
    }

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO | ISIG);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    char error_msg[200] = {0};

    while(1) {
        if (selected >= num_params) selected = num_params - 1;
        if (selected < 0) selected = 0;

        printf("\033[2J\033[H"); // Clear screen
        printf("\033[1;36m>> Interactive FPS Parameter Editor : %s <<\033[0m\n\n", fpsname);
        
        // determine the display window
        int display_count = 20;
        int start_idx = selected - (display_count/2);
        if (start_idx < 0) start_idx = 0;
        if (start_idx + display_count > num_params) start_idx = num_params - display_count;
        if (start_idx < 0) start_idx = 0;
        
        // Render rows
        for (int i = start_idx; i < start_idx + display_count && i < num_params; i++) {
            int pidx = active_pindices[i];
            char valstring[200];
            if (fps.parray[pidx].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps.parray[pidx].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps.parray[pidx], valstring, 200);
            }
            
            const char *display_keyword = fps.parray[pidx].keywordfull;
            int prefix_len = strlen(fps.md->name);
            if (strncmp(display_keyword, fps.md->name, prefix_len) == 0 && display_keyword[prefix_len] == '.') {
                display_keyword += prefix_len + 1;
            }
            
            if (i == selected) {
                printf("\033[1;33m> %-30s : %-20s  (%s)\033[0m\n", display_keyword, valstring, fps.parray[pidx].description);
            } else {
                printf("  %-30s : %-20s  (%s)\n", display_keyword, valstring, fps.parray[pidx].description);
            }
        }
        
        printf("\n\033[2m[Up/Down/PgUp/PgDn] Navigate  [Enter] Edit  [Esc/q] Quit\033[0m\n");
        if (error_msg[0]) {
            printf("\033[1;31mError: %s\033[0m\n", error_msg);
            error_msg[0] = '\0';
        }
        
        // Input loop
        int c = tui_readchar();
        if (c == 27) { // Escape seq
            if (tui_stdin_wait_ms(50)) {
                int b1 = tui_readchar();
                int b2 = tui_readchar();
                if (b1 == '[') {
                    if (b2 == 'A') selected--; // Up
                    else if (b2 == 'B') selected++; // Down
                    else if (b2 == '5') { tui_readchar(); selected -= 10; } // PgUp
                    else if (b2 == '6') { tui_readchar(); selected += 10; } // PgDn
                }
            } else {
                break; // Bare ESC — quit
            }
        } else if (c == 'q' || c == 'Q') {
            break;
        } else if (c == 3 || c == 4) { // Ctrl+C or Ctrl+D
            break;
        } else if (c == 10 || c == 13) {
            // Edit the selected parameter
            int pidx = active_pindices[selected];
            
            // disable raw mode to get input
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
            
            printf("\033[2J\033[H");
            printf("Editing parameter: \033[1;36m%s\033[0m\n", fps.parray[pidx].keywordfull);
            printf("Description: %s\n", fps.parray[pidx].description);
            
            char valstring[200];
            if (fps.parray[pidx].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps.parray[pidx].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps.parray[pidx], valstring, 200);
            }
            printf("Current value: %s\n", valstring);
            printf("New value (leave empty to cancel): ");
            
            char inputbuf[256];
            if (fgets(inputbuf, sizeof(inputbuf), stdin) != NULL) {
                // strip newline
                int len = strlen(inputbuf);
                if (len > 0 && inputbuf[len-1] == '\n') inputbuf[len-1] = '\0';
                
                if (strlen(inputbuf) > 0) {
                    // Update parameter logic
                    int ret = functionparameter_SetParamValue_fromString(&fps, pidx, inputbuf);
                    
                    if (ret != EXIT_SUCCESS) {
                        snprintf(error_msg, sizeof(error_msg), "Invalid value format for the type.");
                    } else {
                        fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                    }
                }
            }
            
            // re-enable raw mode
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        }
    }
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    printf("\033[2J\033[H");
    
    fps_disconnect(&fps);
    return RETURN_SUCCESS;
}

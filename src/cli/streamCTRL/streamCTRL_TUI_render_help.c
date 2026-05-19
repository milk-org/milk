#include "streamCTRL_TUI_render_internal.h"

/**
 * @brief Render the streamCTRL help overlay.
 */
void streamCTRL__render_header_help(streamCTRLarg_struct *streamCTRLdata,
                                    struct streamCTRL_TUI_state *state)
{
//int attrval = A_BOLD;

    DEBUG_TRACEPOINT(" ");


    print_help_entry("x", "Exit");

    TUI_newline();
    TUI_printfw("============ SCREENS");
    TUI_newline();
    print_help_entry("h", "help");
    print_help_entry("F2", "semaphore values");
    print_help_entry("F3", "semaphore read  PIDs");
    print_help_entry("F4", "semaphore write PIDs");
    print_help_entry("F5", "stream process trace");
    print_help_entry("F6", "stream open by processes ...");
    print_help_entry("CTRL+L/R", "cycle between tabs");

    TUI_newline();
    TUI_printfw("============ ACTIONS");
    TUI_newline();
    print_help_entry("CTRL+e", "Erase stream");

    TUI_newline();
    TUI_printfw("============ SCANNING");
    TUI_newline();
    print_help_entry("}", "Increase scan frequency");
    print_help_entry("{", "Decrease scan frequency");
    print_help_entry("o", "output next scan to file");

    TUI_newline();
    TUI_printfw("============ DISPLAY");
    TUI_newline();
    print_help_entry("+/-", "Increase/decrease display frequency");
    print_help_entry("]", "Cycle sort column (name,type,size...)");
    print_help_entry("[", "Toggle sort direction (asc/desc)");
    print_help_entry("1", "Sort by stream name (alphabetical)");
    print_help_entry("2", "Sort by recently updated");
    print_help_entry("3", "Sort by process access");
    print_help_entry("4", "Sort by frequency (descending)");
    print_help_entry("s", "Show 3 semaphores / all semaphores");
    print_help_entry("r", "Force full screen redraw");
    print_help_entry("F", "Set match string pattern");
    print_help_entry("f", "Toggle apply match string to stream");

    TUI_newline();
    TUI_printfw("============ NAVIGATION");
    TUI_newline();
    print_help_entry("UP/DOWN", "Move selection");
    print_help_entry("PgUp/PgDn", "Move selection by 10");
    print_help_entry("LEFT/RIGHT", "Summary / detail view");
    print_help_entry("Click", "Select stream entry");
    print_help_entry("Scroll", "Scroll selection up/down");
}

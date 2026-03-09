/**
 * @file milk-fpsexec-help.c
 * @brief Milk fpsexec help module
 */

#include <stdio.h>

#define C_TITLE "\033[1;36m"
#define C_HDR   "\033[1;34m"
#define C_CMD   "\033[1;32m"
#define C_NOTE  "\033[1;33m"
#define C_BOLD  "\033[1m"
#define C_FPS   "\033[1;35m"
#define C_RST   "\033[0m"
#define C_WARN  "\033[1;31m"

/**
 * milk-fpsexec-help
 *
 * User guide for standalone FPS executables
 * (milk-fpsexec-* and cacao-fpsexec-*).
 * Complements milk-fps-help which covers FPS
 * concepts, inspection, and management tools.
 */
int main(void)
{
    printf("\n");
    printf(C_TITLE
           "========================================"
           "========\n" C_RST);
    printf(C_TITLE
           "    Standalone FPS Executables User Guide"
           "        \n" C_RST);
    printf(C_TITLE
           "========================================"
           "========\n" C_RST);
    printf("\n");

    printf(
        "FPS standalone executables follow"
        " the naming convention:\n"
        "  " C_CMD "milk-fpsexec-<name>" C_RST
        "   or   "
        C_CMD "cacao-fpsexec-<name>" C_RST "\n"
        "\n"
        "Each program manages its own FPS"
        " (Function Parameter Structure)\n"
        "for configuration, execution, and"
        " parameter tuning.\n\n");

    /* ---- Usage ---- */
    printf(C_HDR "General Usage\n" C_RST);
    printf(
        "  $ " C_CMD "<executable> " C_FPS
        "[fpsname:]" C_CMD "<command>"
        " [options]\n" C_RST
        "\n"
        "  If no " C_FPS "fpsname:" C_RST
        " prefix is given, a default name"
        " is used.\n"
        "  Run " C_CMD "<executable> -h" C_RST
        " for command-specific help and"
        " parameters.\n\n");

    /* ---- Commands ---- */
    printf(C_HDR "Commands\n" C_RST);
    printf(
        "  " C_CMD "fpsinit" C_RST
        "      Create the FPS shared-memory"
        " segment\n"
        "  " C_CMD "fps" C_RST
        "          Print current FPS content\n"
        "  " C_CMD "fpslist" C_RST
        "      List FPS instances for this"
        " executable\n"
        "  " C_CMD "confstart" C_RST
        "    Start configuration monitoring"
        " loop\n"
        "  " C_CMD "confstep" C_RST
        "     Run one configuration step\n"
        "  " C_CMD "confstop" C_RST
        "     Stop configuration loop\n"
        "  " C_CMD "runstart" C_RST
        "     Start main processing loop\n"
        "  " C_CMD "runstop" C_RST
        "      Stop main processing loop\n"
        "  " C_CMD "exec" C_RST
        "         Auto-init + run\n\n");

    /* ---- Options ---- */
    printf(C_HDR "Common Options\n" C_RST);
    printf(
        "  " C_FPS "fpsname:" C_RST
        "         Name prefix for the FPS\n"
        "  " C_CMD "-n, --name" C_RST
        "      Specify FPS name\n"
        "  " C_CMD "-tmux" C_RST
        "           Run inside a tmux"
        " session\n"
        "  " C_CMD "-procinfo" C_RST
        "       Enable process monitoring"
        " (for fpsinit)\n"
        "  " C_CMD "-h" C_RST
        "              Show help message\n\n");

    /* ---- Workflow ---- */
    printf(C_HDR "Typical Workflow\n" C_RST);
    printf(
        "  " C_BOLD "1." C_RST
        " Initialize the FPS:\n"
        "     $ " C_CMD
        "milk-fpsexec-clitest " C_FPS
        "myfps00" C_CMD ":fpsinit\n" C_RST
        "  " C_BOLD "2." C_RST
        " Optionally tune parameters:\n"
        "     $ " C_CMD
        "milk-fps-set " C_FPS "myfps00"
        C_CMD ".gain 1.5\n" C_RST
        "  " C_BOLD "3." C_RST
        " Start execution:\n"
        "     $ " C_CMD
        "milk-fpsexec-clitest " C_FPS
        "myfps00" C_CMD ":runstart\n" C_RST
        "  " C_BOLD "4." C_RST
        " Monitor with TUI:\n"
        "     $ " C_CMD
        "milk-fpsCTRL\n" C_RST
        "  " C_BOLD "5." C_RST
        " Stop when done:\n"
        "     $ " C_CMD
        "milk-fpsexec-clitest " C_FPS
        "myfps00" C_CMD ":runstop\n" C_RST
        "\n");

    /* ---- Quick execution ---- */
    printf(C_HDR "Quick Execution (No Workflow)\n"
           C_RST);
    printf(
        "  Use the " C_CMD "exec" C_RST
        " command with positional arguments"
        " to auto-create\n"
        "  the FPS, run, and exit:\n"
        "    $ " C_CMD
        "milk-fpsexec-clitest exec 42 cam01\n"
        C_RST
        "  To use a local-only (non-shared)"
        " FPS, prefix the name with "
        C_BOLD "_" C_RST ":\n"
        "    $ " C_CMD
        "milk-fpsexec-clitest " C_FPS
        "_myfps00" C_CMD ":exec 42 cam01\n"
        C_RST "\n");

    /* ---- tmux ---- */
    printf(C_HDR "Tmux Sessions\n" C_RST);
    printf(
        "  Use " C_CMD "-tmux" C_RST
        " to auto-create a tmux session"
        " and dispatch commands:\n"
        "    $ " C_CMD
        "milk-fpsexec-clitest " C_FPS
        "myfps00" C_CMD
        ":fpsinit -tmux\n" C_RST
        "    $ " C_CMD
        "milk-fpsexec-clitest " C_FPS
        "myfps00" C_CMD
        ":confstart -tmux\n" C_RST
        "    $ " C_CMD
        "milk-fpsexec-clitest " C_FPS
        "myfps00" C_CMD
        ":runstart -tmux\n" C_RST
        "\n");

    /* ---- Alternate tools ---- */
    printf(C_HDR
           "Alternate Tools (After FPS Created)\n"
           C_RST);
    printf(
        "  Once an FPS exists, you can also use"
        " generic FPS tools:\n"
        "    " C_CMD "milk-fps-confstart "
        C_FPS "<fpsname>\n" C_RST
        "    " C_CMD "milk-fps-confstop  "
        C_FPS "<fpsname>\n" C_RST
        "    " C_CMD "milk-fps-runstart  "
        C_FPS "<fpsname>\n" C_RST
        "    " C_CMD "milk-fps-runstop   "
        C_FPS "<fpsname>\n" C_RST
        "    " C_CMD "milk-fps-confstep  "
        C_FPS "<fpsname>\n" C_RST
        "\n");

    /* ---- See also ---- */
    printf(C_NOTE
           "Run " C_CMD "milk-fps-help"
           C_NOTE " for FPS concepts,"
           " inspection, and management tools.\n"
           C_RST "\n");

    return 0;
}

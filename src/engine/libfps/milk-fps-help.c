/**
 * @file milk-fps-help.c
 * @brief Milk fps help module
 */

#include <stdio.h>

#define C_TITLE "\033[1;36m"
#define C_HDR   "\033[1;34m"
#define C_CMD   "\033[1;32m"
#define C_NOTE  "\033[1;33m"
#define C_BOLD  "\033[1m"
#define C_FPS   "\033[1;35m"
#define C_RST   "\033[0m"

/**
 * milk-fps-help
 *
 * Describes FPS (Function Parameter Structures):
 * what they are, how to inspect, modify, and
 * manage them. Complements milk-fpsexec-help
 * which covers fpsexec standalone executables.
 */
int main(void)
{
    printf("\n");
    printf(C_TITLE
           "========================================"
           "========\n" C_RST);
    printf(C_TITLE
           "     Function Parameter Structure (FPS) "
           "        \n" C_RST);
    printf(C_TITLE
           "========================================"
           "========\n" C_RST);
    printf("\n");

    printf(
        "An FPS is a shared-memory parameter"
        " context. It allows\n"
        "users and programs to view, tune, and"
        " steer parameters\n"
        "while a computation runs. It unifies"
        " configuration,\n"
        "control structures, and telemetry into"
        " a single namespace.\n"
        "\n"
        "Each FPS appears as a .fps.shm file"
        " in the FPS shared\n"
        "memory directory.\n\n");

    printf(C_HDR "Listing FPS Instances\n" C_RST);
    printf(
        "  " C_CMD "milk-fps-list" C_RST
        "        List all active shared-memory"
        " FPS\n"
        "  " C_CMD "milk-fps-list -e" C_RST
        "     Show full executable paths\n"
        "  " C_CMD "milk-fps-list -v" C_RST
        "     Verbose (show search directory)\n"
        "\n");

    printf(C_HDR "Inspecting FPS Content\n" C_RST);
    printf(
        "  " C_CMD "milk-fps-info " C_FPS
        "<fpsname>" C_RST
        "    Display parameters and values\n"
        "  " C_CMD "milk-fpsCTRL" C_RST
        "              Interactive TUI for FPS"
        " management\n\n");

    printf(C_HDR "Modifying Parameters\n" C_RST);
    printf(
        "  " C_CMD "milk-fps-set " C_FPS
        "<fpsname>" C_CMD ".<param> <value>"
        C_RST "\n"
        "  Example:\n"
        "    $ " C_CMD "milk-fps-set "
        C_FPS "myfps00" C_CMD ".gain 1.5\n"
        C_RST
        "    $ " C_CMD "milk-fps-set "
        C_FPS "myfps00" C_CMD ".verbose 1\n"
        C_RST "\n");

    printf(C_HDR "Tracking Parameter Changes\n"
           C_RST);
    printf(
        "  " C_CMD "milk-fps-track " C_FPS
        "<fpsname>" C_RST
        "   Monitor live parameter changes\n\n");

    printf(C_HDR "Managing Execution\n" C_RST);
    printf(
        "  Each FPS may have a config loop"
        " (conf) and a run loop:\n"
        "    " C_CMD "milk-fps-confstart "
        C_FPS "<fpsname>" C_RST
        "   Start config loop\n"
        "    " C_CMD "milk-fps-confstop  "
        C_FPS "<fpsname>" C_RST
        "   Stop config loop\n"
        "    " C_CMD "milk-fps-confstep  "
        C_FPS "<fpsname>" C_RST
        "   Single config step\n"
        "    " C_CMD "milk-fps-runstart  "
        C_FPS "<fpsname>" C_RST
        "   Start run loop\n"
        "    " C_CMD "milk-fps-runstop   "
        C_FPS "<fpsname>" C_RST
        "   Stop run loop\n\n");

    printf(C_HDR "Removing FPS\n" C_RST);
    printf(
        "  " C_CMD "milk-fps-rm " C_FPS
        "<fpsname>" C_RST
        "     Remove FPS shared memory\n"
        "  " C_CMD "milk-fps-rm" C_RST
        "              Interactive mode"
        " (select from list)\n\n");

    printf(C_HDR "Local FPS (Non-Shared)\n"
           C_RST);
    printf(
        "  FPS names starting with "
        C_BOLD "_" C_RST
        " (underscore) use process-local\n"
        "  memory instead of shared memory."
        " No .fps.shm file is created.\n"
        "  Up to 64 local FPS instances can"
        " coexist per process.\n"
        C_NOTE "  Note:" C_RST
        " Local FPS are not visible to"
        " milk-fps-list, milk-fps-info,\n"
        "  or milk-fpsCTRL. Use the "
        C_BOLD "?" C_RST
        " query within the milk CLI.\n\n");

    printf(C_HDR
           "CLI Interaction (within milk)\n"
           C_RST);
    printf(
        "  The milk CLI syntax for FPS is:"
        "  cmdkey:" C_FPS "fpsname" C_RST
        ":action\n"
        "  If " C_FPS "fpsname" C_RST
        " is omitted, a default name is"
        " assigned.\n"
        "  Actions: " C_BOLD "init" C_RST
        ", " C_BOLD "initp" C_RST
        " (with processinfo), "
        C_BOLD "?" C_RST " (query)\n"
        "  Example:\n"
        "    milk > " C_CMD
        "modex.fpsclitest:" C_FPS "myfps00"
        C_CMD ":init\n" C_RST
        "    milk > " C_CMD
        "modex.fpsclitest:" C_FPS "myfps00"
        C_CMD ":?\n" C_RST
        "    milk > " C_CMD
        "modex.fpsclitest .gain 1.5\n"
        C_RST "\n");

    printf(C_NOTE
           "Run " C_CMD "milk-fpsexec-help"
           C_NOTE " for a user guide to"
           " standalone FPS executables.\n"
           C_RST "\n");

    return 0;
}

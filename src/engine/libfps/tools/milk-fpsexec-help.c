// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-fpsexec-help.c
 * @brief Milk fpsexec help module
 */

#include <stdio.h>
#include <string.h>

#define C_TITLE "\033[1;36m"
#define C_HDR "\033[1;34m"
#define C_CMD "\033[1;32m"
#define C_NOTE "\033[1;33m"
#define C_BOLD "\033[1m"
#define C_FPS "\033[1;35m"
#define C_RST "\033[0m"
#define C_WARN "\033[1;31m"

/**
 * milk-fpsexec-help
 *
 * User guide for standalone FPS executables
 * (milk-fpsexec-* and cacao-fpsexec-*).
 * Complements milk-fps-help which covers FPS
 * concepts, inspection, and management tools.
 */
int main(int argc, char *argv[])
{
    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h1") == 0 || strcmp(argv[ii], "--help-oneline") == 0)
        {
            printf("guide to standalone FPS executables\n");
            return 0;
        }
    }
    printf("\n");
    printf(C_TITLE "========================================"
                   "========\n" C_RST);
    printf(C_TITLE "    Standalone FPS Executables User Guide"
                   "        \n" C_RST);
    printf(C_TITLE "========================================"
                   "========\n" C_RST);
    printf("\n");

    printf("FPS standalone executables follow"
           " the naming convention:\n"
           "  " C_CMD "milk-fpsexec-<name>" C_RST "   or   " C_CMD "cacao-fpsexec-<name>" C_RST "\n"
           "\n"
           "Each program manages its own FPS"
           " (Function Parameter Structure)\n"
           "for configuration, execution, and"
           " parameter tuning.\n\n");

    /* ---- Usage ---- */
    printf(C_HDR "General Usage\n" C_RST);
    printf("  $ " C_CMD "<executable> " C_RST "[options] " C_FPS "[fpsname:]" C_CMD
           "<command>" C_RST " [values...]\n\n"
           "  If no " C_FPS "fpsname:" C_RST " prefix is given, a default name"
           " is used.\n"
           "  Run " C_CMD "<executable> -h" C_RST " for command-specific help and"
           " parameters.\n\n");

    /* ---- Options ---- */
    printf(C_HDR "Common Options\n" C_RST);
    printf("  " C_FPS "fpsname:" C_RST "         Name prefix for the FPS\n"
           "  " C_CMD "-n, --name" C_RST "      Specify FPS name\n"
           "  " C_CMD "--no-procinfo" C_RST "   One-shot unit: no process monitoring"
           " (fpsinit, exec)\n"
           "  " C_CMD "--fg" C_RST "            Keep confstart/runstart in the foreground\n"
           "  " C_CMD "--timeout S" C_RST "     Readiness timeout of confstart/runstart"
           " (default 10)\n"
           "  " C_CMD "-loops" C_RST "         Infinite loop, stream"
           " semaphore trigger\n"
           "  " C_CMD "-loopd SEC" C_RST "     Infinite loop, delay"
           " trigger (seconds)\n"
           "  " C_CMD "-h, --help" C_RST "      Show help message\n"
           "  " C_CMD "-h1, --help-oneline" C_RST " Show one-line description\n"
           "  " C_CMD "-h2, --help-description" C_RST " Show verbose description\n"
           "  " C_CMD "-hm, --help-mono" C_RST "    Show help message (monochrome)\n"
           "  Some executables add their own options: see the " C_BOLD "Unit Options" C_RST
           " of their -h.\n\n");

    /* ---- Commands ---- */
    printf(C_HDR "Commands\n" C_RST);
    printf("  " C_CMD "fpsinit" C_RST " [values]   Create the FPS shared-memory"
           " segment, then set values\n"
           "  " C_CMD "fps" C_RST "                Print current FPS content\n"
           "  " C_CMD "fpslist" C_RST "            List FPS instances for this"
           " executable\n"
           "  " C_CMD "confstart" C_RST "          Start the configuration loop in the"
           " background\n"
           "  " C_CMD "confstep" C_RST "           Run one configuration step\n"
           "  " C_CMD "confstop" C_RST "           Stop configuration loop\n"
           "  " C_CMD "runstart" C_RST " [values]  Set values, start the processing loop in the"
           " background\n"
           "  " C_CMD "runstop" C_RST "            Stop main processing loop\n"
           "  " C_CMD "run" C_RST " [values]       Set values, run the loop in the foreground\n"
           "  " C_CMD "set" C_RST " [values]       Set values in an existing FPS\n"
           "  " C_CMD "exec" C_RST " [values]      Auto-init + set values + run\n\n");

    /* ---- Values ---- */
    printf(C_HDR "Parameter Values\n" C_RST);
    printf("  The words after the command set FPS parameters, as module commands do:\n"
           "    " C_BOLD "value" C_RST "         the next parameter that has an Idx in -h,"
           " in Idx order\n"
           "    " C_BOLD ".name=value" C_RST "   any parameter, by its keyword (in any order)\n"
           "    " C_BOLD "." C_RST "             keep the current value (also " C_BOLD
           ".name=." C_RST ")\n"
           "  Positional values fill the indexed parameters not set by name. Negative\n"
           "  numbers (-0.5) are values. An unknown parameter, a value that does not\n"
           "  parse for its type, or one value too many is an error (exit status 2):\n"
           "  nothing is changed.\n"
           "    $ " C_CMD "milk-fpsexec-clitest " C_FPS "myfps00" C_CMD
           ":set 42 .p_float64=-1.5\n" C_RST "\n");

    /* ---- Workflow ---- */
    printf(C_HDR "Typical Workflow\n" C_RST);
    printf("  " C_BOLD "1." C_RST " Initialize the FPS:\n"
           "     $ " C_CMD "milk-fpsexec-clitest " C_FPS "myfps00" C_CMD ":fpsinit\n" C_RST
           "  " C_BOLD "2." C_RST " Optionally tune parameters:\n"
           "     $ " C_CMD "milk-fpsexec-clitest " C_FPS "myfps00" C_CMD
           ":set .p_float64=1.5\n" C_RST "     $ " C_CMD "milk-fps-set " C_FPS "myfps00" C_CMD
           ".p_float64 1.5\n" C_RST "  " C_BOLD "3." C_RST " Start execution:\n"
           "     $ " C_CMD "milk-fpsexec-clitest " C_FPS "myfps00" C_CMD ":runstart\n" C_RST
           "  " C_BOLD "4." C_RST " Monitor with TUI:\n"
           "     $ " C_CMD "milk-fpsCTRL\n" C_RST "  " C_BOLD "5." C_RST " Stop when done:\n"
           "     $ " C_CMD "milk-fpsexec-clitest " C_FPS "myfps00" C_CMD ":runstop\n" C_RST "\n");

    /* ---- Quick execution ---- */
    printf(C_HDR "Quick Execution (No Workflow)\n" C_RST);
    printf("  Use the " C_CMD "exec" C_RST " command with values"
           " to auto-create\n"
           "  the FPS, run, and exit:\n"
           "    $ " C_CMD "milk-fpsexec-clitest exec 42 cam01\n" C_RST
           "  To use a local-only (non-shared)"
           " FPS, prefix the name with " C_BOLD "_" C_RST ":\n"
           "    $ " C_CMD "milk-fpsexec-clitest " C_FPS "_myfps00" C_CMD ":exec 42 cam01\n" C_RST
           "\n");

    /* ---- Units without C ---- */
    printf(C_HDR "Compute Units Without C\n" C_RST);
    printf("  " C_CMD "milk-fpsexec-milk-calc" C_RST " and " C_CMD "milk-fpsexec-milk-script" C_RST
           " take their program as unit options\n"
           "  (-e/-c statements, -t trigger, -p declared parameters, -f file); the values\n"
           "  then set the declared parameters, as for any other executable:\n"
           "    $ " C_CMD "milk-fpsexec-milk-calc -n dsub -t cam -p \"gain:float=1\" \\\n"
           "        -e \"out = cam * gain\" exec 2.5\n" C_RST "    $ " C_CMD
           "milk-fpsexec-milk-calc -n dsub set 3.0\n" C_RST "\n");

    /* ---- Background units ---- */
    printf(C_HDR "Units in the Background\n" C_RST);
    printf("  " C_CMD "confstart" C_RST " and " C_CMD "runstart" C_RST
           " start a detached process and return once its loop runs\n"
           "  (exit status 0), fails (1, with its messages) or is not ready in time (124).\n"
           "  Its output goes to <shmdir>/<fpsname>.fps.d/run.out (and conf.out):\n"
           "    $ " C_CMD "milk-fps-log -f " C_FPS "myfps00\n" C_RST
           "  Its environment is the one captured by fpsinit (milk-fps-info --env).\n"
           "  " C_CMD "run" C_RST " or " C_CMD "--fg" C_RST
           " keep the loop in your terminal (gdb, valgrind, a service manager).\n\n");

    /* ---- Alternate tools ---- */
    printf(C_HDR "Alternate Tools (After FPS Created)\n" C_RST);
    printf("  Once an FPS exists, you can also use"
           " generic FPS tools:\n"
           "    " C_CMD "milk-fps-confstart " C_FPS "<fpsname>\n" C_RST "    " C_CMD
           "milk-fps-confstop  " C_FPS "<fpsname>\n" C_RST "    " C_CMD "milk-fps-runstart  " C_FPS
           "<fpsname>\n" C_RST "    " C_CMD "milk-fps-runstop   " C_FPS "<fpsname>\n" C_RST
           "    " C_CMD "milk-fps-confstep  " C_FPS "<fpsname>\n" C_RST "\n");

    /* ---- See also ---- */
    printf(C_NOTE "Run " C_CMD "milk-fps-help" C_NOTE " for FPS concepts,"
                  " inspection, and management tools.\n" C_RST "\n");

    return 0;
}

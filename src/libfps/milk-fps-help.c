#include <stdio.h>

#define C_TITLE "\033[1;36m"   /* Cyan Bold    -> Main section headers / separators */
#define C_HDR   "\033[1;34m"   /* Blue Bold    -> Subheaders inside sections */
#define C_CMD   "\033[1;32m"   /* Green Bold   -> Command names, syntax, execution */
#define C_NOTE  "\033[1;33m"   /* Yellow Bold  -> Tips, Notes, 'run X for more' */
#define C_BOLD  "\033[1m"      /* White Bold   -> Emphasize specific words */
#define C_FPS   "\033[1;35m"   /* Magenta Bold -> FPS names */
#define C_RST   "\033[0m"      /* Reset */

/* Red Bold */
#define C_WARN   "\033[1;31m"

int main()
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "            Function Parameter Structure (FPS)          \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("An FPS provides a shared-memory parameter context, allowing\n");
    printf("users and programs to view, tune, and steer variables externally\n");
    printf("while a computation runs effortlessly. It unifies parameters,\n");
    printf("control structures, and telemetry into a single namespace.\n");
    printf("\n");

    printf(C_HDR "Initializing an FPS\n" C_RST);
    printf("To initialize the parameter space for a module, invoke the\n");
    printf("executable with the " C_BOLD "fpsinit" C_RST " subcommand and provide an FPS name:\n");
    printf("  $ " C_CMD "milk-fpsclitest " C_FPS "myfps00" C_CMD ":fpsinit\n" C_RST);
    printf("  " C_NOTE "Tip:" C_RST " Use the " C_BOLD "-procinfo" C_RST " flag to add daemon monitoring features:\n");
    printf("  $ " C_CMD "milk-fpsclitest " C_FPS "myfps00" C_CMD ":fpsinit -procinfo\n" C_RST);
    printf("\n");

    printf(C_HDR "Viewing FPS Data\n" C_RST);
    printf("You can view the parameters inside an FPS via the terminal:\n");
    printf("  $ " C_CMD "milk-fps-info " C_FPS "myfps00" C_CMD "\n" C_RST);
    printf("Or using the interactive configuration interface (TUI):\n");
    printf("  $ " C_CMD "milk-fpsCTRL\n" C_RST);
    printf("\n");

    printf(C_HDR "Modifying Parameters\n" C_RST);
    printf("Use " C_BOLD "milk-fps-set" C_RST " to change values on-the-fly from bash scripts:\n");
    printf("  $ " C_CMD "milk-fps-set " C_FPS "myfps00" C_CMD ".gain 1.5\n" C_RST);
    printf("  $ " C_CMD "milk-fps-set " C_FPS "myfps00" C_CMD ".verbose 1\n" C_RST);
    printf("\n");

    printf(C_HDR "Managing Execution\n" C_RST);
    printf("A module often has background components like a primary `run`\n");
    printf("loop and a fast configuration supervisor (`conf`).\n");
    printf("  Start process    : $ " C_CMD "milk-fps-runstart " C_FPS "myfps00" C_CMD "\n" C_RST);
    printf("  Stop process     : $ " C_CMD "milk-fps-runstop " C_FPS "myfps00" C_CMD "\n" C_RST);
    printf("\n");
    printf("  Start config loop: $ " C_CMD "milk-fps-confstart " C_FPS "myfps00" C_CMD "\n" C_RST);
    printf("  Stop config loop : $ " C_CMD "milk-fps-confstop " C_FPS "myfps00" C_CMD "\n" C_RST);
    printf("\n");

    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "            Standalone Execution           \n" C_RST);
    printf("\n");
    printf("When providing positional arguments to the CLI, an FPS is\n");
    printf("automatically created/updated, executed once, and disconnected:\n");
    printf("  $ " C_CMD "./milk-fpsclitest 42 cam01\n" C_RST);
    printf("For more detailed about standalone execution, run:\n");
    printf("  $ " C_CMD "./milk-fpsclitest -h\n" C_RST);
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "            Command-line interface (CLI) Execution         \n" C_RST);
    printf("\n");
    printf("  Function help    : $ " C_CMD "cmd? modex.fpsclitest\n" C_RST); 
    printf("\n");
    printf("  The general syntax is:  cmdkey:" C_FPS "fpsname" C_RST ":action\n");
    printf("  If " C_FPS "fpsname" C_RST " is not provided, it is created with default name: " C_FPS "cmdfpsname.CLIsessionname" C_RST "\n");
    printf("  Action is optional, and can be:\n");
    printf("    init : Initialize FPS (create and/or reset to default values)\n");
    printf("    initp: Initialize FPS with processinfo\n");
    printf("    ?    : Show FPS content\n");
    printf(C_WARN "Warning: " C_RST "Commands below creates FPS if it does not exist.\n");
    printf("  Create FPS (no processinfo) : milk > " C_CMD "modex.fpsclitest::init\n" C_RST);
    printf("  Create FPS (processinfo)    : milk > " C_CMD "modex.fpsclitest::initp\n" C_RST);
    printf("  Querry params               : milk > " C_CMD "modex.fpsclitest::?\n" C_RST);
    printf("  Change param value          : milk > " C_CMD "modex.fpsclitest .gain 1.5\n" C_RST);
    printf("  Create named FPS            : milk > " C_CMD "modex.fpsclitest:" C_FPS "myfps00" C_CMD ":init\n" C_RST);
    printf("  Querry named FPS            : milk > " C_CMD "modex.fpsclitest:" C_FPS "myfps00" C_CMD ":?\n" C_RST);
    printf("  Run cmd, default FPS        : milk > " C_CMD "modex.fpsclitest 3 cam01\n" C_RST);
    printf("  Run cmd, named FPS          : milk > " C_CMD "modex.fpsclitest:" C_FPS "myfps00" C_CMD " 3 cam01\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "            FPS Utilities                           \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("  " C_CMD "milk-fps-list" C_RST "   : List all active FPS\n");
    printf("  " C_CMD "milk-fps-info" C_RST "   : Display FPS information and parameters\n");
    printf("  " C_CMD "milk-fps-set" C_RST "    : Set FPS parameter value\n");
    printf("  " C_CMD "milk-fps-rm" C_RST "     : Remove FPS shared memory segment\n");
    printf("  " C_CMD "milk-fps-track" C_RST "  : Monitor FPS parameter changes in real-time\n");
    printf("  " C_CMD "milk-fpsCTRL" C_RST "   : Interactive TUI for FPS management\n");
    printf("\n");
    printf("  Execution Control:\n");
    printf("    " C_CMD "milk-fps-confstart" C_RST "/" C_CMD "stop" C_RST " : Control configuration loop\n");
    printf("    " C_CMD "milk-fps-runstart" C_RST "/" C_CMD "stop" C_RST "  : Control main execution loop\n");
    printf("\n");
    printf(C_NOTE "Run any of these programs with -h for more help." C_RST "\n");
    printf("\n");
    return 0;
}

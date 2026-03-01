#include <stdio.h>

#define C_TITLE "\033[1;36m"   /* Cyan Bold   -> Main section headers / separators */
#define C_HDR   "\033[1;34m"   /* Blue Bold   -> Subheaders inside sections */
#define C_CMD   "\033[1;32m"   /* Green Bold  -> Command names, syntax, execution */
#define C_NOTE  "\033[1;33m"   /* Yellow Bold -> Tips, Notes, 'run X for more' */
#define C_BOLD  "\033[1m"      /* White Bold  -> Emphasize specific words */
#define C_RST   "\033[0m"      /* Reset */

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
    printf("  $ " C_CMD "./milk-fpsclitest myfps00:fpsinit\n" C_RST);
    printf("  " C_NOTE "Tip:" C_RST " Use the " C_BOLD "-procinfo" C_RST " flag to add daemon monitoring features:\n");
    printf("  $ " C_CMD "./milk-fpsclitest myfps00:fpsinit -procinfo\n" C_RST);
    printf("\n");

    printf(C_HDR "Viewing FPS Data\n" C_RST);
    printf("You can view the parameters inside an FPS via the terminal:\n");
    printf("  $ " C_CMD "milk-fps-info myfps00\n" C_RST);
    printf("Or using the interactive configuration interface (TUI):\n");
    printf("  $ " C_CMD "milk-fpsCTRL\n" C_RST);
    printf("\n");

    printf(C_HDR "Modifying Parameters\n" C_RST);
    printf("Use " C_BOLD "milk-fps-set" C_RST " to change values on-the-fly from bash scripts:\n");
    printf("  $ " C_CMD "milk-fps-set myfps00 gain 1.5\n" C_RST);
    printf("  $ " C_CMD "milk-fps-set myfps00 verbose 1\n" C_RST);
    printf("\n");

    printf(C_HDR "Managing Execution\n" C_RST);
    printf("A module often has background components like a primary `run`\n");
    printf("loop and a fast configuration supervisor (`conf`).\n");
    printf("  Start process    : $ " C_CMD "milk-fps-runstart myfps00\n" C_RST);
    printf("  Stop process     : $ " C_CMD "milk-fps-runstop myfps00\n" C_RST);
    printf("\n");
    printf("  Start config loop: $ " C_CMD "milk-fps-confstart myfps00\n" C_RST);
    printf("  Stop config loop : $ " C_CMD "milk-fps-confstop myfps00\n" C_RST);
    printf("\n");

    printf(C_HDR "Standalone Execution\n" C_RST);
    printf("When providing positional arguments to the CLI, an FPS is\n");
    printf("automatically created/updated, executed once, and disconnected:\n");
    printf("  $ " C_CMD "./milk-fpsclitest 2.5 100 1 1 0.75 out.fits\n" C_RST);
    printf("\n");
    return 0;
}

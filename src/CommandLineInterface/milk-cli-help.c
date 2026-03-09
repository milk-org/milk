#include "CLIcore.h"

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    // Initialize data structure

    dcquiet = 1;
    CLI_startup();

    // Call the centralized CLI help function
    print_milk_cli_help();


    return 0;
}

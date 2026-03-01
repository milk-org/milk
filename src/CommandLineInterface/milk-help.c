#include "CLIcore.h"

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    // Initialize data structure

    data.quiet = 1;
    CLI_startup();

    // Call the centralized help function
    print_milk_main_help();


    return 0;
}

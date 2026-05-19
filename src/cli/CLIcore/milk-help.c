/**
 * @file milk-help.c
 * @brief Initialize data structure
 */

#include <string.h>
#include "CLIcore.h"

int main(
    int argc,
    char *argv[])
{
    /* One-line help — before CLI_startup() */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h1") == 0 ||
            strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("milk framework overview\n");
            return 0;
        }
    }

    // Initialize data structure
    dcquiet = 1;
    CLI_startup();

    // Call the centralized framework help function
    print_milk_framework_help();

    return 0;
}

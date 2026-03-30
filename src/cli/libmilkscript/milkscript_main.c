/**
 * @file milkscript_main.c
 * @brief Standalone interpreter executable for milk scripts
 *
 * This binary acts as the non-interactive interpreter for
 * milk-cli scripts. It links ONLY against libmilkscript.so
 * (and core dependencies) with zero dependencies on readline
 * or ncurses.
 */

#include <stdio.h>
#include <string.h>

#include "milkscript.h"

int main(int argc, char **argv)
{
    // Initialize the script engine
    if (milkscript_init(argc, argv) != 0) {
        fprintf(stderr, "milk-script: engine initialization failed\n");
        return 1;
    }

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            printf("Usage: milk-script [SCRIPT_FILE]\n");
            printf("Executes a milk script non-interactively.\n");
            return 0;
        }

        // Run from script file
        FILE *fp = fopen(argv[1], "r");
        if (!fp) {
            fprintf(stderr, "milk-script: cannot open %s\n", argv[1]);
            return 1;
        }
        milkscript_run(fp);
        fclose(fp);
    } else {
        // Run from stdin
        milkscript_run(stdin);
    }

    milkscript_cleanup();
    return 0;
}

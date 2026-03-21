/**
 * @file milk-cli-help.c
 *
 * @brief Entry point for the milk-cli-help standalone helper.
 *
 * Usage:
 *   milk-cli-help           — print topic index
 *   milk-cli-help <topic>   — print help for a single topic
 *
 * Available topics: cmdopts  syntax  commands  variables
 *                   flowcontrol  scripting  milk
 */

#include <stdio.h>
#include "CLIcore.h"

int main(int argc, char *argv[])
{
    dcquiet = 1;
    CLI_startup();

    if (argc >= 2)
    {
        const char *topic = argv[1];
        int ret = help_topic_dispatch(topic);
        if (ret != 0)
        {
            fprintf(stderr,
                    "\033[1;31mmilk-cli-help: unknown topic"
                    " \"%s\"\033[0m\n\n",
                    topic);
            print_help_topic_list();
            return 1;
        }
    }
    else
    {
        print_milk_cli_help();
    }

    return 0;
}


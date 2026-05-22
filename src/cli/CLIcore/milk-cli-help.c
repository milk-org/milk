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

#include <string.h>
#include <stdio.h>
#include "CLIcore.h"

int main(int argc, char *argv[])
{
    /* One-line help — before CLI_startup() to avoid initialization */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h1") == 0 || strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("milk-cli interactive shell help\n");
            return 0;
        }
    }

    dcquiet = 1;
    CLI_startup();


    const char *topic = NULL;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--json") == 0)
        {
            help_format_mode = 1;
        }
        else if (strcmp(argv[i], "--porcelain") == 0)
        {
            help_format_mode = 2;
        }
        else
        {
            topic = argv[i];
        }
    }

    if (topic != NULL)
    {
        int ret = help_topic_dispatch(topic);
        if (ret != 0)
        {
            if (help_format_mode == 0)
            {
                fprintf(stderr,
                        "\033[1;31mmilk-cli-help: unknown topic"
                        " \"%s\"\033[0m\n\n",
                        topic);
                print_help_topic_list();
            }
            return 1;
        }
    }
    else
    {
        print_milk_cli_help();
    }

    return 0;
}

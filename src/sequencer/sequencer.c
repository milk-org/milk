/**
 * @file    sequencer.c
 * @brief   Sequencer CLI Integration Module
 */

#define MODULE_SHORTNAME_DEFAULT "seq"
#define MODULE_DESCRIPTION       "sequencer control module"

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#include "seq_cli.h"

INIT_MODULE_LIB(sequencer)

static errno_t init_module_CLI()
{
    CLIADDCMD_sequencer__seq_cli();
    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */

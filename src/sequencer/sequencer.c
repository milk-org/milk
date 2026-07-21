// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    sequencer.c
 * @brief   Sequencer CLI Integration Module
 */

#define MODULE_SHORTNAME_DEFAULT "seq"
#define MODULE_DESCRIPTION "sequencer control module"

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#    include "seq_cli.h"

static errno_t init_module_CLI()
{
    CLIADDCMD_sequencer__seq_cli();
    return RETURN_SUCCESS;
}

MILK_MODULE(sequencer, init_module_CLI, NULL);
#endif /* MILK_NO_CLI */

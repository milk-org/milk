// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_config.h"
#include <CommandLineInterface/CLIcore.h>
#include <CommandLineInterface/CLIcore/CLIcore_datainit.h>
#include <CommandLineInterface/CLIcore/CLIcore_setSHMdir.h>
#include <CommandLineInterface/processtools.h>

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    // Initialize data
    if(getenv("MILK_QUIET")) {
        data.quiet = 1;
    } else {
        data.quiet = 0;
    }

    strncpy(data.processname, "procCTRL", STRINGMAXLEN_PROCESSNAME - 1);

    // Core initialization
    CLI_startup();
    setSHMdir();
    CLI_data_init();

    // Run the tool
    processinfo_CTRLscreen();

    return 0;
}

#include "milk_config.h"
#include <CommandLineInterface/CLIcore.h>
#include <CommandLineInterface/CLIcore/CLIcore_datainit.h>
#include <CommandLineInterface/CLIcore/CLIcore_setSHMdir.h>
#include "termview.h"

int main(int argc, char *argv[])
{
    // Initialize data
    if(getenv("MILK_QUIET")) {
        data.quiet = 1;
    } else {
        data.quiet = 0;
    }

    strncpy(data.processname, "termview", STRINGMAXLEN_PROCESSNAME - 1);

    if (argc < 2) {
        printf("Usage: %s <image_name>\n", argv[0]);
        return 0;
    }

    // Core initialization
    CLI_startup();
    setSHMdir();
    CLI_data_init();

    // Run the tool
    termview_screen(argv[1]);

    return 0;
}

#include "milk_config.h"
#include "CLIcore.h"
#include "CLIcore_datainit.h"
#include "CLIcore_setSHMdir.h"
#include "streamCTRL_TUI.h"

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

    strncpy(data.processname, "streamCTRL", STRINGMAXLEN_PROCESSNAME - 1);

    // Core initialization
    CLI_startup();
    setSHMdir();
    CLI_data_init();

    // Run the tool
    streamCTRL_CTRLscreen();

    return 0;
}

#include "milk_config.h"
#include "CLIcore.h"
#include "CLIcore_datainit.h"
#include "CLIcore_setSHMdir.h"
#include "streamCTRL_TUI.h"
#include "ImageStreamIO/ImageStreamIO.h"

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    // Silence ImageStreamIO library (suppress stderr warnings/errors in TUI)
    // ImageStreamIO_set_verbosity(0);

    // Initialize data
    if(getenv("MILK_QUIET")) {
        dcquiet = 1;
    } else {
        dcquiet = 0;
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

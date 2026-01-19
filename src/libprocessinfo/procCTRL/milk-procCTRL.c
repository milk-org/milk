#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "processinfo.h"

// Prototypes for functions defined in other procCTRL files
errno_t processinfo_CTRLscreen();

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    // Run the tool
    processinfo_CTRLscreen();

    return 0;
}
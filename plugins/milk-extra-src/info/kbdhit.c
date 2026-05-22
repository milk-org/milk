/**
 * @file kbdhit.c
 * @brief Kbdhit module
 */

/** @file kbdhit.c
 */

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

errno_t kbdhit(void)
{
    struct termios oldt, newt;
    int            ch;
    int            oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        //     ungetc(ch, stdin);
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
}

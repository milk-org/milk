/** @file kbdhit.c
 */

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "CommandLineInterface/CLIcore.h"

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

    if(ch != EOF)
    {
        //     ungetc(ch, stdin);
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
}

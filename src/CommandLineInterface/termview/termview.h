#ifndef _TERMVIEW_H
#define _TERMVIEW_H

#include <stdbool.h>
#include "CommandLineInterface/CLIcore.h"

typedef struct {
    bool force_ascii;
} termview_options_t;

errno_t termview_screen(const char *imagename, termview_options_t options);

#endif

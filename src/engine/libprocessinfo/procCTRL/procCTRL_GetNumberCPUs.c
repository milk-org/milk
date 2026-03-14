/**
 * @file procCTRL_GetNumberCPUs.c
 * @brief Procctrl getnumbercpus module
 */

#include <stdio.h>
#include <unistd.h>

#include "processinfo.h"
#include "processtools.h"
#include "procCTRL_GetNumberCPUs.h"

int GetNumberCPUs(PROCINFOPROC *pinfop)
{
    (void)pinfop;
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
}
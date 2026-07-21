// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file procCTRL_GetNumberCPUs.c
 * @brief Procctrl getnumbercpus module
 */


#include "processtools.h"

/**
 * @brief Count the number of online CPUs.
 *
 * Reads from /proc/stat or sysconf to determine
 * the CPU count for process affinity displays.
 */
int GetNumberCPUs(PROCINFOPROC *pinfop)
{
    (void) pinfop;
    return (int) sysconf(_SC_NPROCESSORS_ONLN);
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    rt_legacy.h
 * @brief   CLI registration functions of the runnabletest_legacy module
 */

#ifndef RT_LEGACY_H
#define RT_LEGACY_H

#include "CLIcore.h"

errno_t CLIADDCMD_runnabletestlegacy__scale();
errno_t CLIADDCMD_runnabletestlegacy__offset();
errno_t CLIADDCMD_runnabletestlegacy__negate();
errno_t CLIADDCMD_runnabletestlegacy__total();

#endif

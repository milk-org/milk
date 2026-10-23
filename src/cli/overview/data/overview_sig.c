// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_sig.c
 * @brief   Stop flags shared by the scan thread and the tools' signal handlers (one definition).
 */

#include "overview_defs.h"

volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;

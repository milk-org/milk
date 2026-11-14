// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_sig.c
 * @brief   Stop flags shared by the scan thread and the tools' signal handlers (one definition).
 *
 * Purpose:     one definition of the SIGINT and SIGTERM flags for the programs that link
 *              milkctrl_model; overview_defs.h declares them and OV_SIG_ANY_SET() tests them.
 * Used by:     the signal handlers of milk-CTRL (tools/milkCTRL.c) and milk-stream-graph
 *              (tools/milk-stream-graph_tui.c) set them; the scan thread (overview_scan.c), the
 *              main loops of those tools and the parameter editor (input/overview_fps_edit.c) stop
 *              when one is set.
 * See also:    overview_defs.h.
 */

#include "overview_defs.h"

volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;

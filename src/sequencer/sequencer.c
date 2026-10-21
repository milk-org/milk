// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    sequencer.c
 * @brief   Sequencer CLI Integration Module
 */

#define MODULE_SHORTNAME_DEFAULT "seq"
#define MODULE_DESCRIPTION "sequencer control module"

#include "CLIcore.h"
#include "seq_cli.h"

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_sequencer_list,  &runnable_sequencer_submit, &runnable_sequencer_status,
    &runnable_sequencer_start, &runnable_sequencer_stop,   &runnable_sequencer_log,
};

MILK_MODULE_RUNNABLES(sequencer, NULL, runnables);

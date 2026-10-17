// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    seq_cli.h
 * @brief   CLI commands for milk-seq Sequencer
 */

#ifndef _SEQ_CLI_H
#define _SEQ_CLI_H

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_sequencer_list;
extern const MILK_RUNNABLE runnable_sequencer_submit;
extern const MILK_RUNNABLE runnable_sequencer_status;
extern const MILK_RUNNABLE runnable_sequencer_start;
extern const MILK_RUNNABLE runnable_sequencer_stop;
extern const MILK_RUNNABLE runnable_sequencer_log;

#endif

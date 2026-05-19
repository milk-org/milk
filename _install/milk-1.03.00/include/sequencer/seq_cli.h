/**
 * @file    seq_cli.h
 * @brief   CLI commands for milk-seq Sequencer
 */

#ifndef _SEQ_CLI_H
#define _SEQ_CLI_H

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

#ifndef MILK_NO_CLI
errno_t CLIADDCMD_sequencer__seq_cli(void);
#endif

#endif

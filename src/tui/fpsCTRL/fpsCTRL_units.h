// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fpsCTRL_units.h
 * @brief   fpsCTRL's unit actions: milk-fps-* tools run without blocking the screen, unit states, the
 *          output view, the clipboard
 */

#ifndef FPSCTRL_UNITS_H
#define FPSCTRL_UNITS_H

#include "fps.h"

/* Display mode of the output view (the others are in fpsCTRL_TUI.c) */
#define DISPLAYMODE_OUTPUT 4

/** fpsctrl_unit_act - run `milk-fps-<op> @name` in the background (fps_unit_job_start()). Return: 0 or -1. */
int fpsctrl_unit_act(const char *name, const char *op);

/** fpsctrl_unit_restart - runstop @name, then runstart it once stopped, in the background. Return: 0 or -1. */
int fpsctrl_unit_restart(const char *name);

/** fpsctrl_units_poll - collect what the tools printed, reap those that ended. Return: 1 if one ended. */
int fpsctrl_units_poll(void);

/** fpsctrl_unit_say - set the message line. */
void fpsctrl_unit_say(int is_error, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

/** fpsctrl_units_message - the message line while it is recent, else NULL; *@is_error for its colour. */
const char *fpsctrl_units_message(int *is_error);

/** fpsctrl_unit_status - state of @role of @fps (U10), refreshed at most twice a second. */
void fpsctrl_unit_status(FPS *fps, FPS_UNIT_ROLE role, FPS_UNIT_STATUS *st);

/** fpsctrl_output_open - follow unit @name in the output view. */
void fpsctrl_output_open(const char *name);

/** fpsctrl_output_name - the unit the output view follows. */
const char *fpsctrl_output_name(void);

/** fpsctrl_output_view - draw the output view: the unit's run state and the end of its run.out. */
void fpsctrl_output_view(void);

/** fpsctrl_copy - put @text in the terminal's clipboard (OSC 52). */
void fpsctrl_copy(const char *text);

#endif

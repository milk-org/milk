// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit.h
 * @brief Command lines of the units without C (milk.calc, milk.script), read as fpsexec's main_impl() does.
 */

#ifndef MILK_UNIT_H
#define MILK_UNIT_H

#include "fps_cli_binding.h"
#include "fps_types.h"

typedef struct
{
    const char *cmd; /* fpsinit, exec, runstart, ... ("run" when there is none) */
    char        name[STRINGMAXLEN_FPS_NAME];
    int         help;
    int         use_loop; /* 1: -loops, 2: -loopd */
    double      loop_delay;
    int         npos;
    char **pos;  /* the values after the command word, as fps_process_cli_and_sync() maps them */
    int   *posi; /* their indices in argv */
} milk_unit_args;

/** milk_unit_args_read() - read @argv; the FPS is named @dflt_name unless -n or NAME:cmd say otherwise. */
void milk_unit_args_read(int argc, char **argv, const char *dflt_name, milk_unit_args *a);

/** milk_unit_pos() - positional value @i, or NULL when missing or `.` (keep the default). */
const char *milk_unit_pos(const milk_unit_args *a, int i);

/** milk_unit_fps_exists() - 1 if FPS @name exists. */
int milk_unit_fps_exists(const char *name);

/** milk_unit_create() - (re)create FPS a->name from bindings @b (their values the defaults), with processinfo
 * entries, then apply -loops or -loopd; 0 or 1. */
int milk_unit_create(const milk_unit_args *a, FPS_APP_INFO *app, FPS_CLI_BINDING *b, int nb);

#endif

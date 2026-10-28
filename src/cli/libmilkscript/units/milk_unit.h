// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit.h
 * @brief Command lines of the units without C (milk.calc, milk.script).
 *
 * The program of a unit (statements, trigger, declared parameters, files) is given with unit
 * options (-e, -t, -p, ...), which are removed from argv before fpsexec's main reads it. The
 * values after the command word then set the declared parameters, as for any fpsexec.
 */

#ifndef MILK_UNIT_H
#define MILK_UNIT_H

#include "fps_cli_binding.h"
#include "fps_types.h"
#include "milk_skinny_runnable.h"

#define MILK_UNIT_MAXOPT 8

/* A unit option, "-x VALUE" or "--long VALUE": it writes fixed binding number @slot */
typedef struct
{
    const char *shortopt;
    const char *longopt;
    int         slot;    /* index of the fixed binding it sets */
    int         is_path; /* 1: a file name, made absolute */
    const char *toolong; /* hint when the value exceeds an FPS string (NULL: none) */
} milk_unit_opt;

typedef struct
{
    const char *cmd; /* fpsinit, exec, runstart, ... ("run" when there is none) */
    char        name[STRINGMAXLEN_FPS_NAME];
    int         help;
    int         use_loop; /* 1: -loops, 2: -loopd */
    double      loop_delay;
    const char *opt[MILK_UNIT_MAXOPT]; /* value of each unit option, NULL when not given */
} milk_unit_args;

/* What milk_unit_main() needs to know about a unit */
typedef struct
{
    const char          *tag; /* "milk.calc", for messages */
    MILK_RUNNABLE       *unit;
    FPS_CLI_BINDING     *bindings; /* fixed .unit.* bindings, then the declared ones */
    int                  nfixed;   /* leading fixed bindings (.unit.*), hidden from Parameters */
    const int           *nbind;    /* bindings in use */
    const milk_unit_opt *opts;
    int                  nopt;
    int                  params_opt;  /* index in @opts of the option that declares parameters */
    int (*declare)(const char *text); /* bind the parameters of a declaration */
    int (*find_params)(const char *fpsname); /* bind the declared parameters of an FPS */
    const char *help_options;                /* "Unit Options" of -h */
} milk_unit_desc;

/** milk_unit_main() - run a unit's command line: unit options, FPS creation, then fpsexec main. */
int milk_unit_main(int argc, char **argv, const milk_unit_desc *d);

/** milk_unit_create() - (re)create FPS a->name from bindings @b (their values the defaults), with
 * processinfo entries, then apply -loops or -loopd; 0 or 1. */
int milk_unit_create(const milk_unit_args *a, const FPS_APP_INFO *app, FPS_CLI_BINDING *b, int nb);

#endif

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_argval.h
 * @brief Typed argument values: text to a value of an FPTYPE parameter and back.
 */

#ifndef MILK_ARGVAL_H
#define MILK_ARGVAL_H

#include <stddef.h>
#include <stdint.h>

#include "libmilkdata/milkdata_clicmd.h"

/** milk_argval_parse() - convert @s to a value of parameter type @type (strict); 0, or -1 if invalid. */
int milk_argval_parse(uint64_t type, const char *s, CLICMDARGDATA *ad);

/** milk_argval_set_number() - store number @v in a numeric parameter; -1 if out of range or not integral. */
int milk_argval_set_number(uint64_t type, double v, CLICMDARGDATA *ad);

/** milk_argval_is_numeric() - 1 if parameter type @type holds a number. */
int milk_argval_is_numeric(uint64_t type);

/** milk_argval_format() - @ad's value as text that milk_argval_parse() reads back. */
void milk_argval_format(const CLICMDARGDATA *ad, char *buf, size_t sz);

#endif

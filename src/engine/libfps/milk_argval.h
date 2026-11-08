// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_argval.h
 * @brief Argument values and words: text to a value of an FPTYPE parameter and back, and the rule that
 *        binds words to parameters (module commands, unit command lines, `@args`, `@set`).
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

/* A declared parameter, "name:type[=default][:help]": @args, and .params of milk.calc / milk.script */
typedef struct
{
    char        name[64];
    const char *tname; /* int uint float bool str stream fits */
    uint64_t    type;  /* FPTYPE_INT64 UINT64 FLOAT64 ONOFF STRING STREAMNAME FITSFILENAME */
    const char *def;   /* NULL: required */
    const char *help;
    const char *val; /* for the caller: the value given */
} milk_argspec_t;

/** milk_argspec_parse() - parse "name:type[=default][:help]" in place (the fields point into @s); 0, or -1. */
int milk_argspec_parse(char *s, milk_argspec_t *a);

/** milk_args_named() - 1 if argument word @s is `.name=value` (name: [A-Za-z_][A-Za-z0-9_.]*). */
int milk_args_named(const char *s);

/* A parameter that argument words bind to */
typedef struct
{
    const char *name;       /* without the leading dot */
    int         positional; /* positional words fill it when it is not named */
} milk_argslot_t;

/** milk_args_bind() - bind words @w to parameters @slots: `.name=value` before @posonly (or before a "--"
 * word when @posonly < 0) by name, the others into the positional parameters not named, in order.
 * @val[i]: the value text of parameter i or NULL ("." keeps the current value); @word[i]: its word index
 * or -1 (@word may be NULL). 0, or -1 with @err (unknown name, given twice, `.name VALUE`, too many). */
int milk_args_bind(int                   nw,
                   char *const          *w,
                   int                   posonly,
                   int                   ns,
                   const milk_argslot_t *slots,
                   const char          **val,
                   int                  *word,
                   char                 *err,
                   size_t                errsz);

#endif

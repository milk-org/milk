// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_engine_internal.h
 * @brief Internal declarations shared between milk_bash_translate.c and milk_bash_coproc.c.
 *
 * Role & Goal:
 *   Shares syntax-checking and translator helpers between the Tier-2 to Tier-1
 *   translator (`milk_bash_translate.c`) and the supervised bash co-process
 *   manager (`milk_bash_coproc.c`).
 */

#ifndef MILK_BASH_ENGINE_INTERNAL_H
#define MILK_BASH_ENGINE_INTERNAL_H

#include "milk_bash_engine.h"

/**
 * @brief Check whether @s contains unclosed quotes or `$()` command substitutions.
 *
 * @param s Input command or script string
 * @return 1 if unclosed syntax is detected, 0 otherwise
 */
int milk_bash_has_unclosed_syntax(const char *s);

#endif /* MILK_BASH_ENGINE_INTERNAL_H */

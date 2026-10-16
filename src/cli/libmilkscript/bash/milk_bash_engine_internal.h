// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_engine_internal.h
 * @brief Internal declarations shared between milk_bash_translate.c and milk_bash_coproc.c.
 *
 * Role & Goal:
 *   Shares syntax-checking and translator helpers between the shorthand-to-bash
 *   translator (`milk_bash_translate.c`) and the supervised bash co-process
 *   manager (`milk_bash_coproc.c`).
 */

#ifndef MILK_BASH_ENGINE_INTERNAL_H
#define MILK_BASH_ENGINE_INTERNAL_H

#include "milk_bash_engine.h"

int milk_bash_funsub_disabled(void);

#endif /* MILK_BASH_ENGINE_INTERNAL_H */

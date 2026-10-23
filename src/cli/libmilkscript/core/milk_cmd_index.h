// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_cmd_index.h
 * @brief Command and module name rules, the hash index of registered commands, the registry generation.
 */

#ifndef MILK_CMD_INDEX_H
#define MILK_CMD_INDEX_H

#include <stdint.h>

/** milk_shortname_check() - 0 if @name may be a module short name ([a-z][a-z0-9]{1,15}, not reserved). */
int milk_shortname_check(const char *name, int quiet);

/** milk_reserve_milk_shortname() - allow the reserved short name `milk` (the built-in module) while @on. */
void milk_reserve_milk_shortname(int on);

/** milk_cmd_key_check() - 0 if the full key @key may be registered for module @midx (-1: no module). */
int milk_cmd_key_check(const char *key, int midx);

/** milk_cmd_lookup() - index in data.cmd[] of the command registered exactly as @key, or -1. */
int milk_cmd_lookup(const char *key);

/** milk_cmd_index_add() - index data.cmd[@cmdi] (and bump the registry generation if uncatalogued). */
void milk_cmd_index_add(unsigned int cmdi);

/** milk_registry_gen() - counts the registered module commands the catalog lacks (shells then resync). */
uint32_t milk_registry_gen(void);

#endif /* MILK_CMD_INDEX_H */

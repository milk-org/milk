// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_cmd_index.h
 * @brief Command and module name rules, the hash index of registered commands, the registry
 *        generation.
 *
 * Purpose:     one place decides which module short names and command keys are valid, and a
 *              command is found by its key without scanning data.cmd[].
 * Does:        checks module short names ([a-z][a-z0-9]{1,15}, not a keyword, an entity
 *              namespace or `milk`) and `module.command` keys; indexes data.cmd[] by key;
 *              counts the registered commands the catalog lacks, so that connected shells know
 *              to fetch the new command functions.
 * Used by:     CLIcore_modules.c (loading a module, registering a command),
 *              milk_builtin_module.c, milk_catalog_build.c, CLIcore_help_command.c, and the
 *              server (milk_server.c, milk_server_admin.c, milk_server_complete.c,
 *              milk_server_dispatch.c).
 * Uses:        milk_keywords.h and milk_entity_ns.h (reserved names), milk_catalog.h.
 * See also:    milk_cmd_index.c; docs/guide/CLIcore.md, "Module Loading".
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

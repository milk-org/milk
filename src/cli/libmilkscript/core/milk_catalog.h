// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_catalog.h
 * @brief The install-time catalog of module commands (`milk-cli --catalog`) and its reader.
 *
 * Purpose:     a shell knows every module command without loading any module, and a session
 *              loads a module only when one of its commands is first called.
 * Does:        built once, as the last install step: share/milk/catalog.d/<module>.tsv
 *              (`key<TAB>loadname<TAB>summary`), share/milk/catalog.bash (one function per
 *              command, sourced by milk.bash in every front end) and libexec/milk/bin, a
 *              read-only farm of milk-call links for external tools. At run time, it tells
 *              which module provides a command and loads that module on first use.
 * Used by:     CLIcore.c (`milk-cli --catalog`), CLIcore_modules.c, milk_cmd_index.c, and the
 *              server (milk_server.c, milk_server_admin.c, milk_server_dispatch.c).
 * Uses:        CLIcore_modules.h (module loading), milk_cmd_index.h (the command registry),
 *              milk_keywords.h (the keywords milk-call implements get links too).
 * See also:    milk_catalog.c (reader), milk_catalog_build.c (writer, autoload);
 *              docs/guide/scripting.md, "Module Commands".
 */

#ifndef MILK_CATALOG_H
#define MILK_CATALOG_H

#include <stddef.h>

/** milk_catalog_build() - load every module under @prefix and write its catalog; 0 or -1. */
int milk_catalog_build(const char *prefix);

/** milk_catalog_has() - 1 if @key is a catalogued module command. */
int milk_catalog_has(const char *key);

/** milk_catalog_module() - load name of the module providing command @key, or NULL. */
const char *milk_catalog_module(const char *key);

/** milk_catalog_autoload() - load the module of command @key; 1 if @key now exists. */
int milk_catalog_autoload(const char *key);

/** milk_catalog_count() - number of catalogued commands; milk_catalog_load_at() - @i-th module. */
size_t      milk_catalog_count(void);
const char *milk_catalog_load_at(size_t i);

/** milk_module_loaded() - 1 if the module with load name @loadname is loaded. */
int milk_module_loaded(const char *loadname);

#endif /* MILK_CATALOG_H */

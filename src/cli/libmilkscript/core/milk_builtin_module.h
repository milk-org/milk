// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_builtin_module.h
 * @brief The built-in module `milk`, the root commands and the default module set:
 *        the same in milk-cli, milk-script and a milk-cli server.
 *
 * Purpose:     every front end starts with the same commands and modules, so start-up is
 *              written once rather than in each program's main().
 * Does:        registers the commands of module `milk` (milk.ci, milk.fpsdump) and the root
 *              command `help`; lets other files register more milk.<name> commands between
 *              milk_module_begin() and milk_module_end(); loads the core modules, then the
 *              modules named in $MILKCLI_ADD_LIBS, if any (the others load on first use).
 * Used by:     milkscript_init() (milkscript_api.c), which milk-cli, milk-script and a
 *              milk-cli server call at start-up; CLIcore_cmd_registry.c, which adds milk-cli's
 *              own milk.<name> commands.
 * Uses:        milk_runnable.h (command definitions), CLIcore_modules.h (module loading),
 *              milk_cmd_index.h (the reserved short name `milk`).
 * See also:    milk_builtin_module.c; docs/guide/CLIcore.md, "Module Loading".
 */

#ifndef MILK_BUILTIN_MODULE_H
#define MILK_BUILTIN_MODULE_H

/** milk_module_begin() - register the following commands as milk.<name> (pair with milk_module_end()). */
void milk_module_begin(void);

/** milk_module_end() - back to registering root commands. */
void milk_module_end(void);

/** milk_register_builtin_commands() - register the built-in milk.* commands and the help commands. */
void milk_register_builtin_commands(void);

/** milk_session_load_modules() - load the core modules, then those $MILKCLI_ADD_LIBS names. */
void milk_session_load_modules(void);

#endif /* MILK_BUILTIN_MODULE_H */

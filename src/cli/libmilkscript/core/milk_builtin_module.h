// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_builtin_module.h
 * @brief The built-in module `milk`, the root commands and the default module set:
 *        the same in milk-cli, milk-script and a milk-cli server.
 */

#ifndef MILK_BUILTIN_MODULE_H
#define MILK_BUILTIN_MODULE_H

/** milk_module_begin() - register the following commands as milk.<name> (pair with milk_module_end()). */
void milk_module_begin(void);

/** milk_module_end() - back to registering root commands. */
void milk_module_end(void);

/** milk_register_builtin_commands() - register the built-in milk.* commands and the help commands. */
void milk_register_builtin_commands(void);

/** milk_session_load_modules() - load the core modules, then $MILKCLI_ADD_LIBS (default list; "none"). */
void milk_session_load_modules(void);

#endif /* MILK_BUILTIN_MODULE_H */

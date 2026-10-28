// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_datainit.h
 *
 * @brief data structure init
 *
 */

#ifndef CLICORE_DATAINIT_H

#define CLICORE_DATAINIT_H

void CLI_data_init();

errno_t CLI_runtime_init(void);

/** CLI_export_own_installdir() - set MILK_INSTALLDIR, when unset, to the install this executable belongs to
 * (<dir>/bin/<exe> with <dir>/bin/milk.bash present): modules, the engine's milk.bash and the units it
 * launches then come from one install, whatever PATH holds. */
void CLI_export_own_installdir(void);

#endif

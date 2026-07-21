// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file breakcube.h
 * @brief Break cube into individual 2D images
 */

errno_t CLIADDCMD_COREMOD_iofits__breakcube();

imageID break_cube(const char *restrict ID_name);

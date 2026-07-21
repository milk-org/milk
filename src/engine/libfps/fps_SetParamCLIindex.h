// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_SetParamCLIindex.h
 * @brief   set parameter CLI index
 */

#ifndef _FPS_SETPARAMCLIINDEX_H
#define _FPS_SETPARAMCLIINDEX_H

/**
 * @brief Set the CLI index for a parameter in an FPS.
 *
 * @param fps Pointer to the Function Parameter Structure.
 * @param pindex Index of the parameter.
 * @param cli_index The CLI argument index (1-indexed usually).
 * @return errno_t RETURN_SUCCESS on success, error code otherwise.
 */
int functionparameter_SetParamCLIindex(FPS *fps, long pindex, int cli_index);

#endif

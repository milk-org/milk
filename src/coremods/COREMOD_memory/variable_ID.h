// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file variable_ID.h
 * @brief Variable id module
 */

/**
 * @file    image_ID.h
 */

variableID variable_ID(const char *name);

variableID next_avail_variable_ID();

/* Record that slot @ID is in use: lookups only scan up to the highest slot recorded. Any code that
 * sets a variable's `used` flag must call this. */
void variable_ID_note_used(variableID ID);

long compute_variable_memory();

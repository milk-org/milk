// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file create_variable.h
 * @brief Create variable module
 */

/**
 * @file    create_variable.h
 */

variableID create_variable_ID(const char *name, double value);

variableID create_variable_long_ID(const char *name, long value);

variableID create_variable_string_ID(const char *name, const char *value);

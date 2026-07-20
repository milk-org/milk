// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    read_keyword.h
 */

int read_keyword(const char *restrict file_name,
                 const char *restrict KEYWORD,
                 char *restrict content);

errno_t read_keyword_alone(const char *restrict file_name,
                           const char *restrict KEYWORD);

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_renames.h
 * @brief Table of removed command spellings and their replacements ("removed, use ..." hints).
 */

#ifndef MILK_RENAMES_H
#define MILK_RENAMES_H

/* How a replacement hint describes the new spelling */
typedef enum
{
    MILK_REN_WORD, /* the replacement takes the place of the word */
    MILK_REN_ARGS, /* the arguments are rearranged too */
    MILK_REN_HINT  /* the replacement is a prose hint for the user */
} milk_ren_how_t;

/** milk_rename_cmd() - replacement of removed command word @word (*@how: how to apply), or NULL. */
const char *milk_rename_cmd(const char *word, milk_ren_how_t *how);

#endif /* MILK_RENAMES_H */

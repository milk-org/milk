// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_renames.h
 * @brief The one table of removed spellings and their replacements (`milk-script --modernize`,
 *        the "removed, use ..." hints).
 */

#ifndef MILK_RENAMES_H
#define MILK_RENAMES_H

#include <stddef.h>

/* How `milk-script --modernize` applies a replacement */
typedef enum
{
    MILK_REN_WORD, /* the replacement takes the place of the word */
    MILK_REN_ARGS, /* the arguments are rearranged too (milk_bash_translate.c) */
    MILK_REN_HINT  /* no mechanical rewrite: the replacement is a hint for the user */
} milk_ren_how_t;

/** milk_rename_cmd() - replacement of the removed command word @word (*@how: how to apply it), or NULL. */
const char *milk_rename_cmd(const char *word, milk_ren_how_t *how);

/** milk_rename_ns() - replacement of the removed entity namespace that starts @path ("seq." -> "q."),
 * or NULL; *@oldlen gets the length of the removed prefix. */
const char *milk_rename_ns(const char *path, size_t *oldlen);

#endif /* MILK_RENAMES_H */

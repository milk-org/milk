// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_renames.h
 * @brief Table of removed command spellings and their replacements ("removed, use ..." hints).
 *
 * Purpose:     an old script that uses a removed spelling fails with a message that names the
 *              replacement, rather than "command not found"; nothing rewrites the script.
 * Does:        maps a removed command word to its replacement, and says whether the
 *              replacement takes the word's place, rearranges the arguments too, or is a hint
 *              in prose.
 * Used by:     the client in every shell (milk_client_stream.c) and the server's
 *              milk_server_dispatch.c.
 * Uses:        libc only.
 * See also:    milk_renames.c; docs/guide/scripting.md, "Migrating Old Scripts";
 *              docs/arch/unit_lifecycle.md, "Removed, with a hint".
 * Constraints: compiled into libmilkbash.so, the thin client every shell loads.
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

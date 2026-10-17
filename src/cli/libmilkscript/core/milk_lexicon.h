// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_lexicon.h
 * @brief The one list of bash words and `@` primitives used by highlighting, completion and milk-call.
 *
 * Registered milk commands come from the command registry (data.cmd); entity namespaces from
 * milk_entity_ns.h. This file holds the rest.
 */

#ifndef MILK_LEXICON_H
#define MILK_LEXICON_H

#define MILK_PRIM_CALL 0x1 /* implemented by milk-call: gets a `@name` symlink in the bin farm */

typedef struct
{
    const char *name;  /* without the sigil */
    unsigned    flags; /* MILK_PRIM_* */
} milk_primitive_t;

/** `@` primitives (calc, get, waitfor, ...), NULL-terminated. Functions, not exported arrays: an
 * executable would otherwise copy-relocate the arrays at their link-time size. */
const milk_primitive_t *milk_lexicon_primitives(void);

/** Prompt-only commands (history, synhl, ...): not exported to bash, reached as `@name`; NULL-terminated. */
const char *const *milk_lexicon_repl_commands(void);

/** Bash reserved words and builtins that the REPL highlights and completes, NULL-terminated. */
const char *const *milk_lexicon_bash_words(void);

#endif /* MILK_LEXICON_H */

/**
 * @file    fps_streamname_parse.h
 * @brief   Parse @X: modifier prefixes from stream names
 *
 * Stream name strings may carry an optional prefix of
 * the form "@XY:name" where X,Y are modifier letters:
 *
 *   L  Local memory only (no SHM)
 *   S  Force shared memory
 *   F  Load from ./conf/shmim.<name>.fits
 *   E  Must exist (error if not found)
 *   N  Must not exist (error if found)
 *
 * Letters are stackable: "@LE:myimg" means local +
 * must-exist.  E and N are mutually exclusive.
 */

#ifndef FPS_STREAMNAME_PARSE_H
#define FPS_STREAMNAME_PARSE_H

/**
 * @brief Parsed stream name with modifier flags.
 */
typedef struct
{
    /** Location modifier: 'L','S','F', or '\0' */
    char        loc;
    /** 1 if 'E' modifier present */
    int         must_exist;
    /** 1 if 'N' modifier present */
    int         must_new;
    /** Pointer to bare name (past prefix) */
    const char *name;
    /** 1 if parse error (e.g. E+N conflict) */
    int         error;
} FPS_STREAMNAME_PARSED;

/**
 * @brief Parse a raw stream name string.
 *
 * If raw starts with '@' and contains ':', the
 * characters between '@' and ':' are scanned for
 * modifier letters.  Unknown letters cause the
 * entire string to be treated as a bare name with
 * no modifiers.
 *
 * @param raw  Raw stream name (possibly prefixed)
 * @return     Parsed result
 */
FPS_STREAMNAME_PARSED fps_streamname_parse(
    const char *raw
);

/**
 * @brief Human-readable label for a modifier set.
 *
 * Writes a short tag like "[L]", "[SE]", or ""
 * into the provided buffer.
 *
 * @param p      Parsed stream name
 * @param buf    Output buffer (>= 8 bytes)
 * @param bufsz  Size of buf
 */
void fps_streamname_modifier_label(
    const FPS_STREAMNAME_PARSED *p,
    char *buf,
    int   bufsz
);

/**
 * @brief Determine shared flag from stream name.
 *
 * Returns 0 if '@L:' prefix (local only).
 * Returns 1 otherwise (default = shared).
 *
 * @param raw  Raw stream name (possibly prefixed)
 * @return     1 for shared, 0 for local
 */
int fps_streamname_is_shared(
    const char *raw
);

#endif

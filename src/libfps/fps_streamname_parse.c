/**
 * @file    fps_streamname_parse.c
 * @brief   Parse @X: modifier prefixes from stream names
 *
 * Scans a raw stream name string for an optional
 * "@XY:name" prefix.  Modifier letters (L, S, E, N)
 * are extracted and returned in a parsed struct.
 *
 * The parser is intentionally stateless and does not
 * allocate memory — it returns a pointer into the
 * original string for the bare name.
 */

#include <string.h>
#include <stdio.h>

#include "fps_streamname_parse.h"

/**
 * @brief Parse a raw stream name for @X: modifiers.
 *
 * Looks for the pattern "@<letters>:<name>".
 * Valid letters: L, S (location — at most one),
 * E, N (existence — mutually exclusive).
 * Unknown letters → treat entire string as bare name.
 *
 * @param raw  Raw stream name string
 * @return     Parsed result with modifier flags
 */
FPS_STREAMNAME_PARSED fps_streamname_parse(
    const char *raw
)
{
    FPS_STREAMNAME_PARSED p;
    memset(&p, 0, sizeof(p));
    p.name = raw;

    if (raw == NULL || raw[0] == '\0')
    {
        return p;
    }

    /* Must start with '@' */
    if (raw[0] != '@')
    {
        return p;
    }

    /* Find the ':' separator */
    const char *colon = strchr(raw, ':');
    if (colon == NULL)
    {
        /* No colon → not a valid prefix */
        return p;
    }

    /* Empty modifier block "@:" → no modifiers */
    int modlen = (int)(colon - raw - 1);
    if (modlen <= 0)
    {
        return p;
    }

    /* Scan modifier letters */
    char loc = '\0';
    int  must_exist = 0;
    int  must_new = 0;
    int  nloc = 0;

    for (int i = 1; i <= modlen; i++)
    {
        switch (raw[i])
        {
        case 'L':
        case 'S':
            loc = raw[i];
            nloc++;
            break;

        case 'E':
            must_exist = 1;
            break;

        case 'N':
            must_new = 1;
            break;

        default:
            /* Unknown letter → error */
            memset(&p, 0, sizeof(p));
            p.name  = raw;
            p.error = 1;
            return p;
        }
    }

    /* Multiple location modifiers is an error */
    if (nloc > 1)
    {
        memset(&p, 0, sizeof(p));
        p.name  = raw;
        p.error = 1;
        return p;
    }

    /* E and N are mutually exclusive */
    if (must_exist && must_new)
    {
        memset(&p, 0, sizeof(p));
        p.name  = raw;
        p.error = 1;
        return p;
    }

    p.loc        = loc;
    p.must_exist = must_exist;
    p.must_new   = must_new;
    p.name       = colon + 1;
    p.error      = 0;

    return p;
}

/**
 * @brief Build a short modifier label string.
 *
 * Produces e.g. "[L]", "[SE]", "[N]", or "" if no
 * modifiers are active.
 *
 * @param p      Parsed stream name
 * @param buf    Output buffer (>= 8 bytes)
 * @param bufsz  Buffer size
 */
void fps_streamname_modifier_label(
    const FPS_STREAMNAME_PARSED *p,
    char *buf,
    int   bufsz
)
{
    if (bufsz < 2)
    {
        return;
    }

    char inner[6];
    int  pos = 0;

    if (p->loc != '\0')
    {
        inner[pos++] = p->loc;
    }
    if (p->must_exist)
    {
        inner[pos++] = 'E';
    }
    if (p->must_new)
    {
        inner[pos++] = 'N';
    }
    inner[pos] = '\0';

    if (pos == 0)
    {
        buf[0] = '\0';
    }
    else
    {
        snprintf(buf, bufsz, "[%s]", inner);
    }
}

/**
 * @brief Return shared flag from stream name prefix.
 *
 * Default (no prefix or @S:) returns 1 (shared).
 * @L: returns 0 (local only).
 *
 * @param raw  Raw stream name (possibly prefixed)
 * @return     1 for shared, 0 for local
 */
int fps_streamname_is_shared(
    const char *raw
)
{
    FPS_STREAMNAME_PARSED p =
        fps_streamname_parse(raw);

    if (p.loc == 'L')
    {
        return 0;
    }
    return 1;
}

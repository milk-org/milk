// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_streamname_parse.c
 * @brief   Parse @X: modifier prefixes from stream names
 *
 * Scans a raw stream name string for an optional
 * "@XY:name" prefix.  Modifier letters (L, S, E, N)
 * are extracted and returned in a parsed struct.
 *
 * The parser is intentionally stateless and does not
 * allocate memory -- it returns a pointer into the
 * original string for the bare name.
 */

#include <ctype.h>
#include <string.h>
#include <stdio.h>

#include "ImageStreamIO/ImageStruct.h"
#include "fps_streamname_parse.h"

/* The @T<type> tokens: the canonical ones (fps_streamname_modifier(), labels) first, then the
 * longer spellings the parser also reads */
static const struct
{
    const char *tok;
    uint8_t     dt;
} type_tokens[] = {
    { "U8", _DATATYPE_UINT8 },
    { "I8", _DATATYPE_INT8 },
    { "U16", _DATATYPE_UINT16 },
    { "I16", _DATATYPE_INT16 },
    { "U32", _DATATYPE_UINT32 },
    { "I32", _DATATYPE_INT32 },
    { "U64", _DATATYPE_UINT64 },
    { "I64", _DATATYPE_INT64 },
    { "F16", _DATATYPE_HALF },
    { "F32", _DATATYPE_FLOAT },
    { "F64", _DATATYPE_DOUBLE },
    { "C64", _DATATYPE_COMPLEX_FLOAT },
    { "C128", _DATATYPE_COMPLEX_DOUBLE },
    { "UI8", _DATATYPE_UINT8 },
    { "SI8", _DATATYPE_INT8 },
    { "UI16", _DATATYPE_UINT16 },
    { "SI16", _DATATYPE_INT16 },
    { "UI32", _DATATYPE_UINT32 },
    { "SI32", _DATATYPE_INT32 },
    { "UI64", _DATATYPE_UINT64 },
    { "SI64", _DATATYPE_INT64 },
};
#define N_TYPE_TOKENS (sizeof(type_tokens) / sizeof(type_tokens[0]))
#define N_CANONICAL_TYPES 13

/**
 * @brief Write modifier prefix @p i into @p buf: "@L:" "@S:" "@F:" "@E:" "@N:", then "@T<type>:" for
 *        each canonical type ("@TU8:" ... "@TC128:"). Letters stack ("@LTF64:").
 *
 * @return 1 if modifier @p i exists, else 0
 */
int fps_streamname_modifier(int i, char *buf, int bufsz)
{
    static const char letters[] = "LSFEN";
    const int         nl        = (int) sizeof(letters) - 1;
    if (i < 0 || i >= nl + N_CANONICAL_TYPES)
    {
        return 0;
    }
    if (i < nl)
    {
        snprintf(buf, (size_t) bufsz, "@%c:", letters[i]);
    }
    else
    {
        snprintf(buf, (size_t) bufsz, "@T%s:", type_tokens[i - nl].tok);
    }
    return 1;
}

/**
 * @brief Return the canonical @T token (without the T) for an ImageStreamIO datatype, or NULL.
 */
static const char *datatype_mod_token(uint8_t dt)
{
    for (int i = 0; i < N_CANONICAL_TYPES; i++)
    {
        if (type_tokens[i].dt == dt)
        {
            return type_tokens[i].tok;
        }
    }
    return NULL;
}

static const char *match_ns_stream_prefix(const char *s, char *out_loc)
{
    size_t plen = 0;
    char   loc  = '\0';
    if (strncmp(s, "s.", 2) == 0)
    {
        plen = 2;
        loc  = 'S';
    }
    else if (strncmp(s, "stream.", 7) == 0)
    {
        plen = 7;
        loc  = 'S';
    }
    else if (strncmp(s, "l.", 2) == 0)
    {
        plen = 2;
        loc  = 'L';
    }
    else if (strncmp(s, "local.", 6) == 0)
    {
        plen = 6;
        loc  = 'L';
    }
    else
    {
        return NULL;
    }

    const char *rest = s + plen;
    if (rest[0] == '\0' || (!isalnum((unsigned char) rest[0]) && rest[0] != '_'))
    {
        return NULL;
    }
    for (const char *q = rest; *q != '\0' && *q != '['; q++)
    {
        if (*q == '.' || *q == ':' || *q == '/')
        {
            return NULL;
        }
    }
    *out_loc = loc;
    return rest;
}

/**
 * @brief Parse a raw stream name for @X: modifiers.
 *
 * Looks for the pattern "@<modifiers>:<name>".
 * Valid modifiers:
 *   L, S, F (location -- at most one)
 *   E, N (existence -- mutually exclusive)
 *   TU8, TU16, TU32, TU64, TI8, TI16, TI32, TI64,
 *   TF16, TF32, TF64, TC64, TC128 (pixel type -- at most one)
 *
 * Unknown letters or malformed @ prefix -> sets p.error = 1.
 *
 * @param raw  Raw stream name string
 * @return     Parsed result with modifier flags
 */
FPS_STREAMNAME_PARSED fps_streamname_parse(const char *raw)
{
    FPS_STREAMNAME_PARSED p;
    memset(&p, 0, sizeof(p));
    p.name = raw;

    if (raw == NULL || raw[0] == '\0')
    {
        return p;
    }

    const char *s_nolead = (raw[0] == '@') ? (raw + 1) : raw;
    if (strchr(s_nolead, ':') == NULL)
    {
        char        ns_loc  = '\0';
        const char *ns_name = match_ns_stream_prefix(s_nolead, &ns_loc);
        if (ns_name != NULL)
        {
            p.loc  = ns_loc;
            p.name = ns_name;
            return p;
        }
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
        /* Starts with '@' but missing ':' -> invalid modifier prefix */
        p.error = 1;
        return p;
    }

    /* Empty modifier block "@:name" or empty bare name "@S:" -> invalid */
    int modlen = (int) (colon - raw - 1);
    if (modlen <= 0 || colon[1] == '\0')
    {
        p.error = 1;
        return p;
    }

    /* Scan modifier letters */
    char    loc        = '\0';
    int     must_exist = 0;
    int     must_new   = 0;
    uint8_t datatype   = 0;
    int     nloc       = 0;
    int     ntype      = 0;

    for (int ii = 1; ii <= modlen; ii++)
    {
        switch (raw[ii])
        {
        case 'L':
        case 'S':
        case 'F':
            loc = raw[ii];
            nloc++;
            break;

        case 'E':
            must_exist = 1;
            break;

        case 'N':
            must_new = 1;
            break;

        case 'T':
        {
            int         rem     = modlen - ii;
            const char *tstr    = &raw[ii + 1];
            int         matched = 0;

            for (size_t ti = 0; ti < N_TYPE_TOKENS; ti++)
            {
                int len = (int) strlen(type_tokens[ti].tok);
                if (rem >= len && strncmp(tstr, type_tokens[ti].tok, (size_t) len) == 0)
                {
                    datatype = type_tokens[ti].dt;
                    ii += len;
                    ntype++;
                    matched = 1;
                    break;
                }
            }
            if (!matched)
            {
                memset(&p, 0, sizeof(p));
                p.name  = raw;
                p.error = 1;
                return p;
            }
            break;
        }

        default:
            /* Unknown letter -> error */
            memset(&p, 0, sizeof(p));
            p.name  = raw;
            p.error = 1;
            return p;
        }
    }

    /* Multiple location or type modifiers is an error */
    if (nloc > 1 || ntype > 1)
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
    p.datatype   = datatype;
    p.name       = colon + 1;
    p.error      = 0;

    return p;
}

/**
 * @brief Build a short modifier label string.
 *
 * Produces e.g. "[L]", "[SE]", "[TF32]", or "" if no
 * modifiers are active.
 *
 * @param p      Parsed stream name
 * @param buf    Output buffer (>= 16 bytes)
 * @param bufsz  Buffer size
 */
void fps_streamname_modifier_label(const FPS_STREAMNAME_PARSED *p, char *buf, int bufsz)
{
    if (bufsz < 2)
    {
        return;
    }

    char inner[16];
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
    if (p->datatype != 0)
    {
        const char *ttok = datatype_mod_token(p->datatype);
        if (ttok != NULL)
        {
            inner[pos++] = 'T';
            for (int ti = 0; ttok[ti] != '\0' && pos < (int) sizeof(inner) - 1; ti++)
            {
                inner[pos++] = ttok[ti];
            }
        }
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
int fps_streamname_is_shared(const char *raw)
{
    FPS_STREAMNAME_PARSED p = fps_streamname_parse(raw);

    if (p.loc == 'L')
    {
        return 0;
    }
    return 1;
}

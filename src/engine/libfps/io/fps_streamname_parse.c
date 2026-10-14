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

#include <string.h>
#include <stdio.h>

#include "ImageStreamIO/ImageStruct.h"
#include "fps_streamname_parse.h"

/**
 * @brief Return short @T...: token for an ImageStreamIO datatype.
 */
static const char *datatype_mod_token(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "TU8";
    case _DATATYPE_INT8:
        return "TI8";
    case _DATATYPE_UINT16:
        return "TU16";
    case _DATATYPE_INT16:
        return "TI16";
    case _DATATYPE_UINT32:
        return "TU32";
    case _DATATYPE_INT32:
        return "TI32";
    case _DATATYPE_UINT64:
        return "TU64";
    case _DATATYPE_INT64:
        return "TI64";
    case _DATATYPE_HALF:
        return "TF16";
    case _DATATYPE_FLOAT:
        return "TF32";
    case _DATATYPE_DOUBLE:
        return "TF64";
    case _DATATYPE_COMPLEX_FLOAT:
        return "TC64";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "TC128";
    default:
        return NULL;
    }
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

            static const struct
            {
                const char *tok;
                int         len;
                uint8_t     dt;
            } tmap[] = {
                { "C128", 4, _DATATYPE_COMPLEX_DOUBLE },
                { "UI16", 4, _DATATYPE_UINT16 },
                { "UI32", 4, _DATATYPE_UINT32 },
                { "UI64", 4, _DATATYPE_UINT64 },
                { "SI16", 4, _DATATYPE_INT16 },
                { "SI32", 4, _DATATYPE_INT32 },
                { "SI64", 4, _DATATYPE_INT64 },
                { "U16", 3, _DATATYPE_UINT16 },
                { "U32", 3, _DATATYPE_UINT32 },
                { "U64", 3, _DATATYPE_UINT64 },
                { "I16", 3, _DATATYPE_INT16 },
                { "I32", 3, _DATATYPE_INT32 },
                { "I64", 3, _DATATYPE_INT64 },
                { "UI8", 3, _DATATYPE_UINT8 },
                { "SI8", 3, _DATATYPE_INT8 },
                { "F16", 3, _DATATYPE_HALF },
                { "F32", 3, _DATATYPE_FLOAT },
                { "F64", 3, _DATATYPE_DOUBLE },
                { "C64", 3, _DATATYPE_COMPLEX_FLOAT },
                { "U8", 2, _DATATYPE_UINT8 },
                { "I8", 2, _DATATYPE_INT8 },
            };

            for (size_t ti = 0; ti < sizeof(tmap) / sizeof(tmap[0]); ti++)
            {
                if (rem >= tmap[ti].len && strncmp(tstr, tmap[ti].tok, (size_t) tmap[ti].len) == 0)
                {
                    datatype = tmap[ti].dt;
                    ii += tmap[ti].len;
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

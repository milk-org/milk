// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_argval.c
 * @brief Typed argument values (see milk_argval.h): no engine state, so the bash client uses them too.
 */

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "fps_types.h"
#include "milk_argval.h"

/**
 * milk_argval_is_numeric - Check whether an FPS/CLI parameter type holds a numeric value
 * @type: `FPTYPE_*` parameter type identifier
 *
 * Classifies integer, floating-point, PID, boolean (`FPTYPE_ONOFF`), and timespec
 * types as numeric so callers know whether to parse numbers/expressions or copy strings.
 *
 * Return: 1 if @type is numeric, 0 if string-like
 */
int milk_argval_is_numeric(uint64_t type)
{
    switch (type)
    {
    case FPTYPE_FLOAT32:
    case FPTYPE_FLOAT64:
    case FPTYPE_TIMESPEC:
    case FPTYPE_INT32:
    case FPTYPE_INT64:
    case FPTYPE_PID:
    case FPTYPE_UINT32:
    case FPTYPE_UINT64:
    case FPTYPE_ONOFF:
        return 1;
    }
    return 0;
}

/**
 * milk_argval_set_number - Store a range-checked floating-point number into a typed argument
 * @type: Target `FPTYPE_*` parameter type
 * @v:    Numeric value to convert and store
 * @ad:   Destination argument data structure
 *
 * Validates that @v fits within the domain of @type (requiring an exact integer and
 * checking bounds for integral and boolean types) before writing the union field in @ad.
 *
 * Return: 0 on success, -1 if @v is out of range or non-integral for an integer type
 */
int milk_argval_set_number(uint64_t type, double v, CLICMDARGDATA *ad)
{
    if (type == FPTYPE_FLOAT32 || type == FPTYPE_FLOAT64 || type == FPTYPE_TIMESPEC)
    {
        if (type == FPTYPE_FLOAT32 && isfinite(v) && fabs(v) > FLT_MAX)
        {
            return -1;
        }
        if (type == FPTYPE_FLOAT32)
        {
            ad->val.f32 = (float) v;
        }
        else
        {
            ad->val.f64 = v;
        }
        return 0;
    }
    if (v != floor(v)) /* also rejects NaN and the infinities */
    {
        return -1;
    }
    switch (type)
    {
    case FPTYPE_INT32:
        if (v < INT32_MIN || v > INT32_MAX)
        {
            return -1;
        }
        ad->val.i32 = (int32_t) v;
        return 0;
    case FPTYPE_INT64:
    case FPTYPE_PID:
        if (v < -9223372036854775808.0 || v >= 9223372036854775808.0)
        {
            return -1;
        }
        ad->val.i64 = (int64_t) v;
        return 0;
    case FPTYPE_UINT32:
        if (v < 0 || v > UINT32_MAX)
        {
            return -1;
        }
        ad->val.ui32 = (uint32_t) v;
        return 0;
    case FPTYPE_UINT64:
        if (v < 0 || v >= 18446744073709551616.0)
        {
            return -1;
        }
        ad->val.ui64 = (uint64_t) v;
        return 0;
    case FPTYPE_ONOFF:
        if (v != 0 && v != 1)
        {
            return -1;
        }
        ad->val.i64 = (int64_t) v;
        return 0;
    }
    return -1;
}

/**
 * milk_argval_parse - Parse a text token into a typed CLI argument value
 * @type: Target `FPTYPE_*` parameter type
 * @s:    Input NUL-terminated string to parse
 * @ad:   Destination argument data structure
 *
 * Parses @s according to @type: copies bounded strings for string types, accepts
 * `on`/`off`/`true`/`false` for `FPTYPE_ONOFF`, uses 64-bit integer parsers to preserve
 * full integer precision, and falls back to `strtod` + `milk_argval_set_number`.
 *
 * Return: 0 on success, -1 if @s is NULL, malformed, or out of range for @type
 */
int milk_argval_parse(uint64_t type, const char *s, CLICMDARGDATA *ad)
{
    char  *end;
    double v;

    if (s == NULL)
    {
        return -1;
    }
    if (!milk_argval_is_numeric(type))
    {
        size_t n = strlen(s);

        if (n >= sizeof(ad->val.s))
        {
            return -1;
        }
        memcpy(ad->val.s, s, n + 1);
        return 0;
    }
    if (s[0] == '\0' || isspace((unsigned char) s[0]))
    {
        return -1;
    }
    if (type == FPTYPE_ONOFF && (strcasecmp(s, "on") == 0 || strcasecmp(s, "true") == 0))
    {
        ad->val.i64 = 1;
        return 0;
    }
    if (type == FPTYPE_ONOFF && (strcasecmp(s, "off") == 0 || strcasecmp(s, "false") == 0))
    {
        ad->val.i64 = 0;
        return 0;
    }

    /* integers exactly (beyond double precision), then any number */
    errno = 0;
    if (type == FPTYPE_UINT64 && s[0] != '-')
    {
        unsigned long long u = strtoull(s, &end, 10);

        if (*end == '\0' && errno == 0)
        {
            ad->val.ui64 = u;
            return 0;
        }
    }
    else if (type == FPTYPE_INT64 || type == FPTYPE_PID)
    {
        long long l = strtoll(s, &end, 10);

        if (*end == '\0' && errno == 0)
        {
            ad->val.i64 = l;
            return 0;
        }
    }
    errno = 0;
    v     = strtod(s, &end);
    if (*end != '\0' || errno == ERANGE)
    {
        return -1;
    }
    return milk_argval_set_number(type, v, ad);
}

/**
 * format_real - Format a float or double with the shortest exact decimal representation
 * @v:      Floating-point value to format
 * @single: Non-zero for 32-bit float precision (`%.6g`/`%.9g`), 0 for 64-bit (`%.15g`/`%.17g`)
 * @buf:    Destination character buffer
 * @sz:     Capacity of @buf in bytes
 *
 * Formats @v at standard precision first and widens to full round-trip precision only
 * if `strtod(buf, NULL)` does not reproduce the original binary value.
 *
 * Return: void
 */
static void format_real(double v, int single, char *buf, size_t sz)
{
    snprintf(buf, sz, "%.*g", single ? 6 : 15, v);
    double back = strtod(buf, NULL);
    if (single ? ((float) back != (float) v) : (back != v))
    {
        snprintf(buf, sz, "%.*g", single ? 9 : 17, v);
    }
}

/**
 * milk_argval_format - Format a typed CLI argument value as a canonical string
 * @ad:  Source argument value with its `type` field set
 * @buf: Destination buffer receiving the formatted string
 * @sz:  Capacity of @buf in bytes
 *
 * Converts the current value stored in @ad into a round-trip-safe text representation
 * suitable for help output, `--no-run` command printing, and preset files.
 *
 * Return: void
 */
void milk_argval_format(const CLICMDARGDATA *ad, char *buf, size_t sz)
{
    switch (ad->type)
    {
    case FPTYPE_FLOAT32:
        format_real(ad->val.f32, 1, buf, sz);
        break;
    case FPTYPE_FLOAT64:
    case FPTYPE_TIMESPEC:
        format_real(ad->val.f64, 0, buf, sz);
        break;
    case FPTYPE_INT32:
        snprintf(buf, sz, "%" PRId32, ad->val.i32);
        break;
    case FPTYPE_INT64:
    case FPTYPE_PID:
        snprintf(buf, sz, "%" PRId64, ad->val.i64);
        break;
    case FPTYPE_UINT32:
        snprintf(buf, sz, "%" PRIu32, ad->val.ui32);
        break;
    case FPTYPE_UINT64:
        snprintf(buf, sz, "%" PRIu64, ad->val.ui64);
        break;
    case FPTYPE_ONOFF:
        snprintf(buf, sz, "%s", ad->val.i64 ? "on" : "off");
        break;
    default:
        snprintf(buf, sz, "%s", ad->val.s);
        break;
    }
}

static const struct
{
    const char *name;
    uint64_t    type;
} spec_types[] = { { "int", FPTYPE_INT64 },         { "uint", FPTYPE_UINT64 },
                   { "float", FPTYPE_FLOAT64 },     { "bool", FPTYPE_ONOFF },
                   { "str", FPTYPE_STRING },        { "stream", FPTYPE_STREAMNAME },
                   { "fits", FPTYPE_FITSFILENAME }, { NULL, 0 } };

/**
 * milk_argspec_parse - Parse an `@args` parameter declaration string in place
 * @s: Writable specification string (`<name>[:<type>[=<default>][:<help>]]`)
 * @a: Output structure populated with pointers into @s and the resolved type
 *
 * Splits @s on `:` and `=` delimiters, validates that the parameter name is a valid
 * identifier, and maps the short type name (`int`, `uint`, `float`, `bool`, `str`,
 * `stream`, `fits`) to its corresponding `FPTYPE_*` constant.
 *
 * Return: 0 on success, -1 if the identifier or type name is invalid
 */
int milk_argspec_parse(char *s, milk_argspec_t *a)
{
    char *t = strchr(s, ':');

    memset(a, 0, sizeof(*a));
    if (t != NULL)
    {
        *t++ = '\0';
    }
    if (!(isalpha((unsigned char) s[0]) || s[0] == '_') || strlen(s) >= sizeof(a->name))
    {
        return -1;
    }
    for (const char *c = s; *c != '\0'; c++)
    {
        if (!(isalnum((unsigned char) *c) || *c == '_'))
        {
            return -1;
        }
    }
    snprintf(a->name, sizeof(a->name), "%s", s);
    a->tname = "str";
    a->type  = FPTYPE_STRING;
    if (t == NULL)
    {
        return 0;
    }
    char *h = strchr(t, ':');
    if (h != NULL)
    {
        *h++    = '\0';
        a->help = h;
    }
    char *d = strchr(t, '=');
    if (d != NULL)
    {
        *d++   = '\0';
        a->def = d;
    }
    for (int i = 0; spec_types[i].name != NULL; i++)
    {
        if (strcmp(t, spec_types[i].name) == 0)
        {
            a->tname = spec_types[i].name;
            a->type  = spec_types[i].type;
            return 0;
        }
    }
    return -1;
}

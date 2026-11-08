// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_argval.c
 * @brief Argument values and words (see milk_argval.h): one reading of text values for module commands,
 *        unit command lines, `@args` and `@set`, and one rule binding words to parameters.
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
 * argval_int_literal - Read @s as an integer literal of the calculator: decimal (leading zeros allowed),
 * 0x hexadecimal, 0o octal or 0b binary, with an optional sign
 * @s:   Text
 * @neg: Out: 1 if the sign is minus
 * @mag: Out: the magnitude
 *
 * Return: 0, or -1 if @s is not such a literal or does not fit 64 bits
 */
static int argval_int_literal(const char *s, int *neg, unsigned long long *mag)
{
    int   base = 10;
    char *end  = NULL;

    *neg = (s[0] == '-');
    s += (s[0] == '-' || s[0] == '+');
    if (s[0] == '0' &&
        (s[1] == 'x' || s[1] == 'X' || s[1] == 'o' || s[1] == 'O' || s[1] == 'b' || s[1] == 'B'))
    {
        base = (s[1] == 'x' || s[1] == 'X') ? 16 : (s[1] == 'o' || s[1] == 'O') ? 8 : 2;
        s += 2;
    }
    if (!isalnum((unsigned char) s[0]))
    {
        return -1;
    }
    errno = 0;
    *mag  = strtoull(s, &end, base);
    return (end != s && *end == '\0' && errno == 0) ? 0 : -1;
}

/**
 * milk_argval_parse - Parse a text token into a typed CLI argument value
 * @type: Target `FPTYPE_*` parameter type
 * @s:    Input NUL-terminated string to parse
 * @ad:   Destination argument data structure
 *
 * Reads @s as the calculator reads a literal: integers in decimal (`010` is 10), `0x`, `0o` or `0b`,
 * exactly over 64 bits; other numbers with `strtod` (an integer type takes `1e3`, not `1.5`); bools as
 * `on`/`off`/`true`/`false` (any case) or 1/0. Strings are copied (bounded). An empty number or one out
 * of the type's range is invalid.
 *
 * Return: 0 on success, -1 if @s is NULL, malformed, or out of range for @type
 */
int milk_argval_parse(uint64_t type, const char *s, CLICMDARGDATA *ad)
{
    char              *end;
    double             v;
    int                neg = 0;
    unsigned long long mag = 0;

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

    if (argval_int_literal(s, &neg, &mag) == 0)
    {
        if (type == FPTYPE_UINT64)
        {
            if (neg && mag != 0)
            {
                return -1;
            }
            ad->val.ui64 = mag;
            return 0;
        }
        if (type == FPTYPE_INT64 || type == FPTYPE_PID)
        {
            if (mag > (unsigned long long) INT64_MAX + (unsigned long long) neg)
            {
                return -1;
            }
            ad->val.i64 = neg ? (int64_t) (0ULL - mag) : (int64_t) mag;
            return 0;
        }
        return milk_argval_set_number(type, neg ? -(double) mag : (double) mag, ad);
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
 * argval_format_real - Format a float or double with the shortest exact decimal representation
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
static void argval_format_real(double v, int single, char *buf, size_t sz)
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
        argval_format_real(ad->val.f32, 1, buf, sz);
        break;
    case FPTYPE_FLOAT64:
    case FPTYPE_TIMESPEC:
        argval_format_real(ad->val.f64, 0, buf, sz);
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
} argval_spec_types[] = { { "int", FPTYPE_INT64 },         { "uint", FPTYPE_UINT64 },
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
    for (int i = 0; argval_spec_types[i].name != NULL; i++)
    {
        if (strcmp(t, argval_spec_types[i].name) == 0)
        {
            a->tname = argval_spec_types[i].name;
            a->type  = argval_spec_types[i].type;
            return 0;
        }
    }
    return -1;
}

/**
 * milk_args_named - Check whether argument word @s is a `.name=value` word
 * @s: Candidate word
 *
 * Return: 1 if @s is `.` and a name (`[A-Za-z_][A-Za-z0-9_.]*`) followed by `=`, 0 otherwise
 */
int milk_args_named(const char *s)
{
    if (s == NULL || s[0] != '.' || !(isalpha((unsigned char) s[1]) || s[1] == '_'))
    {
        return 0;
    }
    for (s += 2; *s != '='; s++)
    {
        if (!isalnum((unsigned char) *s) && *s != '_' && *s != '.')
        {
            return 0;
        }
    }
    return 1;
}

/**
 * argval_find_slot - Index of the parameter named by the @len characters at @name, or -1
 */
static int argval_find_slot(const milk_argslot_t *slots, int ns, const char *name, size_t len)
{
    for (int i = 0; i < ns; i++)
    {
        if (strncmp(slots[i].name, name, len) == 0 && slots[i].name[len] == '\0')
        {
            return i;
        }
    }
    return -1;
}

/**
 * milk_args_bind - Bind argument words to parameters by the module-command rules
 * @nw:      Number of words
 * @w:       Words, after the command name and the caller's own options
 * @posonly: Index of the first word that can only be positional, or -1 to end at a "--" word
 * @ns:      Number of parameters
 * @slots:   Parameters
 * @val:     Out: the value text of parameter i (after the `=` of a `.name=value` word), or NULL
 * @word:    Out: the index in @w of that word, or -1 (may be NULL)
 * @err:     Message buffer
 * @errsz:   Size of @err
 *
 * `.name=value` words before the boundary bind by name; the other words, and every word after it, fill
 * the positional parameters not named, in order. A value "." keeps the current value, for the caller.
 *
 * Return: 0, or -1 with @err: an unknown name, a name given twice, `.name VALUE` (removed), too many
 * values
 */
int milk_args_bind(int                   nw,
                   char *const          *w,
                   int                   posonly,
                   int                   ns,
                   const milk_argslot_t *slots,
                   const char          **val,
                   int                  *word,
                   char                 *err,
                   size_t                errsz)
{
    int dashdash = -1;
    int boundary = (posonly >= 0) ? posonly : nw;
    int i        = 0;

    for (int t = 0; posonly < 0 && t < nw; t++)
    {
        if (strcmp(w[t], "--") == 0)
        {
            dashdash = boundary = t;
            break;
        }
    }
    for (int k = 0; k < ns; k++)
    {
        val[k] = NULL;
        if (word != NULL)
        {
            word[k] = -1;
        }
    }
    for (int t = 0; t < boundary; t++)
    {
        const char *eq = strchr(w[t], '=');
        if (milk_args_named(w[t]))
        {
            int k = argval_find_slot(slots, ns, w[t] + 1, (size_t) (eq - w[t] - 1));
            if (k < 0 || val[k] != NULL)
            {
                snprintf(err, errsz,
                         (k < 0) ? "unknown parameter %.*s" : "parameter %.*s given twice",
                         (int) (eq - w[t]), w[t]);
                return -1;
            }
            val[k] = eq + 1;
            if (word != NULL)
            {
                word[k] = t;
            }
        }
        else if (w[t][0] == '.' && w[t][1] != '\0' &&
                 argval_find_slot(slots, ns, w[t] + 1, strlen(w[t] + 1)) >= 0)
        {
            snprintf(err, errsz, "'%s VALUE' was removed: use %s=VALUE", w[t], w[t]);
            return -1;
        }
    }
    for (int t = 0; t < nw; t++)
    {
        if (t == dashdash || (t < boundary && milk_args_named(w[t])))
        {
            continue;
        }
        while (i < ns && (!slots[i].positional || val[i] != NULL))
        {
            i++;
        }
        if (i >= ns)
        {
            snprintf(err, errsz, "unexpected argument '%s'", w[t]);
            return -1;
        }
        val[i] = w[t];
        if (word != NULL)
        {
            word[i] = t;
        }
        i++;
    }
    return 0;
}

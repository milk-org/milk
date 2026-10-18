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

/* the shortest of %.{6,9}g (float) or %.{15,17}g (double) that reads back exactly */
static void format_real(double v, int single, char *buf, size_t sz)
{
    snprintf(buf, sz, "%.*g", single ? 6 : 15, v);
    double back = strtod(buf, NULL);
    if (single ? ((float) back != (float) v) : (back != v))
    {
        snprintf(buf, sz, "%.*g", single ? 9 : 17, v);
    }
}

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
} spec_types[] = { { "int", FPTYPE_INT64 },  { "uint", FPTYPE_UINT64 }, { "float", FPTYPE_FLOAT64 },
                   { "bool", FPTYPE_ONOFF }, { "str", FPTYPE_STRING },  { NULL, 0 } };

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

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_cmd_args.c
 * @brief Per-call command arguments (see milk_cmd_args.h).
 */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "CLIcore.h"
#include "milk_cmd_args.h"

#define MILK_CALL_DEPTH \
    8 /* nested dispatches tracked; deeper calls get CLI_checkarg_array's fallback */

static milk_call_t g_call[MILK_CALL_DEPTH];
static int         g_ncall;

/* Session defaults, as text, indexed by command then parameter */
typedef struct
{
    int    n;
    char **val;
} sessdef_t;

static sessdef_t *g_sess;
static int        g_nsess;

static int preset_name_ok(const char *s)
{
    size_t n = strlen(s);

    if (n == 0 || n >= sizeof(((milk_call_t *) 0)->save) || s[0] == '.')
    {
        return 0;
    }
    for (; *s != '\0'; s++)
    {
        if (!isalnum((unsigned char) *s) && *s != '_' && *s != '-' && *s != '.')
        {
            return 0;
        }
    }
    return 1;
}

int milk_args_begin(int cmdi, int *argc, char **argv)
{
    CMD         *cmd = &data.cmd[cmdi];
    milk_call_t *c   = (g_ncall < MILK_CALL_DEPTH) ? &g_call[g_ncall] : NULL;
    char         err[256];

    g_ncall++;
    if (c == NULL)
    {
        return 0;
    }
    memset(c, 0, sizeof(*c));
    c->cmdi = cmdi;
    err[0]  = '\0';

    if (cmd->nbparam > 0 && cmd->argdata != NULL && argc != NULL)
    {
        int n       = 1;
        int posonly = -1;

        for (int i = 1; i < *argc && err[0] == '\0'; i++)
        {
            char *a = argv[i];

            if (posonly >= 0)
            {
                argv[n++] = a;
            }
            else if (strcmp(a, "--") == 0)
            {
                posonly = n;
            }
            else if (strcmp(a, "--default") == 0)
            {
                c->set_default = 1;
            }
            else if (strcmp(a, "--no-run") == 0)
            {
                c->no_run = 1;
            }
            else if (strncmp(a, "--save=", 7) == 0 || strncmp(a, "--load=", 7) == 0)
            {
                if (preset_name_ok(a + 7))
                {
                    snprintf(a[2] == 's' ? c->save : c->load, sizeof(c->save), "%s", a + 7);
                }
                else
                {
                    snprintf(err, sizeof(err),
                             "invalid preset name '%.64s' (letters, digits, _ - .)", a + 7);
                }
            }
            else if (strcmp(a, "--save") == 0 || strcmp(a, "--load") == 0)
            {
                snprintf(err, sizeof(err), "%s takes a name: %s=NAME", a, a);
            }
            else
            {
                argv[n++] = a;
            }
        }
        if (err[0] == '\0' && c->set_default && c->save[0] != '\0')
        {
            snprintf(err, sizeof(err), "--save cannot be combined with --default");
        }
        if (err[0] == '\0')
        {
            *argc      = n;
            c->posonly = (posonly >= 0) ? posonly : n;
            c->src     = calloc((size_t) cmd->nbparam, 1);
            if (c->src == NULL)
            {
                snprintf(err, sizeof(err), "out of memory");
            }
        }
    }
    if (err[0] != '\0')
    {
        fprintf(stderr, "milk: %s: %s\n", cmd->key, err);
        g_ncall--;
        return 2;
    }

    c->raw  = argv;
    c->nraw = (argc != NULL) ? *argc : 0;
    milk_args_reset(cmdi);
    return 0;
}

int milk_args_end(void)
{
    if (g_ncall <= 0)
    {
        return 0;
    }
    g_ncall--;
    if (g_ncall >= MILK_CALL_DEPTH)
    {
        return 0;
    }

    milk_call_t *c  = &g_call[g_ncall];
    int          ue = c->usage_error;

    milk_args_reset(c->cmdi); /* help, hints and completion show the defaults between calls */
    free(c->src);
    memset(c, 0, sizeof(*c));
    c->cmdi = -1;
    return ue;
}

milk_call_t *milk_args_call(int cmdi)
{
    if (g_ncall <= 0 || g_ncall > MILK_CALL_DEPTH || g_call[g_ncall - 1].cmdi != cmdi)
    {
        return NULL;
    }
    return &g_call[g_ncall - 1];
}

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

const char *milk_args_tag(const char *fpstag)
{
    return (fpstag[0] == '.') ? fpstag + 1 : fpstag;
}

int milk_args_find(int cmdi, const char *tag, size_t len)
{
    for (int i = 0; i < data.cmd[cmdi].nbparam; i++)
    {
        const char *t = milk_args_tag(data.cmd[cmdi].argdata[i].fpstag);

        if (strncmp(t, tag, len) == 0 && t[len] == '\0')
        {
            return i;
        }
    }
    return -1;
}

/* the declared default: the parameter's example, read as registration always did */
static void argval_declared(CLICMDARGDATA *ad)
{
    const char *ex = ad->example;

    switch (ad->type)
    {
    case FPTYPE_FLOAT32:
        ad->val.f32 = (float) atof(ex);
        break;
    case FPTYPE_FLOAT64:
    case FPTYPE_TIMESPEC:
        ad->val.f64 = atof(ex);
        break;
    case FPTYPE_INT32:
        ad->val.i32 = (int32_t) atol(ex);
        break;
    case FPTYPE_UINT32:
        ad->val.ui32 = (uint32_t) atol(ex);
        break;
    case FPTYPE_INT64:
    case FPTYPE_PID:
    case FPTYPE_ONOFF:
        ad->val.i64 = (int64_t) atol(ex);
        break;
    case FPTYPE_UINT64:
        ad->val.ui64 = (uint64_t) atol(ex);
        break;
    default:
        snprintf(ad->val.s, sizeof(ad->val.s), "%s", ex);
        break;
    }
}

const char *milk_args_session_get(int cmdi, int argi)
{
    if (cmdi < 0 || cmdi >= g_nsess || argi < 0 || argi >= g_sess[cmdi].n)
    {
        return NULL;
    }
    return g_sess[cmdi].val[argi];
}

int milk_args_session_set(int cmdi, int argi, const char *val)
{
    if (cmdi < 0 || argi < 0 || argi >= data.cmd[cmdi].nbparam)
    {
        return -1;
    }
    if (val == NULL && milk_args_session_get(cmdi, argi) == NULL)
    {
        return 0;
    }
    if (cmdi >= g_nsess)
    {
        sessdef_t *p = realloc(g_sess, (size_t) (cmdi + 1) * sizeof(*p));
        if (p == NULL)
        {
            return -1;
        }
        memset(p + g_nsess, 0, (size_t) (cmdi + 1 - g_nsess) * sizeof(*p));
        g_sess  = p;
        g_nsess = cmdi + 1;
    }

    sessdef_t *s = &g_sess[cmdi];
    if (s->val == NULL)
    {
        s->val = calloc((size_t) data.cmd[cmdi].nbparam, sizeof(char *));
        if (s->val == NULL)
        {
            return -1;
        }
        s->n = data.cmd[cmdi].nbparam;
    }
    free(s->val[argi]);
    s->val[argi] = (val != NULL) ? strdup(val) : NULL;
    return (val != NULL && s->val[argi] == NULL) ? -1 : 0;
}

void milk_args_register(int cmdi)
{
    if (cmdi < g_nsess && g_sess[cmdi].val != NULL)
    {
        for (int i = 0; i < g_sess[cmdi].n; i++)
        {
            free(g_sess[cmdi].val[i]);
        }
        free(g_sess[cmdi].val);
        memset(&g_sess[cmdi], 0, sizeof(g_sess[cmdi]));
    }
    milk_args_reset(cmdi);
}

void milk_args_reset(int cmdi)
{
    CMD *c = &data.cmd[cmdi];

    for (int i = 0; i < c->nbparam && c->argdata != NULL; i++)
    {
        const char *sv = milk_args_session_get(cmdi, i);

        argval_declared(&c->argdata[i]);
        if (sv != NULL)
        {
            milk_argval_parse(c->argdata[i].type, sv, &c->argdata[i]); /* checked when it was set */
        }
    }
}

/* ${XDG_STATE_HOME:-$HOME/.local/state}/milk/args/<key> */
static int args_dir(const char *key, char *buf, size_t sz)
{
    const char *st   = getenv("XDG_STATE_HOME");
    const char *home = getenv("HOME");
    int         n;

    if (st != NULL && st[0] == '/')
    {
        n = snprintf(buf, sz, "%s/milk/args/%s", st, key);
    }
    else if (home != NULL && home[0] != '\0')
    {
        n = snprintf(buf, sz, "%s/.local/state/milk/args/%s", home, key);
    }
    else
    {
        return -1;
    }
    return (n > 0 && (size_t) n < sz) ? 0 : -1;
}

static int mkdir_p(char *path)
{
    for (char *p = path + 1; *p != '\0'; p++)
    {
        if (*p == '/')
        {
            *p    = '\0';
            int r = mkdir(path, 0755);
            *p    = '/';
            if (r != 0 && errno != EEXIST)
            {
                return -1;
            }
        }
    }
    return (mkdir(path, 0755) != 0 && errno != EEXIST) ? -1 : 0;
}

int milk_args_preset_save(int cmdi, const char *name)
{
    CMD  *c = &data.cmd[cmdi];
    char  dir[2048];
    char  path[2200];
    char  tmp[2300];
    char  v[STRINGMAXLEN_CLICMDARG + 32];
    FILE *fp;

    if (args_dir(c->key, dir, sizeof(dir)) != 0 || mkdir_p(dir) != 0)
    {
        fprintf(stderr, "milk: %s: cannot create the preset directory %s: %s\n", c->key, dir,
                strerror(errno));
        return -1;
    }
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, (int) getpid());
    fp = fopen(tmp, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "milk: %s: cannot write %s: %s\n", c->key, tmp, strerror(errno));
        return -1;
    }
    fprintf(fp, "# milk preset: %s --load=%s\n", c->key, name);
    for (int i = 0; i < c->nbparam; i++)
    {
        milk_argval_format(&c->argdata[i], v, sizeof(v));
        fprintf(fp, ".%s=%s\n", milk_args_tag(c->argdata[i].fpstag), v);
    }
    if (fclose(fp) != 0 || rename(tmp, path) != 0)
    {
        fprintf(stderr, "milk: %s: cannot write %s: %s\n", c->key, path, strerror(errno));
        unlink(tmp);
        return -1;
    }
    return 0;
}

int milk_args_preset_load(milk_call_t *call)
{
    CMD  *c = &data.cmd[call->cmdi];
    char  dir[2048];
    char  path[2200];
    char  line[STRINGMAXLEN_CLICMDARG + STRINGMAXLEN_FPSCLIARG_TAG + 8];
    int   ln = 0;
    FILE *fp;

    if (args_dir(c->key, dir, sizeof(dir)) != 0)
    {
        fprintf(stderr, "milk: %s: no preset directory (HOME and XDG_STATE_HOME unset)\n", c->key);
        return -1;
    }
    snprintf(path, sizeof(path), "%s/%s", dir, call->load);
    fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "milk: %s: no preset '%s' (%s): save one with %s ... --save=%s\n", c->key,
                call->load, path, c->key, call->load);
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        ln++;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0' || line[0] == '#')
        {
            continue;
        }

        char *eq = strchr(line, '=');
        int   i  = milk_args_named(line)
                       ? milk_args_find(call->cmdi, line + 1, (size_t) (eq - line - 1))
                       : -1;
        if (i < 0 || milk_argval_parse(c->argdata[i].type, eq + 1, &c->argdata[i]) != 0)
        {
            fprintf(stderr, "milk: %s: preset '%s' line %d: invalid entry '%s'\n", c->key,
                    call->load, ln, line);
            fclose(fp);
            return -1;
        }
        call->src[i] = MILK_ARGSRC_PRESET;
    }
    fclose(fp);
    return 0;
}

void milk_args_preset_list(int cmdi, FILE *fp)
{
    char           dir[2048];
    DIR           *dp;
    struct dirent *de;
    int            n = 0;

    if (args_dir(data.cmd[cmdi].key, dir, sizeof(dir)) != 0 || (dp = opendir(dir)) == NULL)
    {
        return;
    }
    while ((de = readdir(dp)) != NULL)
    {
        if (preset_name_ok(de->d_name) && strstr(de->d_name, ".tmp") == NULL)
        {
            fprintf(fp, "%s %s", (n++ == 0) ? "  Presets (--load=NAME):" : "", de->d_name);
        }
    }
    closedir(dp);
    if (n > 0)
    {
        fputc('\n', fp);
    }
}

static void put_quoted(const char *s, FILE *fp)
{
    int plain = (s[0] != '\0');

    for (const char *p = s; *p != '\0' && plain; p++)
    {
        plain = isalnum((unsigned char) *p) || strchr("_./:@%+,=-", *p) != NULL;
    }
    if (plain)
    {
        fputs(s, fp);
        return;
    }
    fputc('\'', fp);
    for (; *s != '\0'; s++)
    {
        if (*s == '\'')
        {
            fputs("'\\''", fp);
        }
        else
        {
            fputc(*s, fp);
        }
    }
    fputc('\'', fp);
}

void milk_args_print_call(int cmdi, int session, FILE *fp)
{
    CMD *c = &data.cmd[cmdi];
    char v[STRINGMAXLEN_CLICMDARG + 32];

    fputs(c->key, fp);
    for (int i = 0; i < c->nbparam; i++)
    {
        const char *sv = milk_args_session_get(cmdi, i);

        if (session && sv == NULL)
        {
            continue;
        }
        milk_argval_format(&c->argdata[i], v, sizeof(v));
        fprintf(fp, " .%s=", milk_args_tag(c->argdata[i].fpstag));
        put_quoted(session ? sv : v, fp);
    }
    fputc('\n', fp);
}

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
#include "CLIcore_script.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "calc/mcalc.h"
#include "milk_cmd_args.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

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

/**
 * preset_name_ok - Validate a preset name for `--save=NAME` and `--load=NAME`
 * @s: Candidate preset name string
 *
 * Rejects empty names, names starting with `.`, names that exceed the `save` buffer,
 * or names containing characters outside `[A-Za-z0-9_.-]` to prevent path traversal.
 *
 * Return: 1 if @s is a valid preset name, 0 otherwise
 */
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

/**
 * milk_args_begin - Push a per-call argument frame and strip call-level meta-flags
 * @cmdi: Index of the command being invoked in `data.cmd[]`
 * @argc: In/out argument count; updated in place after stripping meta-flags
 * @argv: Argument vector; compacted in place with remaining positional/named tokens
 *
 * Extracts `--`, `--default`, `--no-run`, `--save=NAME`, and `--load=NAME` before
 * `CLI_checkarg_array` runs, allocates the per-parameter source tracking array, and
 * resets `data.cmd[cmdi].argdata` to session/declared defaults.
 *
 * Return: 0 on success, or 2 on option usage error
 */
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

/**
 * milk_args_end - Pop the active per-call argument frame and restore defaults
 *
 * Resets `data.cmd[cmdi].argdata` back to session/declared defaults so interactive
 * help and tab completion between calls reflect the default state, frees per-call
 * allocations, and returns whether the call failed with a usage error.
 *
 * Return: 1 if the completed call encountered a usage error, 0 otherwise
 */
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

/**
 * milk_args_call - Return the active `milk_call_t` frame if it matches @cmdi
 * @cmdi: Command index expected at the top of the call stack
 *
 * Retrieves the per-call argument state prepared by `milk_args_begin()`.
 *
 * Return: Pointer to the active `milk_call_t`, or NULL if none matches @cmdi
 */
milk_call_t *milk_args_call(int cmdi)
{
    if (g_ncall <= 0 || g_ncall > MILK_CALL_DEPTH || g_call[g_ncall - 1].cmdi != cmdi)
    {
        return NULL;
    }
    return &g_call[g_ncall - 1];
}

/**
 * milk_args_tag - Strip a leading dot from an FPS parameter tag if present
 * @fpstag: Parameter tag string from `CLICMDARGDEF` or `CLICMDARGDATA`
 *
 * Normalizes parameter tag strings so comparisons work regardless of whether the
 * declaration included a leading `.`.
 *
 * Return: Pointer to the tag name without a leading dot
 */
const char *milk_args_tag(const char *fpstag)
{
    return (fpstag[0] == '.') ? fpstag + 1 : fpstag;
}

/**
 * milk_args_find - Find the parameter index for a tag name on command @cmdi
 * @cmdi: Command index in `data.cmd[]`
 * @tag:  Parameter tag name (without leading dot, not necessarily NUL-terminated)
 * @len:  Length of @tag in bytes
 *
 * Searches `data.cmd[cmdi].argdata` for a parameter whose normalized tag matches
 * the @len bytes at @tag.
 *
 * Return: Parameter index `0 .. nbparam-1` on match, or -1 if not found
 */
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

/**
 * argval_declared - Populate a parameter's value from its declared example string
 * @ad: Command argument data entry whose `example` and `type` are initialized
 *
 * Parses `ad->example` into the appropriate union member of `ad->val` according to
 * `ad->type` when initializing or resetting command argument defaults.
 *
 * Return: void
 */
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

/**
 * milk_args_session_get - Get the session-default override string for a parameter
 * @cmdi: Command index in `data.cmd[]`
 * @argi: Parameter index within the command
 *
 * Looks up any session-level default previously set via `--default`.
 *
 * Return: Session default string for `(cmdi, argi)`, or NULL if none is set
 */
const char *milk_args_session_get(int cmdi, int argi)
{
    if (cmdi < 0 || cmdi >= g_nsess || argi < 0 || argi >= g_sess[cmdi].n)
    {
        return NULL;
    }
    return g_sess[cmdi].val[argi];
}

/**
 * milk_args_session_set - Set or clear the session-default override for a parameter
 * @cmdi: Command index in `data.cmd[]`
 * @argi: Parameter index within the command
 * @val:  Formatted default value string to store, or NULL to clear the override
 *
 * Stores a duplicated string in `g_sess[cmdi].val[argi]` so subsequent invocations
 * in the same session start from @val instead of the declared example value.
 *
 * Return: 0 on success, -1 on invalid indices or allocation failure
 */
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

/**
 * milk_args_register - Initialize parameter defaults when a command is registered
 * @cmdi: Command index in `data.cmd[]`
 *
 * Clears any stale session defaults at slot @cmdi and populates `argdata` from the
 * command's declared example strings.
 *
 * Return: void
 */
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

/**
 * milk_args_reset - Reset a command's `argdata` values to session or declared defaults
 * @cmdi: Command index in `data.cmd[]`
 *
 * Restores every parameter of `data.cmd[cmdi]` to its declared example value and
 * overlays any active session default from `g_sess`.
 *
 * Return: void
 */
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

/**
 * args_dir - Format the per-command preset directory path under XDG state home
 * @key: Command key (`<module>.<cmd>`)
 * @buf: Destination buffer receiving `${XDG_STATE_HOME:-$HOME/.local/state}/milk/args/<key>`
 * @sz:  Capacity of @buf in bytes
 *
 * Resolves the state directory where named argument presets for @key are stored.
 *
 * Return: 0 on success, -1 if neither `XDG_STATE_HOME` nor `HOME` is set or @buf is too small
 */
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

/**
 * mkdir_p - Recursively create a directory path and all missing parent directories
 * @path: Mutable directory path string
 *
 * Walks @path component by component, temporarily NUL-terminating each slash to
 * create intermediate directories with mode `0755`.
 *
 * Return: 0 on success (or if the directory already exists), -1 on error
 */
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

/**
 * milk_args_preset_save - Save the resolved arguments of command @cmdi as a named preset
 * @cmdi: Command index in `data.cmd[]`
 * @name: Validated preset name
 *
 * Writes all parameter `.tag=value` pairs of `data.cmd[cmdi]` to
 * `${XDG_STATE_HOME:-$HOME/.local/state}/milk/args/<key>/<name>` atomically via a
 * temporary file.
 *
 * Return: 0 on success, -1 on directory or file I/O error
 */
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

/**
 * milk_args_preset_load - Load a saved parameter preset into the active call's `argdata`
 * @call: Active per-call context specifying `cmdi` and the `load` preset name
 *
 * Reads `.tag=value` lines from the preset file for `call->load`, parses each value
 * into `data.cmd[call->cmdi].argdata`, and marks its source as `MILK_ARGSRC_PRESET`.
 *
 * Return: 0 on success, -1 if the preset file is missing or contains invalid entries
 */
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

/**
 * milk_args_preset_list - Print available saved preset names for command @cmdi
 * @cmdi: Command index in `data.cmd[]`
 * @fp:   Output stream (e.g. `stdout`)
 *
 * Scans the command's preset directory and prints a one-line summary of saved
 * preset names for inclusion in `<cmd> -h` help output.
 *
 * Return: void
 */
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

/**
 * put_quoted - Write a string to @fp with shell-safe single-quoting when needed
 * @s:  Input NUL-terminated string to print
 * @fp: Output file stream
 *
 * Prints @s verbatim if it is non-empty and consists only of safe shell characters,
 * or wraps it in single quotes (escaping embedded single quotes) otherwise.
 *
 * Return: void
 */
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

/**
 * milk_args_print_call - Print a reproducible command line with `.tag=value` arguments
 * @cmdi:    Command index in `data.cmd[]`
 * @session: Non-zero to print only session-default overrides (`--default`), 0 for all
 * @fp:      Output file stream
 *
 * Formats the command key followed by shell-quoted `.tag=value` arguments for
 * `--no-run` inspection or `--default` display.
 *
 * Return: void
 */
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

static void refresh_shared_image_arg(IMAGE *im)
{
    if (im == NULL || !im->used || im->md == NULL || im->md[0].shared != 1)
    {
        return;
    }
    char        shmpath[512];
    struct stat st;
    if (ImageStreamIO_filename(shmpath, sizeof(shmpath), im->name) == IMAGESTREAMIO_SUCCESS &&
        stat(shmpath, &st) == 0 && st.st_ino != im->md[0].inode)
    {
        IMAGE fresh;
        memset(&fresh, 0, sizeof(fresh));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(im->name, &fresh) == IMAGESTREAMIO_SUCCESS)
        {
            ImageStreamIO_closeIm(im);
            memcpy(im, &fresh, sizeof(IMAGE));
            im->used = 1;
            milk_table_bump();
        }
    }
}

static imageID resolve_arg_image(const char *name, int allow_autoconnect)
{
    if (data.core.image == NULL || data.core.NB_MAX_IMAGE <= 0)
    {
        return -1;
    }
    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(name);
    if (sp.error)
    {
        return -1;
    }
    const char *bare = (sp.name != NULL && sp.name[0] != '\0') ? sp.name : name;
    imageID     id   = image_ID(bare, data.core.image, data.core.NB_MAX_IMAGE);
    if (id != -1)
    {
        refresh_shared_image_arg(&data.core.image[id]);
    }
    if (id != -1 || !allow_autoconnect || sp.loc == 'L')
    {
        return id;
    }
    char shmpath[512];
    if (ImageStreamIO_filename(shmpath, sizeof(shmpath), bare) == IMAGESTREAMIO_SUCCESS &&
        access(shmpath, R_OK) == 0)
    {
        IMAGE tmp_im;
        memset(&tmp_im, 0, sizeof(tmp_im));
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(bare, &tmp_im) == IMAGESTREAMIO_SUCCESS)
        {
            id = next_avail_image_ID(-1);
            if (id != -1)
            {
                memcpy(&data.core.image[id], &tmp_im, sizeof(IMAGE));
                data.core.image[id].used = 1;
            }
            else
            {
                ImageStreamIO_closeIm(&tmp_im);
            }
        }
    }
    return id;
}

static int is_plain_ident_arg(const char *s)
{
    if (s == NULL || s[0] == '\0')
    {
        return 0;
    }
    unsigned char c0 = (unsigned char) s[0];
    if (!(isalpha(c0) || c0 == '_' || c0 == '?' ||
          (c0 == '.' && !isdigit((unsigned char) s[1]) && s[1] != '\0')))
    {
        return 0;
    }
    for (const char *p = s; *p != '\0'; p++)
    {
        unsigned char c = (unsigned char) *p;
        if (!(isalnum(c) || c == '_' || c == '.' || c == '$' || c == ':' || c == '?'))
        {
            return 0;
        }
    }
    return 1;
}

void milk_cmd_classify_arg(const char *arg, int is_slice)
{
    CMDARGTOKEN *tok = &data.cmdargtoken[data.cmdNBarg];
    if (arg == NULL || arg[0] == '\0')
    {
        return;
    }

    size_t len = strlen(arg);
    if (len >= 2 && arg[0] == '"' && arg[len - 1] == '"' && memchr(arg + 1, '"', len - 2) == NULL)
    {
        size_t slen = len - 2;
        if (slen >= STRINGMAXLEN_CMDARGTOKEN_VAL)
        {
            slen = STRINGMAXLEN_CMDARGTOKEN_VAL - 1;
        }
        memcpy(tok->val.string, arg + 1, slen);
        tok->val.string[slen] = '\0';
        tok->type             = CMDARGTOKEN_TYPE_STRING;
        return;
    }

    if (!is_slice && is_plain_ident_arg(arg))
    {
        char sbuf[STRINGMAXLEN_CMDARGTOKEN_VAL];
        if ((arg[0] == 'L' || arg[0] == 'S') && arg[1] == ':')
        {
            snprintf(sbuf, sizeof(sbuf), "@%s", arg);
        }
        else
        {
            snprintf(sbuf, sizeof(sbuf), "%s", arg);
        }

        long svid = variable_ID(sbuf);
        if (svid != -1 && data.core.variable[svid].type == 3 &&
            data.core.variable[svid].value.s[0] != '\0')
        {
            snprintf(sbuf, sizeof(sbuf), "%s", data.core.variable[svid].value.s);
            svid = variable_ID(sbuf);
        }

        if (svid != -1 && data.core.variable[svid].type != 3)
        {
            if (data.core.variable[svid].type == 1)
            {
                tok->type     = CMDARGTOKEN_TYPE_LONG;
                tok->val.numl = data.core.variable[svid].value.l;
            }
            else if (data.core.variable[svid].type == 2)
            {
                snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
                         data.core.variable[svid].value.s);
            }
            else
            {
                tok->type     = CMDARGTOKEN_TYPE_FLOAT;
                tok->val.numf = data.core.variable[svid].value.f;
            }
            return;
        }

        if (resolve_arg_image(sbuf, 0) != -1)
        {
            FPS_STREAMNAME_PARSED sp = fps_streamname_parse(sbuf);
            const char           *bare =
                (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : sbuf;
            tok->type = CMDARGTOKEN_TYPE_EXISTINGIMAGE;
            snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", bare);
            return;
        }

        const char *dot1 = strchr(sbuf, '.');
        if (dot1 != NULL && strchr(dot1 + 1, '.') != NULL &&
            milk_entity_ns_kind(sbuf, NULL) != MILK_NS_NONE)
        {
            double dv     = 0.0;
            long   lv     = 0;
            int    islong = 0;
            if (milk_shm_prop_get_num(sbuf, &dv, &lv, &islong) == 0)
            {
                if (islong)
                {
                    tok->type     = CMDARGTOKEN_TYPE_LONG;
                    tok->val.numl = lv;
                }
                else
                {
                    tok->type     = CMDARGTOKEN_TYPE_FLOAT;
                    tok->val.numf = dv;
                }
                return;
            }
            char pstr[256];
            if (milk_shm_prop_get(sbuf, pstr, sizeof(pstr)) == 0)
            {
                snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", pstr);
                return;
            }
        }

        tok->type = CMDARGTOKEN_TYPE_STRING;
        snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", sbuf);
        return;
    }

    if (is_slice)
    {
        const char *bk = strchr(arg, '[');
        char        bare[256];
        char        sbuf[512];
        size_t      bn = (bk != NULL) ? (size_t) (bk - arg) : 0;
        if (bn == 0 || bn >= sizeof(bare))
        {
            tok->type = CMDARGTOKEN_TYPE_RAWSTRING;
            snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", arg);
            return;
        }
        memcpy(bare, arg, bn);
        bare[bn]  = '\0';
        long bvid = variable_ID(bare);
        if (bvid != -1 && data.core.variable[bvid].type == 3 &&
            data.core.variable[bvid].value.s[0] != '\0')
        {
            snprintf(bare, sizeof(bare), "%s", data.core.variable[bvid].value.s);
        }
        if (resolve_arg_image(bare, 1) == -1)
        {
            tok->type = CMDARGTOKEN_TYPE_STRING;
            snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s%s", bare, bk);
            return;
        }
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(bare);
        const char *rbare = (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : bare;
        snprintf(sbuf, sizeof(sbuf), "%s%s", rbare, bk);

        int    mtype = 0;
        long   mlval = 0;
        double mdval = 0.0;
        if (mcalc_eval_math_to_val(sbuf, &mtype, &mlval, &mdval))
        {
            if (mtype == 1)
            {
                tok->type     = CMDARGTOKEN_TYPE_LONG;
                tok->val.numl = mlval;
            }
            else
            {
                tok->type     = CMDARGTOKEN_TYPE_FLOAT;
                tok->val.numf = mdval;
            }
            return;
        }

        int  eval_rc = 0;
        char outbuf[256];
        outbuf[0] = '\0';
        int ok    = mcalc_eval_str(sbuf, 0, &eval_rc, outbuf, sizeof(outbuf));
        if (ok && eval_rc == 0 && outbuf[0] != '\0' &&
            image_ID(outbuf, data.core.image, data.core.NB_MAX_IMAGE) != -1)
        {
            tok->type = CMDARGTOKEN_TYPE_EXISTINGIMAGE;
            snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", outbuf);
        }
        else
        {
            tok->type = CMDARGTOKEN_TYPE_RAWSTRING;
            snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", arg);
        }
        return;
    }

    int    mtype = 0;
    long   mlval = 0;
    double mdval = 0.0;
    if (mcalc_eval_math_to_val(arg, &mtype, &mlval, &mdval))
    {
        if (mtype == 1)
        {
            tok->type     = CMDARGTOKEN_TYPE_LONG;
            tok->val.numl = mlval;
        }
        else
        {
            tok->type     = CMDARGTOKEN_TYPE_FLOAT;
            tok->val.numf = mdval;
        }
        return;
    }

    tok->type = CMDARGTOKEN_TYPE_RAWSTRING;
    snprintf(tok->val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", arg);
}

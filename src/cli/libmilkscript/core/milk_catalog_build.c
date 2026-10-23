// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_catalog_build.c
 * @brief Writing the install-time catalog (`milk-cli --catalog`), and loading the module of a catalogued
 * command (see milk_catalog.h).
 */

#include <dirent.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_modules.h"
#include "milk_catalog.h"
#include "milk_cmd_index.h"
#include "milk_lexicon.h"

#define CATALOG_PATHMAX 4096

typedef struct
{
    const char *key;
    const char *load; /* module load name ("milk": built into the engine) */
    char        info[256];
} entry_t;

/**
 * elf_defines - Check whether an ELF64 shared object exports a dynamic symbol
 * @path: Path to the `.so` shared library file
 * @sym:  Dynamic symbol name to search for (e.g. `"__milk_module_info"`)
 *
 * Memory-maps @path read-only and inspects its `SHT_DYNSYM` section headers without
 * calling `dlopen()`, avoiding execution of library constructors for non-module `.so` files.
 *
 * Return: 1 if @path defines @sym in its dynamic symbol table, 0 otherwise
 */
static int elf_defines(const char *path, const char *sym)
{
    struct stat st;
    int         found = 0;
    int         fd    = open(path, O_RDONLY | O_CLOEXEC);

    if (fd < 0)
    {
        return 0;
    }
    if (fstat(fd, &st) != 0 || st.st_size < (off_t) sizeof(Elf64_Ehdr))
    {
        close(fd);
        return 0;
    }

    size_t               sz = (size_t) st.st_size;
    const unsigned char *m  = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED)
    {
        return 0;
    }

    const Elf64_Ehdr *eh = (const Elf64_Ehdr *) m;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) == 0 && eh->e_ident[EI_CLASS] == ELFCLASS64 &&
        eh->e_shoff + (size_t) eh->e_shnum * sizeof(Elf64_Shdr) <= sz)
    {
        const Elf64_Shdr *sh = (const Elf64_Shdr *) (m + eh->e_shoff);

        for (int i = 0; i < eh->e_shnum && !found; i++)
        {
            if (sh[i].sh_type != SHT_DYNSYM || sh[i].sh_link >= eh->e_shnum ||
                sh[i].sh_offset + sh[i].sh_size > sz)
            {
                continue;
            }

            const Elf64_Shdr *ss = &sh[sh[i].sh_link];
            if (ss->sh_offset + ss->sh_size > sz)
            {
                continue;
            }

            const Elf64_Sym *s    = (const Elf64_Sym *) (m + sh[i].sh_offset);
            const char      *strs = (const char *) m + ss->sh_offset;
            size_t           len  = strlen(sym) + 1;

            for (size_t k = 0; k < sh[i].sh_size / sizeof(Elf64_Sym) && !found; k++)
            {
                found = s[k].st_shndx != SHN_UNDEF && s[k].st_name + len <= ss->sh_size &&
                        memcmp(strs + s[k].st_name, sym, len) == 0;
            }
        }
    }
    munmap((void *) m, sz);
    return found;
}

/**
 * loads_cleanly - Test in a child process whether a module loads and registers cleanly
 * @load: Module load name to probe
 *
 * Forks a child process to call `load_module_shared(load)` so that a broken shared
 * library (e.g. an unresolved symbol triggered during registration) does not abort
 * the parent `milk-cli --catalog` process.
 *
 * Return: 1 if the child exited with status 0, 0 otherwise
 */
static int loads_cleanly(const char *load)
{
    int   st;
    pid_t pid;

    fflush(NULL);
    if ((pid = fork()) == 0)
    {
        int fd = open("/dev/null", O_WRONLY);
        if (fd >= 0)
        {
            dup2(fd, STDOUT_FILENO);
        }
        _exit((load_module_shared(load) == RETURN_SUCCESS) ? 0 : 1);
    }
    return pid > 0 && waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

/**
 * cmp_str - Compare two `char *` pointers lexicographically for `qsort`
 * @a: Pointer to first `char *` element
 * @b: Pointer to second `char *` element
 *
 * Used to sort shared library filenames deterministically before loading modules.
 *
 * Return: Negative, zero, or positive integer as `*a` is less than, equal to, or
 *         greater than `*b`
 */
static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(char *const *) a, *(char *const *) b);
}

/**
 * milk_module_loaded - Check whether a module with the given load name is already loaded
 * @loadname: Module load name (e.g. `"milkfft"` or `"milkCOREMODmemory"`)
 *
 * Scans `data.module[]` for an entry whose `loadname` matches @loadname.
 *
 * Return: 1 if the module is already loaded, 0 otherwise
 */
int milk_module_loaded(const char *loadname)
{
    for (long m = 0; m < data.NBmodule; m++)
    {
        if (strcmp(data.module[m].loadname, loadname) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * load_installed_modules - Discover and load all milk module libraries in @libdir
 * @libdir: Directory containing installed `lib*.so` shared libraries
 *
 * Scans @libdir for `lib*.so` files that export `__milk_module_info`, verifies each
 * one in a child process via `loads_cleanly()`, and loads healthy modules into the
 * current process so their commands are registered in `data.cmd[]`.
 *
 * Return: void
 */
static void load_installed_modules(const char *libdir)
{
    DIR           *d = opendir(libdir);
    struct dirent *de;
    char         **names = NULL;
    size_t         n     = 0;

    if (d == NULL)
    {
        return;
    }
    while ((de = readdir(d)) != NULL)
    {
        size_t l = strlen(de->d_name);
        char **p;

        if (l <= 6 || strncmp(de->d_name, "lib", 3) != 0 ||
            strcmp(de->d_name + l - 3, ".so") != 0 ||
            (p = realloc(names, (n + 1) * sizeof(*names))) == NULL)
        {
            continue;
        }
        names = p;
        if ((names[n] = strdup(de->d_name)) != NULL)
        {
            n++;
        }
    }
    closedir(d);
    qsort(names, n, sizeof(*names), cmp_str);

    for (size_t i = 0; i < n; i++)
    {
        char path[CATALOG_PATHMAX];
        char load[256];

        snprintf(path, sizeof(path), "%s/%s", libdir, names[i]);
        snprintf(load, sizeof(load), "%.*s", (int) (strlen(names[i]) - 6), names[i] + 3);
        if (milk_module_loaded(load) || !elf_defines(path, "__milk_module_info"))
        {
            free(names[i]);
            continue;
        }
        if (loads_cleanly(load))
        {
            load_module_shared(load);
        }
        else
        {
            fprintf(stderr, "milk: catalog: module %s does not load, its commands are left out\n",
                    load);
        }
        free(names[i]);
    }
    free(names);
}

/**
 * cmp_entry - Compare two catalog entries by module load name, then by command key
 * @a: Pointer to first `entry_t` element
 * @b: Pointer to second `entry_t` element
 *
 * Groups entries belonging to the same module together so `write_tsv()` can emit
 * one `.tsv` file per module in a single pass.
 *
 * Return: Negative, zero, or positive ordering result
 */
static int cmp_entry(const void *a, const void *b)
{
    const entry_t *x = a;
    const entry_t *y = b;
    int            c = strcmp(x->load, y->load);

    return (c != 0) ? c : strcmp(x->key, y->key);
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
 * clear_dir - Unlink all entries in @dir whose filenames end with @suffix
 * @dir:    Directory path to clean
 * @suffix: Required filename suffix to remove (or `""` to remove all files)
 *
 * Removes stale generated catalog files or symlinks before writing fresh ones.
 *
 * Return: void
 */
static void clear_dir(const char *dir, const char *suffix)
{
    DIR           *d = opendir(dir);
    struct dirent *de;
    size_t         sl = strlen(suffix);

    while (d != NULL && (de = readdir(d)) != NULL)
    {
        size_t l = strlen(de->d_name);
        char   p[CATALOG_PATHMAX];

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0 || l < sl ||
            strcmp(de->d_name + l - sl, suffix) != 0)
        {
            continue;
        }
        snprintf(p, sizeof(p), "%s/%s", dir, de->d_name);
        unlink(p);
    }
    if (d != NULL)
    {
        closedir(d);
    }
}

/**
 * write_tsv - Write per-module `.tsv` catalog files into `catalog.d`
 * @dir: Path to `$MILK_INSTALLDIR/share/milk/catalog.d`
 * @e:   Array of catalog entries sorted by `cmp_entry` (grouped by `load`)
 * @n:   Number of entries in @e
 *
 * Writes `<load>.tsv` files containing `<key>\t<load>\t<info>` lines for each
 * module represented in @e.
 *
 * Return: 0 on success, -1 on file I/O error
 */
static int write_tsv(const char *dir, const entry_t *e, size_t n)
{
    FILE *fp = NULL;

    for (size_t i = 0; i < n; i++)
    {
        if (i == 0 || strcmp(e[i].load, e[i - 1].load) != 0)
        {
            char p[CATALOG_PATHMAX];

            if (fp != NULL && fclose(fp) != 0)
            {
                return -1;
            }
            snprintf(p, sizeof(p), "%s/%s.tsv", dir, e[i].load);
            if ((fp = fopen(p, "w")) == NULL)
            {
                return -1;
            }
        }
        fprintf(fp, "%s\t%s\t%s\n", e[i].key, e[i].load, e[i].info);
    }
    return (fp != NULL && fclose(fp) != 0) ? -1 : 0;
}

/**
 * cmp_entry_key - Compare two catalog entries by command key for `qsort`
 * @a: Pointer to first `entry_t` element
 * @b: Pointer to second `entry_t` element
 *
 * Orders `entry_t` records alphabetically by `key` before generating `catalog.bash`.
 *
 * Return: Negative, zero, or positive integer as `a->key` is less than, equal to,
 *         or greater than `b->key`
 */
static int cmp_entry_key(const void *a, const void *b)
{
    return strcmp(((const entry_t *) a)->key, ((const entry_t *) b)->key);
}

/**
 * write_bash - Atomically write `share/milk/catalog.bash` with shell wrapper functions
 * @path: Destination path for `catalog.bash`
 * @e:    Array of catalog entries (sorted in place by `cmp_entry_key`)
 * @n:    Number of entries in @e
 *
 * Generates `__MILK_CATALOG`, `__MILK_SHORT`, and a `function <mod>.<cmd>` wrapper
 * for every catalogued command via a temporary file and atomic `rename()`.
 *
 * Return: 0 on success, -1 on I/O error
 */
static int write_bash(const char *path, entry_t *e, size_t n)
{
    char  tmp[CATALOG_PATHMAX];
    FILE *fp;

    qsort(e, n, sizeof(*e), cmp_entry_key);
    snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, (int) getpid());
    if ((fp = fopen(tmp, "w")) == NULL)
    {
        return -1;
    }
    fprintf(fp,
            "# milk module commands, generated at install by `milk-cli --catalog`: do not edit\n");
    fprintf(fp, "__MILK_CATALOG=(");
    for (size_t i = 0; i < n; i++)
    {
        fprintf(fp, "%s%s", (i % 8 == 0) ? "\n    " : " ", e[i].key);
    }
    fprintf(fp,
            "\n)\n\n# short name -> module commands (hints for the removed @short spellings)\n");
    fprintf(fp, "declare -gA __MILK_SHORT=(");
    for (size_t i = 0; i < n; i++)
    {
        const char *s     = strchr(e[i].key, '.') + 1;
        int         first = 1;

        for (size_t j = 0; j < i && first;
             j++) /* the first command with this short name lists all */
        {
            first = strcmp(strchr(e[j].key, '.') + 1, s) != 0;
        }
        if (!first)
        {
            continue;
        }
        fprintf(fp, "\n    [%s]=\"%s", s, e[i].key);
        for (size_t j = i + 1; j < n; j++)
        {
            if (strcmp(strchr(e[j].key, '.') + 1, s) == 0)
            {
                fprintf(fp, " %s", e[j].key);
            }
        }
        fputc('"', fp);
    }
    fprintf(fp, "\n)\n\n");
    for (size_t i = 0; i < n; i++)
    {
        fprintf(fp, "function %s { milkcall %s \"$@\"; }\n", e[i].key, e[i].key);
    }
    if (fclose(fp) != 0 || rename(tmp, path) != 0)
    {
        unlink(tmp);
        return -1;
    }
    return 0;
}

/**
 * write_farm - Populate `libexec/milk/bin` with `milk-call` symlinks for commands and keywords
 * @dir: Mutable path to `<prefix>/libexec/milk/bin`
 * @e:   Array of catalogued module commands
 * @n:   Number of entries in @e
 *
 * Creates relative symlinks pointing to `../../../bin/milk-call` for every callable
 * `@keyword` and `<module>.<cmd>` so external tools (`xargs`, `find`, `parallel`) can
 * invoke them directly from `$PATH`.
 *
 * Return: 0 on success, non-zero if directory creation or any `symlink()` call failed
 */
static int write_farm(char *dir, const entry_t *e, size_t n)
{
    const char *target = "../../../bin/milk-call";
    char        p[CATALOG_PATHMAX];
    int         rc = 0;

    if (mkdir_p(dir) != 0)
    {
        return -1;
    }
    clear_dir(dir, "");
    for (int i = 0; milk_lexicon_keywords()[i].name != NULL; i++)
    {
        if (milk_lexicon_keywords()[i].callable)
        {
            snprintf(p, sizeof(p), "%s/@%s", dir, milk_lexicon_keywords()[i].name);
            rc |= symlink(target, p);
        }
    }
    for (size_t i = 0; i < n; i++)
    {
        snprintf(p, sizeof(p), "%s/%s", dir, e[i].key);
        rc |= symlink(target, p);
    }
    return rc;
}

/**
 * milk_catalog_build - Build the install-time command catalog under @prefix
 * @prefix: Installation root directory (e.g. `$MILK_INSTALLDIR`)
 *
 * Loads all healthy module shared libraries in `<prefix>/lib`, collects their
 * registered `<module>.<cmd>` commands, and writes `<prefix>/share/milk/catalog.d/*.tsv`,
 * `<prefix>/share/milk/catalog.bash`, and the `<prefix>/libexec/milk/bin` symlink farm.
 *
 * Return: 0 on success, -1 if any catalog artifact could not be written
 */
int milk_catalog_build(const char *prefix)
{
    char     p[CATALOG_PATHMAX];
    entry_t *e = NULL;
    size_t   n = 0;
    int      rc;

    snprintf(p, sizeof(p), "%s/lib", prefix);
    load_installed_modules(p);

    e = calloc((size_t) data.NBcmd + 1, sizeof(*e));
    if (e == NULL)
    {
        return -1;
    }
    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        long midx = data.cmd[i].moduleindex;

        if (strchr(data.cmd[i].key, '.') == NULL) /* root words are keywords (`help`) */
        {
            continue;
        }
        e[n].key  = data.cmd[i].key;
        e[n].load = (midx >= 0 && data.module[midx].loadname[0] != '\0')
                        ? data.module[midx].loadname
                        : "milk";
        snprintf(e[n].info, sizeof(e[n].info), "%s", data.cmd[i].info);
        for (char *q = e[n].info; *q != '\0'; q++)
        {
            *q = (*q == '\t' || *q == '\n' || *q == '\r') ? ' ' : *q;
        }
        n++;
    }
    qsort(e, n, sizeof(*e), cmp_entry);

    snprintf(p, sizeof(p), "%s/share/milk/catalog.d", prefix);
    rc = mkdir_p(p);
    if (rc == 0)
    {
        clear_dir(p, ".tsv");
        rc = write_tsv(p, e, n);
    }
    snprintf(p, sizeof(p), "%s/share/milk/catalog.bash", prefix);
    rc |= write_bash(p, e, n);
    snprintf(p, sizeof(p), "%s/libexec/milk/bin", prefix);
    rc |= write_farm(p, e, n);

    /* the run-time list and farm they replace */
    snprintf(p, sizeof(p), "%s/share/milk/milk-commands.list", prefix);
    unlink(p);
    snprintf(p, sizeof(p), "%s/libexec/milk-bin", prefix);
    clear_dir(p, "");
    rmdir(p);

    if (rc != 0)
    {
        fprintf(stderr, "milk: cannot write the command catalog under %s: %s\n", prefix,
                strerror(errno));
    }
    else
    {
        printf("milk: catalogued %zu commands under %s\n", n, prefix);
    }
    free(e);
    return (rc != 0) ? -1 : 0;
}

/**
 * milk_catalog_autoload - Quietly load the module owning @key if it is in the catalog
 * @key: Full `<module>.<cmd>` command name to resolve
 *
 * Looks up @key in the install-time catalog and, if its owning module is not yet
 * loaded in this session, calls `load_module_shared()` with `dcquiet = 1`.
 *
 * Return: 1 if @key is registered in `data.cmd[]` after autoloading, 0 otherwise
 */
int milk_catalog_autoload(const char *key)
{
    const char *mod = milk_catalog_module(key);
    int         q   = dcquiet;

    if (mod != NULL && strcmp(mod, "milk") != 0 && !milk_module_loaded(mod))
    {
        dcquiet = 1; /* first use of a module: one dlopen, without the LOADED line */
        load_module_shared(mod);
        dcquiet = q;
    }
    return milk_cmd_lookup(key) >= 0;
}

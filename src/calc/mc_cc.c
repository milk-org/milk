// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "mc_plan.h"
#include "mc_prelude.h"

#ifndef MILK_DEFAULT_CC
#    define MILK_DEFAULT_CC "gcc"
#endif

extern char **environ;

#define MC_MEM_CACHE_CAP 4096
#define MC_DISK_MAX_ENTRIES 4096
#define MC_DISK_PRUNE_COUNT 512

typedef struct
{
    uint64_t     h1;
    uint64_t     h2;
    mc_kernel_fn fn;
    char         so_path[512];
} mc_mem_entry_t;

static pthread_mutex_t g_cc_lock = PTHREAD_MUTEX_INITIALIZER;
static mc_mem_entry_t  g_mem_cache[MC_MEM_CACHE_CAP];
static int             g_mem_count = 0;
static int             g_mem_hand  = 0;
static unsigned long   g_tmp_seq   = 0;

static void hash128(const char *cc, const char *src, uint64_t *out_h1, uint64_t *out_h2)
{
    uint64_t h1 = 0xcbf29ce484222325ULL ^ (uint64_t) MC_PRELUDE_VERSION;
    uint64_t h2 = 0x84222325cbf29ce4ULL ^ ((uint64_t) MC_PRELUDE_VERSION << 32);

    for (const unsigned char *p = (const unsigned char *) cc; *p != '\0'; p++)
    {
        h1 = (h1 ^ *p) * 0x100000001b3ULL;
        h2 = (h2 ^ (*p + 0x9eU)) * 0x100000001b3ULL;
    }
    for (const unsigned char *p = (const unsigned char *) src; *p != '\0'; p++)
    {
        h1 = (h1 ^ *p) * 0x100000001b3ULL;
        h2 = (h2 ^ (uint64_t) (*p ^ 0x5aU)) * 0x9e3779b97f4a7c15ULL;
    }
    h1 ^= h1 >> 33;
    h1 *= 0xff51afd7ed558ccdULL;
    h1 ^= h1 >> 33;
    h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53ULL;
    h2 ^= h2 >> 33;
    *out_h1 = h1;
    *out_h2 = h2;
}

static const char *get_compiler(void)
{
    const char *env_cc = getenv("MILK_CALC_CC");
    if (env_cc != NULL && env_cc[0] != '\0')
    {
        return env_cc;
    }
    return MILK_DEFAULT_CC;
}

static int ensure_cache_dir(char *dir_out, size_t dir_sz, char *err, size_t errsz)
{
    const char *env_dir = getenv("MILK_CALC_CACHE");
    if (env_dir != NULL && env_dir[0] != '\0')
    {
        snprintf(dir_out, dir_sz, "%s", env_dir);
    }
    else
    {
        const char *tmp = getenv("TMPDIR");
        if (tmp == NULL || tmp[0] == '\0')
        {
            tmp = "/tmp";
        }
        snprintf(dir_out, dir_sz, "%s/milk-calc-cache-%u", tmp, (unsigned) geteuid());
    }

    struct stat st;
    if (lstat(dir_out, &st) != 0)
    {
        if (mkdir(dir_out, 0700) != 0 && errno != EEXIST)
        {
            snprintf(err, errsz, "cannot create cache dir '%s': %s", dir_out, strerror(errno));
            return -1;
        }
        if (lstat(dir_out, &st) != 0)
        {
            snprintf(err, errsz, "cannot stat cache dir '%s': %s", dir_out, strerror(errno));
            return -1;
        }
    }
    if (!S_ISDIR(st.st_mode) || S_ISLNK(st.st_mode) || st.st_uid != geteuid())
    {
        snprintf(err, errsz, "unsafe cache directory '%s'", dir_out);
        return -1;
    }
    if ((st.st_mode & 0777) != 0700)
    {
        (void) chmod(dir_out, 0700);
    }
    return 0;
}

typedef struct
{
    char   name[64];
    time_t mtime;
} prune_item_t;

static int cmp_prune_mtime(const void *a, const void *b)
{
    const prune_item_t *pa = (const prune_item_t *) a;
    const prune_item_t *pb = (const prune_item_t *) b;
    return (pa->mtime > pb->mtime) - (pa->mtime < pb->mtime);
}

static void maybe_prune_cache(const char *dir_path)
{
    DIR *d = opendir(dir_path);
    if (d == NULL)
    {
        return;
    }
    int           cap = 256, cnt = 0;
    prune_item_t *items = malloc((size_t) cap * sizeof(prune_item_t));
    if (items == NULL)
    {
        closedir(d);
        return;
    }

    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        size_t len = strlen(de->d_name);
        if (len != 35 || strcmp(de->d_name + 32, ".so") != 0)
        {
            continue;
        }
        if (cnt >= cap)
        {
            cap *= 2;
            prune_item_t *ni = realloc(items, (size_t) cap * sizeof(prune_item_t));
            if (ni == NULL)
            {
                break;
            }
            items = ni;
        }
        char        p[600];
        struct stat st;
        snprintf(p, sizeof(p), "%s/%s", dir_path, de->d_name);
        if (stat(p, &st) == 0)
        {
            snprintf(items[cnt].name, sizeof(items[cnt].name), "%.32s", de->d_name);
            items[cnt].mtime = st.st_mtime;
            cnt++;
        }
    }
    closedir(d);

    if (cnt > MC_DISK_MAX_ENTRIES)
    {
        qsort(items, (size_t) cnt, sizeof(prune_item_t), cmp_prune_mtime);
        int to_del = (cnt - MC_DISK_MAX_ENTRIES) + MC_DISK_PRUNE_COUNT;
        if (to_del > cnt)
        {
            to_del = cnt;
        }
        for (int i = 0; i < to_del; i++)
        {
            char p[600];
            snprintf(p, sizeof(p), "%s/%s.so", dir_path, items[i].name);
            (void) unlink(p);
            snprintf(p, sizeof(p), "%s/%s.c", dir_path, items[i].name);
            (void) unlink(p);
        }
    }
    free(items);
}

mc_kernel_fn mc_cc_compile(const char *c_src,
                           char       *so_path_out,
                           size_t      so_path_sz,
                           char       *err,
                           size_t      errsz)
{
    const char *cc = get_compiler();
    uint64_t    h1 = 0, h2 = 0;
    hash128(cc, c_src, &h1, &h2);

    pthread_mutex_lock(&g_cc_lock);
    for (int i = 0; i < g_mem_count; i++)
    {
        if (g_mem_cache[i].h1 == h1 && g_mem_cache[i].h2 == h2)
        {
            if (so_path_out != NULL && so_path_sz > 0)
            {
                snprintf(so_path_out, so_path_sz, "%s", g_mem_cache[i].so_path);
            }
            mc_kernel_fn fn = g_mem_cache[i].fn;
            pthread_mutex_unlock(&g_cc_lock);
            return fn;
        }
    }
    unsigned long seq = ++g_tmp_seq;
    pthread_mutex_unlock(&g_cc_lock);

    char cache_dir[384];
    if (ensure_cache_dir(cache_dir, sizeof(cache_dir), err, errsz) != 0)
    {
        return NULL;
    }

    char key[33];
    snprintf(key, sizeof(key), "%016llx%016llx", (unsigned long long) h1, (unsigned long long) h2);

    char so_path[512], c_path[512];
    snprintf(so_path, sizeof(so_path), "%s/%s.so", cache_dir, key);
    snprintf(c_path, sizeof(c_path), "%s/%s.c", cache_dir, key);

    if (access(so_path, R_OK) != 0)
    {
        char c_tmp[544], so_tmp[544], err_tmp[544];
        int  pid = (int) getpid();
        snprintf(c_tmp, sizeof(c_tmp), "%s/%s.c.tmp.%d.%lu", cache_dir, key, pid, seq);
        snprintf(so_tmp, sizeof(so_tmp), "%s/%s.so.tmp.%d.%lu", cache_dir, key, pid, seq);
        snprintf(err_tmp, sizeof(err_tmp), "%s/%s.err.tmp.%d.%lu", cache_dir, key, pid, seq);

        FILE *fp = fopen(c_tmp, "w");
        if (fp == NULL)
        {
            snprintf(err, errsz, "cannot write kernel source '%s': %s", c_tmp, strerror(errno));
            return NULL;
        }
        size_t slen = strlen(c_src);
        if (fwrite(c_src, 1, slen, fp) != slen || fclose(fp) != 0)
        {
            (void) unlink(c_tmp);
            snprintf(err, errsz, "failed writing kernel source '%s'", c_tmp);
            return NULL;
        }
        (void) rename(c_tmp, c_path);

        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, err_tmp, O_WRONLY | O_CREAT | O_TRUNC,
                                         0600);

        char *argv[] = { (char *) cc,
                         "-O3",
                         "-march=native",
                         "-ffast-math",
                         "-fno-finite-math-only",
                         "-fno-associative-math",
                         "-mno-recip",
                         "-fopenmp-simd",
                         "-fPIC",
                         "-shared",
                         "-std=gnu11",
                         "-x",
                         "c",
                         c_path,
                         "-lm",
                         "-o",
                         so_tmp,
                         NULL };

        pid_t child = 0;
        int sp_rc = (strchr(cc, '/') != NULL) ? posix_spawn(&child, cc, &fa, NULL, argv, environ)
                                              : posix_spawnp(&child, cc, &fa, NULL, argv, environ);
        posix_spawn_file_actions_destroy(&fa);

        int wstatus = 0;
        if (sp_rc == 0)
        {
            while (waitpid(child, &wstatus, 0) < 0 && errno == EINTR)
            {
            }
        }
        if (sp_rc != 0 || !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0)
        {
            char   ebuf[256] = { 0 };
            FILE  *ef        = fopen(err_tmp, "r");
            size_t nr        = (ef != NULL) ? fread(ebuf, 1, sizeof(ebuf) - 1, ef) : 0;
            if (ef != NULL)
            {
                fclose(ef);
            }
            ebuf[nr] = '\0';
            (void) unlink(err_tmp);
            (void) unlink(so_tmp);
            snprintf(err, errsz, "C kernel compiler '%s' failed: %s", cc,
                     ebuf[0] ? ebuf : strerror(sp_rc ? sp_rc : EINVAL));
            return NULL;
        }
        (void) unlink(err_tmp);
        if (rename(so_tmp, so_path) != 0)
        {
            (void) unlink(so_tmp);
            snprintf(err, errsz, "rename '%s' -> '%s' failed: %s", so_tmp, so_path,
                     strerror(errno));
            return NULL;
        }
        maybe_prune_cache(cache_dir);
    }

    void *dl = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (dl == NULL)
    {
        snprintf(err, errsz, "dlopen('%s') failed: %s", so_path, dlerror());
        return NULL;
    }
    mc_kernel_fn fn  = NULL;
    *(void **) (&fn) = dlsym(dl, "mc_k");
    if (fn == NULL)
    {
        dlclose(dl);
        snprintf(err, errsz, "dlsym('mc_k') failed in '%s'", so_path);
        return NULL;
    }

    pthread_mutex_lock(&g_cc_lock);
    int idx = (g_mem_count < MC_MEM_CACHE_CAP) ? g_mem_count++ : (g_mem_hand++ % MC_MEM_CACHE_CAP);
    g_mem_cache[idx].h1 = h1;
    g_mem_cache[idx].h2 = h2;
    g_mem_cache[idx].fn = fn;
    snprintf(g_mem_cache[idx].so_path, sizeof(g_mem_cache[idx].so_path), "%s", so_path);
    pthread_mutex_unlock(&g_cc_lock);

    if (so_path_out != NULL && so_path_sz > 0)
    {
        snprintf(so_path_out, so_path_sz, "%s", so_path);
    }
    return fn;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#if defined(__x86_64__) || defined(__i386__)
#    include <cpuid.h>
#endif
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

#define MC_CC_FLAGS_ID                                                                            \
    "-O3 -pipe -march=native -ffast-math -fno-finite-math-only -fno-associative-math -mno-recip " \
    "-fopenmp-simd -fPIC -shared -std=gnu11"

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

static pthread_mutex_t g_cc_lock  = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_env_once = PTHREAD_ONCE_INIT;
static uint64_t        g_env_h1   = 0;
static uint64_t        g_env_h2   = 0;
static mc_mem_entry_t  g_mem_cache[MC_MEM_CACHE_CAP];
static int             g_mem_hand = 0;
static unsigned long   g_tmp_seq  = 0;

static void hash_bytes(uint64_t *io_h1, uint64_t *io_h2, const void *buf, size_t len)
{
    const unsigned char *p  = (const unsigned char *) buf;
    uint64_t             h1 = *io_h1;
    uint64_t             h2 = *io_h2;
    for (size_t i = 0; i < len; i++)
    {
        h1 = (h1 ^ p[i]) * 0x100000001b3ULL;
        h2 = (h2 ^ (uint64_t) (p[i] ^ 0x5aU)) * 0x9e3779b97f4a7c15ULL;
    }
    *io_h1 = h1;
    *io_h2 = h2;
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

static void init_env_hash(void)
{
    uint64_t h1 = 0xcbf29ce484222325ULL ^ (uint64_t) MC_PRELUDE_VERSION;
    uint64_t h2 = 0x84222325cbf29ce4ULL ^ ((uint64_t) MC_PRELUDE_VERSION << 32);

    hash_bytes(&h1, &h2, MC_CC_FLAGS_ID, sizeof(MC_CC_FLAGS_ID) - 1);

#if defined(__x86_64__) || defined(__i386__)
    {
        unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        {
            uint32_t regs1[4] = { eax, ebx, ecx, edx };
            hash_bytes(&h1, &h2, regs1, sizeof(regs1));
        }
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        {
            uint32_t regs7[4] = { eax, ebx, ecx, edx };
            hash_bytes(&h1, &h2, regs7, sizeof(regs7));
        }
    }
#endif

    const char *cc = get_compiler();
    struct stat st;
    if (strchr(cc, '/') != NULL)
    {
        if (stat(cc, &st) == 0)
        {
            uint64_t meta[3] = { (uint64_t) st.st_mtime, (uint64_t) st.st_size,
                                 (uint64_t) st.st_ino };
            hash_bytes(&h1, &h2, meta, sizeof(meta));
        }
    }
    else
    {
        const char *path_env = getenv("PATH");
        if (path_env != NULL)
        {
            const char *cur = path_env;
            while (*cur != '\0')
            {
                const char *sep = strchr(cur, ':');
                size_t      len = (sep != NULL) ? (size_t) (sep - cur) : strlen(cur);
                char        cand[512];
                if (len > 0 && len + strlen(cc) + 2 < sizeof(cand))
                {
                    snprintf(cand, sizeof(cand), "%.*s/%s", (int) len, cur, cc);
                    if (stat(cand, &st) == 0 && (st.st_mode & S_IXUSR) != 0)
                    {
                        uint64_t meta[3] = { (uint64_t) st.st_mtime, (uint64_t) st.st_size,
                                             (uint64_t) st.st_ino };
                        hash_bytes(&h1, &h2, meta, sizeof(meta));
                        break;
                    }
                }
                if (sep == NULL)
                {
                    break;
                }
                cur = sep + 1;
            }
        }
    }

    g_env_h1 = h1;
    g_env_h2 = h2;
}

static void hash128(const char *cc, const char *src, uint64_t *out_h1, uint64_t *out_h2)
{
    pthread_once(&g_env_once, init_env_hash);
    uint64_t h1 = g_env_h1;
    uint64_t h2 = g_env_h2;

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

static mc_kernel_fn mem_cache_lookup(uint64_t h1, uint64_t h2, char *so_path_out, size_t so_path_sz)
{
    uint32_t mask = (uint32_t) (MC_MEM_CACHE_CAP - 1);
    uint32_t base = (uint32_t) h1 & mask;
    for (uint32_t p = 0; p < 8; p++)
    {
        uint32_t idx = (base + p) & mask;
        if (g_mem_cache[idx].fn == NULL)
        {
            return NULL;
        }
        if (g_mem_cache[idx].h1 == h1 && g_mem_cache[idx].h2 == h2)
        {
            if (so_path_out != NULL && so_path_sz > 0)
            {
                snprintf(so_path_out, so_path_sz, "%s", g_mem_cache[idx].so_path);
            }
            return g_mem_cache[idx].fn;
        }
    }
    return NULL;
}

static void mem_cache_insert(uint64_t h1, uint64_t h2, mc_kernel_fn fn, const char *so_path)
{
    uint32_t mask   = (uint32_t) (MC_MEM_CACHE_CAP - 1);
    uint32_t base   = (uint32_t) h1 & mask;
    uint32_t target = (base + ((uint32_t) (g_mem_hand++) & 7U)) & mask;
    for (uint32_t p = 0; p < 8; p++)
    {
        uint32_t idx = (base + p) & mask;
        if (g_mem_cache[idx].fn == NULL || (g_mem_cache[idx].h1 == h1 && g_mem_cache[idx].h2 == h2))
        {
            target = idx;
            break;
        }
    }
    g_mem_cache[target].h1 = h1;
    g_mem_cache[target].h2 = h2;
    g_mem_cache[target].fn = fn;
    snprintf(g_mem_cache[target].so_path, sizeof(g_mem_cache[target].so_path), "%s", so_path);
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

static int write_source_atomically(const char   *cache_dir,
                                   const char   *key,
                                   unsigned long seq,
                                   const char   *src,
                                   size_t        slen,
                                   char         *c_path_out,
                                   size_t        c_path_sz,
                                   char         *err,
                                   size_t        errsz)
{
    char c_tmp[544];
    snprintf(c_path_out, c_path_sz, "%s/%s.c", cache_dir, key);
    snprintf(c_tmp, sizeof(c_tmp), "%s/%s.c.tmp.%d.%lu", cache_dir, key, (int) getpid(), seq);

    FILE *fp = fopen(c_tmp, "w");
    if (fp == NULL)
    {
        snprintf(err, errsz, "cannot write kernel source '%s': %s", c_tmp, strerror(errno));
        return -1;
    }
    if (fwrite(src, 1, slen, fp) != slen || fclose(fp) != 0)
    {
        (void) unlink(c_tmp);
        snprintf(err, errsz, "failed writing kernel source '%s'", c_tmp);
        return -1;
    }
    (void) rename(c_tmp, c_path_out);
    return 0;
}

static int run_compiler(const char *cc,
                        const char *c_path,
                        const char *so_tmp,
                        const char *err_tmp,
                        char       *err,
                        size_t      errsz)
{
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, err_tmp, O_WRONLY | O_CREAT | O_TRUNC,
                                     0600);

    char *argv[] = { (char *) cc,
                     "-O3",
                     "-pipe",
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
                     (char *) c_path,
                     "-lm",
                     "-o",
                     (char *) so_tmp,
                     NULL };

    pid_t child = 0;
    int   sp_rc = (strchr(cc, '/') != NULL) ? posix_spawn(&child, cc, &fa, NULL, argv, environ)
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
        return -1;
    }
    (void) unlink(err_tmp);
    return 0;
}

static mc_kernel_fn load_so_kernel(const char *so_path, const char *key, char *err, size_t errsz)
{
    void *dl = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (dl == NULL)
    {
        snprintf(err, errsz, "dlopen('%s') failed: %s", so_path, dlerror());
        return NULL;
    }
    mc_kernel_fn fn  = NULL;
    *(void **) (&fn) = dlsym(dl, "mc_k");
    if (fn == NULL && key != NULL)
    {
        char sym[64];
        snprintf(sym, sizeof(sym), "mc_k_%s", key);
        *(void **) (&fn) = dlsym(dl, sym);
    }
    if (fn == NULL)
    {
        dlclose(dl);
        snprintf(err, errsz, "dlsym('mc_k') failed in '%s'", so_path);
        return NULL;
    }
    return fn;
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
    mc_kernel_fn cached_fn = mem_cache_lookup(h1, h2, so_path_out, so_path_sz);
    if (cached_fn != NULL)
    {
        pthread_mutex_unlock(&g_cc_lock);
        return cached_fn;
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

    if (access(so_path, R_OK) != 0)
    {
        if (write_source_atomically(cache_dir, key, seq, c_src, strlen(c_src), c_path,
                                    sizeof(c_path), err, errsz) != 0)
        {
            return NULL;
        }
        char so_tmp[544], err_tmp[544];
        int  pid = (int) getpid();
        snprintf(so_tmp, sizeof(so_tmp), "%s/%s.so.tmp.%d.%lu", cache_dir, key, pid, seq);
        snprintf(err_tmp, sizeof(err_tmp), "%s/%s.err.tmp.%d.%lu", cache_dir, key, pid, seq);

        if (run_compiler(cc, c_path, so_tmp, err_tmp, err, errsz) != 0)
        {
            return NULL;
        }
        if (rename(so_tmp, so_path) != 0)
        {
            (void) unlink(so_tmp);
            snprintf(err, errsz, "rename '%s' -> '%s' failed: %s", so_tmp, so_path,
                     strerror(errno));
            return NULL;
        }
        maybe_prune_cache(cache_dir);
    }

    mc_kernel_fn fn = load_so_kernel(so_path, key, err, errsz);
    if (fn == NULL)
    {
        return NULL;
    }

    pthread_mutex_lock(&g_cc_lock);
    mem_cache_insert(h1, h2, fn, so_path);
    pthread_mutex_unlock(&g_cc_lock);

    if (so_path_out != NULL && so_path_sz > 0)
    {
        snprintf(so_path_out, so_path_sz, "%s", so_path);
    }
    return fn;
}

typedef struct
{
    mc_pass_t *pa;
    uint64_t   h1;
    uint64_t   h2;
    char       key[33];
    char       so_path[512];
    int        uniq_idx;
} pass_job_t;

int mc_cc_compile_prog(mcalc_prog_t *p, char *err, size_t errsz)
{
    int npass = 0;
    for (int st = 0; st < p->nsteps; st++)
    {
        if (p->steps[st].kind == MC_STEP_PASS)
        {
            npass++;
        }
    }
    if (npass == 0)
    {
        return 0;
    }
    if (npass == 1)
    {
        for (int st = 0; st < p->nsteps; st++)
        {
            if (p->steps[st].kind == MC_STEP_PASS)
            {
                mc_pass_t *pa = &p->steps[st].pass;
                pa->fn = mc_cc_compile(pa->c_src, pa->so_path, sizeof(pa->so_path), err, errsz);
                return (pa->fn != NULL) ? 0 : -1;
            }
        }
    }

    pass_job_t  stack_jobs[16];
    pass_job_t *jobs = (npass <= 16) ? stack_jobs : calloc((size_t) npass, sizeof(pass_job_t));
    if (jobs == NULL)
    {
        snprintf(err, errsz, "out of memory compiling passes");
        return -1;
    }

    int         rc        = -1;
    char       *batch_src = NULL;
    const char *cc        = get_compiler();
    int         j         = 0;
    for (int st = 0; st < p->nsteps; st++)
    {
        if (p->steps[st].kind != MC_STEP_PASS)
        {
            continue;
        }
        jobs[j].pa       = &p->steps[st].pass;
        jobs[j].uniq_idx = -1;
        hash128(cc, jobs[j].pa->c_src, &jobs[j].h1, &jobs[j].h2);
        snprintf(jobs[j].key, sizeof(jobs[j].key), "%016llx%016llx",
                 (unsigned long long) jobs[j].h1, (unsigned long long) jobs[j].h2);
        j++;
    }

    int           rem = 0;
    unsigned long seq = 0;
    pthread_mutex_lock(&g_cc_lock);
    for (int i = 0; i < npass; i++)
    {
        jobs[i].pa->fn = mem_cache_lookup(jobs[i].h1, jobs[i].h2, jobs[i].pa->so_path,
                                          sizeof(jobs[i].pa->so_path));
        if (jobs[i].pa->fn == NULL)
        {
            rem++;
        }
    }
    if (rem > 0)
    {
        seq = ++g_tmp_seq;
    }
    pthread_mutex_unlock(&g_cc_lock);

    if (rem == 0)
    {
        rc = 0;
        goto cleanup;
    }

    char cache_dir[384];
    if (ensure_cache_dir(cache_dir, sizeof(cache_dir), err, errsz) != 0)
    {
        goto cleanup;
    }

    int nmiss = 0;
    for (int i = 0; i < npass; i++)
    {
        if (jobs[i].pa->fn != NULL)
        {
            continue;
        }
        snprintf(jobs[i].so_path, sizeof(jobs[i].so_path), "%s/%s.so", cache_dir, jobs[i].key);
        int dup_of = -1;
        for (int k = 0; k < i; k++)
        {
            if (jobs[k].h1 == jobs[i].h1 && jobs[k].h2 == jobs[i].h2)
            {
                dup_of = k;
                break;
            }
        }
        if (dup_of >= 0)
        {
            if (jobs[dup_of].pa->fn != NULL)
            {
                jobs[i].pa->fn = jobs[dup_of].pa->fn;
                snprintf(jobs[i].pa->so_path, sizeof(jobs[i].pa->so_path), "%s",
                         jobs[dup_of].pa->so_path);
            }
            else
            {
                jobs[i].uniq_idx = jobs[dup_of].uniq_idx;
            }
            continue;
        }
        if (access(jobs[i].so_path, R_OK) == 0)
        {
            mc_kernel_fn fn = load_so_kernel(jobs[i].so_path, jobs[i].key, err, errsz);
            if (fn == NULL)
            {
                goto cleanup;
            }
            pthread_mutex_lock(&g_cc_lock);
            mem_cache_insert(jobs[i].h1, jobs[i].h2, fn, jobs[i].so_path);
            pthread_mutex_unlock(&g_cc_lock);
            jobs[i].pa->fn = fn;
            snprintf(jobs[i].pa->so_path, sizeof(jobs[i].pa->so_path), "%s", jobs[i].so_path);
            continue;
        }
        jobs[i].uniq_idx = nmiss++;
    }

    if (nmiss == 0)
    {
        rc = 0;
        goto cleanup;
    }
    if (nmiss == 1)
    {
        for (int i = 0; i < npass; i++)
        {
            if (jobs[i].pa->fn == NULL && jobs[i].uniq_idx == 0)
            {
                mc_kernel_fn fn = mc_cc_compile(jobs[i].pa->c_src, jobs[i].pa->so_path,
                                                sizeof(jobs[i].pa->so_path), err, errsz);
                if (fn == NULL)
                {
                    goto cleanup;
                }
                for (int k = i; k < npass; k++)
                {
                    if (jobs[k].h1 == jobs[i].h1 && jobs[k].h2 == jobs[i].h2)
                    {
                        jobs[k].pa->fn = fn;
                        snprintf(jobs[k].pa->so_path, sizeof(jobs[k].pa->so_path), "%s",
                                 jobs[i].pa->so_path);
                    }
                }
                break;
            }
        }
        rc = 0;
        goto cleanup;
    }

    /* Batch compile >= 2 unique cache-missing passes in a single compiler invocation */
    static const char *const k_sp_defs[10] = {
        "MC_DEF_SPATIAL(u8, uint8_t)",   "MC_DEF_SPATIAL(i8, int8_t)",
        "MC_DEF_SPATIAL(u16, uint16_t)", "MC_DEF_SPATIAL(i16, int16_t)",
        "MC_DEF_SPATIAL(u32, uint32_t)", "MC_DEF_SPATIAL(i32, int32_t)",
        "MC_DEF_SPATIAL(u64, uint64_t)", "MC_DEF_SPATIAL(i64, int64_t)",
        "MC_DEF_SPATIAL(f32, float)",    "MC_DEF_SPATIAL(f64, double)"
    };
    int    need_complex = 0, need_coord = 0, need_spatial = 0, sp_used[10] = { 0 };
    int    last_miss_i = -1;
    size_t batch_cap   = sizeof(MC_PRELUDE_CORE_SRC) + sizeof(MC_PRELUDE_COMPLEX_SRC) +
                         sizeof(MC_PRELUDE_COORD_SHIFT_SRC) + sizeof(MC_PRELUDE_SPATIAL_MACRO_SRC) +
                         sizeof(MC_PRELUDE_SPATIAL_ALL_SRC) + 256;
    for (int i = 0; i < npass; i++)
    {
        if (jobs[i].pa->fn == NULL && jobs[i].uniq_idx >= 0)
        {
            const char *src = jobs[i].pa->c_src;
            batch_cap += strlen(src) + 128;
            need_complex |= (strstr(src, "<complex.h>") != NULL);
            need_coord |= (strstr(src, "mc_cl(") != NULL);
            for (int d = 0; d < 10; d++)
            {
                if (strstr(src, k_sp_defs[d]) != NULL)
                {
                    sp_used[d]   = 1;
                    need_spatial = 1;
                }
            }
        }
    }

    batch_src = malloc(batch_cap);
    if (batch_src == NULL)
    {
        snprintf(err, errsz, "out of memory building batch kernel source");
        goto cleanup;
    }

    size_t blen = (size_t) snprintf(batch_src, batch_cap, "%s%s%s%s", MC_PRELUDE_CORE_SRC,
                                    need_complex ? MC_PRELUDE_COMPLEX_SRC : "",
                                    (need_coord || need_spatial) ? MC_PRELUDE_COORD_SHIFT_SRC : "",
                                    need_spatial ? MC_PRELUDE_SPATIAL_MACRO_SRC : "");
    for (int d = 0; d < 10; d++)
    {
        if (sp_used[d])
        {
            blen += (size_t) snprintf(batch_src + blen, batch_cap - blen, "%s\n", k_sp_defs[d]);
        }
    }

    for (int u = 0; u < nmiss; u++)
    {
        for (int i = 0; i < npass; i++)
        {
            if (jobs[i].pa->fn == NULL && jobs[i].uniq_idx == u)
            {
                last_miss_i = i;
                char c_path[512];
                if (write_source_atomically(cache_dir, jobs[i].key, seq + (unsigned long) u,
                                            jobs[i].pa->c_src, strlen(jobs[i].pa->c_src), c_path,
                                            sizeof(c_path), err, errsz) != 0)
                {
                    goto cleanup;
                }
                const char *fn_pos = strstr(jobs[i].pa->c_src, "void mc_k(");
                const char *body   = (fn_pos != NULL) ? fn_pos : jobs[i].pa->c_src;
                blen +=
                    (size_t) snprintf(batch_src + blen, batch_cap - blen,
                                      "#define mc_k mc_k_%s\n%s\n#undef mc_k\n", jobs[i].key, body);
                break;
            }
        }
    }

    int  pid = (int) getpid();
    char batch_c[544], batch_so_tmp[544], err_tmp[544];
    snprintf(batch_c, sizeof(batch_c), "%s/batch.c.tmp.%d.%lu", cache_dir, pid, seq);
    snprintf(batch_so_tmp, sizeof(batch_so_tmp), "%s/batch.so.tmp.%d.%lu", cache_dir, pid, seq);
    snprintf(err_tmp, sizeof(err_tmp), "%s/batch.err.tmp.%d.%lu", cache_dir, pid, seq);

    FILE *bfp = fopen(batch_c, "w");
    if (bfp == NULL || fwrite(batch_src, 1, blen, bfp) != blen || fclose(bfp) != 0)
    {
        if (bfp != NULL)
        {
            fclose(bfp);
        }
        (void) unlink(batch_c);
        snprintf(err, errsz, "failed writing batch kernel source '%s'", batch_c);
        goto cleanup;
    }
    free(batch_src);
    batch_src = NULL;

    int cc_rc = run_compiler(cc, batch_c, batch_so_tmp, err_tmp, err, errsz);
    (void) unlink(batch_c);
    if (cc_rc != 0)
    {
        goto cleanup;
    }

    for (int u = 0; u < nmiss; u++)
    {
        for (int i = 0; i < npass; i++)
        {
            if (jobs[i].pa->fn == NULL && jobs[i].uniq_idx == u)
            {
                if (i != last_miss_i)
                {
                    (void) unlink(jobs[i].so_path);
                    if (link(batch_so_tmp, jobs[i].so_path) != 0)
                    {
                        (void) unlink(batch_so_tmp);
                        snprintf(err, errsz, "link '%s' -> '%s' failed: %s", batch_so_tmp,
                                 jobs[i].so_path, strerror(errno));
                        goto cleanup;
                    }
                }
                break;
            }
        }
    }
    if (rename(batch_so_tmp, jobs[last_miss_i].so_path) != 0)
    {
        (void) unlink(batch_so_tmp);
        snprintf(err, errsz, "rename '%s' -> '%s' failed: %s", batch_so_tmp,
                 jobs[last_miss_i].so_path, strerror(errno));
        goto cleanup;
    }
    maybe_prune_cache(cache_dir);

    void *dl = dlopen(jobs[last_miss_i].so_path, RTLD_LAZY | RTLD_LOCAL);
    if (dl == NULL)
    {
        snprintf(err, errsz, "dlopen('%s') failed: %s", jobs[last_miss_i].so_path, dlerror());
        goto cleanup;
    }

    for (int u = 0; u < nmiss; u++)
    {
        for (int i = 0; i < npass; i++)
        {
            if (jobs[i].pa->fn == NULL && jobs[i].uniq_idx == u)
            {
                char sym[64];
                snprintf(sym, sizeof(sym), "mc_k_%s", jobs[i].key);
                mc_kernel_fn fn  = NULL;
                *(void **) (&fn) = dlsym(dl, sym);
                if (fn == NULL)
                {
                    dlclose(dl);
                    snprintf(err, errsz, "dlsym('%s') failed in '%s'", sym,
                             jobs[last_miss_i].so_path);
                    goto cleanup;
                }
                pthread_mutex_lock(&g_cc_lock);
                mem_cache_insert(jobs[i].h1, jobs[i].h2, fn, jobs[i].so_path);
                pthread_mutex_unlock(&g_cc_lock);
                for (int k = i; k < npass; k++)
                {
                    if (jobs[k].h1 == jobs[i].h1 && jobs[k].h2 == jobs[i].h2)
                    {
                        jobs[k].pa->fn = fn;
                        snprintf(jobs[k].pa->so_path, sizeof(jobs[k].pa->so_path), "%s",
                                 jobs[i].so_path);
                    }
                }
                break;
            }
        }
    }
    rc = 0;

cleanup:
    free(batch_src);
    if (jobs != stack_jobs)
    {
        free(jobs);
    }
    return rc;
}

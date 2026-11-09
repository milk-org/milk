// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_cc.c
 * @brief C compiler driver (JIT kernels and `-l` extensions) and two-tier (memory + disk) `.so`
 *        cache for libmilkcalc.
 *
 * Purpose:     compile each generated kernel once for a given source, compiler, flags and CPU, and
 *              reuse it in later runs and in other processes.
 * Does:        mc_cc_compile_prog() gives each pass of a planned program its kernel: from the
 *              in-memory table (4096 slots, 8-way probing), from <cache>/<key>.so, or compiled; two
 *              or more misses are compiled in one batch file (each body under
 *              `#define mc_k mc_k_<key>`, one compiler run), whose .so is then hard-linked to each
 *              <key>.so. The key is a 128-bit FNV-1a-style hash (MurmurHash3 finalizer) of the
 *              source and the compiler name, from a seed computed once: MC_PRELUDE_VERSION,
 *              MILK_CC_FLAGS, the JIT's own flags, the CPUID feature bits (the flags hold
 *              -march=native) and the compiler binary's mtime, size and inode; the part of the hash
 *              over the core prelude, which starts every kernel, is also computed once. The cache
 *              directory, $MILK_CALC_CACHE or $TMPDIR/milk-calc-cache-<euid>, must be a real
 *              directory owned by the user (mode 0700); files are written under a temporary name
 *              and renamed, so no process loads a partial file; beyond 4096 .so files the oldest
 *              are deleted. mc_cc_build(), mc_cc_cache_dir() and mc_cc_key() are also the compile
 *              step, cache and key of `-l` C extensions.
 * Used by:     plan/mc_plan.c (mc_cc_compile_prog(), after code generation), ext/mc_ext_loader.c
 *              (`-l` extensions). Declared in plan/mc_plan.h.
 * Uses:        jit/mc_prelude.h (prelude sections, MC_PRELUDE_VERSION), milk_spawn.h (runs the
 *              compiler without a shell), dlopen(); MILK_CC_FLAGS and MILK_DEFAULT_CC, set by CMake
 *              ($MILK_CALC_CC overrides the compiler).
 * See also:    jit/mc_emit.c (writes the sources); scripts/milk-cc (same flags);
 *              docs/guide/calculator.md, "Fusion & Cost Model" and
 *              "Auto-Compilation and Caching (`-l` / `--lib`)".
 */

#if defined(__x86_64__) || defined(__i386__)
#    include <cpuid.h>
#endif
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "libmilkcommon/milk_spawn.h"
#include "plan/mc_plan.h"
#include "jit/mc_prelude.h"

#ifndef MILK_DEFAULT_CC
#    define MILK_DEFAULT_CC "gcc"
#endif

/* MILK_CC_FLAGS is the run-time compile flag list, set once in CMakeLists.txt (milk-cc uses it too) */
#ifndef MILK_CC_FLAGS
#    error "MILK_CC_FLAGS is set by CMakeLists.txt"
#endif

/* The JIT adds these: no reassociation, so kernels round as the scalar evaluator does, and its
 * generated code (also the batch file, which has no .c suffix) is GNU C11 */
static const char *const k_jit_args[] = { "-fno-associative-math", "-std=gnu11", "-x", "c", NULL };

#define MC_MEM_CACHE_CAP 4096
#define MC_DISK_MAX_ENTRIES 4096
#define MC_DISK_PRUNE_COUNT 512

/* ========================================================================== */
/* In-Memory Cache State & 128-Bit Environment Hashing                        */
/* ========================================================================== */

/**
 * struct mc_mem_entry_t - Entry in the in-memory compiled kernel hash table
 * @h1:      Low 64 bits of the 128-bit kernel hash
 * @h2:      High 64 bits of the 128-bit kernel hash
 * @fn:      Loaded `mc_kernel_fn` function pointer (NULL if slot is empty)
 * @so_path: Filesystem path to the backing `.so` file in the disk cache
 */
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

/**
 * hash_bytes - Mix an arbitrary byte buffer into a pair of 64-bit hash accumulators
 * @io_h1: First 64-bit hash state (updated in place)
 * @io_h2: Second 64-bit hash state (updated in place)
 * @buf:   Input byte buffer
 * @len:   Number of bytes to hash
 *
 * Uses two independent 64-bit FNV/golden-ratio multipliers so @io_h1 and @io_h2
 * accumulate uncorrelated 64-bit lanes.
 *
 * Return: void
 */
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

/**
 * get_compiler - Return the C compiler command or path to invoke
 *
 * Checks the `MILK_CALC_CC` environment variable first, falling back to the
 * build-time `MILK_DEFAULT_CC` (the compiler milk was built with).
 *
 * Return: Non-NULL compiler command string
 */
static const char *get_compiler(void)
{
    const char *env_cc = getenv("MILK_CALC_CC");
    if (env_cc != NULL && env_cc[0] != '\0')
    {
        return env_cc;
    }
    return MILK_DEFAULT_CC;
}

/**
 * init_env_hash - Compute the one-time environment seed for kernel cache keys
 *
 * Invoked via `pthread_once(&g_env_once, init_env_hash)`. Mixes:
 * - `MC_PRELUDE_VERSION`, `MILK_CC_FLAGS` and `k_jit_args`
 * - Host CPUID feature registers (leaves 1 and 7 on x86/x86_64)
 * - Compiler executable `(st_mtime, st_size, st_ino)` resolved directly or via `$PATH`
 *
 * This guarantees that upgrading GCC, modifying prelude macros, or migrating a
 * shared cache across different CPU microarchitectures automatically invalidates
 * stale `.so` cache entries.
 *
 * Return: void
 */
static void init_env_hash(void)
{
    uint64_t h1 = 0xcbf29ce484222325ULL ^ (uint64_t) MC_PRELUDE_VERSION;
    uint64_t h2 = 0x84222325cbf29ce4ULL ^ ((uint64_t) MC_PRELUDE_VERSION << 32);

    hash_bytes(&h1, &h2, MILK_CC_FLAGS, sizeof(MILK_CC_FLAGS) - 1);
    for (int i = 0; k_jit_args[i] != NULL; i++)
    {
        hash_bytes(&h1, &h2, k_jit_args[i], strlen(k_jit_args[i]) + 1);
    }

#if defined(__x86_64__) || defined(__i386__)
    {
        unsigned eax = 0;
        unsigned ebx = 0;
        unsigned ecx = 0;
        unsigned edx = 0;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        {
            /* EBX[31:24] is the APIC ID of the core this thread runs on, not a CPU feature */
            uint32_t regs1[4] = { eax, ebx & 0x00FFFFFFu, ecx, edx };
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

static void hash_src(uint64_t *h1, uint64_t *h2, const unsigned char *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        *h1 = (*h1 ^ p[i]) * 0x100000001b3ULL;
        *h2 = (*h2 ^ (uint64_t) (p[i] ^ 0x5aU)) * 0x9e3779b97f4a7c15ULL;
    }
}

/* Hash state after the environment and MC_PRELUDE_CORE_SRC, which every kernel source starts with */
static pthread_once_t g_pre_once = PTHREAD_ONCE_INIT;
static uint64_t       g_pre_h1   = 0;
static uint64_t       g_pre_h2   = 0;

static void init_pre_hash(void)
{
    pthread_once(&g_env_once, init_env_hash);
    g_pre_h1 = g_env_h1;
    g_pre_h2 = g_env_h2;
    hash_src(&g_pre_h1, &g_pre_h2, (const unsigned char *) MC_PRELUDE_CORE_SRC,
             sizeof(MC_PRELUDE_CORE_SRC) - 1);
}

/**
 * hash128 - Compute the 128-bit cache key for a compiler command and C source string
 * @cc:     Compiler command string
 * @src:    Complete C kernel source code
 * @out_h1: Output low 64-bit hash lane
 * @out_h2: Output high 64-bit hash lane
 *
 * Seeds from `g_env_h1`/`g_env_h2`, hashes @src then @cc, and finishes with a
 * MurmurHash3 64-bit avalanche finalizer on both lanes. The state after
 * `MC_PRELUDE_CORE_SRC`, which every kernel source starts with, is computed once.
 *
 * Return: void
 */
static void hash128(const char *cc, const char *src, uint64_t *out_h1, uint64_t *out_h2)
{
    pthread_once(&g_pre_once, init_pre_hash);
    size_t   plen = sizeof(MC_PRELUDE_CORE_SRC) - 1;
    int      pre = (strncmp(src, MC_PRELUDE_CORE_SRC, plen) == 0); /* hashed once, not per kernel */
    uint64_t h1  = pre ? g_pre_h1 : g_env_h1;
    uint64_t h2  = pre ? g_pre_h2 : g_env_h2;

    src += pre ? plen : 0;
    hash_src(&h1, &h2, (const unsigned char *) src, strlen(src));
    for (const unsigned char *p = (const unsigned char *) cc; *p != '\0'; p++)
    {
        h1 = (h1 ^ *p) * 0x100000001b3ULL;
        h2 = (h2 ^ (*p + 0x9eU)) * 0x100000001b3ULL;
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

/**
 * mc_cc_key - Format the 32-hex-digit cache key of @text
 * @text: Text to key (a C source, or a description of a source file)
 * @key:  Output key
 *
 * The key also covers the compiler, its flags, the CPU and `MC_PRELUDE_VERSION` (init_env_hash()).
 *
 * Return: void
 */
void mc_cc_key(const char *text, char key[33])
{
    uint64_t h1 = 0;
    uint64_t h2 = 0;
    hash128(get_compiler(), text, &h1, &h2);
    snprintf(key, 33, "%016llx%016llx", (unsigned long long) h1, (unsigned long long) h2);
}

/**
 * mem_cache_lookup - Look up a compiled kernel in the Tier-1 in-memory cache
 * @h1:          First 64-bit hash word
 * @h2:          Second 64-bit hash word
 * @so_path_out: Optional buffer receiving the `.so` path on a cache hit
 * @so_path_sz:  Size of @so_path_out in bytes
 *
 * Context: Caller must hold `g_cc_lock`. Probes up to 8 consecutive slots
 * starting at `h1 & (MC_MEM_CACHE_CAP - 1)`.
 *
 * Return: Cached `mc_kernel_fn` on hit, or NULL on miss
 */
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

/**
 * mem_cache_insert - Insert a compiled kernel into the Tier-1 in-memory cache
 * @h1:      First 64-bit hash word
 * @h2:      Second 64-bit hash word
 * @fn:      Resolved `mc_kernel_fn` pointer
 * @so_path: Filesystem path to the compiled `.so`
 *
 * Context: Caller must hold `g_cc_lock`. Uses an 8-way set-associative window
 * with round-robin eviction (`g_mem_hand`) if all 8 probe slots are occupied.
 *
 * Return: void
 */
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

/* ========================================================================== */
/* On-Disk Cache Directory Management & Pruning                               */
/* ========================================================================== */

/**
 * mc_cc_cache_dir - Resolve, create, and validate the on-disk kernel and extension cache directory
 * @dir_out: Output buffer receiving the validated directory path
 * @dir_sz:  Size of @dir_out in bytes
 * @err:     Error message buffer
 * @errsz:   Size of @err in bytes
 *
 * Uses `$MILK_CALC_CACHE` if set, otherwise `$TMPDIR/milk-calc-cache-<euid>`
 * (defaulting to `/tmp`). Verifies with `lstat()` that the path is a real
 * directory (not a symlink) owned by `geteuid()` with mode `0700` to prevent
 * symlink/privilege-escalation attacks in `/tmp`.
 *
 * Return: 0 on success, -1 on error
 */
int mc_cc_cache_dir(char *dir_out, size_t dir_sz, char *err, size_t errsz)
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

/**
 * struct prune_item_t - Disk cache entry candidate for mtime-based eviction
 * @name:  32-character hex key stem (without `.so` / `.c` extension)
 * @mtime: Last modification timestamp (`st_mtime`)
 */
typedef struct
{
    char   name[64];
    time_t mtime;
} prune_item_t;

/**
 * cmp_prune_mtime - `qsort` comparator ordering `prune_item_t` from oldest to newest
 * @a: First `prune_item_t` pointer
 * @b: Second `prune_item_t` pointer
 *
 * Return: -1, 0, or 1
 */
static int cmp_prune_mtime(const void *a, const void *b)
{
    const prune_item_t *pa = (const prune_item_t *) a;
    const prune_item_t *pb = (const prune_item_t *) b;
    return (pa->mtime > pb->mtime) - (pa->mtime < pb->mtime);
}

/**
 * maybe_prune_cache - Evict oldest `.so`/`.c` pairs if the disk cache exceeds capacity
 * @dir_path: Path to the kernel cache directory
 *
 * When the count of `<32-hex>.so` files in @dir_path exceeds `MC_DISK_MAX_ENTRIES`
 * (4096), sorts entries by `st_mtime` and unlinks the oldest entries down to
 * `MC_DISK_MAX_ENTRIES - MC_DISK_PRUNE_COUNT` to amortize directory scans.
 *
 * Return: void
 */
static void maybe_prune_cache(const char *dir_path)
{
    DIR *d = opendir(dir_path);
    if (d == NULL)
    {
        return;
    }
    int           cap   = 256;
    int           cnt   = 0;
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

/* ========================================================================== */
/* Atomic Source Writer, Subprocess Compiler & Dynamic Loader                 */
/* ========================================================================== */

/**
 * write_source_atomically - Write C source to a temporary file and rename into place
 * @cache_dir:  Target cache directory path
 * @key:        32-character hex cache key
 * @seq:        Unique per-invocation sequence number
 * @src:        C source string to write
 * @slen:       Length of @src in bytes
 * @c_path_out: Output buffer receiving the final `<cache_dir>/<key>.c` path
 * @c_path_sz:  Size of @c_path_out in bytes
 * @err:        Error message buffer
 * @errsz:      Size of @err in bytes
 *
 * Writes to `<key>.c.tmp.<pid>.<seq>` first and atomically renames to `<key>.c`
 * so concurrent compilations of the same kernel never truncate an active source
 * file being read by another `gcc` process.
 *
 * Return: 0 on success, -1 on write error
 */
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
    int wok = (fwrite(src, 1, slen, fp) == slen);
    if (fclose(fp) != 0 || !wok || rename(c_tmp, c_path_out) != 0)
    {
        (void) unlink(c_tmp);
        snprintf(err, errsz, "failed writing kernel source '%s'", c_path_out);
        return -1;
    }
    return 0;
}

/**
 * first_error_line - Copy the compiler's first error line (else its first line) from @path
 * @path:  File holding the compiler's stderr
 * @out:   Output buffer (empty when the file is empty or unreadable)
 * @outsz: Size of @out in bytes
 *
 * Return: void
 */
static void first_error_line(const char *path, char *out, size_t outsz)
{
    char   buf[4096];
    FILE  *f = fopen(path, "r");
    size_t n = (f != NULL) ? fread(buf, 1, sizeof(buf) - 1, f) : 0;
    if (f != NULL)
    {
        fclose(f);
    }
    buf[n]   = '\0';
    char *ln = strstr(buf, "error:");
    while (ln != NULL && ln > buf && ln[-1] != '\n')
    {
        ln--;
    }
    ln = (ln != NULL) ? ln : buf;
    snprintf(out, outsz, "%.*s", (int) strcspn(ln, "\n"), ln);
}

/**
 * mc_cc_build - Compile C source file @c_path into shared object @so_path
 * @c_path:  C source file
 * @so_path: Output `.so`, built as `<so_path>.tmp.<pid>.<seq>` and renamed into place
 * @extra:   NULL-terminated arguments placed before @c_path (`-I...`, `-std=...`), or NULL
 * @err:     Error message buffer
 * @errsz:   Size of @err in bytes
 *
 * The one compile step for JIT kernels and `-l` extensions: runs get_compiler() with
 * `MILK_CC_FLAGS -fPIC -shared` through milkspawn (no shell, stdin on /dev/null, no inherited
 * descriptors, EINTR-safe wait) and reports the compiler's first error line on failure.
 *
 * Return: 0 on success, -1 on error
 */
int mc_cc_build(const char        *c_path,
                const char        *so_path,
                const char *const *extra,
                char              *err,
                size_t             errsz)
{
    const char *cc       = get_compiler();
    char        flags[]  = MILK_CC_FLAGS;
    const char *argv[64] = { cc };
    int         ac       = 1;
    char       *save     = NULL;
    for (char *t = strtok_r(flags, " ", &save); t != NULL && ac < 40;
         t       = strtok_r(NULL, " ", &save))
    {
        argv[ac++] = t;
    }
    argv[ac++] = "-fPIC";
    argv[ac++] = "-shared";
    for (int i = 0; extra != NULL && extra[i] != NULL && ac < 58; i++)
    {
        argv[ac++] = extra[i];
    }

    char so_tmp[600];
    char err_tmp[600];
    pthread_mutex_lock(&g_cc_lock);
    unsigned long seq = ++g_tmp_seq;
    pthread_mutex_unlock(&g_cc_lock);
    snprintf(so_tmp, sizeof(so_tmp), "%s.tmp.%d.%lu", so_path, (int) getpid(), seq);
    snprintf(err_tmp, sizeof(err_tmp), "%s.err.tmp.%d.%lu", so_path, (int) getpid(), seq);
    argv[ac++] = c_path;
    argv[ac++] = "-lm";
    argv[ac++] = "-o";
    argv[ac++] = so_tmp;
    argv[ac]   = NULL;

    int efd = open(err_tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (efd < 0)
    {
        snprintf(err, errsz, "cannot create '%s': %s", err_tmp, strerror(errno));
        return -1;
    }
    pid_t pid    = milkspawn_start(argv, -1, efd);
    int   sp_err = errno;
    close(efd);
    int rc = (pid < 0) ? -1 : milkspawn_wait(pid);
    if (rc != 0)
    {
        char line[256];
        first_error_line(err_tmp, line, sizeof(line));
        if (pid < 0)
        {
            snprintf(err, errsz, "cannot run C compiler '%s': %s", cc, strerror(sp_err));
        }
        else if (line[0] != '\0')
        {
            snprintf(err, errsz, "C compiler failed: %s", line);
        }
        else
        {
            snprintf(err, errsz, "C compiler '%s' failed on '%s' (status %d)", cc, c_path, rc);
        }
        (void) unlink(err_tmp);
        (void) unlink(so_tmp);
        return -1;
    }
    (void) unlink(err_tmp);
    if (rename(so_tmp, so_path) != 0)
    {
        snprintf(err, errsz, "cannot rename '%s': %s", so_tmp, strerror(errno));
        (void) unlink(so_tmp);
        return -1;
    }
    return 0;
}

/**
 * load_so_kernel - `dlopen` a compiled `.so` and resolve its kernel entry point
 * @so_path: Path to the shared object file
 * @key:     32-character hex key (used to probe `mc_k_<key>` in batch-compiled `.so` files)
 * @err:     Error message buffer
 * @errsz:   Size of @err in bytes
 *
 * Probes `mc_k` first (single-pass compilation) and falls back to `mc_k_<key>`
 * (when the `.so` was produced by a multi-pass batch compilation and hard-linked
 * into the disk cache).
 *
 * Return: Resolved `mc_kernel_fn` pointer, or NULL on error
 */
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

/* ========================================================================== */
/* Public Single-Pass & Multi-Pass Batch Compilation APIs                     */
/* ========================================================================== */

/**
 * mc_cc_compile - Compile a single C kernel source string (with memory + disk caching)
 * @c_src:       Complete C11 kernel source code
 * @so_path_out: Optional buffer receiving the cached `.so` path
 * @so_path_sz:  Size of @so_path_out in bytes
 * @err:         Error message buffer
 * @errsz:       Size of @err in bytes
 *
 * Checks the Tier-1 in-memory hash table first, then checks the Tier-2 on-disk
 * cache directory for `<key>.so`. On a full cache miss, writes `<key>.c`,
 * invokes the C compiler to produce a temporary `.so`, atomically renames it to
 * `<key>.so`, loads the symbol via `dlopen`/`dlsym`, and inserts it into the
 * in-memory cache.
 *
 * Return: Compiled `mc_kernel_fn` pointer on success, or NULL on error
 */
static mc_kernel_fn mc_cc_compile(const char *c_src,
                                  char       *so_path_out,
                                  size_t      so_path_sz,
                                  char       *err,
                                  size_t      errsz)
{
    const char *cc = get_compiler();
    uint64_t    h1 = 0;
    uint64_t    h2 = 0;
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
    if (mc_cc_cache_dir(cache_dir, sizeof(cache_dir), err, errsz) != 0)
    {
        return NULL;
    }

    char key[33];
    snprintf(key, sizeof(key), "%016llx%016llx", (unsigned long long) h1, (unsigned long long) h2);

    char so_path[512];
    char c_path[512];
    snprintf(so_path, sizeof(so_path), "%s/%s.so", cache_dir, key);

    if (access(so_path, R_OK) != 0)
    {
        if (write_source_atomically(cache_dir, key, seq, c_src, strlen(c_src), c_path,
                                    sizeof(c_path), err, errsz) != 0 ||
            mc_cc_build(c_path, so_path, k_jit_args, err, errsz) != 0)
        {
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

/**
 * struct pass_job_t - Per-pass compilation tracking descriptor for `mc_cc_compile_prog`
 * @pa:       Pointer to the elementwise pass in `p->steps[]`
 * @h1:       Low 64-bit hash of `pa->c_src`
 * @h2:       High 64-bit hash of `pa->c_src`
 * @key:      32-character hex cache key
 * @so_path:  Target `<cache_dir>/<key>.so` path
 * @uniq_idx: Unique cache-miss index (`>= 0` for the first occurrence of a missing key,
 *            or `-1` if satisfied from cache)
 */
typedef struct
{
    mc_pass_t *pa;
    uint64_t   h1;
    uint64_t   h2;
    char       key[33];
    char       so_path[512];
    int        uniq_idx;
} pass_job_t;

/**
 * mc_cc_compile_prog - Compile all elementwise passes in a program, batching cache misses
 * @p:     Planned program whose `MC_STEP_PASS` steps have `c_src` populated
 * @err:   Error message buffer
 * @errsz: Size of @err in bytes
 *
 * Resolves all passes that hit in the in-memory or on-disk caches first. If 0
 * or 1 unique passes miss the cache, delegates directly to `mc_cc_compile()`.
 * When 2 or more unique passes miss the cache, combines the union of their
 * required prelude sections and all `void mc_k(...)` bodies (renamed via
 * `#define mc_k mc_k_<key>`) into a single temporary batch `.c` file, invokes
 * the compiler once, hard-links the resulting `.so` to each `<key>.so` path,
 * `dlopen`s it once, and resolves each `mc_k_<key>` symbol.
 *
 * Return: 0 on success, -1 on error
 */
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
    if (mc_cc_cache_dir(cache_dir, sizeof(cache_dir), err, errsz) != 0)
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
    int    need_complex = 0;
    int    need_coord   = 0;
    int    need_spatial = 0;
    int    sp_used[10]  = { 0 };
    int    last_miss_i  = -1;
    size_t batch_cap = sizeof(MC_PRELUDE_CORE_SRC) + sizeof(MC_PRELUDE_COMPLEX_SRC) +
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
    char batch_c[544];
    snprintf(batch_c, sizeof(batch_c), "%s/batch.c.tmp.%d.%lu", cache_dir, pid, seq);

    FILE *bfp = fopen(batch_c, "w");
    int   wok = (bfp != NULL && fwrite(batch_src, 1, blen, bfp) == blen);
    if ((bfp != NULL && fclose(bfp) != 0) || !wok)
    {
        (void) unlink(batch_c);
        snprintf(err, errsz, "failed writing batch kernel source '%s'", batch_c);
        goto cleanup;
    }
    free(batch_src);
    batch_src = NULL;

    int cc_rc = mc_cc_build(batch_c, jobs[last_miss_i].so_path, k_jit_args, err, errsz);
    (void) unlink(batch_c);
    if (cc_rc != 0)
    {
        goto cleanup;
    }

    /* The other keys name the same library: hard-link it under a temp name, then rename */
    for (int u = 0; u < nmiss; u++)
    {
        for (int i = 0; i < npass; i++)
        {
            if (jobs[i].pa->fn == NULL && jobs[i].uniq_idx == u)
            {
                char so_tmp[600];
                snprintf(so_tmp, sizeof(so_tmp), "%s.tmp.%d.%lu", jobs[i].so_path, pid, seq);
                if (i != last_miss_i && (link(jobs[last_miss_i].so_path, so_tmp) != 0 ||
                                         rename(so_tmp, jobs[i].so_path) != 0))
                {
                    snprintf(err, errsz, "cannot link '%s': %s", jobs[i].so_path, strerror(errno));
                    (void) unlink(so_tmp);
                    goto cleanup;
                }
                break;
            }
        }
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

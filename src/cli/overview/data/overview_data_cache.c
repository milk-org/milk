#include <limits.h>
#include <pthread.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "overview_data_internal.h"
#include "overview_data_index.h"

#undef STRINGMAXLEN_DIRNAME
#undef STRINGMAXLEN_FULLFILENAME
#undef STRINGMAXLEN_COMMAND
#undef PRINT_ERROR
#include "fps_types.h"
#include "fps_paramvalue.h"
#include "fps_printparameter_valuestring.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_save2disk.h"

/* =========================================================
 * Directories: each one comes from the resolver its writers use
 * ========================================================= */

static char           s_shmdir[PATH_MAX];
static char           s_procdir[PATH_MAX];
static char           s_fpsdir[PATH_MAX];
static pthread_once_t s_dirs_once = PTHREAD_ONCE_INIT;

/** dirs_resolve - ImageStreamIO, libprocessinfo and libfps resolvers, run once. */
static void dirs_resolve(void)
{
    char  fname[PATH_MAX];
    char *slash;

    ImageStreamIO_filename(fname, sizeof(fname), "x");
    slash = strrchr(fname, '/');
    if (slash != NULL)
    {
        *slash = '\0';
    }
    snprintf(s_shmdir, sizeof(s_shmdir), "%s", fname);
    processinfo_procdirname(s_procdir);
    function_parameter_struct_shmdirname(s_fpsdir);
}

/** ov_dirs_init - resolve all directories now (call before raw mode: resolvers may print). */
void ov_dirs_init(void)
{
    pthread_once(&s_dirs_once, dirs_resolve);
}

/** ov_get_shmdir - stream directory, as ImageStreamIO opens it. */
const char *ov_get_shmdir(void)
{
    ov_dirs_init();
    return s_shmdir;
}

/** ov_get_procdir - processinfo directory, as libprocessinfo creates entries. */
const char *ov_get_procdir(void)
{
    ov_dirs_init();
    return s_procdir;
}

/** ov_get_fpsdir - FPS directory, as libfps creates FPS files. */
const char *ov_get_fpsdir(void)
{
    ov_dirs_init();
    return s_fpsdir;
}

/* =========================================================
 * Persistent SHM mapping caches
 *
 * These caches keep SHM file descriptors and mappings
 * alive across scan ticks to avoid the overhead of
 * open/mmap/munmap/close on every cycle.
 *
 * Staleness: each tick checks stat() inode; a mismatch
 * or missing file triggers remap/eviction.
 * ========================================================= */

/* --- Stream cache --- */


ov_stream_cache_t s_scache[OV_MAX_STREAMS];
int               s_scache_nb    = 0;
pthread_mutex_t   s_scache_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * Cache indexes: name (or slot+PID) -> cache position. A lookup first indexes the entries appended since
 * the last one; an eviction moves the last entry, so it marks the index stale (rebuilt on next lookup).
 */
static OV_INDEX_DEF(32768) s_scache_ix;
static int s_scache_ix_n = -1; /* entries indexed, -1 = stale */

/** scache_match - cache entry @ci is stream @ctx. */
static int scache_match(const void *ctx, int32_t ci)
{
    return strcmp(s_scache[ci].name, (const char *) ctx) == 0;
}

/**
 * scache_find - find stream in cache by name.
 *
 * Return: cache index, or -1 if not found.
 */
int scache_find(const char *name)
{
    if (s_scache_ix_n < 0 || s_scache_ix_n > s_scache_nb)
    {
        ov_ix_clear(&s_scache_ix);
        s_scache_ix_n = 0;
    }
    for (; s_scache_ix_n < s_scache_nb; s_scache_ix_n++)
    {
        ov_ix_put(&s_scache_ix, ov_hash_str(s_scache[s_scache_ix_n].name), s_scache_ix_n);
    }
    return ov_ix_find(&s_scache_ix, ov_hash_str(name), scache_match, name);
}

/**
 * scache_evict_locked - close mapping and compact array (caller holds s_scache_mutex).
 */
void scache_evict_locked(int ci)
{
    s_scache_ix_n = -1;
    if (s_scache[ci].mapped)
    {
        ImageStreamIO_closeIm(&s_scache[ci].img);
    }
    s_scache_nb--;
    if (ci < s_scache_nb)
    {
        s_scache[ci] = s_scache[s_scache_nb];
    }
}

/**
 * scache_evict - close mapping and compact array.
 */
void scache_evict(int ci)
{
    pthread_mutex_lock(&s_scache_mutex);
    scache_evict_locked(ci);
    pthread_mutex_unlock(&s_scache_mutex);
}

/**
 * scache_evict_by_name - find and evict stream by name under lock.
 */
void scache_evict_by_name(const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return;
    }
    pthread_mutex_lock(&s_scache_mutex);
    int ci = scache_find(name);
    if (ci >= 0)
    {
        scache_evict_locked(ci);
    }
    pthread_mutex_unlock(&s_scache_mutex);
}

/* --- FPS cache --- */

pthread_mutex_t      s_fcache_mutex = PTHREAD_MUTEX_INITIALIZER;
static OV_FPS_PARAMS s_active_fps_params;

ov_fps_cache_t s_fcache[OV_MAX_FPS];
int            s_fcache_nb = 0;

static OV_INDEX_DEF(512) s_fcache_ix;
static int s_fcache_ix_n = -1; /* entries indexed, -1 = stale */

/** fcache_match - cache entry @ci is FPS @ctx. */
static int fcache_match(const void *ctx, int32_t ci)
{
    return strcmp(s_fcache[ci].fname, (const char *) ctx) == 0;
}

/**
 * @brief Look up an FPS entry in the connection cache.
 */
int fcache_find(const char *name)
{
    if (s_fcache_ix_n < 0 || s_fcache_ix_n > s_fcache_nb)
    {
        ov_ix_clear(&s_fcache_ix);
        s_fcache_ix_n = 0;
    }
    for (; s_fcache_ix_n < s_fcache_nb; s_fcache_ix_n++)
    {
        ov_ix_put(&s_fcache_ix, ov_hash_str(s_fcache[s_fcache_ix_n].fname), s_fcache_ix_n);
    }
    return ov_ix_find(&s_fcache_ix, ov_hash_str(name), fcache_match, name);
}

/**
 * @brief Evict and disconnect an FPS cache entry (caller must hold s_fcache_mutex).
 */
void fcache_evict_locked(int ci)
{
    s_fcache_ix_n = -1;
    ov_shm_unmap(&s_fcache[ci].map);
    s_fcache_nb--;
    if (ci < s_fcache_nb)
    {
        s_fcache[ci] = s_fcache[s_fcache_nb];
    }
}

/* --- Proc cache --- */


ov_proc_cache_t s_pcache[OV_MAX_PROCS];
int             s_pcache_nb = 0;


static OV_INDEX_DEF(1024) s_pcache_ix;
static int s_pcache_ix_n = -1; /* entries indexed, -1 = stale */

/** pcache_key - hash of a registry slot and PID. */
static uint64_t pcache_key(long pindex, pid_t pid)
{
    return ov_hash_u64(((uint64_t) pindex << 32) ^ (uint32_t) pid);
}

/** pcache_match - cache entry @ci is the registry slot and PID in @ctx (a long[2]). */
static int pcache_match(const void *ctx, int32_t ci)
{
    const long *k = ctx;
    return s_pcache[ci].pindex == k[0] && s_pcache[ci].pid == (pid_t) k[1];
}

/**
 * pcache_find - Look up a registry entry by slot and PID.
 */
int pcache_find(long pindex, pid_t pid)
{
    const long k[2] = { pindex, (long) pid };
    if (s_pcache_ix_n < 0 || s_pcache_ix_n > s_pcache_nb)
    {
        ov_ix_clear(&s_pcache_ix);
        s_pcache_ix_n = 0;
    }
    for (; s_pcache_ix_n < s_pcache_nb; s_pcache_ix_n++)
    {
        ov_ix_put(&s_pcache_ix,
                  pcache_key(s_pcache[s_pcache_ix_n].pindex, s_pcache[s_pcache_ix_n].pid),
                  s_pcache_ix_n);
    }
    return ov_ix_find(&s_pcache_ix, pcache_key(pindex, pid), pcache_match, k);
}

/**
 * pcache_evict - Unmap a cached processinfo entry and move the last entry into its slot.
 */
void pcache_evict(int ci)
{
    s_pcache_ix_n = -1;
    ov_shm_unmap(&s_pcache[ci].map);
    s_pcache_nb--;
    if (ci < s_pcache_nb)
    {
        s_pcache[ci] = s_pcache[s_pcache_nb];
    }
}

/* --- Shared memory file mappings --- */

ov_shm_map_t s_plist_map;

/**
 * ov_shm_map - map an existing regular file of at least @minsize bytes, read-only unless @rw.
 * The fd is closed once mapped. Return: 0; -1 cannot open, -2 not a regular file of that size, -3 mmap failed
 * (*m zeroed).
 */
int ov_shm_map(ov_shm_map_t *m, const char *path, int rw, size_t minsize)
{
    struct stat st;
    memset(m, 0, sizeof(*m));
    int fd = open(path, (rw ? O_RDWR : O_RDONLY) | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0)
    {
        return -1;
    }
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
        (size_t) st.st_size < minsize)
    {
        close(fd);
        return -2;
    }
    void *a = mmap(NULL, (size_t) st.st_size, PROT_READ | (rw ? PROT_WRITE : 0), MAP_SHARED, fd, 0);
    close(fd);
    if (a == MAP_FAILED)
    {
        return -3;
    }
    m->addr = a;
    m->len  = (size_t) st.st_size;
    m->dev  = st.st_dev;
    m->ino  = st.st_ino;
    return 0;
}

/** ov_shm_unmap - unmap exactly what ov_shm_map() mapped; no-op on an empty map. */
void ov_shm_unmap(ov_shm_map_t *m)
{
    if (m->addr != NULL)
    {
        munmap(m->addr, m->len);
    }
    memset(m, 0, sizeof(*m));
}

/** ov_shm_same_file - 1 if @m maps the file described by @st (same device, inode and size). */
int ov_shm_same_file(const ov_shm_map_t *m, const struct stat *st)
{
    return m->addr != NULL && m->dev == st->st_dev && m->ino == st->st_ino &&
           m->len == (size_t) st->st_size;
}

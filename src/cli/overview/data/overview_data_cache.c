#include <limits.h>
#include <pthread.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "overview_data_internal.h"

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

/**
 * scache_find - find stream in cache by name.
 *
 * Return: cache index, or -1 if not found.
 */
int scache_find(const char *name)
{
    for (int i = 0; i < s_scache_nb; i++)
    {
        if (strcmp(s_scache[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * scache_evict_locked - close mapping and compact array (caller holds s_scache_mutex).
 */
void scache_evict_locked(int ci)
{
    ImageStreamIO_closeIm(&s_scache[ci].img);
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

/**
 * @brief Look up an FPS entry in the connection cache.
 */
int fcache_find(const char *name)
{
    for (int i = 0; i < s_fcache_nb; i++)
    {
        if (strcmp(s_fcache[i].fname, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Evict and disconnect an FPS cache entry (caller must hold s_fcache_mutex).
 */
void fcache_evict_locked(int ci)
{
    fps_disconnect(&s_fcache[ci].fps);
    s_fcache_nb--;
    if (ci < s_fcache_nb)
    {
        s_fcache[ci] = s_fcache[s_fcache_nb];
    }
}

/**
 * @brief Evict and disconnect an FPS cache entry.
 */
void fcache_evict(int ci)
{
    pthread_mutex_lock(&s_fcache_mutex);
    fcache_evict_locked(ci);
    pthread_mutex_unlock(&s_fcache_mutex);
}

/* --- Proc cache --- */


ov_proc_cache_t s_pcache[OV_MAX_PROCS];
int             s_pcache_nb = 0;


/**
 * @brief Look up a process by PID in the cache.
 */
int pcache_find_pid(pid_t pid)
{
    for (int i = 0; i < s_pcache_nb; i++)
    {
        if (s_pcache[i].pid == pid)
        {
            return i;
        }
    }
    return -1;
}

/**
 * pcache_find - Look up a registry entry by slot and PID.
 */
int pcache_find(long pindex, pid_t pid)
{
    for (int i = 0; i < s_pcache_nb; i++)
    {
        if (s_pcache[i].pindex == pindex && s_pcache[i].pid == pid)
        {
            return i;
        }
    }
    return -1;
}

/**
 * pcache_evict - Unmap a cached processinfo entry and move the last entry into its slot.
 */
void pcache_evict(int ci)
{
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
 * The fd is closed once mapped. Return: 0, or -1 with *m zeroed.
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
        return -1;
    }
    void *a = mmap(NULL, (size_t) st.st_size, PROT_READ | (rw ? PROT_WRITE : 0), MAP_SHARED, fd, 0);
    close(fd);
    if (a == MAP_FAILED)
    {
        return -1;
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

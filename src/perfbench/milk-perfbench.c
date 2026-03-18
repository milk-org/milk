/**
 * @file    milk-perfbench.c
 * @brief   FPS compute unit benchmark harness
 *
 * Replaces the milk-perfbench bash script.
 * Uses perf_event_open(2) for hardware counters
 * and mmaps processinfo SHM directly while the
 * benchmark subprocess runs.
 *
 * Usage:
 *   milk-perfbench [opts] <fpsexec> <nbiter>
 *   Options:
 *     -w N   Warmup iterations
 *     -o DIR Output directory (default: ./perfresults)
 *     -a STR Extra args to fpsexec set
 *     -s CMD Setup command run before benchmark
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <signal.h>
#include <time.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <limits.h>

#include "processinfo.h"
#include "ImageStreamIO/ImageStruct.h"

/**
 * Suppress -Wunused-result on calls where we
 * intentionally discard the return value.
 */
#define IGNORE_RESULT(x) \
    do { if (x) {} } while (0)

/* ================================================================
 * Constants
 * ============================================================= */

#define MAX_CMD           2048
#define MAX_PATH          512
#define MAX_LABEL         64
#define POLL_INTERVAL_MS  10
#define POLL_TIMEOUT_MS   30000

/* ================================================================
 * Hardware counter table
 * ============================================================= */

typedef struct
{
    const char   *name;   /**< human-readable label      */
    const char   *json;   /**< JSON key                  */
    uint32_t      type;   /**< perf_event_attr.type      */
    uint64_t      config; /**< perf_event_attr.config    */
} perf_ev_t;

/* All events we open.  Must stay in sync with idx_* enums. */
static const perf_ev_t PERF_EVS[] = {
    {"cycles",            "cycles",
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
    {"bus-cycles",        "bus_cycles",
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES},
    {"instructions",      "instructions",
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
    /* L1d */
    {"L1-dcache-loads",   "L1_dcache_loads",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"L1-dcache-load-misses", "L1_dcache_misses",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {"L1-dcache-stores",  "L1_dcache_stores",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D)
        | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    /* L1i */
    {"L1-icache-loads",   "L1_icache_loads",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1I)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"L1-icache-load-misses", "L1_icache_misses",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1I)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    /* iTLB */
    {"iTLB-load-misses",  "iTLB_misses",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_ITLB)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    /* LLC */
    {"LLC-loads",         "LLC_loads",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_LL)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"LLC-load-misses",   "LLC_misses",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_LL)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {"LLC-stores",        "LLC_stores",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_LL)
        | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"LLC-store-misses",  "LLC_store_misses",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_LL)
        | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    /* dTLB */
    {"dTLB-loads",        "dTLB_loads",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_DTLB)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"dTLB-load-misses",  "dTLB_misses",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_DTLB)
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {"dTLB-store-misses", "dTLB_store_misses",
        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_DTLB)
        | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    /* stalls */
    {"stalled-cycles-frontend",
        "stalled_cycles_frontend",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
    {"stalled-cycles-backend",
        "stalled_cycles_backend",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
    /* branch */
    {"branches",          "branches",
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
    {"branch-misses",     "branch_misses",
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
    /* software */
    {"page-faults",       "page_faults",
        PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS},
    {"minor-faults",      "minor_faults",
        PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN},
    {"major-faults",      "major_faults",
        PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ},
    {"cpu-migrations",    "cpu_migrations",
        PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS},
    {"context-switches",  "context_switches",
        PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES},
    {"task-clock",        "task_clock_ns",
        PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
};

#define N_PERF_EVS \
    ((int)(sizeof(PERF_EVS) / sizeof(PERF_EVS[0])))

/* Named indices into PERF_EVS[] */
enum {
    IDX_CYCLES = 0, IDX_BUS_CYCLES, IDX_INSTRUCTIONS,
    IDX_L1D_LOADS, IDX_L1D_MISSES, IDX_L1D_STORES,
    IDX_L1I_LOADS, IDX_L1I_MISSES,
    IDX_ITLB_MISSES,
    IDX_LLC_LOADS, IDX_LLC_MISSES,
    IDX_LLC_STORES, IDX_LLC_STORE_MISSES,
    IDX_DTLB_LOADS, IDX_DTLB_MISSES, IDX_DTLB_ST_MISSES,
    IDX_STALL_FE, IDX_STALL_BE,
    IDX_BRANCHES, IDX_BRANCH_MISSES,
    IDX_PAGE_FAULTS, IDX_MINOR_FAULTS, IDX_MAJOR_FAULTS,
    IDX_CPU_MIGRATIONS, IDX_CTX_SWITCHES,
    IDX_TASK_CLOCK
};

/* ================================================================
 * Data structures
 * ============================================================= */

/** Raw counter values for one phase */
typedef struct
{
    long long v[N_PERF_EVS]; /**< one slot per PERF_EVS[] */
    int       valid;          /**< 1 if perf_event_open worked */
} hw_phase_t;

/** Processinfo-derived stats */
typedef struct
{
    /* timing percentiles */
    long p50_iter,  p95_iter,  p99_iter;
    long p999_iter, max_iter;
    long p50_exec,  p95_exec,  p99_exec;
    long p999_exec, max_exec;
    /* derived jitter (p99 - p50) */
    long jitter_iter, jitter_exec;
    long loopcnt;
    /* memory */
    long vmpeak_kb, vmhwm_kb, vmrss_kb;
    long anon_huge_kb; /* anonymous huge pages */
    /* OS scheduling */
    long vol_ctxt;  /* voluntary context switches   */
    long nvol_ctxt; /* non-voluntary context switches */
    /* CPU frequency during run (kHz) */
    long cpu_freq_min_khz;
    long cpu_freq_max_khz;
    /* RAPL energy (micro-joules, -1 if unavailable) */
    long long rapl_uj;
    long exe_size; /* bytes */
    int  valid;
} pi_stats_t;

/** Full benchmark config */
typedef struct
{
    char fpsexec[MAX_PATH];
    char fpsname[MAX_LABEL]; /* unique per run  */
    char outdir[MAX_PATH];
    char fpsargs[MAX_CMD];
    char setupcmd[MAX_CMD];
    int  nbiter;
    int  warmup;
    char procdir[MAX_PATH];
    char result_file[MAX_PATH];
    char git_commit[64];
    char build_tags[256]; /* extracted from binary */
} bench_cfg_t;

/* ================================================================
 * Globals for cleanup
 * ============================================================= */

static bench_cfg_t g_cfg;
static int         g_initialized = 0;

/* ================================================================
 * Utility: run command, return exit code
 * ============================================================= */

/**
 * @brief Run a shell command, discarding output.
 *
 * @param fmt  printf-style format string
 * @return     exit code of command, -1 on error
 */
static int run_cmd(const char *fmt, ...)
{
    char buf[MAX_CMD];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return system(buf);
}


/* ================================================================
 * Arg parsing
 * ============================================================= */

static void usage(const char *prog)
{
    printf(
        "Usage: %s [options] <fpsexec> <nbiter>\n\n"
        "Options:\n"
        "  -w N, --warmup N    Warmup iterations"
        " (default: 0)\n"
        "  -o D, --outdir D    Output directory"
        " (default: ./perfresults)\n"
        "  -a S, --fpsargs S   Extra args to"
        " fpsexec set\n"
        "  -s C, --setup C     Setup command"
        " (run before benchmark)\n"
        "  -h, --help          Show this help\n\n"
        "Example:\n"
        "  %s milk-fpsexec-imggen-mkrandom 1000"
        " -w 200 -o /tmp/results\n",
        prog, prog);
}

/**
 * @brief Parse command-line arguments into cfg.
 * @return 0 on success, 1 on error
 */
static int parse_args(
    int argc, char *argv[], bench_cfg_t *cfg)
{
    static const struct option long_opts[] = {
        {"warmup",  required_argument, 0, 'w'},
        {"outdir",  required_argument, 0, 'o'},
        {"fpsargs", required_argument, 0, 'a'},
        {"setup",   required_argument, 0, 's'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv,
                             "w:o:a:s:h",
                             long_opts, NULL)) != -1)
    {
        switch (c)
        {
            case 'w':
                cfg->warmup = atoi(optarg);
                break;
            case 'o':
                strncpy(cfg->outdir, optarg,
                        sizeof(cfg->outdir) - 1);
                break;
            case 'a':
                strncpy(cfg->fpsargs, optarg,
                        sizeof(cfg->fpsargs) - 1);
                break;
            case 's':
                strncpy(cfg->setupcmd, optarg,
                        sizeof(cfg->setupcmd) - 1);
                break;
            case 'h':
                usage(argv[0]);
                exit(0);
            default:
                return 1;
        }
    }

    if (optind + 2 > argc)
    {
        fprintf(stderr,
                "Error: fpsexec and nbiter required\n");
        usage(argv[0]);
        return 1;
    }

    strncpy(cfg->fpsexec, argv[optind],
            sizeof(cfg->fpsexec) - 1);
    cfg->nbiter = atoi(argv[optind + 1]);

    if (cfg->nbiter <= 0)
    {
        fprintf(stderr,
                "Error: nbiter must be positive\n");
        return 1;
    }
    if (cfg->warmup >= cfg->nbiter)
    {
        fprintf(stderr,
                "Error: warmup must be < nbiter\n");
        return 1;
    }

    return 0;
}

/* ================================================================
 * Environment helpers
 * ============================================================= */

/**
 * @brief Resolve the process SHM directory.
 *
 * Checks MILK_SHM_DIR env var, then /milk/shm, then /tmp.
 */
static void resolve_procdir(bench_cfg_t *cfg)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env && strlen(env) > 0)
    {
        strncpy(cfg->procdir, env,
                sizeof(cfg->procdir) - 1);
        return;
    }

    struct stat st;
    if (stat("/milk/shm", &st) == 0
        && S_ISDIR(st.st_mode))
    {
        strncpy(cfg->procdir, "/milk/shm",
                sizeof(cfg->procdir) - 1);
        return;
    }

    strncpy(cfg->procdir, "/tmp",
            sizeof(cfg->procdir) - 1);
}

/**
 * @brief Get SHM directory (same as MILK_SHM_DIR).
 */
static void resolve_shmdir(char *shmdir, size_t sz)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env && strlen(env) > 0)
    {
        strncpy(shmdir, env, sz - 1);
        return;
    }

    struct stat st;
    if (stat("/milk/shm", &st) == 0
        && S_ISDIR(st.st_mode))
    {
        strncpy(shmdir, "/milk/shm", sz - 1);
        return;
    }

    strncpy(shmdir, "/tmp", sz - 1);
}

/**
 * @brief Get short git commit hash.
 */
static void resolve_git_commit(bench_cfg_t *cfg)
{
    FILE *fp = popen(
        "git rev-parse --short HEAD 2>/dev/null",
        "r");
    if (!fp)
    {
        strncpy(cfg->git_commit, "unknown",
                sizeof(cfg->git_commit) - 1);
        return;
    }
    if (!fgets(cfg->git_commit,
               sizeof(cfg->git_commit) - 1, fp))
    {
        strncpy(cfg->git_commit, "unknown",
                sizeof(cfg->git_commit) - 1);
    }
    else
    {
        /* strip trailing newline */
        cfg->git_commit[strcspn(
            cfg->git_commit, "\n")] = '\0';
    }
    pclose(fp);
}

/**
 * @brief Get executable size in bytes.
 */
static long exe_size(const char *exe)
{
    /* find full path */
    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
             "command -v %s 2>/dev/null", exe);
    FILE *fp = popen(cmd, "r");
    if (!fp)
        return 0;
    char path[MAX_PATH] = {0};
    if (!fgets(path, sizeof(path) - 1, fp))
    {
        pclose(fp);
        return 0;
    }
    pclose(fp);
    path[strcspn(path, "\n")] = '\0';

    if (strlen(path) == 0)
        return 0;

    struct stat st;
    if (stat(path, &st) != 0)
        return 0;
    return (long) st.st_size;
}

/**
 * @brief Extract MILK_BUILD sentinel string from binary.
 *
 * Uses `strings | grep` to find the embedded tag
 * written by MILK_EMBED_BUILD_TAG() in fps.h.
 * Parses and formats relevant fields into @p out.
 *
 * @param exe   fpsexec name or full path
 * @param out   output buffer
 * @param outsz size of output buffer
 */
static void read_build_tags(
    const char *exe,
    char       *out,
    size_t      outsz)
{
    out[0] = '\0';

    /* Resolve full path */
    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
             "command -v '%s' 2>/dev/null", exe);
    FILE *fp = popen(cmd, "r");
    if (!fp)
        return;
    char path[MAX_PATH] = {0};
    if (!fgets(path, sizeof(path) - 1, fp))
    {
        pclose(fp);
        return;
    }
    pclose(fp);
    path[strcspn(path, "\n")] = '\0';
    if (strlen(path) == 0)
        return;

    /* Extract the sentinel via strings(1) */
    snprintf(cmd, sizeof(cmd),
        "strings '%s' 2>/dev/null"
        " | grep 'MILK_BUILD:'",
        path);
    fp = popen(cmd, "r");
    if (!fp)
        return;
    char raw[512] = {0};
    if (!fgets(raw, sizeof(raw) - 1, fp))
    {
        pclose(fp);
        out[0] = '\0';
        return;
    }
    pclose(fp);
    raw[strcspn(raw, "\n")] = '\0';

    /* Locate payload after "MILK_BUILD:" prefix */
    char *payload = strstr(raw, "MILK_BUILD:");
    if (!payload)
        return;
    payload += strlen("MILK_BUILD:");

    /* Build a compact human-readable summary */
    char summary[256] = {0};
    size_t slen = 0;

    if (strstr(payload, "OPT=3"))
        slen += (size_t) snprintf(
            summary + slen,
            sizeof(summary) - slen,
            "O3 ");
    if (strstr(payload, "PGO=USE"))
        slen += (size_t) snprintf(
            summary + slen,
            sizeof(summary) - slen,
            "PGO ");
    else if (strstr(payload, "PGO=GENERATE"))
        slen += (size_t) snprintf(
            summary + slen,
            sizeof(summary) - slen,
            "PGO-instr ");
    if (strstr(payload, "LTO=STATIC"))
        slen += (size_t) snprintf(
            summary + slen,
            sizeof(summary) - slen,
            "LTO-static ");
    else if (strstr(payload, "LTO=1"))
        slen += (size_t) snprintf(
            summary + slen,
            sizeof(summary) - slen,
            "LTO ");

    /* Extract architecture field */
    {
        char *ap = strstr(payload, "ARCH=");
        if (ap)
        {
            ap += 5;
            char arch[32] = {0};
            size_t ai = 0;
            while (*ap && *ap != ',' && ai < 31)
                arch[ai++] = *ap++;
            slen += (size_t) snprintf(
                summary + slen,
                sizeof(summary) - slen,
                "[%s]", arch);
        }
    }

    if (slen == 0)
        snprintf(summary, sizeof(summary),
                 "default (no PGO/LTO)");

    /* Trim trailing space */
    while (slen > 0 && summary[slen - 1] == ' ')
        summary[--slen] = '\0';

    snprintf(out, outsz, "%s", summary);
}

/* ================================================================
 * Unique FPS name
 * ============================================================= */

static void make_fpsname(char *out, size_t sz)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    unsigned long seed =
        (unsigned long) ts.tv_nsec ^ (unsigned long) ts.tv_sec;
    snprintf(out, sz, "pb%07lu", seed % 10000000UL);
}

/* ================================================================
 * FPS lifecycle helpers
 * ============================================================= */

/**
 * @brief Run milk-fps-set for one parameter.
 *
 * @param fpsname    FPS instance name
 * @param param_tag  parameter tag (e.g. "procinfo.enabled")
 * @param value      value string
 */
static void fps_set(
    const char *fpsname,
    const char *param_tag,
    const char *value)
{
    run_cmd("milk-fps-set %s.%s %s"
            " >/dev/null 2>&1",
            fpsname, param_tag, value);
}

/**
 * @brief Initialize FPS and configure procinfo.
 *
 * Runs fpsinit + confstep, enables procinfo and
 * MeasureTiming, sets triggermode to IMMEDIATE.
 */
static void fps_setup(const bench_cfg_t *cfg)
{
    /* fpsinit with -procinfo flag */
    run_cmd("%s %s:fpsinit -procinfo"
            " >/dev/null 2>&1",
            cfg->fpsexec, cfg->fpsname);

    /* confstep */
    run_cmd("%s %s:confstep"
            " >/dev/null 2>&1",
            cfg->fpsexec, cfg->fpsname);

    /* enable processinfo */
    fps_set(cfg->fpsname,
            "procinfo.enabled", "ON");
    fps_set(cfg->fpsname,
            "procinfo.MeasureTiming", "ON");
    /* triggermode 0 = IMMEDIATE */
    fps_set(cfg->fpsname,
            "procinfo.triggermode", "0");

    /* apply extra positional args if any */
    if (cfg->fpsargs[0] != '\0')
    {
        run_cmd("%s %s:set %s"
                " >/dev/null 2>&1",
                cfg->fpsexec, cfg->fpsname,
                cfg->fpsargs);
    }
}

/**
 * @brief Auto-create missing SHM streams.
 *
 * Queries FPS for STREAMNAME entries and creates
 * any streams that don't have a .im.shm file yet.
 */
static void fps_create_streams(const bench_cfg_t *cfg)
{
    char shmdir[MAX_PATH];
    resolve_shmdir(shmdir, sizeof(shmdir));

    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
        "%s %s:fps 2>/dev/null"
        " | sed 's/\\x1b\\[[0-9;]*m//g'"
        " | awk '$3==\"STREAMNAME\" && NF>=8"
             " {print $4}'",
        cfg->fpsexec, cfg->fpsname);

    FILE *fp = popen(cmd, "r");
    if (!fp)
        return;

    char sname[256];
    while (fgets(sname, sizeof(sname), fp))
    {
        sname[strcspn(sname, "\n")] = '\0';
        if (strlen(sname) == 0)
            continue;

        char impath[MAX_PATH];
        snprintf(impath, sizeof(impath),
                 "%s/%s.im.shm", shmdir, sname);

        struct stat st;
        if (stat(impath, &st) != 0)
        {
            printf("  Creating stream: %s (32x32)\n",
                   sname);
            run_cmd("milk-perfbench-mkstream"
                    " %s 32 32 >/dev/null 2>&1",
                    sname);
        }
        else
        {
            printf("  Stream exists: %s\n", sname);
        }
    }
    pclose(fp);
}

/**
 * @brief Cleanup: remove FPS SHM files.
 */
static void fps_cleanup(const bench_cfg_t *cfg)
{
    char shmdir[MAX_PATH];
    resolve_shmdir(shmdir, sizeof(shmdir));

    run_cmd("rm -f '%s/fps.%s'*.shm"
            " '%s/%s.fps.datadir'"
            " 2>/dev/null",
            shmdir, cfg->fpsname,
            shmdir, cfg->fpsname);
}

/* ================================================================
 * perf_event_open syscall wrapper
 * ============================================================= */

static long perf_event_open(
    struct perf_event_attr *attr,
    pid_t                   pid,
    int                     cpu,
    int                     group_fd,
    unsigned long           flags)
{
    return syscall(__NR_perf_event_open,
                   attr, pid, cpu, group_fd, flags);
}

/* ================================================================
 * Processinfo SHM helpers
 * ============================================================= */

/** Compare two longs for qsort */
static int cmp_long(const void *a, const void *b)
{
    long la = *(const long *) a;
    long lb = *(const long *) b;
    return (la > lb) - (la < lb);
}

/**
 * @brief Find proc.*.shm in procdir matching a PID.
 *
 * @param procdir  Directory to scan
 * @param pid      PID to match (0 = any)
 * @param out      Output path buffer
 * @param outsz    Size of out
 * @return 1 if found, 0 otherwise
 */
static int find_proc_shm(
    const char *procdir,
    pid_t       pid,
    char       *out,
    size_t      outsz)
{
    DIR *d = opendir(procdir);
    if (!d)
        return 0;

    int found = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        if (strncmp(de->d_name, "proc.", 5) != 0)
            continue;
        if (!strstr(de->d_name, ".shm"))
            continue;

        if (pid != 0)
        {
            /* extract PID field: proc.NAME.PID.shm */
            const char *p = strrchr(de->d_name, '.');
            if (!p)
                continue;
            /* step back over ".shm" */
            /* format: proc.<name>.<pid>.shm */
            /* find second-to-last dot */
            char tmp[256];
            strncpy(tmp, de->d_name, sizeof(tmp)-1);
            /* remove trailing ".shm" */
            char *dot = strrchr(tmp, '.');
            if (!dot) continue;
            *dot = '\0';
            /* now find the PID field */
            dot = strrchr(tmp, '.');
            if (!dot) continue;
            pid_t fpid = (pid_t) atoi(dot + 1);
            if (fpid != pid)
                continue;
        }

        snprintf(out, outsz, "%s/%s",
                 procdir, de->d_name);
        found = 1;
        break;
    }
    closedir(d);
    return found;
}

/**
 * @brief Read memory + scheduling stats from
 *        /proc/PID/status.
 */
static void read_proc_mem(
    pid_t  pid,
    long  *vmpeak_kb,
    long  *vmhwm_kb,
    long  *vmrss_kb,
    long  *vol_ctxt,
    long  *nvol_ctxt)
{
    *vmpeak_kb = -1;
    *vmhwm_kb  = -1;
    *vmrss_kb  = -1;
    *vol_ctxt  = -1;
    *nvol_ctxt = -1;

    char path[128];
    snprintf(path, sizeof(path),
             "/proc/%d/status", (int) pid);

    FILE *fp = fopen(path, "r");
    if (!fp)
        return;

    char line[256];
    while (fgets(line, sizeof(line), fp))
    {
        if (strncmp(line, "VmPeak:", 7) == 0)
            sscanf(line + 7, " %ld", vmpeak_kb);
        else if (strncmp(line, "VmHWM:", 6) == 0)
            sscanf(line + 6, " %ld", vmhwm_kb);
        else if (strncmp(line, "VmRSS:", 6) == 0)
            sscanf(line + 6, " %ld", vmrss_kb);
        else if (strncmp(line,
                         "voluntary_ctxt_switches:",
                         24) == 0)
            sscanf(line + 24, " %ld", vol_ctxt);
        else if (strncmp(line,
                         "nonvoluntary_ctxt_switches:",
                         27) == 0)
            sscanf(line + 27, " %ld", nvol_ctxt);
    }
    fclose(fp);
}

/**
 * @brief Read anonymous huge-page usage from
 *        /proc/PID/smaps_rollup.
 *
 * @param pid   Target process
 * @return      AnonHugePages in kB, or 0 if
 *              unavailable
 */
static long read_smaps_huge(pid_t pid)
{
    char path[128];
    snprintf(path, sizeof(path),
             "/proc/%d/smaps_rollup", (int) pid);
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;
    char line[256];
    long val = 0;
    while (fgets(line, sizeof(line), fp))
    {
        if (strncmp(line, "AnonHugePages:", 14) == 0)
        {
            sscanf(line + 14, " %ld", &val);
            break;
        }
    }
    fclose(fp);
    return val;
}

/**
 * @brief Sample CPU frequency from sysfs.
 *
 * Reads scaling_cur_freq for every online CPU,
 * returns min and max observed values in kHz.
 * Falls back to cpuinfo_cur_freq if scaling_cur
 * is absent.
 */
static void read_cpu_freq(
    long *freq_min_khz,
    long *freq_max_khz)
{
    *freq_min_khz = -1;
    *freq_max_khz = -1;
    long fmin = LONG_MAX;
    long fmax = 0;
    int  found = 0;

    for (int cpu = 0; cpu < 1024; cpu++)
    {
        char path[160];
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/"
            "cpu%d/cpufreq/scaling_cur_freq",
            cpu);
        FILE *fp = fopen(path, "r");
        if (!fp)
        {
            /* Try cpuinfo_cur_freq */
            snprintf(path, sizeof(path),
                "/sys/devices/system/cpu/"
                "cpu%d/cpufreq/cpuinfo_cur_freq",
                cpu);
            fp = fopen(path, "r");
        }
        if (!fp)
        {
            if (found)
                break; /* no more CPUs */
            continue;
        }
        long f = 0;
        if (fscanf(fp, "%ld", &f) == 1 && f > 0)
        {
            found = 1;
            if (f < fmin)
                fmin = f;
            if (f > fmax)
                fmax = f;
        }
        fclose(fp);
    }
    if (found)
    {
        *freq_min_khz = fmin;
        *freq_max_khz = fmax;
    }
}

/**
 * @brief Read RAPL package energy counter.
 *
 * Reads /sys/class/powercap/intel-rapl/intel-rapl:0/
 * energy_uj.  Returns -1 if unavailable or if
 * access is denied.
 *
 * @return Energy in micro-joules, or -1
 */
static long long read_rapl_energy(void)
{
    const char *rapl =
        "/sys/class/powercap/intel-rapl/"
        "intel-rapl:0/energy_uj";
    FILE *fp = fopen(rapl, "r");
    if (!fp)
        return -1LL;
    long long val = -1LL;
    IGNORE_RESULT(fscanf(fp, "%lld", &val));
    fclose(fp);
    return val;
}

/**
 * @brief Compute percentile stats from PROCESSINFO.
 *
 * Maps the proc SHM file and extracts p50/p95/p99
 * for both iteration time and execution time.
 */
static void read_procinfo_stats(
    const char *shm_path,
    pid_t       child_pid,
    pi_stats_t *out,
    long long   rapl_start)
{
    out->valid = 0;

    int fd = open(shm_path, O_RDONLY);
    if (fd < 0)
        return;

    struct stat st;
    if (fstat(fd, &st) < 0)
    {
        close(fd);
        return;
    }

    PROCESSINFO *pi = (PROCESSINFO *) mmap(
        NULL, (size_t) st.st_size,
        PROT_READ, MAP_SHARED, fd, 0);
    if (pi == MAP_FAILED)
    {
        close(fd);
        return;
    }

    /*
     * Determine how many ring-buffer entries are valid.
     * timingbuffercnt > 0: ring has wrapped at least once,
     * all PROCESSINFO_NBtimer slots are valid.
     * timingbuffercnt == 0: ring has not wrapped yet,
     * timerindex is the number of entries written so far.
     */
    int nbsam;
    if (pi->timingbuffercnt > 0)
        nbsam = PROCESSINFO_NBtimer;
    else
        nbsam = pi->timerindex;
    if (nbsam > PROCESSINFO_NBtimer)
        nbsam = PROCESSINFO_NBtimer;

    long iter_ns[PROCESSINFO_NBtimer];
    long exec_ns[PROCESSINFO_NBtimer];
    int  nv = 0;

    for (int i = 1; i < nbsam; i++)
    {
        long dt_exec =
            (pi->texecend[i].tv_sec
             - pi->texecstart[i].tv_sec)
            * 1000000000L
            + (pi->texecend[i].tv_nsec
               - pi->texecstart[i].tv_nsec);
        long dt_iter =
            (pi->texecstart[i].tv_sec
             - pi->texecstart[i-1].tv_sec)
            * 1000000000L
            + (pi->texecstart[i].tv_nsec
               - pi->texecstart[i-1].tv_nsec);
        /* Reject negative, zero, or implausibly
         * large values (stale ring buffer entries
         * from a previous FPS session). */
        if (dt_exec > 0 && dt_iter > 0
            && dt_exec < 10000000000L
            && dt_iter < 10000000000L)
        {
            exec_ns[nv] = dt_exec;
            iter_ns[nv] = dt_iter;
            nv++;
        }
    }

    out->loopcnt = pi->loopcnt;

    /* Read memory + scheduling stats */
    read_proc_mem(child_pid,
                  &out->vmpeak_kb,
                  &out->vmhwm_kb,
                  &out->vmrss_kb,
                  &out->vol_ctxt,
                  &out->nvol_ctxt);

    /* Anonymous huge pages */
    out->anon_huge_kb = read_smaps_huge(child_pid);

    /* CPU frequency (min/max across all CPUs) */
    read_cpu_freq(&out->cpu_freq_min_khz,
                  &out->cpu_freq_max_khz);

    /* RAPL energy delta */
    {
        long long rapl_end = read_rapl_energy();
        if (rapl_start >= 0 && rapl_end >= 0)
        {
            /* Handle counter wrap (max_energy_range_uj)
             * by taking absolute diff */
            out->rapl_uj =
                (rapl_end >= rapl_start)
                ? (rapl_end - rapl_start)
                : rapl_end; /* wrapped: use end value */
        }
        else
        {
            out->rapl_uj = -1LL;
        }
    }

    munmap(pi, (size_t) st.st_size);
    close(fd);

    if (nv == 0)
        return;

    qsort(exec_ns, (size_t) nv,
          sizeof(long), cmp_long);
    qsort(iter_ns,  (size_t) nv,
          sizeof(long), cmp_long);

    /* Compute percentile index, clamped to [0,nv-1] */
#define PCTILE(arr, pct) \
    (arr)[((nv * (pct) / 100) < nv \
           ? (nv * (pct) / 100) : (nv - 1))]
/* p99.9: need 1000-based arithmetic */
#define PCTILE999(arr) \
    (arr)[((nv * 999 / 1000) < nv \
           ? (nv * 999 / 1000) : (nv - 1))]

    out->p50_exec  = PCTILE(exec_ns, 50);
    out->p95_exec  = PCTILE(exec_ns, 95);
    out->p99_exec  = PCTILE(exec_ns, 99);
    out->p999_exec = PCTILE999(exec_ns);
    out->max_exec  = exec_ns[nv - 1];
    out->p50_iter  = PCTILE(iter_ns, 50);
    out->p95_iter  = PCTILE(iter_ns, 95);
    out->p99_iter  = PCTILE(iter_ns, 99);
    out->p999_iter = PCTILE999(iter_ns);
    out->max_iter  = iter_ns[nv - 1];

    /* Jitter: tail spread above median */
    out->jitter_iter = out->p99_iter - out->p50_iter;
    out->jitter_exec = out->p99_exec - out->p50_exec;

    out->valid = 1;

#undef PCTILE
#undef PCTILE999
}

/* ================================================================
 * Open perf counters for a set of events
 * ============================================================= */

/**
 * @brief Open all perf_event fds for a child PID.
 *
 * Opens N_PERF_EVS file descriptors tracking the
 * given pid.  Descriptors are stored in fds[].
 * Returns number of successfully opened fds.
 * Events that fail (no kernel support or permissions)
 * are stored as -1 and skipped gracefully.
 *
 * @param pid    Process to monitor
 * @param fds    Output array of size N_PERF_EVS
 * @return       Number of fds successfully opened
 */
static int perf_open_all(pid_t pid, int *fds)
{
    int nok = 0;
    for (int i = 0; i < N_PERF_EVS; i++)
    {
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.type           = PERF_EVS[i].type;
        attr.size           = sizeof(attr);
        attr.config         = PERF_EVS[i].config;
        attr.disabled       = 1;
        attr.exclude_kernel = 0;
        attr.exclude_hv     = 1;
        attr.inherit        = 1;

        fds[i] = (int) perf_event_open(
            &attr, pid, -1, -1, 0);
        if (fds[i] >= 0)
        {
            ioctl(fds[i],
                  PERF_EVENT_IOC_RESET, 0);
            ioctl(fds[i],
                  PERF_EVENT_IOC_ENABLE, 0);
            nok++;
        }
    }
    return nok;
}

/**
 * @brief Disable, read, and close all perf fds.
 *
 * @param fds    Array of N_PERF_EVS fds
 * @param phase  Output hw_phase_t to fill
 */
static void perf_read_close(
    int *fds, hw_phase_t *phase)
{
    phase->valid = 0;
    for (int i = 0; i < N_PERF_EVS; i++)
    {
        phase->v[i] = 0;
        if (fds[i] < 0)
            continue;
        ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
        IGNORE_RESULT(
            read(fds[i], &phase->v[i],
                 sizeof(long long)));
        close(fds[i]);
        fds[i] = -1;
        phase->valid = 1;
    }
}

/* ================================================================
 * run_phase: fork+exec the benchmark, collect data
 * ============================================================= */

/**
 * @brief Run one benchmark phase.
 *
 * Forks the FPS executable, opens perf counters
 * on the child, and mmaps the proc SHM while child
 * runs (polling every 10 ms), keeping the most
 * recent stats.  Waits for child to exit then reads
 * final counter values.
 *
 * @param cfg         Benchmark configuration
 * @param iters       Number of iterations for this phase
 * @param phase       Output perf counter data
 * @param pi          Output processinfo stats
 * @param collect_pi  1 = poll proc SHM, 0 = skip
 * @param wall_ns     Output wall-clock nanoseconds
 */
static void run_phase(
    const bench_cfg_t *cfg,
    int                iters,
    hw_phase_t        *phase,
    pi_stats_t        *pi,
    int                collect_pi,
    long long         *wall_ns)
{
    /* Capture RAPL energy baseline for delta */
    long long rapl_start = read_rapl_energy();

    char iters_str[32];
    snprintf(iters_str, sizeof(iters_str),
             "%d", iters);
    run_cmd("milk-fps-set %s.procinfo.loopcntMax"
            " %s >/dev/null 2>&1",
            cfg->fpsname, iters_str);

    /* Build argv for child */
    char runarg[MAX_CMD];
    snprintf(runarg, sizeof(runarg),
             "%s:runstart", cfg->fpsname);

    /* Fork child */
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    pid_t child = fork();
    if (child < 0)
    {
        perror("fork");
        *wall_ns = 0;
        phase->valid = 0;
        if (pi) pi->valid = 0;
        return;
    }

    if (child == 0)
    {
        /* Child: redirect stdout+stderr to /dev/null */
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0)
        {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        execlp(cfg->fpsexec,
               cfg->fpsexec, runarg, NULL);
        _exit(127);
    }

    /* Parent: open perf counters for child */
    int fds[N_PERF_EVS];
    memset(fds, -1, sizeof(fds));
    perf_open_all(child, fds);

    /* Monitor proc SHM while child runs */
    if (collect_pi && pi)
    {
        pi->valid = 0;
        char shm_path[MAX_PATH];
        struct timespec poll_ts;
        long total_poll_ms = 0;

        while (total_poll_ms < POLL_TIMEOUT_MS)
        {
            /* Check if child is still running */
            int status;
            pid_t ret = waitpid(
                child, &status, WNOHANG);
            if (ret == child)
            {
                /* Child exited: do a final read */
                if (find_proc_shm(
                        cfg->procdir, child,
                        shm_path, sizeof(shm_path)))
                {
                    read_procinfo_stats(
                        shm_path, child,
                        pi, rapl_start);
                }
                perf_read_close(fds, phase);
                clock_gettime(CLOCK_MONOTONIC,
                              &ts_end);
                *wall_ns =
                    (long long)(ts_end.tv_sec
                       - ts_start.tv_sec)
                    * 1000000000LL
                    + (ts_end.tv_nsec
                       - ts_start.tv_nsec);
                return;
            }

            /* Poll proc SHM */
            if (find_proc_shm(
                    cfg->procdir, child,
                    shm_path, sizeof(shm_path)))
            {
                /* snapshot the most recent state */
                pi_stats_t tmp;
                memset(&tmp, 0, sizeof(tmp));
                read_procinfo_stats(
                    shm_path, child,
                    &tmp, rapl_start);
                if (tmp.valid)
                    *pi = tmp;
            }

            poll_ts.tv_sec  = 0;
            poll_ts.tv_nsec =
                POLL_INTERVAL_MS * 1000000L;
            nanosleep(&poll_ts, NULL);
            total_poll_ms += POLL_INTERVAL_MS;
        }

        /* Timeout: just wait */
    }

    /* Wait for child and read counters */
    int status;
    waitpid(child, &status, 0);
    perf_read_close(fds, phase);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    *wall_ns =
        (long long)(ts_end.tv_sec
                    - ts_start.tv_sec)
        * 1000000000LL
        + (ts_end.tv_nsec - ts_start.tv_nsec);
}

/* ================================================================
 * Counter math helpers
 * ============================================================= */

static double ipc(
    const hw_phase_t *p)
{
    if (!p->valid || p->v[IDX_CYCLES] == 0)
        return 0.0;
    return (double) p->v[IDX_INSTRUCTIONS]
         / (double) p->v[IDX_CYCLES];
}

static double miss_rate(
    long long misses, long long loads)
{
    if (loads == 0)
        return 0.0;
    return 100.0 * (double) misses
                 / (double) loads;
}

/**
 * @brief Compute measured = total - warmup.
 *
 * @param m    Output (measured phase)
 * @param t    Total phase
 * @param w    Warmup phase
 */
static void sub_phase(
    hw_phase_t       *m,
    const hw_phase_t *t,
    const hw_phase_t *w)
{
    m->valid = t->valid;
    for (int i = 0; i < N_PERF_EVS; i++)
        m->v[i] = t->v[i] - w->v[i];
}

/* ================================================================
 * JSON output
 * ============================================================= */

/**
 * @brief Write full JSON result to file.
 *
 * @param cfg      Benchmark configuration
 * @param t        Total phase counters
 * @param w        Warmup phase counters
 * @param measured Number of measured iterations
 * @param t_ns     Total wall-clock ns
 * @param w_ns     Warmup wall-clock ns
 * @param pi       Total-run processinfo stats
 * @param pi_w     Warmup processinfo stats
 * @param exe_sz   Executable size in bytes
 */
static void write_json(
    const bench_cfg_t *cfg,
    const hw_phase_t  *t,
    const hw_phase_t  *w,
    int                measured,
    long long          t_ns,
    long long          w_ns,
    const pi_stats_t  *pi,
    const pi_stats_t  *pi_w,
    long               exe_sz)
{
    FILE *fp = fopen(cfg->result_file, "w");
    if (!fp)
    {
        fprintf(stderr,
                "ERROR: cannot write %s: %s\n",
                cfg->result_file, strerror(errno));
        return;
    }

    /* timestamp */
    time_t now = time(NULL);
    struct tm utc;
    gmtime_r(&now, &utc);
    char ts[32];
    strftime(ts, sizeof(ts),
             "%Y-%m-%dT%H:%M:%SZ", &utc);

    fprintf(fp, "{\n");
    fprintf(fp, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(fp, "  \"compute_unit\": \"%s\",\n",
            cfg->fpsexec);
    fprintf(fp, "  \"exe_size_bytes\": %ld,\n",
            exe_sz);
    fprintf(fp, "  \"git_commit\": \"%s\",\n",
            cfg->git_commit);
    fprintf(fp, "  \"build_tags\": \"%s\",\n",
            cfg->build_tags[0]
                ? cfg->build_tags
                : "default");
    fprintf(fp, "  \"iterations\": %d,\n",
            cfg->nbiter);
    fprintf(fp, "  \"warmup_iterations\": %d,\n",
            cfg->warmup);
    fprintf(fp, "  \"measured_iterations\": %d,\n",
            measured);
    fprintf(fp,
            "  \"wall_clock_s\": %.9f,\n",
            (double) t_ns / 1e9);
    fprintf(fp,
            "  \"warmup_s\": %.9f,\n",
            (double) w_ns / 1e9);

    /* hw_counters */
    fprintf(fp, "  \"hw_counters\": {\n");
    if (t->valid)
    {
        for (int i = 0; i < N_PERF_EVS; i++)
        {
            fprintf(fp,
                    "    \"%s\": %lld%s\n",
                    PERF_EVS[i].json,
                    t->v[i],
                    (i < N_PERF_EVS - 1)
                        ? "," : "");
        }
    }
    fprintf(fp, "  },\n");

    /* warmup_counters */
    if (cfg->warmup > 0 && w->valid)
    {
        fprintf(fp, "  \"warmup_counters\": {\n");
        fprintf(fp,
                "    \"cycles\": %lld,\n"
                "    \"instructions\": %lld,\n"
                "    \"L1_dcache_misses\": %lld,\n"
                "    \"LLC_misses\": %lld,\n"
                "    \"branch_misses\": %lld,\n"
                "    \"page_faults\": %lld\n"
                "  },\n",
                w->v[IDX_CYCLES],
                w->v[IDX_INSTRUCTIONS],
                w->v[IDX_L1D_MISSES],
                w->v[IDX_LLC_MISSES],
                w->v[IDX_BRANCH_MISSES],
                w->v[IDX_PAGE_FAULTS]);
    }

    /* processinfo - total */
    if (pi && pi->valid)
    {
        fprintf(fp,
            "  \"processinfo\": {\n"
            "    \"loopcnt\": %ld,\n"
            "    \"p50_iter_ns\": %ld,\n"
            "    \"p95_iter_ns\": %ld,\n"
            "    \"p99_iter_ns\": %ld,\n"
            "    \"p999_iter_ns\": %ld,\n"
            "    \"max_iter_ns\": %ld,\n"
            "    \"jitter_iter_ns\": %ld,\n"
            "    \"p50_exec_ns\": %ld,\n"
            "    \"p95_exec_ns\": %ld,\n"
            "    \"p99_exec_ns\": %ld,\n"
            "    \"p999_exec_ns\": %ld,\n"
            "    \"max_exec_ns\": %ld,\n"
            "    \"jitter_exec_ns\": %ld,\n"
            "    \"vmpeak_kb\": %ld,\n"
            "    \"vmhwm_kb\": %ld,\n"
            "    \"vmrss_kb\": %ld,\n"
            "    \"anon_huge_kb\": %ld,\n"
            "    \"vol_ctxt_switches\": %ld,\n"
            "    \"nvol_ctxt_switches\": %ld,\n"
            "    \"cpu_freq_min_khz\": %ld,\n"
            "    \"cpu_freq_max_khz\": %ld,\n"
            "    \"rapl_energy_uj\": %lld\n"
            "  },\n",
            pi->loopcnt,
            pi->p50_iter, pi->p95_iter,
            pi->p99_iter,
            pi->p999_iter, pi->max_iter,
            pi->jitter_iter,
            pi->p50_exec, pi->p95_exec,
            pi->p99_exec,
            pi->p999_exec, pi->max_exec,
            pi->jitter_exec,
            pi->vmpeak_kb, pi->vmhwm_kb,
            pi->vmrss_kb,
            pi->anon_huge_kb,
            pi->vol_ctxt, pi->nvol_ctxt,
            pi->cpu_freq_min_khz,
            pi->cpu_freq_max_khz,
            pi->rapl_uj);
    }
    else
    {
        fprintf(fp,
                "  \"processinfo\": null,\n");
    }
    /* processinfo - warmup */
    if (pi_w && pi_w->valid)
    {
        fprintf(fp,
            "  \"processinfo_warmup\": {\n"
            "    \"loopcnt\": %ld,\n"
            "    \"p50_iter_ns\": %ld,\n"
            "    \"p95_iter_ns\": %ld,\n"
            "    \"p99_iter_ns\": %ld,\n"
            "    \"p999_iter_ns\": %ld,\n"
            "    \"max_iter_ns\": %ld,\n"
            "    \"jitter_iter_ns\": %ld,\n"
            "    \"p50_exec_ns\": %ld,\n"
            "    \"p95_exec_ns\": %ld,\n"
            "    \"p99_exec_ns\": %ld,\n"
            "    \"p999_exec_ns\": %ld,\n"
            "    \"max_exec_ns\": %ld,\n"
            "    \"jitter_exec_ns\": %ld,\n"
            "    \"vol_ctxt_switches\": %ld,\n"
            "    \"nvol_ctxt_switches\": %ld,\n"
            "    \"cpu_freq_min_khz\": %ld,\n"
            "    \"cpu_freq_max_khz\": %ld,\n"
            "    \"rapl_energy_uj\": %lld\n"
            "  }\n",
            pi_w->loopcnt,
            pi_w->p50_iter, pi_w->p95_iter,
            pi_w->p99_iter,
            pi_w->p999_iter, pi_w->max_iter,
            pi_w->jitter_iter,
            pi_w->p50_exec, pi_w->p95_exec,
            pi_w->p99_exec,
            pi_w->p999_exec, pi_w->max_exec,
            pi_w->jitter_exec,
            pi_w->vol_ctxt, pi_w->nvol_ctxt,
            pi_w->cpu_freq_min_khz,
            pi_w->cpu_freq_max_khz,
            pi_w->rapl_uj);
    }
    else
    {
        fprintf(fp,
                "  \"processinfo_warmup\": null\n");
    }

    fprintf(fp, "}\n");
    fclose(fp);
}

/* ================================================================
 * Human-readable summary
 * ============================================================= */

#define COL1W  26
#define COL2W  14

/**
 * @brief Print a counter row: label, total,
 *        [warmup,] per-measured-iter.
 */
static void print_row(
    const char *label,
    long long   total,
    long long   warmup_v,
    int         measured,
    int         has_warmup,
    int         decimals)
{
    double per_iter =
        (measured > 0 && total > warmup_v)
        ? (double)(total - warmup_v)
          / (double) measured
        : 0.0;

    if (has_warmup)
    {
        if (decimals == 6)
            printf("  %-*s %*lld %*lld %*.6f/iter\n",
                   COL1W, label,
                   COL2W, total,
                   COL2W, warmup_v,
                   COL2W, per_iter);
        else
            printf("  %-*s %*lld %*lld %*.1f/iter\n",
                   COL1W, label,
                   COL2W, total,
                   COL2W, warmup_v,
                   COL2W, per_iter);
    }
    else
    {
        if (decimals == 6)
            printf("  %-*s %*lld %*.6f/iter\n",
                   COL1W, label,
                   COL2W, total,
                   COL2W, per_iter);
        else
            printf("  %-*s %*lld %*.1f/iter\n",
                   COL1W, label,
                   COL2W, total,
                   COL2W, per_iter);
    }
}

static void print_rate(
    const char *label,
    double      rate_t,
    double      rate_w,
    double      rate_m,
    int         has_warmup)
{
    if (has_warmup)
        printf("  %-*s %*.6f%% %*.6f%% %*.6f%%\n",
               COL1W, label,
               COL2W - 1, rate_t,
               COL2W - 1, rate_w,
               COL2W - 1, rate_m);
    else
        printf("  %-*s %*.6f%%\n",
               COL1W, label,
               COL2W - 1, rate_t);
}

static void print_sep(void)
{
    printf("  %s\n",
           "------------------------------------------"
           "--------------------");
}

/**
 * @brief Print the full human-readable summary.
 */
static void print_summary(
    const bench_cfg_t *cfg,
    int                measured,
    const hw_phase_t  *t,
    const hw_phase_t  *w,
    long long          t_ns,
    long long          w_ns,
    const pi_stats_t  *pi,
    const pi_stats_t  *pi_w,
    long               exe_sz)
{
    int hw = (cfg->warmup > 0);
    hw_phase_t m;
    sub_phase(&m, t, w);

    printf("\n");
    printf("======================================================\n");
    printf("  Benchmark Results\n");
    printf("  Total: %d  Warmup: %d  Measured: %d\n",
           cfg->nbiter, cfg->warmup, measured);
    printf("======================================================\n");

    if (hw)
        printf("  %-*s %*s %*s %*s\n",
               COL1W, "",
               COL2W, "Total",
               COL2W, "Warmup",
               COL2W, "Measured");
    else
        printf("  %-*s %*s %*s\n",
               COL1W, "",
               COL2W, "Total",
               COL2W, "Per-iter");

    print_sep();

    /* Wall clock */
    {
        long long meas_ns = t_ns - w_ns;
        double    meas_pi =
            (measured > 0)
            ? (double) meas_ns / measured
            : 0.0;

        if (hw)
            printf("  %-*s %*.6f s %*.6f s %*.1f ns/iter\n",
                   COL1W, "Wall clock",
                   COL2W - 2, (double) t_ns / 1e9,
                   COL2W - 2, (double) w_ns / 1e9,
                   COL2W, meas_pi);
        else
            printf("  %-*s %*.6f s %*.1f ns/iter\n",
                   COL1W, "Wall clock",
                   COL2W - 2, (double) t_ns / 1e9,
                   COL2W, (double) t_ns / measured);
    }

#define C(idx) t->v[idx], w->v[idx]
    print_row("Cycles",
        C(IDX_CYCLES), measured, hw, 1);
    print_row("Bus cycles",
        C(IDX_BUS_CYCLES), measured, hw, 1);
    print_row("Instructions",
        C(IDX_INSTRUCTIONS), measured, hw, 1);

    /* IPC */
    {
        double ipc_t = ipc(t);
        double ipc_w = ipc(w);
        double ipc_m = ipc(&m);
        if (hw)
            printf("  %-*s %*.3f %*.3f %*.3f\n",
                   COL1W, "Instr per Cycle (IPC)",
                   COL2W, ipc_t,
                   COL2W, ipc_w,
                   COL2W, ipc_m);
        else
            printf("  %-*s %*.3f\n",
                   COL1W, "Instr per Cycle (IPC)",
                   COL2W, ipc_t);
    }

    print_sep();
    printf("  --- L1 Data Cache ---\n");
    print_row("  Loads",
        C(IDX_L1D_LOADS), measured, hw, 1);
    print_row("  Load misses",
        C(IDX_L1D_MISSES), measured, hw, 1);
    {
        double mr_t = miss_rate(
            t->v[IDX_L1D_MISSES],
            t->v[IDX_L1D_LOADS]);
        double mr_w = miss_rate(
            w->v[IDX_L1D_MISSES],
            w->v[IDX_L1D_LOADS]);
        double mr_m = miss_rate(
            m.v[IDX_L1D_MISSES],
            m.v[IDX_L1D_LOADS]);
        print_rate("    Miss rate",
                   mr_t, mr_w, mr_m, hw);
    }
    print_row("  Stores",
        C(IDX_L1D_STORES), measured, hw, 1);

    printf("  --- L1 Instruction Cache ---\n");
    print_row("  Loads",
        C(IDX_L1I_LOADS), measured, hw, 1);
    print_row("  Load misses",
        C(IDX_L1I_MISSES), measured, hw, 1);
    {
        double mr_t = miss_rate(
            t->v[IDX_L1I_MISSES],
            t->v[IDX_L1I_LOADS]);
        double mr_w = miss_rate(
            w->v[IDX_L1I_MISSES],
            w->v[IDX_L1I_LOADS]);
        double mr_m = miss_rate(
            m.v[IDX_L1I_MISSES],
            m.v[IDX_L1I_LOADS]);
        print_rate("    Miss rate",
                   mr_t, mr_w, mr_m, hw);
    }

    printf("  --- Instruction TLB ---\n");
    print_row("  Load misses",
        C(IDX_ITLB_MISSES), measured, hw, 1);

    printf("  --- Last Level Cache (LLC) ---\n");
    print_row("  Loads",
        C(IDX_LLC_LOADS), measured, hw, 1);
    print_row("  Load misses",
        C(IDX_LLC_MISSES), measured, hw, 1);
    {
        double mr_t = miss_rate(
            t->v[IDX_LLC_MISSES],
            t->v[IDX_LLC_LOADS]);
        double mr_w = miss_rate(
            w->v[IDX_LLC_MISSES],
            w->v[IDX_LLC_LOADS]);
        double mr_m = miss_rate(
            m.v[IDX_LLC_MISSES],
            m.v[IDX_LLC_LOADS]);
        print_rate("    Load miss rate",
                   mr_t, mr_w, mr_m, hw);
    }
    print_row("  Stores",
        C(IDX_LLC_STORES), measured, hw, 1);
    print_row("  Store misses",
        C(IDX_LLC_STORE_MISSES), measured, hw, 1);
    {
        double mr_t = miss_rate(
            t->v[IDX_LLC_STORE_MISSES],
            t->v[IDX_LLC_STORES]);
        double mr_w = miss_rate(
            w->v[IDX_LLC_STORE_MISSES],
            w->v[IDX_LLC_STORES]);
        double mr_m = miss_rate(
            m.v[IDX_LLC_STORE_MISSES],
            m.v[IDX_LLC_STORES]);
        print_rate("    Store miss rate",
                   mr_t, mr_w, mr_m, hw);
    }

    printf("  --- Data TLB ---\n");
    print_row("  Loads",
        C(IDX_DTLB_LOADS), measured, hw, 1);
    print_row("  Load misses",
        C(IDX_DTLB_MISSES), measured, hw, 1);
    {
        double mr_t = miss_rate(
            t->v[IDX_DTLB_MISSES],
            t->v[IDX_DTLB_LOADS]);
        double mr_w = miss_rate(
            w->v[IDX_DTLB_MISSES],
            w->v[IDX_DTLB_LOADS]);
        double mr_m = miss_rate(
            m.v[IDX_DTLB_MISSES],
            m.v[IDX_DTLB_LOADS]);
        print_rate("    Load miss rate",
                   mr_t, mr_w, mr_m, hw);
    }
    print_row("  Store misses",
        C(IDX_DTLB_ST_MISSES), measured, hw, 1);

    print_sep();
    /* Stall cycles */
    print_row("Stalled cyc (frontend)",
        C(IDX_STALL_FE), measured, hw, 1);
    print_row("Stalled cyc (backend)",
        C(IDX_STALL_BE), measured, hw, 1);

    print_sep();
    print_row("Branch misses",
        C(IDX_BRANCH_MISSES), measured, hw, 1);

    print_sep();
    print_row("Page faults",
        C(IDX_PAGE_FAULTS), measured, hw, 6);
    print_row("  Minor (in page cache)",
        C(IDX_MINOR_FAULTS), measured, hw, 6);
    print_row("  Major (disk I/O)",
        C(IDX_MAJOR_FAULTS), measured, hw, 6);
    print_row("CPU core migrations",
        C(IDX_CPU_MIGRATIONS), measured, hw, 6);
    print_row("Context switches",
        C(IDX_CTX_SWITCHES), measured, hw, 6);
#undef C

    /* Processinfo timing */
    if (pi && pi->valid)
    {
        /*
         * print_pi_row — one processinfo timing row.
         *
         * When warmup active: shows Total value in
         * col 2, Warmup value in col 3. Measured
         * col is omitted (procinfo is a run snapshot,
         * not per-iteration arithmetic).
         *
         * When no warmup: single value in col 2.
         */
#define print_pi_row(label, tot, wrm, unit) \
        do { \
            long _wv = (long)(wrm); \
            if (hw && pi_w && pi_w->valid) \
                printf( \
                    "  %-*s %*ld %-3s%*ld %-3s\n", \
                    COL1W, (label), \
                    COL2W, (long)(tot), (unit), \
                    COL2W - 1, _wv, (unit)); \
            else \
                printf( \
                    "  %-*s %*ld %-3s\n", \
                    COL1W, (label), \
                    COL2W, (long)(tot), (unit)); \
        } while (0)

        print_sep();
        printf("  --- Timing (processinfo) ---\n");
        print_pi_row("  Iterations counted",
                     pi->loopcnt,
                     (pi_w ? pi_w->loopcnt : 0L),
                     "");
        print_pi_row("  Iter time  p50",
                     pi->p50_iter,
                     (pi_w ? pi_w->p50_iter : 0L),
                     "ns");
        print_pi_row("  Exec time  p50",
                     pi->p50_exec,
                     (pi_w ? pi_w->p50_exec : 0L),
                     "ns");
        print_pi_row("  Iter time  p95",
                     pi->p95_iter,
                     (pi_w ? pi_w->p95_iter : 0L),
                     "ns");
        print_pi_row("  Exec time  p95",
                     pi->p95_exec,
                     (pi_w ? pi_w->p95_exec : 0L),
                     "ns");
        print_pi_row("  Iter time  p99",
                     pi->p99_iter,
                     (pi_w ? pi_w->p99_iter : 0L),
                     "ns");
        print_pi_row("  Exec time  p99",
                     pi->p99_exec,
                     (pi_w ? pi_w->p99_exec : 0L),
                     "ns");
        print_pi_row("  Iter time  p99.9",
                     pi->p999_iter,
                     (pi_w ? pi_w->p999_iter : 0L),
                     "ns");
        print_pi_row("  Exec time  p99.9",
                     pi->p999_exec,
                     (pi_w ? pi_w->p999_exec : 0L),
                     "ns");
        print_pi_row("  Iter time  max",
                     pi->max_iter,
                     (pi_w ? pi_w->max_iter : 0L),
                     "ns");
        print_pi_row("  Exec time  max",
                     pi->max_exec,
                     (pi_w ? pi_w->max_exec : 0L),
                     "ns");
        print_pi_row("  Iter jitter (p99-p50)",
                     pi->jitter_iter,
                     (pi_w ? pi_w->jitter_iter : 0L),
                     "ns");
        print_pi_row("  Exec jitter (p99-p50)",
                     pi->jitter_exec,
                     (pi_w ? pi_w->jitter_exec : 0L),
                     "ns");
        printf("  --- Memory ---\n");
        print_pi_row("  Peak virtual memory",
                     pi->vmpeak_kb, 0L, "kB");
        print_pi_row("  Peak RSS (VmHWM)",
                     pi->vmhwm_kb, 0L, "kB");
        print_pi_row("  Final RSS (VmRSS)",
                     pi->vmrss_kb, 0L, "kB");
        print_pi_row("  Anonymous huge pages",
                     pi->anon_huge_kb,
                     (pi_w ? pi_w->anon_huge_kb : 0L),
                     "kB");
        printf("  --- Scheduling ---\n");
        print_pi_row("  Voluntary ctx switches",
                     pi->vol_ctxt,
                     (pi_w ? pi_w->vol_ctxt : 0L),
                     "");
        print_pi_row("  Nonvol ctx switches",
                     pi->nvol_ctxt,
                     (pi_w ? pi_w->nvol_ctxt : 0L),
                     "");
        printf("  --- CPU Frequency ---\n");
        if (pi->cpu_freq_min_khz > 0)
        {
            print_pi_row(
                "  Min across cores",
                pi->cpu_freq_min_khz / 1000L,
                (pi_w ? pi_w->cpu_freq_min_khz
                            / 1000L
                      : 0L),
                "MHz");
            print_pi_row(
                "  Max across cores",
                pi->cpu_freq_max_khz / 1000L,
                (pi_w ? pi_w->cpu_freq_max_khz
                            / 1000L
                      : 0L),
                "MHz");
        }
        else
            printf("  %-*s N/A\n",
                   COL1W, "  (unavailable)");
        printf("  --- Power (RAPL) ---\n");
        if (pi->rapl_uj >= 0)
        {
            print_pi_row(
                "  Package energy",
                (long)(pi->rapl_uj / 1000LL),
                (pi_w && pi_w->rapl_uj >= 0
                    ? (long)(pi_w->rapl_uj / 1000LL)
                    : 0L),
                "mJ");
        }
        else
            printf("  %-*s N/A (need root or"
                   " perf_event_paranoid<=0)\n",
                   COL1W, "  Package energy");

#undef print_pi_row
    }

    printf("  %-*s %*ld B\n",
           COL1W, "Executable size",
           COL2W, exe_sz);

    printf("======================================================\n");
    printf("  Results: %s\n", cfg->result_file);
    printf("======================================================\n");
    printf("\n");
}

/* ================================================================
 * Cleanup / signal handling
 * ============================================================= */

static void cleanup(void)
{
    if (g_initialized)
        fps_cleanup(&g_cfg);
}

static void sig_handler(int sig)
{
    (void) sig;
    cleanup();
    _exit(128 + sig);
}

/* ================================================================
 * main()
 * ============================================================= */

/**
 * @brief Entry point.
 *
 * Parses arguments, runs FPS lifecycle, collects
 * hardware counters and processinfo data, and writes
 * JSON + human-readable summary.
 */
int main(int argc, char *argv[])
{
    bench_cfg_t *cfg = &g_cfg;
    memset(cfg, 0, sizeof(*cfg));

    /* Defaults */
    strncpy(cfg->outdir, "./perfresults",
            sizeof(cfg->outdir) - 1);

    if (parse_args(argc, argv, cfg) != 0)
        return 1;

    /* Resolve environment */
    resolve_procdir(cfg);
    resolve_git_commit(cfg);
    make_fpsname(cfg->fpsname,
                 sizeof(cfg->fpsname));

    /* Output path */
    {
        time_t now = time(NULL);
        struct tm lt;
        localtime_r(&now, &lt);
        char ts[32];
        strftime(ts, sizeof(ts),
                 "%Y%m%dT%H%M%S", &lt);
        char exe_base[MAX_LABEL];
        strncpy(exe_base, cfg->fpsexec,
                sizeof(exe_base) - 1);
        /* basename in-place */
        char *slash = strrchr(exe_base, '/');
        if (slash)
            memmove(exe_base, slash + 1,
                    strlen(slash));
        snprintf(cfg->result_file,
                 sizeof(cfg->result_file),
                 "%s/%s-%s.json",
                 cfg->outdir, exe_base, ts);
    }

    /* Create output dir */
    run_cmd("mkdir -p '%s'", cfg->outdir);

    long sz = exe_size(cfg->fpsexec);
    int  measured = cfg->nbiter - cfg->warmup;

    /* Read build tags from binary before printing header */
    read_build_tags(
        cfg->fpsexec,
        cfg->build_tags,
        sizeof(cfg->build_tags));

    /* Print header */
    printf("\n");
    printf("===========================================\n");
    printf("  milk-perfbench\n");
    printf("===========================================\n");
    printf("  Command   : %s\n", cfg->fpsexec);
    printf("  FPS name  : %s\n", cfg->fpsname);
    printf("  Build     : %s\n",
           cfg->build_tags[0]
               ? cfg->build_tags
               : "default (no PGO/LTO)");
    printf("  Iterations: %d\n", cfg->nbiter);
    if (cfg->warmup > 0)
    {
        printf("  Warmup    : %d\n", cfg->warmup);
        printf("  Measured  : %d\n", measured);
    }
    printf("  Output    : %s\n",
           cfg->result_file);
    printf("===========================================\n");
    printf("\n");

    /* Register cleanup */
    atexit(cleanup);
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    /* FPS setup */
    printf("[1/4] fpsinit -procinfo ...\n");
    fps_setup(cfg);
    g_initialized = 1;

    /* Stream auto-creation */
    printf("[4/5] Creating missing streams ...\n");
    fps_create_streams(cfg);

    /* Setup command */
    if (cfg->setupcmd[0] != '\0')
    {
        printf("[5/5] Setup: %s\n",
               cfg->setupcmd);
        IGNORE_RESULT(system(cfg->setupcmd));
    }

    /* ---- WARMUP PHASE ---- */
    hw_phase_t ph_warmup, ph_total;
    pi_stats_t pi, pi_warmup;
    long long  w_ns = 0, t_ns = 0;

    memset(&ph_warmup,  0, sizeof(ph_warmup));
    memset(&ph_total,   0, sizeof(ph_total));
    memset(&pi,         0, sizeof(pi));
    memset(&pi_warmup,  0, sizeof(pi_warmup));

    if (cfg->warmup > 0)
    {
        printf("[6/7] Running benchmark ...\n");
        printf("  [warmup] %d iterations ...\n",
               cfg->warmup);
        run_phase(cfg, cfg->warmup,
                  &ph_warmup, &pi_warmup, 1, &w_ns);

        /* Re-init FPS for 2nd run */
        run_cmd("%s %s:fpsinit -procinfo"
                " >/dev/null 2>&1",
                cfg->fpsexec, cfg->fpsname);
        run_cmd("%s %s:confstep >/dev/null 2>&1",
                cfg->fpsexec, cfg->fpsname);
        fps_set(cfg->fpsname,
                "procinfo.enabled", "ON");
        fps_set(cfg->fpsname,
                "procinfo.MeasureTiming", "ON");
        fps_set(cfg->fpsname,
                "procinfo.triggermode", "0");

        printf("  [total ] %d iterations ...\n",
               cfg->nbiter);
        run_phase(cfg, cfg->nbiter,
                  &ph_total, &pi, 1, &t_ns);
    }
    else
    {
        printf("[6/6] Running %d iterations ...\n",
               cfg->nbiter);
        run_phase(cfg, cfg->nbiter,
                  &ph_total, &pi, 1, &t_ns);
        measured = cfg->nbiter;
    }

    /* JSON */
    write_json(cfg,
               &ph_total, &ph_warmup,
               measured, t_ns, w_ns,
               &pi, &pi_warmup, sz);

    /* Summary */
    print_summary(cfg,
                  measured,
                  &ph_total, &ph_warmup,
                  t_ns, w_ns,
                  &pi, &pi_warmup, sz);

    return 0;
}

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef PERFBENCH_H
#define PERFBENCH_H

#include <stdint.h>
#include <linux/perf_event.h>
#include <sys/types.h>

/* ================================================================
 * Constants
 * ============================================================= */

#define MAX_CMD 8192
#define MAX_PATH 2048
#define MAX_LABEL 64
#define POLL_INTERVAL_MS 10
#define POLL_TIMEOUT_MS 30000

/* ================================================================
 * Hardware counter table
 * ============================================================= */

typedef struct
{
    const char *name;   /**< human-readable label      */
    const char *json;   /**< JSON key                  */
    uint32_t    type;   /**< perf_event_attr.type      */
    uint64_t    config; /**< perf_event_attr.config    */
} perf_ev_t;

extern const perf_ev_t PERF_EVS[];

/* Named indices into PERF_EVS[] */
enum
{
    IDX_CYCLES = 0,
    IDX_BUS_CYCLES,
    IDX_INSTRUCTIONS,
    IDX_L1D_LOADS,
    IDX_L1D_MISSES,
    IDX_L1D_STORES,
    IDX_L1I_LOADS,
    IDX_L1I_MISSES,
    IDX_ITLB_MISSES,
    IDX_LLC_LOADS,
    IDX_LLC_MISSES,
    IDX_LLC_STORES,
    IDX_LLC_STORE_MISSES,
    IDX_DTLB_LOADS,
    IDX_DTLB_MISSES,
    IDX_DTLB_ST_MISSES,
    IDX_STALL_FE,
    IDX_STALL_BE,
    IDX_BRANCHES,
    IDX_BRANCH_MISSES,
    IDX_PAGE_FAULTS,
    IDX_MINOR_FAULTS,
    IDX_MAJOR_FAULTS,
    IDX_CPU_MIGRATIONS,
    IDX_CTX_SWITCHES,
    IDX_TASK_CLOCK,
    N_PERF_EVS
};

/* ================================================================
 * Data structures
 * ============================================================= */

/** Raw counter values for one phase */
typedef struct
{
    long long v[N_PERF_EVS]; /**< one slot per PERF_EVS[] */
    int       valid;         /**< 1 if perf_event_open worked */
} hw_phase_t;

/** Processinfo-derived stats */
typedef struct
{
    /* timing percentiles */
    int64_t p50_iter, p95_iter, p99_iter;
    int64_t p999_iter, max_iter;
    int64_t p50_exec, p95_exec, p99_exec;
    int64_t p999_exec, max_exec;
    /* derived jitter (p99 - p50) */
    int64_t jitter_iter, jitter_exec;
    int64_t loopcnt;
    /* memory */
    int64_t vmpeak_kb, vmhwm_kb, vmrss_kb;
    int64_t anon_huge_kb; /* anonymous huge pages */
    /* OS scheduling */
    int64_t vol_ctxt;  /* voluntary context switches   */
    int64_t nvol_ctxt; /* non-voluntary context switches */
    /* CPU frequency during run (kHz) */
    int64_t cpu_freq_min_khz;
    int64_t cpu_freq_max_khz;
    /* RAPL energy (micro-joules, -1 if unavailable) */
    int64_t rapl_uj;
    int64_t exe_size; /* bytes */
    int     valid;
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
    char result_file[MAX_PATH + MAX_LABEL + 64];
    char git_commit[64];
    char build_tags[256]; /* extracted from binary */
} bench_cfg_t;

/* ================================================================
 * Prototypes
 * ============================================================= */

// setup.c
int     run_cmd(const char *fmt, ...);
void    resolve_procdir(bench_cfg_t *cfg);
void    resolve_shmdir(char *shmdir, size_t sz);
void    resolve_git_commit(bench_cfg_t *cfg);
int64_t exe_size(const char *exe);
void    read_build_tags(const char *exe, char *out, size_t outsz);
void    make_fpsname(char *out, size_t sz);
void    fps_set(const char *fpsname, const char *key, const char *val);
void    fps_setup(bench_cfg_t *cfg);
void    fps_create_streams(bench_cfg_t *cfg);
void    fps_cleanup(bench_cfg_t *cfg);

// sysmon.c
void    find_proc_shm(bench_cfg_t *cfg, pid_t pid, char *shm_path, size_t sz);
void    read_proc_mem(pid_t pid, pi_stats_t *st);
void    read_smaps_huge(pid_t pid, pi_stats_t *st);
void    read_cpu_freq(pi_stats_t *st);
int64_t read_rapl_energy(void);
void    read_procinfo_stats(bench_cfg_t *cfg, pid_t pid, pi_stats_t *st);

// perf.c
int  perf_open_all(pid_t pid, int *fds);
void perf_read_close(int *fds, hw_phase_t *hw);

// report.c
void write_json(const bench_cfg_t *cfg,
                const hw_phase_t  *t,
                const hw_phase_t  *w,
                int                measured,
                long long          t_ns,
                long long          w_ns,
                const pi_stats_t  *pi,
                const pi_stats_t  *pi_w,
                int64_t            exe_size);
void print_summary(const bench_cfg_t *cfg,
                   int                measured,
                   const hw_phase_t  *t,
                   const hw_phase_t  *w,
                   long long          t_ns,
                   long long          w_ns,
                   const pi_stats_t  *pi,
                   const pi_stats_t  *pi_w,
                   int64_t            exe_size);

#endif /* PERFBENCH_H */

// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "perfbench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>

/* ================================================================
 * Counter math helpers
 * ============================================================= */

static double ipc(const hw_phase_t *p)
{
    if (!p->valid || p->v[IDX_CYCLES] == 0)
    {
        return 0.0;
    }
    return (double) p->v[IDX_INSTRUCTIONS] / (double) p->v[IDX_CYCLES];
}

static double miss_rate(long long misses, long long loads)
{
    if (loads == 0)
    {
        return 0.0;
    }
    return 100.0 * (double) misses / (double) loads;
}

/**
 * @brief Compute measured = total - warmup.
 *
 * @param m    Output (measured phase)
 * @param t    Total phase
 * @param w    Warmup phase
 */
static void sub_phase(hw_phase_t *m, const hw_phase_t *t, const hw_phase_t *w)
{
    m->valid = t->valid;
    for (int i = 0; i < N_PERF_EVS; i++)
    {
        m->v[i] = t->v[i] - w->v[i];
    }
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
void write_json(const bench_cfg_t *cfg,
                const hw_phase_t  *t,
                const hw_phase_t  *w,
                int                measured,
                long long          t_ns,
                long long          w_ns,
                const pi_stats_t  *pi,
                const pi_stats_t  *pi_w,
                int64_t            exe_sz)
{
    FILE *fp = fopen(cfg->result_file, "w");
    if (!fp)
    {
        fprintf(stderr, "ERROR: cannot write %s: %s\n", cfg->result_file, strerror(errno));
        return;
    }

    /* timestamp */
    time_t    now = time(NULL);
    struct tm utc;
    gmtime_r(&now, &utc);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &utc);

    fprintf(fp, "{\n");
    fprintf(fp, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(fp, "  \"compute_unit\": \"%s\",\n", cfg->fpsexec);
    fprintf(fp, "  \"exe_size_bytes\": %" PRId64 ",\n", exe_sz);
    fprintf(fp, "  \"git_commit\": \"%s\",\n", cfg->git_commit);
    fprintf(fp, "  \"build_tags\": \"%s\",\n", cfg->build_tags[0] ? cfg->build_tags : "default");
    fprintf(fp, "  \"iterations\": %d,\n", cfg->nbiter);
    fprintf(fp, "  \"warmup_iterations\": %d,\n", cfg->warmup);
    fprintf(fp, "  \"measured_iterations\": %d,\n", measured);
    fprintf(fp, "  \"wall_clock_s\": %.9f,\n", (double) t_ns / 1e9);
    fprintf(fp, "  \"warmup_s\": %.9f,\n", (double) w_ns / 1e9);

    /* hw_counters */
    fprintf(fp, "  \"hw_counters\": {\n");
    if (t->valid)
    {
        for (int i = 0; i < N_PERF_EVS; i++)
        {
            fprintf(fp, "    \"%s\": %lld%s\n", PERF_EVS[i].json, t->v[i],
                    (i < N_PERF_EVS - 1) ? "," : "");
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
                w->v[IDX_CYCLES], w->v[IDX_INSTRUCTIONS], w->v[IDX_L1D_MISSES],
                w->v[IDX_LLC_MISSES], w->v[IDX_BRANCH_MISSES], w->v[IDX_PAGE_FAULTS]);
    }

    /* processinfo - total */
    if (pi && pi->valid)
    {
        fprintf(fp,
                "  \"processinfo\": {\n"
                "    \"loopcnt\": %" PRId64 ",\n"
                "    \"p50_iter_ns\": %" PRId64 ",\n"
                "    \"p95_iter_ns\": %" PRId64 ",\n"
                "    \"p99_iter_ns\": %" PRId64 ",\n"
                "    \"p999_iter_ns\": %" PRId64 ",\n"
                "    \"max_iter_ns\": %" PRId64 ",\n"
                "    \"jitter_iter_ns\": %" PRId64 ",\n"
                "    \"p50_exec_ns\": %" PRId64 ",\n"
                "    \"p95_exec_ns\": %" PRId64 ",\n"
                "    \"p99_exec_ns\": %" PRId64 ",\n"
                "    \"p999_exec_ns\": %" PRId64 ",\n"
                "    \"max_exec_ns\": %" PRId64 ",\n"
                "    \"jitter_exec_ns\": %" PRId64 ",\n"
                "    \"vmpeak_kb\": %" PRId64 ",\n"
                "    \"vmhwm_kb\": %" PRId64 ",\n"
                "    \"vmrss_kb\": %" PRId64 ",\n"
                "    \"anon_huge_kb\": %" PRId64 ",\n"
                "    \"vol_ctxt_switches\": %" PRId64 ",\n"
                "    \"nvol_ctxt_switches\": %" PRId64 ",\n"
                "    \"cpu_freq_min_khz\": %" PRId64 ",\n"
                "    \"cpu_freq_max_khz\": %" PRId64 ",\n"
                "    \"rapl_energy_uj\": %" PRId64 "\n"
                "  },\n",
                pi->loopcnt, pi->p50_iter, pi->p95_iter, pi->p99_iter, pi->p999_iter, pi->max_iter,
                pi->jitter_iter, pi->p50_exec, pi->p95_exec, pi->p99_exec, pi->p999_exec,
                pi->max_exec, pi->jitter_exec, pi->vmpeak_kb, pi->vmhwm_kb, pi->vmrss_kb,
                pi->anon_huge_kb, pi->vol_ctxt, pi->nvol_ctxt, pi->cpu_freq_min_khz,
                pi->cpu_freq_max_khz, pi->rapl_uj);
    }
    else
    {
        fprintf(fp, "  \"processinfo\": null,\n");
    }
    /* processinfo - warmup */
    if (pi_w && pi_w->valid)
    {
        fprintf(fp,
                "  \"processinfo_warmup\": {\n"
                "    \"loopcnt\": %" PRId64 ",\n"
                "    \"p50_iter_ns\": %" PRId64 ",\n"
                "    \"p95_iter_ns\": %" PRId64 ",\n"
                "    \"p99_iter_ns\": %" PRId64 ",\n"
                "    \"p999_iter_ns\": %" PRId64 ",\n"
                "    \"max_iter_ns\": %" PRId64 ",\n"
                "    \"jitter_iter_ns\": %" PRId64 ",\n"
                "    \"p50_exec_ns\": %" PRId64 ",\n"
                "    \"p95_exec_ns\": %" PRId64 ",\n"
                "    \"p99_exec_ns\": %" PRId64 ",\n"
                "    \"p999_exec_ns\": %" PRId64 ",\n"
                "    \"max_exec_ns\": %" PRId64 ",\n"
                "    \"jitter_exec_ns\": %" PRId64 ",\n"
                "    \"vol_ctxt_switches\": %" PRId64 ",\n"
                "    \"nvol_ctxt_switches\": %" PRId64 ",\n"
                "    \"cpu_freq_min_khz\": %" PRId64 ",\n"
                "    \"cpu_freq_max_khz\": %" PRId64 ",\n"
                "    \"rapl_energy_uj\": %" PRId64 "\n"
                "  }\n",
                pi_w->loopcnt, pi_w->p50_iter, pi_w->p95_iter, pi_w->p99_iter, pi_w->p999_iter,
                pi_w->max_iter, pi_w->jitter_iter, pi_w->p50_exec, pi_w->p95_exec, pi_w->p99_exec,
                pi_w->p999_exec, pi_w->max_exec, pi_w->jitter_exec, pi_w->vol_ctxt, pi_w->nvol_ctxt,
                pi_w->cpu_freq_min_khz, pi_w->cpu_freq_max_khz, pi_w->rapl_uj);
    }
    else
    {
        fprintf(fp, "  \"processinfo_warmup\": null\n");
    }

    fprintf(fp, "}\n");
    fclose(fp);
}

/* ================================================================
 * Human-readable summary
 * ============================================================= */

/*
 * ANSI escape sequences for terminal styling.
 * All output gracefully degrades if stdout is
 * redirected (no color codes in piped output).
 */
#define ANSI_RESET "\033[0m"
#define ANSI_BOLD "\033[1m"
#define ANSI_DIM "\033[2m"
#define ANSI_CYAN "\033[36m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_GREEN "\033[32m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_WHITE "\033[97m"

/* Check once at startup whether stdout is a tty */
static int g_use_color = 0;

/* Width constants */
#define COL1W 28 /* label column */
#define COL2W 14 /* value column */

/* Box-drawing strings */
#define BOX_HEAVY "\342\224\201" /* ━ */
#define BOX_LIGHT "\342\224\200" /* ─ */
#define SEP_WIDE                                                                                   \
    "  " BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT \
        BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT  \
            BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT        \
                BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT    \
                    BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT          \
                        BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT BOX_LIGHT      \
                            BOX_LIGHT

/* color helper macros — emit codes only when tty */
#define C(code) (g_use_color ? code : "")
#define CR C(ANSI_RESET)
#define CB C(ANSI_BOLD)
#define CD C(ANSI_DIM)
#define CCY C(ANSI_CYAN)
#define CYL C(ANSI_YELLOW)
#define CGR C(ANSI_GREEN)
#define CMG C(ANSI_MAGENTA)
#define CWH C(ANSI_WHITE)

/**
 * @brief Print a heavy separator line (section boundary).
 */
static void print_heavy_sep(void)
{
    printf("%s  ", CB);
    for (int i = 0; i < 54; i++)
    {
        printf("%s", BOX_HEAVY);
    }
    printf("%s\n", CR);
}

/**
 * @brief Print a light separator line.
 */
static void print_sep(void)
{
    printf("%s%s%s\n", CD, SEP_WIDE, CR);
}

/**
 * @brief Print a colored section header.
 *
 * @param title  Section label, e.g. "L1 Data Cache"
 */
static void print_section(const char *title)
{
    printf("\n  %s\342\226\270 %s%s\n", CCY, title, CR);
}

/**
 * @brief Print a counter row: label, total,
 *        [warmup,] per-measured-iter.
 *
 * @param paired_warmup_misses  Companion "misses" counter warmup
 *   value.  When > 0 and warmup_v == 0, the loads counter was
 *   multiplexed out by the PMU scheduler.  Pass 0 to disable
 *   detection (e.g. for misses rows themselves).
 */
static void print_row(const char *label,
                      long long   total,
                      long long   warmup_v,
                      int         measured,
                      int         has_warmup,
                      int         decimals,
                      long long   paired_warmup_misses)
{
    /* Detect PMU multiplexing: loads counter got zero samples
     * during warmup while its companion misses counter did not. */
    int mux_out = has_warmup && (warmup_v == 0) && (paired_warmup_misses > 0);

    double per_iter =
        (measured > 0 && total > warmup_v) ? (double) (total - warmup_v) / (double) measured : 0.0;

    if (has_warmup)
    {
        if (mux_out)
        {
            /* Warmup column: show "n/a" to flag multiplexed-out
             * PMU counter rather than a misleading zero. */
            if (decimals == 6)
            {
                printf("  %s%-*s%s %*lld %s%*s%s %s%*.6f%s/iter\n", CD, COL1W, label, CR, COL2W,
                       total, CD, COL2W, "n/a", CR, CYL, COL2W, per_iter, CR);
            }
            else
            {
                printf("  %s%-*s%s %*lld %s%*s%s %s%*.1f%s/iter\n", CD, COL1W, label, CR, COL2W,
                       total, CD, COL2W, "n/a", CR, CYL, COL2W, per_iter, CR);
            }
        }
        else
        {
            if (decimals == 6)
            {
                printf("  %s%-*s%s %*lld %*lld %s%*.6f%s/iter\n", CD, COL1W, label, CR, COL2W,
                       total, COL2W, warmup_v, CYL, COL2W, per_iter, CR);
            }
            else
            {
                printf("  %s%-*s%s %*lld %*lld %s%*.1f%s/iter\n", CD, COL1W, label, CR, COL2W,
                       total, COL2W, warmup_v, CYL, COL2W, per_iter, CR);
            }
        }
    }
    else
    {
        if (decimals == 6)
        {
            printf("  %s%-*s%s %*lld %s%*.6f%s/iter\n", CD, COL1W, label, CR, COL2W, total, CYL,
                   COL2W, per_iter, CR);
        }
        else
        {
            printf("  %s%-*s%s %*lld %s%*.1f%s/iter\n", CD, COL1W, label, CR, COL2W, total, CYL,
                   COL2W, per_iter, CR);
        }
    }
}

/**
 * @brief Print a miss-rate percentage row.
 *
 * @param loads_mux_out  When non-zero the loads counter was
 *   multiplexed out during warmup, so the warmup miss rate
 *   cannot be computed — show "n/a" instead of 0.000%%.
 */
static void print_rate(const char *label,
                       double      rate_t,
                       double      rate_w,
                       double      rate_m,
                       int         has_warmup,
                       int         loads_mux_out)
{
    if (has_warmup)
    {
        if (loads_mux_out)
        {
            printf("  %s%-*s%s %s%*.3f%%%s "
                   "%s%*s%s %s%*.3f%%%s\n",
                   CD, COL1W, label, CR, CD, COL2W - 1, rate_t, CR, CD, COL2W, "n/a", CR, CMG,
                   COL2W - 1, rate_m, CR);
        }
        else
        {
            printf("  %s%-*s%s %s%*.3f%%%s "
                   "%s%*.3f%%%s %s%*.3f%%%s\n",
                   CD, COL1W, label, CR, CD, COL2W - 1, rate_t, CR, CD, COL2W - 1, rate_w, CR, CMG,
                   COL2W - 1, rate_m, CR);
        }
    }
    else
    {
        printf("  %s%-*s%s %s%*.3f%%%s\n", CD, COL1W, label, CR, CMG, COL2W - 1, rate_t, CR);
    }
}

/**
 * @brief Print the full human-readable summary.
 */
void print_summary(const bench_cfg_t *cfg,
                   int                measured,
                   const hw_phase_t  *t,
                   const hw_phase_t  *w,
                   long long          t_ns,
                   long long          w_ns,
                   const pi_stats_t  *pi,
                   const pi_stats_t  *pi_w,
                   int64_t            exe_sz)
{
    /* detect color support once */
    g_use_color = isatty(STDOUT_FILENO);

    int        hw = (cfg->warmup > 0);
    hw_phase_t m;
    sub_phase(&m, t, w);

    print_heavy_sep();
    printf("  %s%-*s%s %s%-*s%s %s%-*s%s\n", CB, COL1W, "", CR, CB, COL2W, "Total", CR,
           (hw ? CB : CD), COL2W, (hw ? "Warmup" : ""), CR);
    printf("  %s%-*s%s %s%-*s%s %s%-*s%s\n", CB, COL1W, "Benchmark Results", CR, CD, COL2W, "", CR,
           CB, COL2W, "Measured", CR);
    printf("  %s%-*s%s  %s%d iter%s, "
           "%sBuild:%s %s\n",
           CD, COL1W,
           cfg->fpsexec[0] == '/' || cfg->fpsexec[0] == '.' ? (strrchr(cfg->fpsexec, '/') + 1)
                                                            : cfg->fpsexec,
           CR, CD, measured, CR, CD, CR, cfg->build_tags[0] ? cfg->build_tags : "default");
    print_heavy_sep();

    /* Column header */
    printf("  %s%-*s%s %s%*s%s %s%*s%s %s%*s%s\n", CD, COL1W, "", CR, CD, COL2W, "total", CR, CD,
           COL2W, hw ? "warmup" : "", CR, CB, COL2W, "per-iter", CR);
    print_sep();

    /* Wall clock */
    {
        long long meas_ns = t_ns - w_ns;
        double    meas_pi = (measured > 0) ? (double) meas_ns / measured : 0.0;
        /* display timing in µs when >= 1000 ns */
        int         use_us = (meas_pi >= 1000.0);
        double      scale  = use_us ? 1e3 : 1.0;
        const char *unit   = use_us ? "µs" : "ns";

        if (hw)
        {
            printf("  %s%-*s%s %*.3f s %*.3f s "
                   "%s%*.1f %s/iter%s\n",
                   CB, COL1W, "Wall clock", CR, COL2W - 2, (double) t_ns / 1e9, COL2W - 2,
                   (double) w_ns / 1e9, CWH, COL2W - 2, meas_pi / scale, unit, CR);
        }
        else
        {
            printf("  %s%-*s%s %*.3f s %s%*.1f %s/iter%s\n", CB, COL1W, "Wall clock", CR, COL2W - 2,
                   (double) t_ns / 1e9, CWH, COL2W - 2, (double) t_ns / measured / scale, unit, CR);
        }
    }

#define C_VAL(idx) t->v[idx], w->v[idx]
    print_row("Cycles", C_VAL(IDX_CYCLES), measured, hw, 1, 0);
    print_row("Instructions", C_VAL(IDX_INSTRUCTIONS), measured, hw, 1, 0);

    /* IPC */
    {
        double ipc_t = ipc(t);
        double ipc_w = ipc(w);
        double ipc_m = ipc(&m);
        if (hw)
        {
            printf("  %s%-*s%s %*.3f %*.3f %s%*.3f%s\n", CD, COL1W, "Instr per Cycle (IPC)", CR,
                   COL2W, ipc_t, COL2W, ipc_w, CGR, COL2W, ipc_m, CR);
        }
        else
        {
            printf("  %s%-*s%s %s%*.3f%s\n", CD, COL1W, "Instr per Cycle (IPC)", CR, CGR, COL2W,
                   ipc_t, CR);
        }
    }
    print_row("Branch misses", C_VAL(IDX_BRANCH_MISSES), measured, hw, 1, 0);
    {
        long long stall_fe = t->v[IDX_STALL_FE];
        long long stall_be = t->v[IDX_STALL_BE];
        if (stall_fe > 0 || stall_be > 0)
        {
            print_row("Stalled cyc (FE)", C_VAL(IDX_STALL_FE), measured, hw, 1, 0);
            print_row("Stalled cyc (BE)", C_VAL(IDX_STALL_BE), measured, hw, 1, 0);
        }
    }

    print_section("Cache");
    print_sep();
    /* L1 Data */
    printf("    %sL1 Data%s\n", CD, CR);
    print_row("      Loads", C_VAL(IDX_L1D_LOADS), measured, hw, 1, w->v[IDX_L1D_MISSES]);
    print_row("      Load misses", C_VAL(IDX_L1D_MISSES), measured, hw, 1, 0);
    {
        int    l1d_mux = hw && (w->v[IDX_L1D_LOADS] == 0) && (w->v[IDX_L1D_MISSES] > 0);
        double mr_t    = miss_rate(t->v[IDX_L1D_MISSES], t->v[IDX_L1D_LOADS]);
        double mr_w    = miss_rate(w->v[IDX_L1D_MISSES], w->v[IDX_L1D_LOADS]);
        double mr_m    = miss_rate(m.v[IDX_L1D_MISSES], m.v[IDX_L1D_LOADS]);
        print_rate("        miss rate", mr_t, mr_w, mr_m, hw, l1d_mux);
    }
    /* L1 Instruction */
    printf("    %sL1 Instruction%s\n", CD, CR);
    print_row("      Loads", C_VAL(IDX_L1I_LOADS), measured, hw, 1, w->v[IDX_L1I_MISSES]);
    print_row("      Load misses", C_VAL(IDX_L1I_MISSES), measured, hw, 1, 0);
    {
        int    l1i_mux = hw && (w->v[IDX_L1I_LOADS] == 0) && (w->v[IDX_L1I_MISSES] > 0);
        double mr_t    = miss_rate(t->v[IDX_L1I_MISSES], t->v[IDX_L1I_LOADS]);
        double mr_w    = miss_rate(w->v[IDX_L1I_MISSES], w->v[IDX_L1I_LOADS]);
        double mr_m    = miss_rate(m.v[IDX_L1I_MISSES], m.v[IDX_L1I_LOADS]);
        print_rate("        miss rate", mr_t, mr_w, mr_m, hw, l1i_mux);
    }
    /* LLC */
    printf("    %sLast Level Cache%s\n", CD, CR);
    print_row("      Loads", C_VAL(IDX_LLC_LOADS), measured, hw, 1, w->v[IDX_LLC_MISSES]);
    print_row("      Load misses", C_VAL(IDX_LLC_MISSES), measured, hw, 1, 0);
    {
        int    llc_mux = hw && (w->v[IDX_LLC_LOADS] == 0) && (w->v[IDX_LLC_MISSES] > 0);
        double mr_t    = miss_rate(t->v[IDX_LLC_MISSES], t->v[IDX_LLC_LOADS]);
        double mr_w    = miss_rate(w->v[IDX_LLC_MISSES], w->v[IDX_LLC_LOADS]);
        double mr_m    = miss_rate(m.v[IDX_LLC_MISSES], m.v[IDX_LLC_LOADS]);
        print_rate("        load miss rate", mr_t, mr_w, mr_m, hw, llc_mux);
    }

    print_section("TLB");
    print_sep();
    printf("    %sData TLB%s\n", CD, CR);
    print_row("      Loads", C_VAL(IDX_DTLB_LOADS), measured, hw, 1, w->v[IDX_DTLB_MISSES]);
    print_row("      Load misses", C_VAL(IDX_DTLB_MISSES), measured, hw, 1, 0);
    {
        int    dtlb_mux = hw && (w->v[IDX_DTLB_LOADS] == 0) && (w->v[IDX_DTLB_MISSES] > 0);
        double mr_t     = miss_rate(t->v[IDX_DTLB_MISSES], t->v[IDX_DTLB_LOADS]);
        double mr_w     = miss_rate(w->v[IDX_DTLB_MISSES], w->v[IDX_DTLB_LOADS]);
        double mr_m     = miss_rate(m.v[IDX_DTLB_MISSES], m.v[IDX_DTLB_LOADS]);
        print_rate("        load miss rate", mr_t, mr_w, mr_m, hw, dtlb_mux);
    }
    printf("    %sInstruction TLB%s\n", CD, CR);
    print_row("      Load misses", C_VAL(IDX_ITLB_MISSES), measured, hw, 1, 0);

    print_section("OS Events");
    print_sep();
    print_row("  Page faults (minor)", C_VAL(IDX_MINOR_FAULTS), measured, hw, 6, 0);
    print_row("  Page faults (major)", C_VAL(IDX_MAJOR_FAULTS), measured, hw, 6, 0);
    print_row("  CPU migrations", C_VAL(IDX_CPU_MIGRATIONS), measured, hw, 6, 0);
    print_row("  Context switches", C_VAL(IDX_CTX_SWITCHES), measured, hw, 6, 0);
#undef C_VAL

    /* Processinfo timing */
    if (pi && pi->valid)
    {
        /*
         * print_pi_row — one processinfo timing row.
         */
#define print_pi_row(label, tot, wrm, unit)                                                       \
    do                                                                                            \
    {                                                                                             \
        int64_t _wv = (int64_t) (wrm);                                                            \
        if (hw && pi_w && pi_w->valid)                                                            \
            printf("  %s%-*s%s %s%*" PRId64 "%s %-3s%*" PRId64 " %-3s\n", CD, COL1W, (label), CR, \
                   CWH, COL2W, (int64_t) (tot), CR, (unit), COL2W - 1, _wv, (unit));              \
        else                                                                                      \
            printf("  %s%-*s%s %s%*" PRId64 "%s %-3s\n", CD, COL1W, (label), CR, CWH, COL2W,      \
                   (int64_t) (tot), CR, (unit));                                                  \
    } while (0)

        print_section("Timing  (processinfo ring buffer)");
        print_sep();
        printf("  %s%-*s%s %s%*" PRId64 "%s\n", CD, COL1W, "  Iterations counted", CR, CWH, COL2W,
               (int64_t) pi->loopcnt, CR);

        /* Scale ns → µs for display if values > 5000 ns */
        int64_t     scale = (pi->p50_iter > 5000LL) ? 1000LL : 1LL;
        const char *tunit = (scale == 1000LL) ? "µs" : "ns";

#define print_timing_row(lbl, tot, wrm)                                                            \
    do                                                                                             \
    {                                                                                              \
        int64_t _t = (int64_t) (tot) / scale;                                                      \
        int64_t _w = (int64_t) (wrm) / scale;                                                      \
        if (hw && pi_w && pi_w->valid)                                                             \
            printf("  %s%-*s%s %s%*" PRId64 "%s %-3s%*" PRId64 " %-3s\n", CD, COL1W, (lbl), CR,    \
                   CWH, COL2W, _t, CR, tunit, COL2W - 1, _w, tunit);                               \
        else                                                                                       \
            printf("  %s%-*s%s %s%*" PRId64 "%s %-3s\n", CD, COL1W, (lbl), CR, CWH, COL2W, _t, CR, \
                   tunit);                                                                         \
    } while (0)

        /* Column header for timing table */
        printf("  %s  %-28s %14s %14s %s%14s%s\n", CD, "", "p50", "p95", CWH, "p99", CR);
        /* Iter row */
        printf("  %s%-*s%s %s%14" PRId64 "%s %-3s%14" PRId64 " %-3s"
               "%s%14" PRId64 "%s %-3s\n",
               CD, COL1W, "  Iter time", CR, CD, (int64_t) pi->p50_iter / scale, CR, tunit,
               (int64_t) pi->p95_iter / scale, tunit, CWH, (int64_t) pi->p99_iter / scale, CR,
               tunit);
        /* Exec row */
        printf("  %s%-*s%s %s%14" PRId64 "%s %-3s%14" PRId64 " %-3s"
               "%s%14" PRId64 "%s %-3s\n",
               CD, COL1W, "  Exec time", CR, CD, (int64_t) pi->p50_exec / scale, CR, tunit,
               (int64_t) pi->p95_exec / scale, tunit, CWH, (int64_t) pi->p99_exec / scale, CR,
               tunit);
        /* Jitter + max */
        printf("  %s%-*s%s %s%14" PRId64 "%s %-3s   "
               "%s%-*s%s %s%14" PRId64 "%s %-3s\n",
               CD, COL1W, "  Iter jitter (p99-p50)", CR, CWH, (int64_t) pi->jitter_iter / scale, CR,
               tunit, CD, COL1W, "  Exec jitter (p99-p50)", CR, CWH,
               (int64_t) pi->jitter_exec / scale, CR, tunit);
        printf("  %s%-*s%s %s%14" PRId64 "%s %-3s   "
               "%s%-*s%s %s%14" PRId64 "%s %-3s\n",
               CD, COL1W, "  Iter max", CR, CWH, (int64_t) pi->max_iter / scale, CR, tunit, CD,
               COL1W, "  Exec max", CR, CWH, (int64_t) pi->max_exec / scale, CR, tunit);
#undef print_timing_row

        print_section("Memory & OS");
        print_sep();
        print_pi_row("  Peak RSS", pi->vmhwm_kb, 0LL, "kB");
        print_pi_row("  Virtual peak", pi->vmpeak_kb, 0LL, "kB");
        if (pi->anon_huge_kb > 0)
        {
            print_pi_row("  Anon huge pages", pi->anon_huge_kb, 0LL, "kB");
        }
        print_pi_row("  Vol ctx switches", pi->vol_ctxt, (pi_w ? pi_w->vol_ctxt : 0LL), "");
        print_pi_row("  Nonvol ctx switches", pi->nvol_ctxt, (pi_w ? pi_w->nvol_ctxt : 0LL), "");
        if (pi->cpu_freq_min_khz > 0)
        {
            print_pi_row("  CPU freq (min)", pi->cpu_freq_min_khz / 1000LL,
                         (pi_w ? pi_w->cpu_freq_min_khz / 1000LL : 0LL), "MHz");
            print_pi_row("  CPU freq (max)", pi->cpu_freq_max_khz / 1000LL,
                         (pi_w ? pi_w->cpu_freq_max_khz / 1000LL : 0LL), "MHz");
        }
        if (pi->rapl_uj >= 0)
        {
            print_pi_row("  RAPL (package)", (int64_t) (pi->rapl_uj / 1000LL),
                         (pi_w && pi_w->rapl_uj >= 0 ? (int64_t) (pi_w->rapl_uj / 1000LL) : 0LL),
                         "mJ");
        }
        else
        {
            printf("  %s%-*s%s %sN/A%s "
                   "%s(need root or "
                   "perf_event_paranoid<=0)%s\n",
                   CD, COL1W, "  RAPL (package)", CR, CYL, CR, CD, CR);
        }

#undef print_pi_row
    }

    /* Executable info */
    print_heavy_sep();
    printf("  %s%-*s%s %s%*" PRId64 "%s B   "
           "%s%-*s%s %s%s%s\n",
           CD, COL1W, "Executable size", CR, CWH, COL2W, exe_sz, CR, CD, COL1W - 4, "Results", CR,
           CCY, cfg->result_file, CR);
    print_heavy_sep();
    printf("\n");
}

#include "perfbench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <limits.h>
#include <inttypes.h>

#include "processinfo.h"

#ifndef IGNORE_RESULT
#define IGNORE_RESULT(x) do { if (x) {} } while (0)
#endif

/* ================================================================
 * Processinfo SHM helpers
 * ============================================================= */

/** Compare two int64_t for qsort */
static int cmp_int64(const void *a, const void *b)
{
    int64_t la = *(const int64_t *) a;
    int64_t lb = *(const int64_t *) b;
    return (la > lb) - (la < lb);
}

/**
 * @brief Find proc.*.shm in procdir matching a PID.
 *
 * @param cfg      Benchmark configuration (uses procdir)
 * @param pid      PID to match (0 = any)
 * @param out      Output path buffer
 * @param outsz    Size of out
 * @return 1 if found, 0 otherwise
 */
void find_proc_shm(
    bench_cfg_t *cfg,
    pid_t       pid,
    char        *out,
    size_t      outsz)
{
    DIR *d = opendir(cfg->procdir);
    if (!d)
        return;

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
            tmp[sizeof(tmp)-1] = '\0';
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
                 cfg->procdir, de->d_name);
        found = 1;
        break;
    }
    closedir(d);
    
    // In order to make it return something, since I changed the prototype:
    if (!found) {
        out[0] = '\0';
    }
}

/**
 * @brief Read memory + scheduling stats from
 *        /proc/PID/status.
 */
void read_proc_mem(
    pid_t      pid,
    pi_stats_t *st)
{
    st->vmpeak_kb = -1;
    st->vmhwm_kb  = -1;
    st->vmrss_kb  = -1;
    st->vol_ctxt  = -1;
    st->nvol_ctxt = -1;

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
            sscanf(line + 7, " %" SCNd64, &st->vmpeak_kb);
        else if (strncmp(line, "VmHWM:", 6) == 0)
            sscanf(line + 6, " %" SCNd64, &st->vmhwm_kb);
        else if (strncmp(line, "VmRSS:", 6) == 0)
            sscanf(line + 6, " %" SCNd64, &st->vmrss_kb);
        else if (strncmp(line,
                         "voluntary_ctxt_switches:",
                         24) == 0)
            sscanf(line + 24, " %" SCNd64, &st->vol_ctxt);
        else if (strncmp(line,
                         "nonvoluntary_ctxt_switches:",
                         27) == 0)
            sscanf(line + 27, " %" SCNd64, &st->nvol_ctxt);
    }
    fclose(fp);
}

/**
 * @brief Read anonymous huge-page usage from
 *        /proc/PID/smaps_rollup.
 *
 * @param pid   Target process
 * @param st    Stats structure to fill
 */
void read_smaps_huge(pid_t pid, pi_stats_t *st)
{
    st->anon_huge_kb = 0;
    char path[128];
    snprintf(path, sizeof(path),
             "/proc/%d/smaps_rollup", (int) pid);
    FILE *fp = fopen(path, "r");
    if (!fp)
        return;
    char line[256];
    int64_t val = 0;
    while (fgets(line, sizeof(line), fp))
    {
        if (strncmp(line, "AnonHugePages:", 14) == 0)
        {
            sscanf(line + 14, " %" SCNd64, &val);
            break;
        }
    }
    fclose(fp);
    st->anon_huge_kb = val;
}

/**
 * @brief Sample CPU frequency from sysfs.
 *
 * Reads scaling_cur_freq for every online CPU,
 * returns min and max observed values in kHz.
 * Falls back to cpuinfo_cur_freq if scaling_cur
 * is absent.
 */
void read_cpu_freq(pi_stats_t *st)
{
    st->cpu_freq_min_khz = -1;
    st->cpu_freq_max_khz = -1;
    int64_t fmin = INT64_MAX;
    int64_t fmax = 0;
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
        int64_t f = 0;
        if (fscanf(fp, "%" SCNd64, &f) == 1 && f > 0)
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
        st->cpu_freq_min_khz = fmin;
        st->cpu_freq_max_khz = fmax;
    }
}

/**
 * @brief Read RAPL package energy counter.
 *
 * Reads /sys/class/powercap/intel-rapl/intel-rapl:0/
 * energy_uj.  Returns -1 if unavailable or if
 * access is denied.
 *
 * @param uj Pointer to store result
 */
int64_t read_rapl_energy(void)
{
    const char *rapl =
        "/sys/class/powercap/intel-rapl/"
        "intel-rapl:0/energy_uj";
    FILE *fp = fopen(rapl, "r");
    if (!fp)
        return -1LL;
    int64_t val = -1LL;
    IGNORE_RESULT(fscanf(fp, "%" SCNd64, &val));
    fclose(fp);
    return val;
}

/**
 * @brief Compute percentile stats from PROCESSINFO.
 *
 * Maps the proc SHM file and extracts p50/p95/p99
 * for both iteration time and execution time.
 */
void read_procinfo_stats(
    bench_cfg_t *cfg,
    pid_t       child_pid,
    pi_stats_t  *out)
{
    out->valid = 0;
    
    char shm_path[MAX_PATH];
    find_proc_shm(cfg, child_pid, shm_path, sizeof(shm_path));
    if (shm_path[0] == '\0') {
        return;
    }
    
    /* Capture RAPL energy baseline for delta */
    long long rapl_start = read_rapl_energy();

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

    int64_t iter_ns[PROCESSINFO_NBtimer];
    int64_t exec_ns[PROCESSINFO_NBtimer];
    int  nv = 0;

    for (int i = 1; i < nbsam; i++)
    {
        int64_t dt_exec =
            (int64_t)(pi->texecend[i].tv_sec
             - pi->texecstart[i].tv_sec)
            * 1000000000LL
            + (int64_t)(pi->texecend[i].tv_nsec
               - pi->texecstart[i].tv_nsec);
        int64_t dt_iter =
            (int64_t)(pi->texecstart[i].tv_sec
             - pi->texecstart[i-1].tv_sec)
            * 1000000000LL
            + (int64_t)(pi->texecstart[i].tv_nsec
               - pi->texecstart[i-1].tv_nsec);
        /* Reject negative, zero, or implausibly
         * large values (stale ring buffer entries
         * from a previous FPS session). */
        if (dt_exec > 0 && dt_iter > 0
            && dt_exec < 10000000000LL
            && dt_iter < 10000000000LL)
        {
            exec_ns[nv] = dt_exec;
            iter_ns[nv] = dt_iter;
            nv++;
        }
    }

    out->loopcnt = pi->loopcnt;

    /* Read memory + scheduling stats */
    read_proc_mem(child_pid, out);

    /* Anonymous huge pages */
    read_smaps_huge(child_pid, out);

    /* CPU frequency (min/max across all CPUs) */
    read_cpu_freq(out);

    /* RAPL energy delta */
    {
        int64_t rapl_end = read_rapl_energy();
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
          sizeof(int64_t), cmp_int64);
    qsort(iter_ns,  (size_t) nv,
          sizeof(int64_t), cmp_int64);

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

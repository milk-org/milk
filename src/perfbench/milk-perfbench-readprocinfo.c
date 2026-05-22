/**
 * @file    milk-perfbench-readprocinfo.c
 * @brief   Read processinfo SHM and output timing
 *          metrics as JSON.
 *
 * This utility maps a processinfo shared-memory file
 * and extracts timing data (iteration/exec medians,
 * timing circular buffer percentiles, loop count).
 * It also reads VmPeak from /proc/<PID>/status.
 *
 * Output is JSON on stdout, intended for consumption
 * by the milk-perfbench harness script.
 *
 * Usage:
 *   milk-perfbench-readprocinfo <procinfo_shm_path>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <libgen.h>

#include "milk_help.h"
#include "processinfo.h"

/**
 * @brief Compare two longs for qsort.
 */
static int cmp_long(const void *a, const void *b)
{
    long la = *(const long *) a;
    long lb = *(const long *) b;

    if (la < lb)
    {
        return -1;
    }
    if (la > lb)
    {
        return 1;
    }
    return 0;
}


/** Memory stats from /proc/PID/status */
struct proc_mem
{
    long vmpeak_kb; /**< peak virtual memory */
    long vmhwm_kb;  /**< peak resident set size */
    long vmrss_kb;  /**< current RSS at readout */
};


/**
 * @brief Read memory stats from /proc/PID/status.
 *
 * @param pid   Process ID to query
 * @param[out] mem  Filled with VmPeak, VmHWM, VmRSS
 * @return      0 on success, -1 on failure
 */
static int read_proc_memory(pid_t pid, struct proc_mem *mem)
{
    char path[128];
    snprintf(path, sizeof(path), "/proc/%d/status", (int) pid);

    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return -1;
    }

    mem->vmpeak_kb = -1;
    mem->vmhwm_kb  = -1;
    mem->vmrss_kb  = -1;

    char line[256];
    int  found = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (strncmp(line, "VmPeak:", 7) == 0)
        {
            sscanf(line + 7, " %ld", &mem->vmpeak_kb);
            found++;
        }
        else if (strncmp(line, "VmHWM:", 6) == 0)
        {
            sscanf(line + 6, " %ld", &mem->vmhwm_kb);
            found++;
        }
        else if (strncmp(line, "VmRSS:", 6) == 0)
        {
            sscanf(line + 6, " %ld", &mem->vmrss_kb);
            found++;
        }
        if (found >= 3)
        {
            break;
        }
    }

    fclose(fp);
    return 0;
}


/**
 * @brief Compute elapsed nanoseconds between two
 *        timespec values.
 */
static long timespec_diff_ns(struct timespec start, struct timespec end)
{
    long sec  = end.tv_sec - start.tv_sec;
    long nsec = end.tv_nsec - start.tv_nsec;

    return sec * 1000000000L + nsec;
}


static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, "read processinfo shared memory and output timing metrics as JSON",
                     mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s<procinfo_shm_path>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  Maps a processinfo shared-memory file and extracts timing data (iteration/exec "
           "medians,\n"
           "  timing circular buffer percentiles, loop count). It also reads VmPeak from "
           "/proc/<PID>/status.\n"
           "  Output is JSON on stdout, intended for consumption by the milk-perfbench harness "
           "script.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");

    const char *see_also[] = { "milk-perfbench:run milk performance benchmarks",
                               "milk-procinfo-info:inspect processinfo memory contents" };
    milk_help_see_also(see_also, 2, mh_color);
}


/**
 * @brief Main entry point.
 *
 * Opens the processinfo SHM file specified on the
 * command line, reads timing data, and prints a
 * JSON object to stdout.
 */
int main(int argc, char *argv[])
{
    const char *progname = basename(argv[0]);

    int action = milk_help_init(argc, argv,
                                "read processinfo shared memory and output timing metrics as JSON",
                                "Maps a processinfo shared-memory file and extracts timing data. "
                                "Output is JSON on stdout.");
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        return 0;
    }

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <procinfo_shm_path>\n", progname);
        return 1;
    }

    const char *shm_path = argv[1];

    /* Open and map the processinfo SHM file */
    PROCESSINFO *pinfo;
    size_t       pinfo_size;
    {
        int fd = open(shm_path, O_RDONLY);
        if (fd == -1)
        {
            fprintf(stderr, "ERROR: cannot open '%s': %s\n", shm_path, strerror(errno));
            return 1;
        }

        struct stat st;
        if (fstat(fd, &st) == -1)
        {
            fprintf(stderr, "ERROR: fstat failed: %s\n", strerror(errno));
            close(fd);
            return 1;
        }
        pinfo_size = (size_t) st.st_size;

        pinfo = (PROCESSINFO *) mmap(NULL, pinfo_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd); // fd no longer needed after mmap

        if (pinfo == MAP_FAILED)
        {
            fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
            return 1;
        }
    }

    /* ----- Extract timing percentiles ----- */

    int  nvalid   = 0;
    long p50_exec = 0, p95_exec = 0, p99_exec = 0;
    long p50_iter = 0, p95_iter = 0, p99_iter = 0;

    {
        int nbsamples = pinfo->timerindex;
        if (pinfo->timingbuffercnt > 0)
        {
            nbsamples = PROCESSINFO_NBtimer;
        }

        /* Compute exec/iter durations from the
         * circular buffer for percentile analysis */
        long exec_ns[PROCESSINFO_NBtimer];
        long iter_ns[PROCESSINFO_NBtimer];

        for (int i = 1; i < nbsamples; i++)
        {
            long dt_exec = timespec_diff_ns(pinfo->texecstart[i], pinfo->texecend[i]);
            long dt_iter = timespec_diff_ns(pinfo->texecstart[i - 1], pinfo->texecstart[i]);

            if (dt_exec > 0 && dt_iter > 0)
            {
                exec_ns[nvalid] = dt_exec;
                iter_ns[nvalid] = dt_iter;
                nvalid++;
            }
        }

        if (nvalid > 0)
        {
            qsort(exec_ns, nvalid, sizeof(long), cmp_long);
            qsort(iter_ns, nvalid, sizeof(long), cmp_long);

            p50_exec = exec_ns[nvalid / 2];
            p95_exec = exec_ns[(nvalid * 95) / 100];
            p99_exec = exec_ns[(nvalid * 99) / 100];

            p50_iter = iter_ns[nvalid / 2];
            p95_iter = iter_ns[(nvalid * 95) / 100];
            p99_iter = iter_ns[(nvalid * 99) / 100];
        }
    }

    /* Read memory stats from /proc */
    struct proc_mem mem;
    read_proc_memory(pinfo->PID, &mem);

    /* ----- Output JSON ----- */

    printf("{\n");
    printf("  \"process_name\": \"%s\",\n", pinfo->name);
    printf("  \"pid\": %d,\n", (int) pinfo->PID);
    printf("  \"loopcnt\": %ld,\n", pinfo->loopcnt);
    printf("  \"timing_samples\": %d,\n", nvalid);
    printf("  \"timing\": {\n");
    printf("    \"median_iter_ns\": %ld,\n", pinfo->dtmedian_iter_ns);
    printf("    \"median_exec_ns\": %ld,\n", pinfo->dtmedian_exec_ns);
    printf("    \"p50_exec_ns\": %ld,\n", p50_exec);
    printf("    \"p95_exec_ns\": %ld,\n", p95_exec);
    printf("    \"p99_exec_ns\": %ld,\n", p99_exec);
    printf("    \"p50_iter_ns\": %ld,\n", p50_iter);
    printf("    \"p95_iter_ns\": %ld,\n", p95_iter);
    printf("    \"p99_iter_ns\": %ld\n", p99_iter);
    printf("  },\n");
    printf("  \"memory\": {\n");
    printf("    \"vmpeak_kb\": %ld,\n", mem.vmpeak_kb);
    printf("    \"vmhwm_kb\": %ld,\n", mem.vmhwm_kb);
    printf("    \"vmrss_kb\": %ld\n", mem.vmrss_kb);
    printf("  }\n");
    printf("}\n");

    munmap(pinfo, pinfo_size);

    return 0;
}

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


/**
 * @brief Read VmPeak from /proc/PID/status.
 *
 * @param pid   Process ID to query
 * @return      VmPeak in kB, or -1 on failure
 */
static long read_vmpeak(pid_t pid)
{
    char path[128];
    snprintf(path, sizeof(path),
             "/proc/%d/status", (int) pid);

    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return -1;
    }

    char line[256];
    long vmpeak_kb = -1;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (strncmp(line, "VmPeak:", 7) == 0)
        {
            if (sscanf(line + 7, " %ld", &vmpeak_kb)
                != 1)
            {
                vmpeak_kb = -1;
            }
            break;
        }
    }

    fclose(fp);
    return vmpeak_kb;
}


/**
 * @brief Compute elapsed nanoseconds between two
 *        timespec values.
 */
static long timespec_diff_ns(
    struct timespec start,
    struct timespec end)
{
    long sec  = end.tv_sec  - start.tv_sec;
    long nsec = end.tv_nsec - start.tv_nsec;

    return sec * 1000000000L + nsec;
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
    if (argc < 2)
    {
        fprintf(stderr,
                "Usage: %s <procinfo_shm_path>\n",
                argv[0]);
        return 1;
    }

    const char *shm_path = argv[1];

    int fd = open(shm_path, O_RDONLY);
    if (fd == -1)
    {
        fprintf(stderr,
                "ERROR: cannot open '%s': %s\n",
                shm_path, strerror(errno));
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) == -1)
    {
        fprintf(stderr,
                "ERROR: fstat failed: %s\n",
                strerror(errno));
        close(fd);
        return 1;
    }

    PROCESSINFO *pinfo = (PROCESSINFO *)
        mmap(NULL, st.st_size,
             PROT_READ, MAP_SHARED, fd, 0);

    if (pinfo == MAP_FAILED)
    {
        fprintf(stderr,
                "ERROR: mmap failed: %s\n",
                strerror(errno));
        close(fd);
        return 1;
    }

    /* ----- Extract timing data ----- */

    int nbsamples = pinfo->timerindex;
    if (pinfo->timingbuffercnt > 0)
    {
        nbsamples = PROCESSINFO_NBtimer;
    }

    /* Compute exec durations from the circular
     * buffer for percentile analysis */
    long exec_ns[PROCESSINFO_NBtimer];
    long iter_ns[PROCESSINFO_NBtimer];
    int  nvalid = 0;

    for (int i = 1; i < nbsamples; i++)
    {
        long dt_exec = timespec_diff_ns(
            pinfo->texecstart[i],
            pinfo->texecend[i]);
        long dt_iter = timespec_diff_ns(
            pinfo->texecstart[i - 1],
            pinfo->texecstart[i]);

        if (dt_exec > 0 && dt_iter > 0)
        {
            exec_ns[nvalid] = dt_exec;
            iter_ns[nvalid] = dt_iter;
            nvalid++;
        }
    }

    /* Sort for percentile computation */
    long p50_exec = 0;
    long p95_exec = 0;
    long p99_exec = 0;
    long p50_iter = 0;
    long p95_iter = 0;
    long p99_iter = 0;

    if (nvalid > 0)
    {
        qsort(exec_ns, nvalid, sizeof(long),
              cmp_long);
        qsort(iter_ns, nvalid, sizeof(long),
              cmp_long);

        p50_exec = exec_ns[nvalid / 2];
        p95_exec = exec_ns[(nvalid * 95) / 100];
        p99_exec = exec_ns[(nvalid * 99) / 100];

        p50_iter = iter_ns[nvalid / 2];
        p95_iter = iter_ns[(nvalid * 95) / 100];
        p99_iter = iter_ns[(nvalid * 99) / 100];
    }

    /* Read VmPeak */
    long vmpeak_kb = read_vmpeak(pinfo->PID);

    /* ----- Output JSON ----- */

    printf("{\n");
    printf("  \"process_name\": \"%s\",\n",
           pinfo->name);
    printf("  \"pid\": %d,\n", (int) pinfo->PID);
    printf("  \"loopcnt\": %ld,\n", pinfo->loopcnt);
    printf("  \"timing_samples\": %d,\n", nvalid);
    printf("  \"timing\": {\n");
    printf("    \"median_iter_ns\": %ld,\n",
           pinfo->dtmedian_iter_ns);
    printf("    \"median_exec_ns\": %ld,\n",
           pinfo->dtmedian_exec_ns);
    printf("    \"p50_exec_ns\": %ld,\n", p50_exec);
    printf("    \"p95_exec_ns\": %ld,\n", p95_exec);
    printf("    \"p99_exec_ns\": %ld,\n", p99_exec);
    printf("    \"p50_iter_ns\": %ld,\n", p50_iter);
    printf("    \"p95_iter_ns\": %ld,\n", p95_iter);
    printf("    \"p99_iter_ns\": %ld\n", p99_iter);
    printf("  },\n");
    printf("  \"memory\": {\n");
    printf("    \"vmpeak_kb\": %ld\n", vmpeak_kb);
    printf("  }\n");
    printf("}\n");

    munmap(pinfo, st.st_size);
    close(fd);

    return 0;
}

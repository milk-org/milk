/**
 * @file milk-procCTRL-scan.c
 * @brief Milk procctrl scan module
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <getopt.h>
#include <dirent.h>
#include <time.h>

#include "processinfo.h"
#include "processtools.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_list_create.h"
#include "processinfo_shm_link.h"
#include "processinfo_shm_close.h"
#include "processinfo_shm_create.h"
#include "milkDebugTools.h"
#include "procCTRL_PIDcollectSystemInfo.h"
#include "processinfo_scan_shm.h"
#include "quicksort.h"

static int loopOK = 1;

void handle_signal(int sig) {
    (void)sig;
    loopOK = 0;
}

// Global mappings for the scanner to avoid repeated map/unmap
PROCESSINFO *pinfo_mappings[PROCESSINFOLISTSIZE];
int          pinfo_fds[PROCESSINFOLISTSIZE];

#include "procCTRL_scan_internal.h"

void *create_scan_shm(const char *name, size_t size, int *fd) {
    *fd = open(name, O_RDWR | O_CREAT, 0666);
    if (*fd == -1) return MAP_FAILED;
    // Do NOT ftruncate here to avoid zeroing if it already exists
    struct stat st;
    if (fstat(*fd, &st) == 0 && (size_t)st.st_size < size) {
        if (ftruncate(*fd, size) == -1) {
            close(*fd);
            return MAP_FAILED;
        }
    }
    void *ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    if (ptr == MAP_FAILED) close(*fd);
    return ptr;
}

int main(int argc, char *argv[]) {
    /* One-line help — before daemon check and SHM init */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h1") == 0 ||
            strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("background process info scanner daemon\n");
            return 0;
        }
    }

    double rate = 10.0;
    int verbose = 0;
    int rebuild = 0;
    int opt;
    
    // --- DAEMON CHECK ---
    pid_t my_pid = getpid();
    int other_running = 0;
    FILE *fp = popen("pgrep \"milk-procCTRL-s\"", "r");
    if (fp != NULL) {
        char pid_str[64];
        while (fgets(pid_str, sizeof(pid_str), fp) != NULL) {
            pid_t pid = (pid_t)atoi(pid_str);
            if (pid != my_pid) {
                printf("Scanner milk-procCTRL-scan is already running (PID %d)\n", (int)pid);
                other_running = 1;
                break;
            }
        }
        pclose(fp);
    }
    if (other_running) return 0;
    // --- END DAEMON CHECK ---

    static struct option long_options[] = {
        {"help",    no_argument,       0, 'h'},
        {"rate",    required_argument, 0, 'r'},
        {"verbose", no_argument,       0, 'v'},
        {"rebuild", no_argument,       0, 'R'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "hr:vR", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: %s [-r rate_hz] [-v] [-R]\n", argv[0]);
                return 0;
            case 'r':
                rate = atof(optarg);
                break;
            case 'v':
                verbose = 1;
                break;
            case 'R':
                rebuild = 1;
                break;
        }
    }

    {
        struct sigaction sa;
        sa.sa_handler = handle_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
    }

    // 1. Create/Open Process List SHM
    if (processinfo_shm_list_create() == -1) {
        fprintf(stderr, "Error connecting to process list shared memory\n");
        return 1;
    }

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    // 2. Create/Open Scan Result SHM
    char scan_shm_name[STRINGMAXLEN_FULLFILENAME];
    snprintf(scan_shm_name, sizeof(scan_shm_name), "%s/%s", procdname, PROCESSINFO_SCAN_SHM_NAME);
    
    int fd_scan;
    PROCSCAN_SHM *scan_shm = (PROCSCAN_SHM *) create_scan_shm(scan_shm_name, sizeof(PROCSCAN_SHM), &fd_scan);
    if (scan_shm == MAP_FAILED) {
         fprintf(stderr, "Error creating scan shared memory: %s\n", scan_shm_name);
         return 1;
    }

    if (rebuild) {
        rebuild_process_list(procdname, scan_shm);
    }

    printf("Scanner running at %.1f Hz\n", rate);

    long usleep_time = (long)(1000000.0 / rate);
    long scan_counter = 0;

    double *timearray = (double *) malloc(sizeof(double) * PROCESSINFOLISTSIZE);
    long   *indexarray = (long *)   malloc(sizeof(long)  * PROCESSINFOLISTSIZE);

    while (loopOK) {
        int active_count = 0;
        int stopped_count = 0;
        int crashed_count = 0;
        int serviced_count = 0;
        int listcnt = 0;

        if (scan_counter % 10 == 0) {
            FILE *pp = popen("pgrep -x milk-procCTRL | wc -l", "r");
            if (pp) {
                int nb;
                if (fscanf(pp, "%d", &nb) == 1) scan_shm->NBreaders = nb;
                pclose(pp);
            }
        }

        scan_update_pinfolist_status(procdname, scan_shm, &active_count, &stopped_count, &crashed_count, timearray, indexarray, &listcnt);

        if (listcnt > 0) {
            quick_sort2l_double(timearray, indexarray, listcnt);
            for (int j = 0; j < listcnt; j++) scan_shm->sorted_pindex[j] = (int)indexarray[j];
        }
        scan_shm->NBactive = listcnt;

        Scan_GetCPUloads(scan_shm);

        scan_update_process_details(procdname, scan_shm, &serviced_count);

        if (verbose) {
            printf("Scan %ld: Act:%d Stp:%d Cra:%d Srv:%d Readers:%d   \r", 
                   scan_counter, active_count, stopped_count, crashed_count, serviced_count, scan_shm->NBreaders);
            fflush(stdout);
        }

        scan_counter++;
        usleep(usleep_time);
    }
    
    for(
        long i=0; i<PROCESSINFOLISTSIZE; i++) if(pinfo_mappings[i]) processinfo_shm_close(pinfo_mappings[i],
        pinfo_fds[i]);
    free(timearray);
    free(indexarray);
    munmap(scan_shm, sizeof(PROCSCAN_SHM));
    close(fd_scan);

    return 0;
}
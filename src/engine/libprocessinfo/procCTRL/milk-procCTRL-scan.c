/**
 * @file milk-procCTRL-scan.c
 * @brief Milk procctrl scan module
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <signal.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <getopt.h>
#include <dirent.h>

#include "processtools.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_list_create.h"
#include "milkDebugTools.h"
#include "procCTRL_PIDcollectSystemInfo.h"
#include "processinfo_scan_shm.h"
#include "quicksort.h"

static int loopOK = 1;

/**
 * @brief Signal handler for the process scanner daemon.
 *
 * Sets the exit flag on SIGINT/SIGTERM.
 */
void handle_signal(int sig)
{
    (void)sig;
    loopOK = 0;
}

// Global mappings for the scanner to avoid repeated map/unmap
static PROCESSINFO *pinfo_mappings[PROCESSINFOLISTSIZE];
static int          pinfo_fds[PROCESSINFOLISTSIZE];

// Inline simplified CPU load collector
/**
 * @brief Read per-CPU load from /proc/stat.
 *
 * Computes user/system/idle percentages for
 * each online CPU.
 */
void Scan_GetCPUloads(PROCSCAN_SHM *scan_shm)
{
    static long long prev_user[MAXNBCPU], prev_nice[MAXNBCPU], prev_system[MAXNBCPU];
    static long long prev_idle[MAXNBCPU], prev_iowait[MAXNBCPU], prev_irq[MAXNBCPU],
           prev_softirq[MAXNBCPU], prev_steal[MAXNBCPU];
    static int initialized = 0;

    FILE *fp = fopen("/proc/stat", "r");
    if(!fp)
    {
        return;
    }

    char line[1024];
    int cpu_idx = 0;
    while(fgets(line, sizeof(line), fp) && cpu_idx < MAXNBCPU)
    {
        if(strncmp(line, "cpu", 3) == 0 && line[3] != ' ')    // cpu0, cpu1...
        {
            long long user, nice, system, idle, iowait, irq, softirq, steal;
            if(sscanf(line + 3, "%lld %lld %lld %lld %lld %lld %lld %lld",
                      &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) == 8)
            {

                if(initialized)
                {
                    long long total_prev = prev_user[cpu_idx] + prev_nice[cpu_idx] + prev_system[cpu_idx] +
                                           prev_idle[cpu_idx] +
                                           prev_iowait[cpu_idx] + prev_irq[cpu_idx] + prev_softirq[cpu_idx] + prev_steal[cpu_idx];
                    long long total_cur = user + nice + system + idle + iowait + irq + softirq + steal;
                    long long total_diff = total_cur - total_prev;
                    long long idle_diff = idle - prev_idle[cpu_idx];

                    if(total_diff > 0)
                    {
                        scan_shm->CPUload[cpu_idx] = (float)(total_diff - idle_diff) / total_diff;
                    }
                }

                prev_user[cpu_idx] = user;
                prev_nice[cpu_idx] = nice;
                prev_system[cpu_idx] = system;
                prev_idle[cpu_idx] = idle;
                prev_iowait[cpu_idx] = iowait;
                prev_irq[cpu_idx] = irq;
                prev_softirq[cpu_idx] = softirq;
                prev_steal[cpu_idx] = steal;

                cpu_idx++;
            }
        }
    }
    scan_shm->NBcpus = cpu_idx;
    initialized = 1;
    fclose(fp);
}

/**
 * @brief Create the scanner shared memory segment.
 *
 * Allocates the SHM file for communicating process
 * listing data to procCTRL TUI clients.
 */
void *create_scan_shm(const char *name, size_t size, int *fd)
{
    *fd = open(name, O_RDWR | O_CREAT, 0666);
    if(*fd == -1)
    {
        return MAP_FAILED;
    }
    // Do NOT ftruncate here to avoid zeroing if it already exists
    struct stat st;
    if(fstat(*fd, &st) == 0 && (size_t)st.st_size < size)
    {
        if(ftruncate(*fd, size) == -1)
        {
            close(*fd);
            return MAP_FAILED;
        }
    }
    void *ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    if(ptr == MAP_FAILED)
    {
        close(*fd);
    }
    return ptr;
}

/**
 * @brief Rebuild the process list from /proc.
 *
 * Scans all processinfo SHM files and updates
 * the scanner shared memory with current data.
 */
void rebuild_process_list(
    const char   *procdname,
    PROCSCAN_SHM *scan_shm)
{
    printf("Rebuilding process list from %s...\n", procdname);
    for(long i = 0; i < PROCESSINFOLISTSIZE; i++)
    {
        pinfolist->active[i] = 0;
    }
    DIR *d = opendir(procdname);
    if(d)
    {
        long pindex = 0;
        struct dirent *dir;
        while((dir = readdir(d)) != NULL)
        {
            if(strncmp(dir->d_name, "proc.", 5) == 0 && strstr(dir->d_name, ".shm"))
            {
                char workbuf[512];
                strncpy(workbuf, dir->d_name, 511);
                workbuf[511] = '\0';

                char fullpath[STRINGMAXLEN_FULLFILENAME];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, dir->d_name);

                char *shm_ext = strstr(workbuf, ".shm");
                if(shm_ext)
                {
                    *shm_ext = '\0';
                }
                char *pid_str = strrchr(workbuf, '.');
                if(pid_str)
                {
                    *pid_str = '\0';
                    pid_str++;
                    int pid = atoi(pid_str);
                    char *name = workbuf + 5;

                    if(pindex < PROCESSINFOLISTSIZE)
                    {
                        int fd;
                        PROCESSINFO *pinfo = processinfo_shm_link(fullpath,
                                             &fd);
                        if(pinfo != (PROCESSINFO *)MAP_FAILED)
                        {
                            // Check if still alive
                            if(kill(pid, 0) == 0)
                            {
                                pinfolist->active[pindex] = 1;
                            }
                            else
                            {
                                if(pinfo->loopstat == 3)
                                {
                                    pinfolist->active[pindex] = 2;    // STOPPED
                                }
                                else
                                {
                                    pinfolist->active[pindex] = 3;    // CRASHED
                                }
                            }

                            pinfolist->PIDarray[pindex] = pid;
                            strncpy(pinfolist->pnamearray[pindex],
                                    name,
                                    STRINGMAXLEN_PROCESSINFO_NAME - 1);
                            pinfolist->createtime[pindex] = 1.0 * pinfo->createtime.tv_sec + 1.0e-9 * pinfo->createtime.tv_nsec;

                            // Restore stats in scan_shm
                            scan_shm->pinfodisp[pindex].PID = pid;
                            scan_shm->pinfodisp[pindex].loopcnt = pinfo->loopcnt;
                            strncpy(scan_shm->pinfodisp[pindex].name,
                                    pinfo->name,
                                    39);
                            strncpy(scan_shm->pinfodisp[pindex].statusmsg,
                                    pinfo->statusmsg,
                                    199);

                            processinfo_shm_close(pinfo, fd);
                            pindex++;
                        }
                    }
                }
            }
        }
        closedir(d);
    }
}

int main(
    int argc,
    char *argv[])
{
    /* One-line help — before daemon check and SHM init */
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-h1") == 0 ||
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
    if(fp != NULL)
    {
        char pid_str[64];
        while(fgets(pid_str, sizeof(pid_str), fp) != NULL)
        {
            pid_t pid = (pid_t)atoi(pid_str);
            if(pid != my_pid)
            {
                printf("Scanner milk-procCTRL-scan is already running (PID %d)\n", (int)pid);
                other_running = 1;
                break;
            }
        }
        pclose(fp);
    }
    if(other_running)
    {
        return 0;
    }
    // --- END DAEMON CHECK ---

    static struct option long_options[] =
    {
        {"help",    no_argument,       0, 'h'},
        {"rate",    required_argument, 0, 'r'},
        {"verbose", no_argument,       0, 'v'},
        {"rebuild", no_argument,       0, 'R'},
        {0, 0, 0, 0}
    };

    while((opt = getopt_long(argc, argv, "hr:vR", long_options, NULL)) != -1)
    {
        switch(opt)
        {
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
        case '?':
        default:
            printf("\n\033[1;31mERROR\033[0m: Invalid option.\n\n");
            printf("Usage: %s [-r rate_hz] [-v] [-R]\n", argv[0]);
            return 1;
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
    {
        long pindex_unused;
        if(processinfo_shm_list_create(&pindex_unused)
                != RETURN_SUCCESS)
        {
            PRINT_ERROR(
                "Error connecting to process list shared memory");
            return 1;
        }
    }

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    // 2. Create/Open Scan Result SHM
    char scan_shm_name[STRINGMAXLEN_FULLFILENAME];
    snprintf(scan_shm_name, sizeof(scan_shm_name), "%s/%s", procdname, PROCESSINFO_SCAN_SHM_NAME);

    int fd_scan;
    PROCSCAN_SHM *scan_shm = (PROCSCAN_SHM *) create_scan_shm(scan_shm_name, sizeof(PROCSCAN_SHM),
                             &fd_scan);
    if(scan_shm == MAP_FAILED)
    {
        PRINT_ERROR("Error creating scan shared memory: %s", scan_shm_name);
        return 1;
    }

    if(rebuild)
    {
        rebuild_process_list(procdname, scan_shm);
    }

    printf("Scanner running at %.1f Hz\n", rate);

    long usleep_time = (long)(1000000.0 / rate);
    long scan_counter = 0;

    double *timearray = (double *) malloc(sizeof(double) * PROCESSINFOLISTSIZE);
    long   *indexarray = (long *)   malloc(sizeof(long)  * PROCESSINFOLISTSIZE);

    while(loopOK)
    {
        int active_count = 0;
        int stopped_count = 0;
        int crashed_count = 0;
        int serviced_count = 0;
        int listcnt = 0;

        if(scan_counter % 10 == 0)
        {
            FILE *pp = popen("pgrep -x milk-procCTRL | wc -l", "r");
            if(pp)
            {
                int nb;
                if(fscanf(pp, "%d", &nb) == 1)
                {
                    scan_shm->NBreaders = nb;
                }
                pclose(pp);
            }
        }

        if(pinfolist != NULL)
        {
            for(long i = 0; i < PROCESSINFOLISTSIZE; i++)
            {
                if(pinfolist->active[i] == 1)
                {
                    if(kill(pinfolist->PIDarray[i], 0) == -1 && errno == ESRCH)
                    {
                        char SM_fname[STRINGMAXLEN_FULLFILENAME];
                        snprintf(SM_fname, sizeof(SM_fname), "%s/proc.%s.%06d.shm", procdname, pinfolist->pnamearray[i],
                                 pinfolist->PIDarray[i]);
                        int fd;
                        PROCESSINFO *pinfo = processinfo_shm_link(SM_fname,
                                             &fd);
                        if(pinfo != (PROCESSINFO *)MAP_FAILED)
                        {
                            if(pinfo->loopstat == 3)
                            {
                                pinfolist->active[i] = 2;    // STOPPED
                            }
                            else
                            {
                                pinfolist->active[i] = 3;    // CRASHED
                            }

                            // FINAL SNAPSHOT
                            scan_shm->pinfodisp[i].loopcnt = pinfo->loopcnt;
                            strncpy(scan_shm->pinfodisp[i].statusmsg,
                                    pinfo->statusmsg,
                                    199);
                            strncpy(scan_shm->pinfodisp[i].name,
                                    pinfo->name,
                                    39);

                            processinfo_shm_close(pinfo, fd);
                            if(pinfo_mappings[i])
                            {
                                // If we had it mapped, close it
                                // (Actually handled by the mapping logic below too)
                            }
                        }
                        else
                        {
                            pinfolist->active[i] = 2;
                        }
                    }
                }

                if(pinfolist->active[i] != 0)
                {
                    if(pinfolist->active[i] == 1)
                    {
                        active_count++;
                    }
                    else if(pinfolist->active[i] == 2)
                    {
                        stopped_count++;
                    }
                    else if(pinfolist->active[i] == 3)
                    {
                        crashed_count++;
                    }

                    if(listcnt < PROCESSINFOLISTSIZE)
                    {
                        indexarray[listcnt] = i;
                        timearray[listcnt] = -1.0 * pinfolist->createtime[i];
                        listcnt++;
                    }
                }
            }
        }

        if(listcnt > 0)
        {
            quick_sort2l_double(timearray, indexarray, listcnt);
            for(int j = 0; j < listcnt; j++)
            {
                scan_shm->sorted_pindex[j] = (int)indexarray[j];
            }
        }
        scan_shm->NBactive = listcnt;

        Scan_GetCPUloads(scan_shm);

        // Update Process Details
        for(long i = 0; i < PROCESSINFOLISTSIZE; i++)
        {
            if(pinfolist->active[i] == 1)
            {

                scan_shm->pinfodisp[i].pindex = i;
                // Track CPU usage for all active processes every cycle
                PIDcollectSystemInfo(&scan_shm->pinfodisp[i], 0);

                if(scan_shm->request_scan[i] == 1)
                {
                    // Link if not already linked
                    if(pinfo_mappings[i] == NULL)
                    {
                        char SM_fname[STRINGMAXLEN_FULLFILENAME];
                        snprintf(SM_fname, sizeof(SM_fname), "%s/proc.%s.%06d.shm", procdname, pinfolist->pnamearray[i],
                                 pinfolist->PIDarray[i]);
                        pinfo_mappings[i] = processinfo_shm_link(SM_fname,
                                            &pinfo_fds[i]);
                        if(pinfo_mappings[i] == (PROCESSINFO *)MAP_FAILED)
                        {
                            pinfo_mappings[i] = NULL;
                        }
                    }

                    if(pinfo_mappings[i] != NULL)
                    {
                        // Update live stats in scan_shm for readers
                        scan_shm->pinfodisp[i].PID = pinfolist->PIDarray[i];
                        scan_shm->pinfodisp[i].loopcnt = pinfo_mappings[i]->loopcnt;
                        scan_shm->pinfodisp[i].loopcntMax = pinfo_mappings[i]->loopcntMax;
                        scan_shm->pinfodisp[i].loopstat = pinfo_mappings[i]->loopstat;
                        scan_shm->pinfodisp[i].rt_priority = pinfo_mappings[i]->RT_priority;
                        strncpy(
                            scan_shm->pinfodisp[i].statusmsg,
                            pinfo_mappings[i]->statusmsg,
                            199);
                        strncpy(scan_shm->pinfodisp[i].name,
                                pinfo_mappings[i]->name,
                                39);

                        // Triggering info
                        scan_shm->pinfodisp[i].triggermode = pinfo_mappings[i]->triggermode;
                        strncpy(
                            scan_shm->pinfodisp[i].triggerstreamname,
                            pinfo_mappings[i]->triggerstreamname,
                            79);
                        scan_shm->pinfodisp[i].triggersem = pinfo_mappings[i]->triggersem;
                        scan_shm->pinfodisp[i].triggerstreamcnt = pinfo_mappings[i]->triggerstreamcnt;
                        scan_shm->pinfodisp[i].triggertimeout = pinfo_mappings[i]->triggertimeout;
                        scan_shm->pinfodisp[i].triggermissedframe = pinfo_mappings[i]->triggermissedframe;
                        scan_shm->pinfodisp[i].triggermissedframe_cumul = pinfo_mappings[i]->triggermissedframe_cumul;
                        // Timing info
                        scan_shm->pinfodisp[i].MeasureTiming = pinfo_mappings[i]->MeasureTiming;

                        if(pinfo_mappings[i]->MeasureTiming != 0)
                        {
                            long dtiter_array[PROCESSINFO_NBtimer - 1];
                            long dtexec_array[PROCESSINFO_NBtimer - 1];

                            for(int tindex = 0; tindex < PROCESSINFO_NBtimer - 1; tindex++)
                            {
                                int ti1 = (pinfo_mappings[i]->timerindex - tindex + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;
                                int ti0 = (ti1 - 1 + PROCESSINFO_NBtimer) % PROCESSINFO_NBtimer;

                                dtiter_array[tindex] = (pinfo_mappings[i]->texecstart[ti1].tv_nsec -
                                                        pinfo_mappings[i]->texecstart[ti0].tv_nsec) +
                                                       1000000000L * (pinfo_mappings[i]->texecstart[ti1].tv_sec -
                                                                      pinfo_mappings[i]->texecstart[ti0].tv_sec);

                                dtexec_array[tindex] = (pinfo_mappings[i]->texecend[ti0].tv_nsec -
                                                        pinfo_mappings[i]->texecstart[ti0].tv_nsec) +
                                                       1000000000L * (pinfo_mappings[i]->texecend[ti0].tv_sec - pinfo_mappings[i]->texecstart[ti0].tv_sec);
                            }

                            quick_sort_long(dtiter_array,
                                            PROCESSINFO_NBtimer - 1);
                            quick_sort_long(dtexec_array,
                                            PROCESSINFO_NBtimer - 1);

                            pinfo_mappings[i]->dtmedian_iter_ns = dtiter_array[(PROCESSINFO_NBtimer - 1) / 2];
                            pinfo_mappings[i]->dtmedian_exec_ns = dtexec_array[(PROCESSINFO_NBtimer - 1) / 2];
                        }

                        scan_shm->pinfodisp[i].dtmedian_iter_ns = pinfo_mappings[i]->dtmedian_iter_ns;
                        scan_shm->pinfodisp[i].dtmedian_exec_ns = pinfo_mappings[i]->dtmedian_exec_ns;
                    }

                    scan_shm->request_scan[i] = 0;
                    serviced_count++;
                }
                else
                {
                    // ... close if inactive ...
                    if(pinfolist->active[i] != 1 && pinfo_mappings[i] != NULL)
                    {
                        processinfo_shm_close(pinfo_mappings[i], pinfo_fds[i]);
                        pinfo_mappings[i] = NULL;
                    }
                }
            }
        }

        if(verbose)
        {
            printf("Scan %ld: Act:%d Stp:%d Cra:%d Srv:%d Readers:%d   \r",
                   scan_counter, active_count, stopped_count, crashed_count, serviced_count, scan_shm->NBreaders);
            fflush(stdout);
        }

        scan_counter++;
        usleep(usleep_time);
    }

    for(
        long i = 0; i < PROCESSINFOLISTSIZE;
        i++) if(pinfo_mappings[i]) processinfo_shm_close(pinfo_mappings[i],
                        pinfo_fds[i]);
    free(timearray);
    free(indexarray);
    munmap(scan_shm, sizeof(PROCSCAN_SHM));
    close(fd_scan);

    return 0;
}

/**
 * @file    stream_monproc_shm.c
 * @brief   Shared memory connection routines for stream monitor
 * 
 * This file is isolated to minimize dependencies for standalone viewers
 * like stream-monproc-disp, avoiding full CLIcore/milkinfo_compute linkage.
 */

#include "stream_monproc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "libmilkdata/milkdata.h"

STREAM_MON_STRUCT* stream_monitor_connect(const char *streamname, int create)
{
    char shmname[STRINGMAXLEN_FULLFILENAME];
    int fd;
    STREAM_MON_STRUCT *smon = NULL;

    // Fallback to /milk/shm if milk_data.shmdir is empty
    const char *dir = (milk_data.shmdir[0] != '\0') ? milk_data.shmdir : "/milk/shm";
    snprintf(shmname, sizeof(shmname), "%s/%s.mon.shm", dir, streamname);

    int flags = O_RDWR;
    if (create) {
        flags |= O_CREAT;
    }

    fd = open(shmname, flags, 0666);
    if (fd == -1) {
        if (create) {
            perror("Error opening/creating monitor SHM file");
        }
        return NULL;
    }

    if (create) {
        if (ftruncate(fd, sizeof(STREAM_MON_STRUCT)) == -1) {
            perror("Error truncating monitor SHM file");
            close(fd);
            return NULL;
        }
    }

    smon = (STREAM_MON_STRUCT*) mmap(NULL, sizeof(STREAM_MON_STRUCT), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (smon == MAP_FAILED) {
        perror("Error mapping monitor SHM file");
        return NULL;
    }

    if (create) {
        smon->size = STREAM_MON_MAX_SAMPLES;
        smon->cnt = 0;
        smon->cindex = 0;
        smon->hist_nbins = STREAM_MON_MAX_HIST_BINS;
        memset(smon->flux, 0, sizeof(smon->flux));
        memset(smon->time, 0, sizeof(smon->time));
        memset(smon->hist_min_buf, 0, sizeof(smon->hist_min_buf));
        memset(smon->hist_max_buf, 0, sizeof(smon->hist_max_buf));
        memset(smon->hist_counts, 0, sizeof(smon->hist_counts));
    }

    return smon;
}

void stream_monitor_detach(STREAM_MON_STRUCT *smon)
{
    if (smon) {
        munmap(smon, sizeof(STREAM_MON_STRUCT));
    }
}

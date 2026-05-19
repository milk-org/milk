/**
 * @file fpsseq_shm.c
 * @brief Shared memory lifecycle for the milk-seq FPS Sequencer
 *
 * Handles creation, connection, and destruction of the /dev/shm
 * mapped state structs used by sequencer instances.
 */

#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include "milkDebugTools.h"
#include "fpsseq.h"

#define SHM_PREFIX "/milkseq."
#define FIFO_PREFIX "/tmp/milkseq."

/**
 * build_shm_name - Format the POSIX SHM path for a sequencer
 * @dest:  Destination buffer
 * @size:  Buffer size
 * @name:  Sequencer name
 */
static void build_shm_name(
    char *dest,
    size_t size,
    const char *name)
{
    snprintf(dest, size, "%s%s.shm", SHM_PREFIX, name);
}

/**
 * build_fifo_name - Format the FIFO path for a sequencer
 * @dest:  Destination buffer
 * @size:  Buffer size
 * @name:  Sequencer name
 */
static void build_fifo_name(
    char *dest,
    size_t size,
    const char *name)
{
    snprintf(dest, size, "%s%s.fifo", FIFO_PREFIX, name);
}

/**
 * milkseq_create - Create a new sequencer instance in shared memory
 * @name:  Sequencer name (used to derive SHM and FIFO paths)
 *
 * Allocates a POSIX shared memory segment for the MILKSEQ_STATE
 * struct, initializes all fields to zero, sets the default queue
 * priority, and creates the command FIFO in /tmp/.
 *
 * Return: Pointer to the mapped state, or NULL on error
 */
MILKSEQ_STATE *milkseq_create(const char *name)
{
    if(!name || strlen(name) == 0)
    {
        return NULL;
    }

    char shm_name[256];
    build_shm_name(shm_name, sizeof(shm_name), name);

    // Unlink old if it exists
    shm_unlink(shm_name);

    int fd = shm_open(shm_name, O_CREAT | O_RDWR | O_EXCL, 0666);
    if(fd == -1)
    {
        PRINT_ERROR("shm_open milkseq: %s", strerror(errno));
        return NULL;
    }

    if(ftruncate(fd, sizeof(MILKSEQ_STATE)) == -1)
    {
        PRINT_ERROR("ftruncate milkseq: %s", strerror(errno));
        close(fd);
        shm_unlink(shm_name);
        return NULL;
    }

    MILKSEQ_STATE *state = mmap(NULL, sizeof(MILKSEQ_STATE), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if(state == MAP_FAILED)
    {
        PRINT_ERROR("mmap milkseq: %s", strerror(errno));
        shm_unlink(shm_name);
        return NULL;
    }

    // Initialize state
    memset(state, 0, sizeof(MILKSEQ_STATE));
    strncpy(state->name, name, FPSSEQ_NAME_MAX - 1);
    state->status = MILKSEQ_STATUS_IDLE;
    state->pid = getpid();
    clock_gettime(CLOCK_REALTIME, &state->starttime);

    // Set max limits based on configured arrays
    state->NBtasks_max = NB_FPSCTRL_TASK_MAX;

    // Let the default queue (index 0) have an active priority
    state->queuelist[0].priority = 10;

    // Create FIFO
    build_fifo_name(state->fifo_path, sizeof(state->fifo_path), name);
    unlink(state->fifo_path); // remove stale
    if(mkfifo(state->fifo_path, 0666) == -1)
    {
        PRINT_ERROR("mkfifo milkseq: %s", strerror(errno));
        // Non-fatal, but warn
    }

    return state;
}

/**
 * milkseq_connect - Attach to an existing sequencer's shared memory
 * @name:  Sequencer name
 *
 * Opens the named SHM segment in read-write mode and maps it.
 * Does not create the segment if it does not exist.
 *
 * Return: Pointer to the mapped state, or NULL if not found
 */
MILKSEQ_STATE *milkseq_connect(const char *name)
{
    if(!name || strlen(name) == 0)
    {
        return NULL;
    }

    char shm_name[256];
    build_shm_name(shm_name, sizeof(shm_name), name);

    int fd = shm_open(shm_name, O_RDWR, 0); // Allow write for status updates
    if(fd == -1)
    {
        return NULL; // Not found
    }

    MILKSEQ_STATE *state = mmap(NULL, sizeof(MILKSEQ_STATE), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if(state == MAP_FAILED)
    {
        PRINT_ERROR("mmap milkseq connect: %s", strerror(errno));
        return NULL;
    }

    return state;
}

/**
 * milkseq_disconnect - Unmap a sequencer's shared memory
 * @state:  Mapped pointer obtained from milkseq_create/connect
 *
 * Return: 0 on success, -1 if state is NULL
 */
int milkseq_disconnect(MILKSEQ_STATE *state)
{
    if(!state)
    {
        return -1;
    }
    return munmap(state, sizeof(MILKSEQ_STATE));
}

/**
 * milkseq_destroy - Remove a sequencer's SHM segment and FIFO
 * @name:  Sequencer name
 *
 * Unlinks both the POSIX shared memory object and the /tmp/ FIFO.
 *
 * Return: 0 if both were removed, -1 otherwise
 */
int milkseq_destroy(const char *name)
{
    if(!name || strlen(name) == 0)
    {
        return -1;
    }

    char shm_name[256];
    char fifo_name[256];
    build_shm_name(shm_name, sizeof(shm_name), name);
    build_fifo_name(fifo_name, sizeof(fifo_name), name);

    int err1 = shm_unlink(shm_name);
    int err2 = unlink(fifo_name);

    return (err1 == 0 && err2 == 0) ? 0 : -1;
}

/**
 * milkseq_list - Enumerate active sequencer instances
 * @names:     Array of name buffers to fill
 * @maxcount:  Maximum entries to return
 *
 * Scans /dev/shm/ for files matching "milkseq.*.shm" and
 * extracts the sequencer name from each filename.
 *
 * Return: Number of sequencers found (<= maxcount)
 */
int milkseq_list(
    char names[][FPSSEQ_NAME_MAX],
    int maxcount)
{
    int count = 0;
    DIR *d;
    struct dirent *dir;
    d = opendir("/dev/shm");
    if(d)
    {
        while((dir = readdir(d)) != NULL)
        {
            // Match milkseq.*.shm
            if(strncmp(dir->d_name, "milkseq.", 8) == 0)
            {
                char *ext = strstr(dir->d_name, ".shm");
                if(ext)
                {
                    if(count < maxcount)
                    {
                        size_t namelen = ext - (dir->d_name + 8);
                        if(namelen < FPSSEQ_NAME_MAX)
                        {
                            strncpy(names[count], dir->d_name + 8, namelen);
                            names[count][namelen] = '\0';
                            count++;
                        }
                    }
                    else
                    {
                        break; // Array full
                    }
                }
            }
        }
        closedir(d);
    }
    return count;
}

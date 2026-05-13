/**
 * @file    milk-shmimpurge.c
 * @brief   Purge orphan shared memory image streams
 *
 * Replaces the bash script milk-shmimpurge.
 * Detects orphan streams by scanning /proc/<pid>/fd/ and
 * removes them via ImageStreamIO_destroyIm() + unlink().
 *
 * Usage:
 *   milk-shmimpurge [-f filter] [-y] [-n] [-v]
 */

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#include "libmilkcommon/milkDebugTools.h"

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_help.h"

#define SP_DESC \
    "purge orphan shared memory streams and files"

#define SP_DESC_LONG \
    "Scan the shared memory directory for image streams with\n" \
    "no live process holding them open, and remove them.\n" \
    "\n" \
    "Orphan detection: each stream's .im.shm inode is compared\n" \
    "against /proc/<pid>/fd/ symlinks. A stream is considered\n" \
    "an orphan only if no process has it open.\n" \
    "\n" \
    "Without -y, lists orphans and prompts before removal.\n" \
    "With -n (dry-run), lists orphans without removing them."

#define SP_MAX_STREAMS 4096

/**
 * print_help() - Print usage information
 * @progname: argv[0]
 * @mh_color: non-zero for ANSI color output
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, SP_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s]\n\n",
           mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SP_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-f, --filter <str>",
           mh_color ? MH_RST : "",
           "Only consider streams whose name contains <str>");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-y, --force",
           mh_color ? MH_RST : "",
           "Skip confirmation prompt");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-n, --dry-run",
           mh_color ? MH_RST : "",
           "List orphans without removing them");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-v, --verbose",
           mh_color ? MH_RST : "",
           "Show which PIDs hold each live stream");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "",
           "One-line description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-shmimpurge%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-shmimpurge%s %s-f dm -v%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-shmimpurge%s %s-n%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-stream-rm", "milk-stream-list" };
    milk_help_see_also(see_also, 2, mh_color);
}

/**
 * get_shmdir() - Determine the shared memory directory
 *
 * Checks MILK_SHM_DIR, then /milk/shm, then /dev/shm.
 */
static const char *get_shmdir(void)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env != NULL) {
        return env;
    }

    static const char fallback1[] = "/milk/shm";
    struct stat st;

    if (stat(fallback1, &st) == 0 && S_ISDIR(st.st_mode)) {
        return fallback1;
    }
    return "/dev/shm";
}

/**
 * pid_has_inode_open() - Check if a PID has a given inode open
 * @pid:   process ID to check
 * @inode: inode number of the SHM file
 *
 * Scans /proc/<pid>/fd/ symlinks for a stat inode match.
 * Returns 1 if the process has the inode open, 0 otherwise.
 */
static int pid_has_inode_open(pid_t pid, ino_t inode)
{
    char fddir[64];
    snprintf(fddir, sizeof(fddir), "/proc/%d/fd", (int)pid);

    DIR *d = opendir(fddir);
    if (d == NULL) {
        return 0;
    }

    struct dirent *de;
    int found = 0;

    while (!found && (de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') {
            continue;
        }

        char fdpath[128];
        snprintf(fdpath, sizeof(fdpath),
                 "/proc/%d/fd/%s", (int)pid, de->d_name);

        struct stat st;
        if (stat(fdpath, &st) == 0 && st.st_ino == inode) {
            found = 1;
        }
    } // while !found

    closedir(d);
    return found;
}

/**
 * is_stream_orphan() - Check whether no process holds the stream open
 * @fullpath: full path to the .im.shm file
 * @verbose:  if non-zero, print which PIDs hold the stream open
 *
 * Returns 1 if orphan (no live holder), 0 if live.
 */
static int is_stream_orphan(const char *fullpath, int verbose)
{
    struct stat st;
    if (stat(fullpath, &st) != 0) {
        return 1; /* cannot stat — treat as orphan */
    }

    ino_t inode = st.st_ino;

    DIR *proc = opendir("/proc");
    if (proc == NULL) {
        PRINT_ERROR("opendir(/proc): %s", strerror(errno));
        return 0; /* fail-safe: assume live */
    }

    int any_live = 0;
    struct dirent *de;

    while ((de = readdir(proc)) != NULL) {
        char *ep;
        long pid_l = strtol(de->d_name, &ep, 10);
        if (*ep != '\0' || pid_l <= 0) {
            continue;
        }
        pid_t pid = (pid_t)pid_l;

        if (pid_has_inode_open(pid, inode)) {
            any_live = 1;
            if (verbose) {
                printf("    held by PID %d\n", (int)pid);
            } else {
                break; /* early exit when not verbose */
            }
        }
    } // while readdir(proc)

    closedir(proc);
    return !any_live;
}

/**
 * read_confirm() - Ask y/N prompt; return 1 if user confirms
 */
static int read_confirm(void)
{
    printf("  Remove orphan streams? [y/N] ");
    fflush(stdout);

    struct termios old_t;
    int is_tty = isatty(STDIN_FILENO);

    if (is_tty) {
        tcgetattr(STDIN_FILENO, &old_t);
        struct termios t = old_t;
        t.c_lflag |= (ICANON | ECHO);
        t.c_iflag |= ICRNL;
        tcsetattr(STDIN_FILENO, TCSANOW, &t);
    }

    char buf[32];
    int ok = (fgets(buf, sizeof(buf), stdin) != NULL);

    if (is_tty) {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_t);
    }

    return ok && (buf[0] == 'y' || buf[0] == 'Y');
}

/**
 * remove_orphan() - Remove a single orphan stream
 * @shmdir: shared memory directory
 * @sname:  stream name (without .im.shm suffix)
 *
 * Returns 0 on success, 1 on error.
 */
static int remove_orphan(const char *shmdir, const char *sname)
{
    char fullpath[512];
    snprintf(fullpath, sizeof(fullpath),
             "%s/%s.im.shm", shmdir, sname);

    IMAGE image = {0};
    errno_t ret =
        ImageStreamIO_read_sharedmem_image_toIMAGE(sname, &image);

    if (ret == IMAGESTREAMIO_SUCCESS) {
        for (int i = 0; i < image.md->sem; i++) {
            sem_destroy(image.semptr[i]);
        }
        ImageStreamIO_closeIm(&image);
    }

    if (unlink(fullpath) != 0) {
        fprintf(stderr,
                "  \033[1;31mFAILED\033[0m: unlink '%s': %s\n",
                fullpath, strerror(errno));
        return 1;
    }

    printf("  Removed: %s\n", sname);
    return 0;
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv, SP_DESC, SP_DESC_LONG);

    if (action == MH_ACTION_H1 || action == MH_ACTION_H2) {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);

    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO) {
        print_help(argv[0], mh_color);
        return 0;
    }

    /* Parse arguments */
    const char *filter  = NULL;
    int         force   = 0;
    int         dry_run = 0;
    int         verbose = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-f") == 0 ||
             strcmp(argv[i], "--filter") == 0) &&
            i + 1 < argc)
        {
            filter = argv[++i];
        } else if (strcmp(argv[i], "-y") == 0 ||
                   strcmp(argv[i], "--force") == 0) {
            force = 1;
        } else if (strcmp(argv[i], "-n") == 0 ||
                   strcmp(argv[i], "--dry-run") == 0) {
            dry_run = 1;
        } else if (strcmp(argv[i], "-v") == 0 ||
                   strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else {
            fprintf(stderr,
                    "\n\033[1;31mERROR\033[0m:"
                    " unknown option '%s'.\n\n", argv[i]);
            print_help(argv[0], 1);
            return 1;
        }
    } // for i

    const char *shmdir = get_shmdir();

    /* Scan SHM directory */
    DIR *d = opendir(shmdir);
    if (d == NULL) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m:"
                " cannot open '%s': %s\n",
                shmdir, strerror(errno));
        return 1;
    }

    static char names[SP_MAX_STREAMS][256];
    int total = 0;
    struct dirent *de;

    while ((de = readdir(d)) != NULL &&
           total < SP_MAX_STREAMS)
    {
        char *suf = strstr(de->d_name, ".im.shm");
        if (suf == NULL) {
            continue;
        }
        size_t nl = (size_t)(suf - de->d_name);
        if (nl == 0 || nl >= sizeof(names[0])) {
            continue;
        }
        if (suf != de->d_name + strlen(de->d_name) - 7) {
            continue;
        }
        if (filter != NULL &&
            strstr(de->d_name, filter) == NULL) {
            continue;
        }

        memcpy(names[total], de->d_name, nl);
        names[total][nl] = '\0';
        total++;
    } // while readdir

    closedir(d);

    if (total == 0) {
        printf("No streams found");
        if (filter != NULL) {
            printf(" matching '%s'", filter);
        }
        printf(".\n");
        return 0;
    }

    /* Classify as orphan / live */
    static int is_orphan[SP_MAX_STREAMS];
    int n_orphan = 0;

    printf("Scanning %d stream(s)...\n\n", total);

    for (int i = 0; i < total; i++) {
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath),
                 "%s/%s.im.shm", shmdir, names[i]);

        if (verbose) {
            printf("  %s\n", names[i]);
        }

        is_orphan[i] = is_stream_orphan(fullpath, verbose);
        if (is_orphan[i]) {
            n_orphan++;
        }
    } // for i

    if (n_orphan == 0) {
        printf("No orphan streams found.\n");
        return 0;
    }

    printf("\n  Orphan streams (%d):\n\n", n_orphan);
    for (int i = 0; i < total; i++) {
        if (is_orphan[i]) {
            printf("    %s\n", names[i]);
        }
    }
    printf("\n");

    if (dry_run) {
        printf("  [dry-run] No streams removed.\n");
        return 0;
    }

    if (!force && !read_confirm()) {
        printf("  Cancelled.\n");
        return 0;
    }

    int errors = 0;
    for (int i = 0; i < total; i++) {
        if (is_orphan[i]) {
            errors += remove_orphan(shmdir, names[i]);
        }
    }

    if (errors > 0) {
        fprintf(stderr, "%d stream(s) failed to remove.\n", errors);
        return 1;
    }

    printf("  Done.\n");
    return 0;
}

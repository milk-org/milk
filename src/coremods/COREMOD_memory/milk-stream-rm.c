#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <termios.h>
#include <regex.h>

#include "multiselect_parse.h"

#include "ImageStreamIO/ImageStreamIO.h"

static void print_help(const char *progname)
{
    printf(
        "Usage: %s [options] [STREAM_NAME]\n"
        "\n"
        "Remove ImageStreamIO shared memory"
        " stream(s).\n"
        "\n"
        "Modes:\n"
        "  (default)     Exact match. Removes the"
        " single stream\n"
        "                whose name matches"
        " STREAM_NAME exactly.\n"
        "  -r, --regex   Regex match. Treats"
        " STREAM_NAME as an\n"
        "                Extended Regular Expression"
        " and shows\n"
        "                all matching streams before"
        " asking\n"
        "                for confirmation.\n"
        "\n"
        "If no STREAM_NAME is given, lists all"
        " existing\n"
        "streams and prompts for interactive"
        " selection.\n"
        "Multiple items can be selected using"
        " numbers,\n"
        "ranges, or 'all'"
        " (e.g. 1 3 5-7).\n"
        "\n"
        "Options:\n"
        "  -r, --regex   Use regex matching"
        " (may match multiple streams)\n"
        "  -f, --force   Force removal without"
        " confirmation (with -r)\n"
        "  -v, --verbose Verbose mode\n"
        "  -h, --help    Show this help message\n"
        "\n"
        "Examples:\n"
        "  %s dm00disp    "
        "  # exact: remove 'dm00disp'\n"
        "  %s -r 'dm.*'   "
        "  # regex: list dm* streams, confirm\n"
        "  %s -r -f 'dm.*'"
        "  # regex + force: remove immediately\n"
        "  %s             "
        "  # interactive selection\n",
        progname, progname, progname,
        progname, progname);
}

/**
 * remove_single_stream() - Remove one stream
 * @shmdir: shared memory directory path
 * @sname:  stream name (without .im.shm suffix)
 * @verbose: print extra info if true
 *
 * Handles both symlinks and real SHM files.
 * Returns 0 on success, 1 on failure.
 */
static int remove_single_stream(
    const char *shmdir,
    const char *sname,
    int verbose)
{
    char fullpath[512];

    snprintf(fullpath, sizeof(fullpath),
             "%s/%s.im.shm", shmdir, sname);

    /* Check if symlink — just unlink */
    struct stat lbuf;

    if (lstat(fullpath, &lbuf) == 0
        && S_ISLNK(lbuf.st_mode))
    {
        if (verbose) {
            printf("Removing symlink '%s'...\n",
                   sname);
        }
        if (unlink(fullpath) != 0) {
            perror("unlink");
            return 1;
        }
        printf("  Stream symlink '%s'"
               " removed.\n", sname);
        return 0;
    }

    /* Open and destroy the stream */
    IMAGE image = {0};
    errno_t ret =
        ImageStreamIO_read_sharedmem_image_toIMAGE(
            sname, &image);

    if (ret != IMAGESTREAMIO_SUCCESS) {
        fprintf(stderr,
                "Error: cannot open stream"
                " '%s'.\n", sname);
        return 1;
    }

    if (verbose) {
        printf("Removing stream '%s'...\n",
               sname);
    }

    /* Destroy semaphores */
    for (int si = 0;
         si < image.md->sem; si++)
    {
        sem_destroy(image.semptr[si]);
    }

    /* Close (unmap + close fd) */
    ImageStreamIO_closeIm(&image);

    /* Unlink the SHM file */
    if (unlink(fullpath) != 0) {
        perror("unlink");
        return 1;
    }

    printf("  Stream '%s' removed.\n", sname);
    return 0;
}

/**
 * scan_streams() - Scan SHM directory for streams
 * @shmdir:     shared memory directory
 * @names:      output array of stream names
 * @capacity:   initial array capacity (doubled as needed)
 *
 * Returns number of streams found.
 * Caller must free each name and the array.
 */
static int scan_streams(
    const char *shmdir,
    char ***names,
    int capacity)
{
    DIR *d = opendir(shmdir);

    if (!d) {
        fprintf(stderr,
                "Error opening directory %s\n",
                shmdir);
        return -1;
    }

    int count = 0;

    *names = calloc(capacity, sizeof(char *));

    struct dirent *dir;

    while ((dir = readdir(d)) != NULL) {
        char *pch =
            strstr(dir->d_name, ".im.shm");
        if (!pch) {
            continue;
        }
        int suffix_pos =
            (int)(pch - dir->d_name);
        int name_len =
            (int)strlen(dir->d_name);
        if (suffix_pos != name_len - 7) {
            continue;
        }

        char sname[256];

        snprintf(sname, sizeof(sname),
                 "%.*s", suffix_pos,
                 dir->d_name);

        if (count >= capacity) {
            capacity *= 2;
            *names = realloc(*names,
                             capacity
                             * sizeof(char *));
        }
        (*names)[count] = strdup(sname);
        count++;
    }
    closedir(d);

    return count;
}

/**
 * read_confirmation() - Ask user Y/N question
 *
 * Returns 1 if user confirms, 0 otherwise.
 */
static int read_confirmation(void)
{
    printf("  Proceed? [y/N] ");
    fflush(stdout);

    struct termios old_term;
    int is_tty = isatty(STDIN_FILENO);

    if (is_tty) {
        tcgetattr(STDIN_FILENO, &old_term);
        struct termios t = old_term;

        t.c_lflag |= (ICANON | ECHO);
        t.c_iflag |= ICRNL;
        tcsetattr(STDIN_FILENO,
                  TCSANOW, &t);
    }

    char linebuf[64];
    int ok =
        (fgets(linebuf, sizeof(linebuf),
               stdin) != NULL);

    if (is_tty) {
        tcsetattr(STDIN_FILENO,
                  TCSANOW, &old_term);
    }

    if (!ok) {
        return 0;
    }

    /* Strip trailing whitespace */
    {
        char *p = linebuf + strlen(linebuf);

        while (p > linebuf
               && (*(p - 1) == '\n'
                   || *(p - 1) == '\r'))
        {
            *(--p) = '\0';
        }
    }

    return (linebuf[0] == 'y'
            || linebuf[0] == 'Y');
}

/**
 * read_line() - Read a line from stdin
 * @buf:  output buffer
 * @size: buffer size
 *
 * Handles terminal mode setup and strips
 * trailing whitespace.  Returns 1 on success.
 */
static int read_line(char *buf, int size)
{
    struct termios old_term;
    int is_tty = isatty(STDIN_FILENO);

    if (is_tty) {
        tcgetattr(STDIN_FILENO, &old_term);
        struct termios t = old_term;

        t.c_lflag |= (ICANON | ECHO);
        t.c_iflag |= ICRNL;
        tcsetattr(STDIN_FILENO,
                  TCSANOW, &t);
    }

    int ok =
        (fgets(buf, size, stdin) != NULL);

    if (is_tty) {
        tcsetattr(STDIN_FILENO,
                  TCSANOW, &old_term);
    }

    if (!ok) {
        return 0;
    }

    /* Strip trailing whitespace */
    {
        char *p = buf + strlen(buf);

        while (p > buf
               && (*(p - 1) == '\n'
                   || *(p - 1) == '\r'))
        {
            *(--p) = '\0';
        }
    }

    return 1;
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    int use_regex = 0;
    int force = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument, 0, 'v'},
        {"regex",   no_argument, 0, 'r'},
        {"force",   no_argument, 0, 'f'},
        {"help",    no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv,
                              "vrfh",
                              long_options,
                              NULL)) != -1)
    {
        switch (opt) {
        case 'v':
            verbose = 1;
            break;
        case 'r':
            use_regex = 1;
            break;
        case 'f':
            force = 1;
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        default:
            print_help(argv[0]);
            return 1;
        }
    }

    const char *pattern = NULL;

    if (optind < argc) {
        pattern = argv[optind];
    }

    /* Determine SHM directory */
    const char *shmdir = getenv("MILK_SHM_DIR");

    if (shmdir == NULL) {
        shmdir = "/milk/shm";
        struct stat st;

        if (stat(shmdir, &st) != 0) {
            shmdir = "/dev/shm";
        }
    }

    /* ------------------------------------------
     * Mode 1: Exact match (default with pattern)
     * ----------------------------------------*/
    if (pattern != NULL && !use_regex) {
        return remove_single_stream(
            shmdir, pattern, verbose);
    }

    /* ------------------------------------------
     * Scan all streams for regex/interactive mode
     * ----------------------------------------*/
    int capacity = 64;
    char **all_names = NULL;
    int total =
        scan_streams(shmdir, &all_names,
                     capacity);

    if (total < 0) {
        return 1;
    }

    if (total == 0) {
        printf("No streams found.\n");
        free(all_names);
        return 1;
    }

    /* ------------------------------------------
     * Mode 2: Regex match (with -r flag)
     * ----------------------------------------*/
    if (use_regex && pattern != NULL) {
        regex_t regex;
        int ret = regcomp(&regex, pattern,
                          REG_EXTENDED
                          | REG_NOSUB);
        if (ret != 0) {
            fprintf(stderr,
                    "Error: invalid regex"
                    " '%s'.\n", pattern);
            for (int i = 0; i < total; i++) {
                free(all_names[i]);
            }
            free(all_names);
            return 1;
        }

        /* Filter matching streams */
        int match_cap = 64;
        int match_count = 0;
        char **match_names =
            calloc(match_cap, sizeof(char *));

        for (int i = 0; i < total; i++) {
            if (regexec(&regex, all_names[i],
                        0, NULL, 0) == 0)
            {
                if (match_count >= match_cap) {
                    match_cap *= 2;
                    match_names = realloc(
                        match_names,
                        match_cap
                        * sizeof(char *));
                }
                match_names[match_count] =
                    all_names[i];
                match_count++;
            } else {
                free(all_names[i]);
            }
        }
        free(all_names);
        regfree(&regex);

        if (match_count == 0) {
            fprintf(stderr,
                    "No streams matching"
                    " '%s' found.\n",
                    pattern);
            free(match_names);
            return 1;
        }

        /* Show matches */
        printf("\n  Streams matching '%s'"
               " (%d):\n\n",
               pattern, match_count);
        for (int i = 0;
             i < match_count; i++)
        {
            printf("    %s\n", match_names[i]);
        }
        printf("\n");

        /* Confirm unless -f */
        if (!force) {
            if (!read_confirmation()) {
                printf("Cancelled.\n");
                for (int i = 0;
                     i < match_count; i++)
                {
                    free(match_names[i]);
                }
                free(match_names);
                return 0;
            }
        }

        /* Remove all matched streams */
        int errors = 0;

        for (int i = 0;
             i < match_count; i++)
        {
            errors += remove_single_stream(
                shmdir, match_names[i],
                verbose);
            free(match_names[i]);
        }
        free(match_names);

        if (errors > 0) {
            fprintf(stderr,
                    "%d stream(s) failed"
                    " to remove.\n",
                    errors);
            return 1;
        }
        return 0;
    }

    /* ------------------------------------------
     * Mode 3: Interactive selection (no pattern)
     * ----------------------------------------*/
    printf("\n  Streams:\n\n");
    for (int i = 0; i < total; i++) {
        printf("  %3d  %s\n",
               i + 1, all_names[i]);
    }

    printf("\n  Enter number(s) to remove"
           " (e.g. 1 3 5-7, 'all',"
           " 0 to cancel): ");
    fflush(stdout);

    char linebuf[512];

    if (!read_line(linebuf,
                   sizeof(linebuf)))
    {
        printf("Cancelled.\n");
        for (int i = 0; i < total; i++) {
            free(all_names[i]);
        }
        free(all_names);
        return 0;
    }

    int *selected = calloc(total, sizeof(int));
    int nsel =
        parse_multiselect(linebuf,
                          selected, total);

    if (nsel <= 0) {
        printf("Cancelled.\n");
        free(selected);
        for (int i = 0; i < total; i++) {
            free(all_names[i]);
        }
        free(all_names);
        return 0;
    }

    int errors = 0;

    for (int i = 0; i < total; i++) {
        if (selected[i]) {
            errors += remove_single_stream(
                shmdir, all_names[i],
                verbose);
        }
    }

    free(selected);
    for (int i = 0; i < total; i++) {
        free(all_names[i]);
    }
    free(all_names);

    if (errors > 0) {
        fprintf(stderr,
                "%d stream(s) failed"
                " to remove.\n", errors);
        return 1;
    }
    return 0;
}

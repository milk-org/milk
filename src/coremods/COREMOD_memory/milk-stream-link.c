/**
 * @file    milk-stream-link.c
 * @brief   Establish a SHM symlink between streams
 *
 * Replaces the milk-streamlink bash script.
 *
 * Reads conf/streamlink.<linkname>.name.txt to find the source stream,
 * creates a symlink $MILK_SHM_DIR/<prefix><linkname>.im.shm →
 * $MILK_SHM_DIR/<sourcename>.im.shm, then writes the image size to
 * conf/streamlink.<linkname>.imsize.txt using ImageStreamIO.
 *
 * Usage:
 *   milk-stream-link [-p <prefix>] <streamname>
 */

#define _GNU_SOURCE

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_help.h"

#define LSL_DESC      "establish shared memory stream symlink"
#define LSL_DESC_LONG \
    "Reads conf/streamlink.<linkname>.name.txt to identify the source\n" \
    "stream, then creates a symlink:\n\n" \
    "  $MILK_SHM_DIR/<prefix><linkname>.im.shm\n" \
    "      → $MILK_SHM_DIR/<sourcename>.im.shm\n\n" \
    "If the source stream exists, also writes its pixel dimensions to\n" \
    "conf/streamlink.<linkname>.imsize.txt."

/** Return MILK_SHM_DIR or fallback */
static const char *get_shmdir(void)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env != NULL) {
        return env;
    }

    static const char fallback[] = "/milk/shm";
    struct stat st;

    if (stat(fallback, &st) == 0 && S_ISDIR(st.st_mode)) {
        return fallback;
    }
    return "/dev/shm";
}

/**
 * print_help() - Print usage information
 * @progname: argv[0]
 * @color:    non-zero for ANSI color
 */
static void print_help(const char *progname, int color)
{
    milk_help_banner(progname, LSL_DESC, color);
    milk_help_section("Usage", color);
    printf("  %s%s%s %s[-p prefix]%s %s<streamname>%s\n\n",
           color ? MH_CMD : "", progname, color ? MH_RST : "",
           color ? MH_OPT : "", color ? MH_RST : "",
           color ? MH_ARG : "", color ? MH_RST : "");
    milk_help_section("Description", color);
    printf("  %s\n\n", LSL_DESC_LONG);
    milk_help_section("Options", color);
    printf("  %s%-25s%s %s\n",
           color ? MH_OPT : "", "-p <prefix>",
           color ? MH_RST : "",
           "Prefix for the link name (default: none)");
    printf("  %s%-25s%s %s\n",
           color ? MH_OPT : "", "-h, --help",
           color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n\n",
           color ? MH_OPT : "", "-h1, --help-oneline",
           color ? MH_RST : "",
           "One-line description and exit");
    milk_help_section("Examples", color);
    printf("  %s$ milk-stream-link%s %sircam0%s\n",
           color ? MH_CMD : "", color ? MH_RST : "",
           color ? MH_ARG : "", color ? MH_RST : "");
    printf("  %s$ milk-stream-link%s %s-p myprefix- ircam0%s\n\n",
           color ? MH_CMD : "", color ? MH_RST : "",
           color ? MH_ARG : "", color ? MH_RST : "");
    const char *see_also[] = {
        "milk-stream-list:list active shared memory streams",
        "milk-stream-create:create a new shared memory stream"
    };
    milk_help_see_also(see_also, 2, color);
}

/**
 * write_imsize() - Write image pixel dimensions to a text file
 * @shmdir:       shared memory directory
 * @srcname:      source stream name (without .im.shm)
 * @linkname:     link stream name (for the output filename)
 *
 * Opens the source stream via ImageStreamIO and writes
 * "xsize ysize\n" to conf/streamlink.<linkname>.imsize.txt.
 */
static void write_imsize(
    const char *shmdir,
    const char *srcname,
    const char *linkname)
{
    (void)shmdir; /* ImageStreamIO uses MILK_SHM_DIR internally */

    IMAGE img;
    int   ret = ImageStreamIO_openIm(&img, srcname);

    if (ret != 0) {
        fprintf(stderr,
                "  [warn] could not open stream '%s'"
                " — skipping imsize write\n", srcname);
        return;
    }

    /* Ensure conf/ directory exists */
    if (mkdir("conf", 0755) != 0 && errno != EEXIST) {
        fprintf(stderr,
                "  [warn] cannot create conf/ directory: %s\n",
                strerror(errno));
        ImageStreamIO_closeIm(&img);
        return;
    }

    char outfile[512];
    snprintf(outfile, sizeof(outfile),
             "conf/streamlink.%s.imsize.txt", linkname);

    FILE *fp = fopen(outfile, "w");
    if (fp == NULL) {
        fprintf(stderr,
                "  [warn] cannot write '%s': %s\n",
                outfile, strerror(errno));
        ImageStreamIO_closeIm(&img);
        return;
    }

    uint8_t naxis = img.md->naxis;

    for (uint8_t i = 0; i < naxis; i++) {
        fprintf(fp, "%u ", (unsigned)img.md->size[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);

    ImageStreamIO_closeIm(&img);

    printf("  imsize written → %s\n", outfile);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(
        argc, argv, LSL_DESC, LSL_DESC_LONG);

    if (action == MH_ACTION_H1 || action == MH_ACTION_H2) {
        return 0;
    }

    int color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO) {
        print_help(argv[0], color);
        return 0;
    }

    /* Parse optional -p <prefix> */
    const char *prefix    = "";
    const char *linkname  = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prefix = argv[++i];
        } else if (argv[i][0] != '-') {
            linkname = argv[i];
        }
    }

    if (linkname == NULL) {
        fprintf(stderr,
                "\n\033[1;31mERROR\033[0m:"
                " <streamname> argument required.\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    /* Read conf/streamlink.<linkname>.name.txt */
    char conffile[512];
    snprintf(conffile, sizeof(conffile),
             "conf/streamlink.%s.name.txt", linkname);

    FILE *fp = fopen(conffile, "r");
    if (fp == NULL) {
        fprintf(stderr,
                "\n\033[1;31mERROR\033[0m:"
                " cannot open '%s': %s\n"
                "  Nothing to do.\n\n",
                conffile, strerror(errno));
        return 1;
    }

    char srcname[256];
    if (fgets(srcname, sizeof(srcname), fp) == NULL) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: '%s' is empty.\n", conffile);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    /* Strip trailing newline */
    srcname[strcspn(srcname, "\r\n")] = '\0';

    const char *shmdir = get_shmdir();
    printf("linking %s%s\n", prefix, linkname);
    printf("  SHMDIR = %s\n", shmdir);

    /* Build paths */
    char linkpath[512], srcpath[512];
    snprintf(linkpath, sizeof(linkpath),
             "%s/%s%s.im.shm", shmdir, prefix, linkname);
    snprintf(srcpath, sizeof(srcpath),
             "%s/%s.im.shm", shmdir, srcname);

    /* Remove stale link/file */
    struct stat st;
    if (lstat(linkpath, &st) == 0) {
        if (unlink(linkpath) != 0) {
            fprintf(stderr,
                    "\033[1;31mERROR\033[0m:"
                    " cannot remove '%s': %s\n",
                    linkpath, strerror(errno));
            return 1;
        }
    }

    /* Create symlink */
    if (symlink(srcpath, linkpath) != 0) {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m:"
                " symlink('%s' → '%s'): %s\n",
                srcpath, linkpath, strerror(errno));
        return 1;
    }

    printf("  Linking %s%s:\n"
           "    ln -s %s\n"
           "       -> %s\n",
           prefix, linkname, srcpath, linkpath);

    /* If source stream exists, write imsize */
    if (stat(srcpath, &st) == 0) {
        write_imsize(shmdir, srcname, linkname);
    } else {
        printf("  [warn] source '%s' not found"
               " — skipping imsize write\n", srcpath);
    }

    return 0;
}

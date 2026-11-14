// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk-stream-link.c
 * @brief   Establish or atomically swap a shared-memory stream symlink
 */

#define _GNU_SOURCE

#include <errno.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_help.h"

#define LSL_DESC "establish or atomically swap a shared-memory stream symlink"
#define LSL_DESC_LONG                                                                 \
    "Create or atomically update a shared-memory stream symlink:\n\n"                 \
    "  $MILK_SHM_DIR/<prefix><linkname>.im.shm -> $MILK_SHM_DIR/<srcname>.im.shm\n\n" \
    "In direct mode (-s <srcname>), links directly from CLI arguments, validates\n"   \
    "source stream readiness and optional --expect-size dimensions, refuses to\n"     \
    "clobber non-symlink regular .im.shm files, and swaps symlinks atomically via\n"  \
    "rename() so concurrent real-time readers never observe ENOENT.\n\n"              \
    "Without -s, reads conf/streamlink.<linkname>.name.txt for compatibility."

static const char *get_shmdir(void)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env != NULL && env[0] != '\0')
    {
        return env;
    }

    static const char fallback[] = "/milk/shm";
    struct stat       st;
    if (stat(fallback, &st) == 0 && S_ISDIR(st.st_mode))
    {
        return fallback;
    }
    return "/dev/shm";
}

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, LSL_DESC, mh_color);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s-p <prefix>%s] %s<linkname>%s\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s%s%s %s-s <srcstream>%s [%soptions%s] %s<linkname>%s\n\n", mh_color ? MH_CMD : "",
           progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", LSL_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_ARG, "<linkname>", 26),
           "Alias stream name (without <prefix> or .im.shm)");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-s, --source <src>", 26),
           "Source stream name (direct mode without conf/ input)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-p <prefix>", 26),
           "Prefix prepended to <linkname> (default: none)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--expect-size <WxH[xD]>", 26),
           "Require source stream dimensions to match");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--allow-missing", 26),
           "Allow linking before source stream exists (with -s)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--no-conf", 26),
           "Do not write conf/streamlink.<linkname>.* metadata");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-q, --quiet", 26),
           "Suppress informational stdout messages");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h, --help", 26), "Show this help and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h1, --help-oneline", 26),
           "Print one-line description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h2, --help-description", 26),
           "Print verbose description and exit");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "-hm, --help-mono", 26),
           "Full help without ANSI color");

    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s%s %s-s dm10disp -p aol7_ dmC_alias%s\n", mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s$ %s%s%s %s-s ocam2d --expect-size 120x120 aol7_wfsim%s\n\n",
           mh_color ? MH_DIM : "", mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    const char *see_also[] = { "milk-stream-wait:wait for shared memory streams",
                               "milk-shmimmon:monitor shared memory stream activity" };
    milk_help_see_also(see_also, 2, mh_color);
} // print_help

static int parse_dims(const char *str, uint32_t dims[3], uint8_t *ndims_out)
{
    uint8_t     nd  = 0;
    const char *cur = str;

    while (*cur != '\0')
    {
        if (nd >= 3 || *cur < '1' || *cur > '9')
        {
            return -1;
        }
        char         *end = NULL;
        unsigned long val = strtoul(cur, &end, 10);
        if (end == cur || val == 0 || val > UINT32_MAX)
        {
            return -1;
        }
        dims[nd++] = (uint32_t) val;
        if (*end == '\0')
        {
            break;
        }
        if (*end != 'x' && *end != 'X')
        {
            return -1;
        }
        cur = end + 1;
        if (*cur == '\0')
        {
            return -1;
        }
    }
    *ndims_out = nd;
    return (nd > 0) ? 0 : -1;
} // parse_dims

static void write_conf_files(const char  *srcname,
                             const char  *linkname,
                             const IMAGE *opened_img,
                             int          create_conf_dir,
                             int          write_name_file,
                             int          quiet)
{
    struct stat st;
    if (!create_conf_dir)
    {
        if (stat("conf", &st) != 0 || !S_ISDIR(st.st_mode))
        {
            return;
        }
    }
    else if (mkdir("conf", 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "  [warn] cannot create conf/ directory: %s\n", strerror(errno));
        return;
    }

    if (write_name_file)
    {
        char namefile[512];
        snprintf(namefile, sizeof(namefile), "conf/streamlink.%s.name.txt", linkname);
        FILE *nfp = fopen(namefile, "w");
        if (nfp != NULL)
        {
            fprintf(nfp, "%s\n", srcname);
            fclose(nfp);
        }
    }

    if (opened_img == NULL || opened_img->md == NULL)
    {
        return;
    }

    char outfile[512];
    snprintf(outfile, sizeof(outfile), "conf/streamlink.%s.imsize.txt", linkname);
    FILE *fp = fopen(outfile, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "  [warn] cannot write '%s': %s\n", outfile, strerror(errno));
        return;
    }

    uint8_t naxis = opened_img->md->naxis;
    for (uint8_t ii = 0; ii < naxis; ii++)
    {
        fprintf(fp, "%u ", (unsigned) opened_img->md->size[ii]);
    }
    fprintf(fp, "\n");
    fclose(fp);

    if (!quiet)
    {
        printf("  imsize written -> %s\n", outfile);
    }
} // write_conf_files

int main(int argc, char *argv[])
{
    char       *progbuf  = strdup(argv[0]);
    const char *progname = (progbuf != NULL) ? basename(progbuf) : argv[0];

    int action = milk_help_init(argc, argv, LSL_DESC, LSL_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        free(progbuf);
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        free(progbuf);
        return 0;
    }

    const char *prefix        = "";
    const char *src_opt       = NULL;
    const char *expect_str    = NULL;
    const char *linkname      = NULL;
    int         allow_missing = 0;
    int         no_conf       = 0;
    int         quiet         = 0;
    int         opts_done     = 0;
    uint32_t    exp_dims[3]   = { 0, 0, 0 };
    uint8_t     exp_ndims     = 0;

    for (int ii = 1; ii < argc; ii++)
    {
        const char *arg = argv[ii];
        if (!opts_done && strcmp(arg, "--") == 0)
        {
            opts_done = 1;
            continue;
        }
        if (!opts_done && strcmp(arg, "-p") == 0)
        {
            if (ii + 1 >= argc)
            {
                fprintf(stderr, "%s: option '-p' requires an argument\n", progname);
                free(progbuf);
                return 2;
            }
            prefix = argv[++ii];
            continue;
        }
        if (!opts_done && (strcmp(arg, "-s") == 0 || strcmp(arg, "--source") == 0))
        {
            if (ii + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires an argument\n", progname, arg);
                free(progbuf);
                return 2;
            }
            src_opt = argv[++ii];
            continue;
        }
        if (!opts_done && strcmp(arg, "--expect-size") == 0)
        {
            if (ii + 1 >= argc)
            {
                fprintf(stderr, "%s: option '--expect-size' requires an argument\n", progname);
                free(progbuf);
                return 2;
            }
            expect_str = argv[++ii];
            if (parse_dims(expect_str, exp_dims, &exp_ndims) != 0)
            {
                fprintf(stderr, "%s: invalid --expect-size '%s'\n", progname, expect_str);
                free(progbuf);
                return 2;
            }
            continue;
        }
        if (!opts_done && strcmp(arg, "--allow-missing") == 0)
        {
            allow_missing = 1;
            continue;
        }
        if (!opts_done && strcmp(arg, "--no-conf") == 0)
        {
            no_conf = 1;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0))
        {
            quiet = 1;
            continue;
        }
        if (!opts_done && arg[0] == '-')
        {
            fprintf(stderr, "%s: unknown option '%s'\n", progname, arg);
            free(progbuf);
            return 2;
        }
        if (linkname != NULL)
        {
            fprintf(stderr, "%s: unexpected extra argument '%s'\n", progname, arg);
            free(progbuf);
            return 2;
        }
        linkname = arg;
    } // for ii

    if (linkname == NULL || linkname[0] == '\0')
    {
        fprintf(stderr, "%s: <linkname> argument required\n", progname);
        print_help(progname, 1);
        free(progbuf);
        return 2;
    }
    if (allow_missing && exp_ndims > 0)
    {
        fprintf(stderr, "%s: --allow-missing cannot be combined with --expect-size\n", progname);
        free(progbuf);
        return 2;
    }

    char srcname_buf[256];
    if (src_opt != NULL)
    {
        if (src_opt[0] == '\0' || strlen(src_opt) >= sizeof(srcname_buf))
        {
            fprintf(stderr, "%s: invalid source stream name '%s'\n", progname, src_opt);
            free(progbuf);
            return 2;
        }
        snprintf(srcname_buf, sizeof(srcname_buf), "%s", src_opt);
    }
    else
    {
        char conffile[512];
        snprintf(conffile, sizeof(conffile), "conf/streamlink.%s.name.txt", linkname);
        FILE *fp = fopen(conffile, "r");
        if (fp == NULL)
        {
            fprintf(stderr, "%s: cannot open '%s': %s\n", progname, conffile, strerror(errno));
            free(progbuf);
            return 1;
        }
        if (fgets(srcname_buf, sizeof(srcname_buf), fp) == NULL)
        {
            fprintf(stderr, "%s: '%s' is empty\n", progname, conffile);
            fclose(fp);
            free(progbuf);
            return 1;
        }
        fclose(fp);
        srcname_buf[strcspn(srcname_buf, "\r\n")] = '\0';
        if (srcname_buf[0] == '\0')
        {
            fprintf(stderr, "%s: '%s' contains an empty stream name\n", progname, conffile);
            free(progbuf);
            return 1;
        }
    } // if/else src_opt

    const char *shmdir = get_shmdir();
    char        full_link[512];
    char        linkpath[1024];
    char        srcpath[1024];
    char        tmppath[1024];

    snprintf(full_link, sizeof(full_link), "%s%s", prefix, linkname);
    if (strcmp(full_link, srcname_buf) == 0)
    {
        fprintf(stderr, "%s: refusing self-link '%s' -> '%s'\n", progname, full_link, srcname_buf);
        free(progbuf);
        return 1;
    }

    snprintf(linkpath, sizeof(linkpath), "%s/%s.im.shm", shmdir, full_link);
    snprintf(srcpath, sizeof(srcpath), "%s/%s.im.shm", shmdir, srcname_buf);
    snprintf(tmppath, sizeof(tmppath), "%s/.tmp.%s.%d.im.shm", shmdir, full_link, (int) getpid());

    /* Refuse to clobber a non-symlink regular SHM stream file */
    struct stat st;
    if (lstat(linkpath, &st) == 0 && !S_ISLNK(st.st_mode))
    {
        fprintf(stderr, "%s: refusing to overwrite regular SHM file '%s' (not a symlink)\n",
                progname, linkpath);
        free(progbuf);
        return 1;
    }

    IMAGE src_img;
    int   src_opened = 0;
    memset(&src_img, 0, sizeof(src_img));
    ImageStreamIO_set_printError(NULL);

    if (src_opt != NULL && !allow_missing)
    {
        if (access(srcpath, F_OK) != 0 ||
            ImageStreamIO_openIm(&src_img, srcname_buf) != IMAGESTREAMIO_SUCCESS ||
            src_img.md == NULL || src_img.md->write != 0 || src_img.md->naxis == 0)
        {
            if (src_img.md != NULL)
            {
                ImageStreamIO_closeIm(&src_img);
            }
            fprintf(stderr,
                    "%s: source stream '%s' is missing or not ready (use --allow-missing)\n",
                    progname, srcname_buf);
            free(progbuf);
            return 1;
        }
        src_opened = 1;
    }
    else if (access(srcpath, F_OK) == 0 &&
             ImageStreamIO_openIm(&src_img, srcname_buf) == IMAGESTREAMIO_SUCCESS &&
             src_img.md != NULL)
    {
        src_opened = 1;
    }

    if (exp_ndims > 0)
    {
        if (!src_opened)
        {
            fprintf(stderr, "%s: cannot verify --expect-size: source '%s' not open\n", progname,
                    srcname_buf);
            free(progbuf);
            return 1;
        }
        int mismatch = (src_img.md->naxis != exp_ndims);
        for (uint8_t k = 0; !mismatch && k < exp_ndims; k++)
        {
            if (src_img.md->size[k] != exp_dims[k])
            {
                mismatch = 1;
            }
        }
        if (mismatch)
        {
            fprintf(stderr, "%s: source '%s' dimensions do not match --expect-size %s\n", progname,
                    srcname_buf, expect_str);
            ImageStreamIO_closeIm(&src_img);
            free(progbuf);
            return 1;
        }
    } // if exp_ndims > 0

    /* Atomic symlink swap via temporary symlink + rename() */
    unlink(tmppath);
    if (symlink(srcpath, tmppath) != 0)
    {
        fprintf(stderr, "%s: symlink('%s' -> '%s') failed: %s\n", progname, srcpath, tmppath,
                strerror(errno));
        if (src_opened)
        {
            ImageStreamIO_closeIm(&src_img);
        }
        free(progbuf);
        return 1;
    }
    if (rename(tmppath, linkpath) != 0)
    {
        fprintf(stderr, "%s: rename('%s' -> '%s') failed: %s\n", progname, tmppath, linkpath,
                strerror(errno));
        unlink(tmppath);
        if (src_opened)
        {
            ImageStreamIO_closeIm(&src_img);
        }
        free(progbuf);
        return 1;
    }

    if (!quiet)
    {
        printf("  Linking %s:\n    ln -s %s\n       -> %s\n", full_link, srcpath, linkpath);
    }

    if (!no_conf)
    {
        if (src_opt == NULL)
        {
            if (src_opened)
            {
                write_conf_files(srcname_buf, linkname, &src_img, 1, 0, quiet);
            }
            else if (!quiet)
            {
                printf("  [warn] source '%s' not found -- skipping imsize write\n", srcpath);
            }
        }
        else
        {
            write_conf_files(srcname_buf, linkname, src_opened ? &src_img : NULL, 0, 1, quiet);
        }
    }

    if (src_opened)
    {
        ImageStreamIO_closeIm(&src_img);
    }
    free(progbuf);
    return 0;
} // main

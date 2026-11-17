// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk-stream-link.c
 * @brief   Establish or atomically swap shared-memory stream symlinks
 */

#define _GNU_SOURCE

#include <ctype.h>
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

#define LSL_DESC "establish or atomically swap shared-memory stream symlinks"
#define LSL_DESC_LONG                                                                  \
    "Create or atomically update one or more shared-memory stream symlinks:\n\n"       \
    "  $MILK_SHM_DIR/<prefix><linkname>.im.shm -> $MILK_SHM_DIR/<srcname>.im.shm\n\n"  \
    "In direct mode (-s <srcname> <linkname>, <src>=<link>..., or -b <file>), links\n" \
    "directly from CLI/batch arguments, validates all source streams and optional\n"   \
    "--expect-size dimensions before modifying any target, refuses to clobber\n"       \
    "non-symlink regular .im.shm files, and swaps symlinks atomically via rename()\n"  \
    "so concurrent real-time readers never observe ENOENT.\n\n"                        \
    "With a single <linkname> and no -s/-b, reads conf/streamlink.<linkname>.name.txt."

typedef struct
{
    char     src[256];
    char     link[256];
    char     full_link[512];
    char     linkpath[1024];
    char     srcpath[1024];
    int      is_direct;
    int      has_imsize;
    uint8_t  src_naxis;
    uint32_t src_size[3];
} link_pair_t;

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
} // get_shmdir

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, LSL_DESC, mh_color);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s-p <prefix>%s] %s<linkname>%s\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s%s%s %s-s <srcstream>%s [%soptions%s] %s<linkname>%s\n", mh_color ? MH_CMD : "",
           progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    printf("  %s%s%s [%soptions%s] %s<src1>=<link1>%s [%s<src2>=<link2> ...%s]\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s%s%s %s-b <file|->%s [%soptions%s]\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", LSL_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_ARG, "<linkname>", 26),
           "Alias stream name (without <prefix> or .im.shm)");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_ARG, "<src>=<link>", 26),
           "Batch source-to-alias stream pair (<prefix> applies to <link>)");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-s, --source <src>", 26),
           "Source stream name (single-pair direct mode)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-b, --batch <file|->", 26),
           "Read '<src>=<link>' or '<src> <link>' lines from file or stdin");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-p, --prefix <prefix>", 26),
           "Prefix prepended to each <linkname> (default: none)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--expect-size <WxH[xD]>", 26),
           "Require source stream dimensions to match");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--allow-missing", 26),
           "Allow linking before source stream exists (direct/batch mode)");
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
    printf("  %s$ %s%s%s %s-q --no-conf -p aol7_ dm00disp00=dmO dm00disp03=dmC%s\n\n",
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

static void write_conf_files(const link_pair_t *pair,
                             int                create_conf_dir,
                             int                write_name_file,
                             int                quiet)
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
        snprintf(namefile, sizeof(namefile), "conf/streamlink.%s.name.txt", pair->link);
        FILE *nfp = fopen(namefile, "w");
        if (nfp != NULL)
        {
            fprintf(nfp, "%s\n", pair->src);
            fclose(nfp);
        }
    }

    if (!pair->has_imsize)
    {
        return;
    }

    char outfile[512];
    snprintf(outfile, sizeof(outfile), "conf/streamlink.%s.imsize.txt", pair->link);
    FILE *fp = fopen(outfile, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "  [warn] cannot write '%s': %s\n", outfile, strerror(errno));
        return;
    }

    for (uint8_t dim_idx = 0; dim_idx < pair->src_naxis; dim_idx++)
    {
        fprintf(fp, "%u ", (unsigned) pair->src_size[dim_idx]);
    }
    fprintf(fp, "\n");
    fclose(fp);

    if (!quiet)
    {
        printf("  imsize written -> %s\n", outfile);
    }
} // write_conf_files

static int append_pair(link_pair_t **pairs,
                       size_t       *npairs,
                       size_t       *cap,
                       const char   *progname,
                       const char   *src,
                       const char   *link,
                       int           is_direct)
{
    if (src == NULL || src[0] == '\0' || strlen(src) >= sizeof((*pairs)[0].src) ||
        strchr(src, '/') != NULL)
    {
        fprintf(stderr, "%s: invalid source stream name '%s'\n", progname, src ? src : "");
        return -1;
    }
    if (link == NULL || link[0] == '\0' || strlen(link) >= sizeof((*pairs)[0].link) ||
        strchr(link, '/') != NULL)
    {
        fprintf(stderr, "%s: invalid link stream name '%s'\n", progname, link ? link : "");
        return -1;
    }

    if (*npairs >= *cap)
    {
        size_t       new_cap = (*cap == 0) ? 16 : (*cap * 2);
        link_pair_t *grown   = realloc(*pairs, new_cap * sizeof(*grown));
        if (grown == NULL)
        {
            fprintf(stderr, "%s: out of memory\n", progname);
            return -1;
        }
        *pairs = grown;
        *cap   = new_cap;
    }

    link_pair_t *entry = &(*pairs)[*npairs];
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->src, sizeof(entry->src), "%s", src);
    snprintf(entry->link, sizeof(entry->link), "%s", link);
    entry->is_direct = is_direct;
    (*npairs)++;
    return 0;
} // append_pair

static int parse_pair_token(const char   *progname,
                            const char   *token,
                            link_pair_t **pairs,
                            size_t       *npairs,
                            size_t       *cap)
{
    const char *eq = strchr(token, '=');
    if (eq == NULL || strchr(eq + 1, '=') != NULL)
    {
        fprintf(stderr, "%s: invalid '<src>=<link>' pair '%s'\n", progname, token);
        return -1;
    }

    size_t src_len = (size_t) (eq - token);
    while (src_len > 0 && isspace((unsigned char) token[src_len - 1]))
    {
        src_len--;
    }
    const char *rhs = eq + 1;
    while (*rhs != '\0' && isspace((unsigned char) *rhs))
    {
        rhs++;
    }
    if (src_len == 0 || *rhs == '\0')
    {
        fprintf(stderr, "%s: invalid '<src>=<link>' pair '%s'\n", progname, token);
        return -1;
    }

    char src_buf[256];
    if (src_len >= sizeof(src_buf))
    {
        fprintf(stderr, "%s: source stream name too long in '%s'\n", progname, token);
        return -1;
    }
    memcpy(src_buf, token, src_len);
    src_buf[src_len] = '\0';

    return append_pair(pairs, npairs, cap, progname, src_buf, rhs, 1);
} // parse_pair_token

static int load_batch_file(const char   *progname,
                           const char   *path,
                           link_pair_t **pairs,
                           size_t       *npairs,
                           size_t       *cap)
{
    FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "%s: cannot open batch file '%s': %s\n", progname, path, strerror(errno));
        return -1;
    }

    char   line[1024];
    size_t lineno = 0;
    int    rc     = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        lineno++;
        line[strcspn(line, "\r\n")] = '\0';

        /* Strip leading whitespace */
        char *cur = line;
        while (*cur != '\0' && isspace((unsigned char) *cur))
        {
            cur++;
        }
        if (*cur == '\0' || *cur == '#')
        {
            continue;
        }

        /* Strip inline '#...' comments preceded by whitespace */
        for (char *scan = cur; *scan != '\0'; scan++)
        {
            if (*scan == '#' && scan > cur && isspace((unsigned char) scan[-1]))
            {
                *scan = '\0';
                break;
            }
        }

        /* Trim trailing whitespace */
        size_t len = strlen(cur);
        while (len > 0 && isspace((unsigned char) cur[len - 1]))
        {
            cur[--len] = '\0';
        }
        if (len == 0)
        {
            continue;
        }

        const char *eq = strchr(cur, '=');
        if (eq != NULL)
        {
            if (parse_pair_token(progname, cur, pairs, npairs, cap) != 0)
            {
                fprintf(stderr, "%s: %s:%zu: invalid batch entry\n", progname, path, lineno);
                rc = -1;
                break;
            }
        }
        else
        {
            char *tok1 = cur;
            while (*cur != '\0' && !isspace((unsigned char) *cur))
            {
                cur++;
            }
            if (*cur == '\0')
            {
                fprintf(stderr, "%s: %s:%zu: expected '<src>=<link>' or '<src> <link>'\n", progname,
                        path, lineno);
                rc = -1;
                break;
            }
            *cur++ = '\0';
            while (*cur != '\0' && isspace((unsigned char) *cur))
            {
                cur++;
            }
            char *tok2 = cur;
            while (*cur != '\0' && !isspace((unsigned char) *cur))
            {
                cur++;
            }
            if (*cur != '\0')
            {
                fprintf(stderr, "%s: %s:%zu: unexpected extra token after '%s'\n", progname, path,
                        lineno, tok2);
                rc = -1;
                break;
            }
            if (append_pair(pairs, npairs, cap, progname, tok1, tok2, 1) != 0)
            {
                rc = -1;
                break;
            }
        }
    } // while fgets

    if (fp != stdin)
    {
        fclose(fp);
    }
    return rc;
} // load_batch_file

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

    const char  *prefix        = "";
    const char  *src_opt       = NULL;
    const char  *batch_file    = NULL;
    const char  *expect_str    = NULL;
    int          allow_missing = 0;
    int          no_conf       = 0;
    int          quiet         = 0;
    int          opts_done     = 0;
    uint32_t     exp_dims[3]   = { 0, 0, 0 };
    uint8_t      exp_ndims     = 0;
    const char **pos_args      = calloc((size_t) argc, sizeof(*pos_args));
    int          npos          = 0;

    if (pos_args == NULL)
    {
        free(progbuf);
        return 1;
    }

    for (int arg_idx = 1; arg_idx < argc; arg_idx++)
    {
        const char *arg = argv[arg_idx];
        if (!opts_done && strcmp(arg, "--") == 0)
        {
            opts_done = 1;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-p") == 0 || strcmp(arg, "--prefix") == 0))
        {
            if (arg_idx + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires an argument\n", progname, arg);
                free(pos_args);
                free(progbuf);
                return 2;
            }
            prefix = argv[++arg_idx];
            continue;
        }
        if (!opts_done && (strcmp(arg, "-s") == 0 || strcmp(arg, "--source") == 0))
        {
            if (arg_idx + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires an argument\n", progname, arg);
                free(pos_args);
                free(progbuf);
                return 2;
            }
            src_opt = argv[++arg_idx];
            continue;
        }
        if (!opts_done && (strcmp(arg, "-b") == 0 || strcmp(arg, "--batch") == 0))
        {
            if (arg_idx + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires an argument\n", progname, arg);
                free(pos_args);
                free(progbuf);
                return 2;
            }
            batch_file = argv[++arg_idx];
            continue;
        }
        if (!opts_done && strcmp(arg, "--expect-size") == 0)
        {
            if (arg_idx + 1 >= argc)
            {
                fprintf(stderr, "%s: option '--expect-size' requires an argument\n", progname);
                free(pos_args);
                free(progbuf);
                return 2;
            }
            expect_str = argv[++arg_idx];
            if (parse_dims(expect_str, exp_dims, &exp_ndims) != 0)
            {
                fprintf(stderr, "%s: invalid --expect-size '%s'\n", progname, expect_str);
                free(pos_args);
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
        if (!opts_done && arg[0] == '-' && arg[1] != '\0')
        {
            fprintf(stderr, "%s: unknown option '%s'\n", progname, arg);
            free(pos_args);
            free(progbuf);
            return 2;
        }
        pos_args[npos++] = arg;
    } // for arg_idx

    if (allow_missing && exp_ndims > 0)
    {
        fprintf(stderr, "%s: --allow-missing cannot be combined with --expect-size\n", progname);
        free(pos_args);
        free(progbuf);
        return 2;
    }
    if (src_opt != NULL && batch_file != NULL)
    {
        fprintf(stderr, "%s: -s/--source cannot be combined with -b/--batch\n", progname);
        free(pos_args);
        free(progbuf);
        return 2;
    }

    link_pair_t *pairs  = NULL;
    size_t       npairs = 0;
    size_t       cap    = 0;

    if (batch_file != NULL)
    {
        if (load_batch_file(progname, batch_file, &pairs, &npairs, &cap) != 0)
        {
            free(pairs);
            free(pos_args);
            free(progbuf);
            return 2;
        }
        for (int pos_idx = 0; pos_idx < npos; pos_idx++)
        {
            if (parse_pair_token(progname, pos_args[pos_idx], &pairs, &npairs, &cap) != 0)
            {
                free(pairs);
                free(pos_args);
                free(progbuf);
                return 2;
            }
        }
    }
    else if (src_opt != NULL)
    {
        if (npos != 1)
        {
            fprintf(stderr, "%s: -s/--source requires exactly one <linkname> argument\n", progname);
            free(pos_args);
            free(progbuf);
            return 2;
        }
        if (append_pair(&pairs, &npairs, &cap, progname, src_opt, pos_args[0], 1) != 0)
        {
            free(pairs);
            free(pos_args);
            free(progbuf);
            return 2;
        }
    }
    else if (npos == 1 && strchr(pos_args[0], '=') == NULL)
    {
        /* Legacy single-link mode reading conf/streamlink.<linkname>.name.txt */
        const char *linkname = pos_args[0];
        if (linkname[0] == '\0')
        {
            fprintf(stderr, "%s: <linkname> argument required\n", progname);
            free(pos_args);
            free(progbuf);
            return 2;
        }
        char conffile[512];
        snprintf(conffile, sizeof(conffile), "conf/streamlink.%s.name.txt", linkname);
        FILE *fp = fopen(conffile, "r");
        if (fp == NULL)
        {
            fprintf(stderr, "%s: cannot open '%s': %s\n", progname, conffile, strerror(errno));
            free(pos_args);
            free(progbuf);
            return 1;
        }
        char srcname_buf[256];
        if (fgets(srcname_buf, sizeof(srcname_buf), fp) == NULL)
        {
            fprintf(stderr, "%s: '%s' is empty\n", progname, conffile);
            fclose(fp);
            free(pos_args);
            free(progbuf);
            return 1;
        }
        fclose(fp);
        srcname_buf[strcspn(srcname_buf, "\r\n")] = '\0';
        if (srcname_buf[0] == '\0')
        {
            fprintf(stderr, "%s: '%s' contains an empty stream name\n", progname, conffile);
            free(pos_args);
            free(progbuf);
            return 1;
        }
        if (append_pair(&pairs, &npairs, &cap, progname, srcname_buf, linkname, 0) != 0)
        {
            free(pairs);
            free(pos_args);
            free(progbuf);
            return 2;
        }
    }
    else if (npos >= 1)
    {
        for (int pos_idx = 0; pos_idx < npos; pos_idx++)
        {
            if (parse_pair_token(progname, pos_args[pos_idx], &pairs, &npairs, &cap) != 0)
            {
                free(pairs);
                free(pos_args);
                free(progbuf);
                return 2;
            }
        }
    }
    else
    {
        fprintf(stderr, "%s: <linkname> or <src>=<link> argument required\n", progname);
        print_help(progname, 1);
        free(pos_args);
        free(progbuf);
        return 2;
    } // mode dispatch

    free(pos_args);

    if (npairs == 0)
    {
        free(pairs);
        free(progbuf);
        return 0;
    }

    const char *shmdir = get_shmdir();
    ImageStreamIO_set_printError(NULL);

    /* Phase 1: Validate all pairs before modifying any target symlink */
    for (size_t pair_idx = 0; pair_idx < npairs; pair_idx++)
    {
        link_pair_t *pair = &pairs[pair_idx];
        snprintf(pair->full_link, sizeof(pair->full_link), "%s%s", prefix, pair->link);
        if (strcmp(pair->full_link, pair->src) == 0)
        {
            fprintf(stderr, "%s: refusing self-link '%s' -> '%s'\n", progname, pair->full_link,
                    pair->src);
            free(pairs);
            free(progbuf);
            return 1;
        }

        snprintf(pair->linkpath, sizeof(pair->linkpath), "%s/%s.im.shm", shmdir, pair->full_link);
        snprintf(pair->srcpath, sizeof(pair->srcpath), "%s/%s.im.shm", shmdir, pair->src);

        struct stat st;
        if (lstat(pair->linkpath, &st) == 0 && !S_ISLNK(st.st_mode))
        {
            fprintf(stderr, "%s: refusing to overwrite regular SHM file '%s' (not a symlink)\n",
                    progname, pair->linkpath);
            free(pairs);
            free(progbuf);
            return 1;
        }

        IMAGE src_img;
        int   src_opened = 0;
        memset(&src_img, 0, sizeof(src_img));

        if (pair->is_direct && !allow_missing)
        {
            if (access(pair->srcpath, F_OK) != 0 ||
                ImageStreamIO_openIm(&src_img, pair->src) != IMAGESTREAMIO_SUCCESS ||
                src_img.md == NULL || src_img.md->write != 0 || src_img.md->naxis == 0)
            {
                if (src_img.md != NULL)
                {
                    ImageStreamIO_closeIm(&src_img);
                }
                fprintf(stderr,
                        "%s: source stream '%s' is missing or not ready (use --allow-missing)\n",
                        progname, pair->src);
                free(pairs);
                free(progbuf);
                return 1;
            }
            src_opened = 1;
        }
        else if (access(pair->srcpath, F_OK) == 0 &&
                 ImageStreamIO_openIm(&src_img, pair->src) == IMAGESTREAMIO_SUCCESS &&
                 src_img.md != NULL)
        {
            src_opened = 1;
        }

        if (src_opened)
        {
            pair->has_imsize = 1;
            pair->src_naxis  = src_img.md->naxis;
            for (uint8_t dim_idx = 0; dim_idx < 3 && dim_idx < src_img.md->naxis; dim_idx++)
            {
                pair->src_size[dim_idx] = src_img.md->size[dim_idx];
            }
        }

        if (exp_ndims > 0)
        {
            if (!src_opened)
            {
                fprintf(stderr, "%s: cannot verify --expect-size: source '%s' not open\n", progname,
                        pair->src);
                free(pairs);
                free(progbuf);
                return 1;
            }
            int mismatch = (src_img.md->naxis != exp_ndims);
            for (uint8_t dim_idx = 0; !mismatch && dim_idx < exp_ndims; dim_idx++)
            {
                if (src_img.md->size[dim_idx] != exp_dims[dim_idx])
                {
                    mismatch = 1;
                }
            }
            if (mismatch)
            {
                fprintf(stderr, "%s: source '%s' dimensions do not match --expect-size %s\n",
                        progname, pair->src, expect_str);
                ImageStreamIO_closeIm(&src_img);
                free(pairs);
                free(progbuf);
                return 1;
            }
        } // if exp_ndims > 0

        if (src_opened)
        {
            ImageStreamIO_closeIm(&src_img);
        }
    } // for pair_idx (Phase 1)

    /* Phase 2: Atomic symlink swap via temporary symlink + rename() */
    int pid_val = (int) getpid();
    for (size_t pair_idx = 0; pair_idx < npairs; pair_idx++)
    {
        const link_pair_t *pair = &pairs[pair_idx];
        char               tmppath[1024];
        snprintf(tmppath, sizeof(tmppath), "%s/.tmp.%s.%d.%zu.im.shm", shmdir, pair->full_link,
                 pid_val, pair_idx);

        unlink(tmppath);
        if (symlink(pair->srcpath, tmppath) != 0)
        {
            fprintf(stderr, "%s: symlink('%s' -> '%s') failed: %s\n", progname, pair->srcpath,
                    tmppath, strerror(errno));
            free(pairs);
            free(progbuf);
            return 1;
        }
        if (rename(tmppath, pair->linkpath) != 0)
        {
            fprintf(stderr, "%s: rename('%s' -> '%s') failed: %s\n", progname, tmppath,
                    pair->linkpath, strerror(errno));
            unlink(tmppath);
            free(pairs);
            free(progbuf);
            return 1;
        }

        if (!quiet)
        {
            printf("  Linking %s:\n    ln -s %s\n       -> %s\n", pair->full_link, pair->srcpath,
                   pair->linkpath);
        }

        if (!no_conf)
        {
            if (!pair->is_direct)
            {
                if (pair->has_imsize)
                {
                    write_conf_files(pair, 1, 0, quiet);
                }
                else if (!quiet)
                {
                    printf("  [warn] source '%s' not found -- skipping imsize write\n",
                           pair->srcpath);
                }
            }
            else
            {
                write_conf_files(pair, 0, 1, quiet);
            }
        }
    } // for pair_idx (Phase 2)

    free(pairs);
    free(progbuf);
    return 0;
} // main

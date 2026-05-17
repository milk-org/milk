/**
 * @file milk-stream-list.c
 * @brief Milk stream list module
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <regex.h>

#include "libmilkcommon/milkDebugTools.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_help.h"

#define C_TITLE MH_TITLE
#define C_HDR   MH_HDR
#define C_NAME  MH_CMD
#define C_TYPE  MH_NOTE
#define C_SIZE  MH_BOLD
#define C_CNT   MH_ARG
#define C_SEM   MH_DFLT
#define C_LINK  MH_DFLT
#define C_ERR   MH_ERR
#define C_DIM   MH_DIM
#define C_RST   MH_RST

#define STRINGMAXLEN_FULLFILENAME 512

#define SL_DESC "list shared memory image streams"
#define SL_DESC_LONG \
    "Scan /dev/shm for ImageStreamIO (.im.shm) files and print a\n" \
    "summary table. Each row shows stream name, pixel type, dimensions,\n" \
    "and write counter. Use -a for full details including semaphores.\n" \
    "An optional regex pattern filters which streams are listed."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, SL_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sregex%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SL_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-a, --all",
           mh_color ? MH_RST : "",
           "Show all details (verbose, includes semaphores)");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-stream-list%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-stream-list%s %sdm.*%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = {
        "milk-stream-info:inspect stream metadata and data",
        "milk-stream-rm:remove shared memory streams",
        "milk-stream-create:create a new shared memory stream"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                SL_DESC, SL_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
        return 0;
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int show_all = 0;
    int opt;

    static struct option long_options[] = {
        {"all",     no_argument,       0, 'a'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "ah", long_options, NULL)) != -1) {
        switch (opt) {
            case 'a':
                show_all = 1;
                break;
            case 'h': break; /* handled above */
            default:
                printf("\n\033[1;31mERROR\033[0m: Invalid option.\n\n");
                print_help(argv[0], 1);
                return 1;
        }
    }

    const char *pattern = NULL;
    regex_t regex;
    int use_regex = 0;

    if (optind < argc) {
        pattern = argv[optind];
        int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
        if (ret != 0) {
            char error_msg[128];
            regerror(ret, &regex, error_msg, sizeof(error_msg));
            PRINT_ERROR("Error: Invalid regular expression. %s", error_msg);
            return 1;
        }
        use_regex = 1;
    }

    const char *shmdir = getenv("MILK_SHM_DIR");
    if (shmdir == NULL) {
        shmdir = "/milk/shm"; 
        struct stat st;
        if (stat(shmdir, &st) != 0) {
             shmdir = "/dev/shm"; 
        }
    }

    DIR *d;
    struct dirent *dir;

    d = opendir(shmdir);
    if(d)
    {
        // Header
        printf(C_TITLE "%-30s %-12s %-20s %-12s"
               C_RST,
               "Stream Name",
               "Type",
               "Size",
               "Cnt0");
        if (show_all) {
            printf(C_TITLE " %-40s" C_RST,
                   "Semaphores (up to 10)");
        }
        printf("\n");

        int total_width = 30 + 1 + 12 + 1 + 20 + 1 + 12;
        if (show_all)
            total_width += 1 + 40;
        printf(C_DIM);
        for (int i = 0; i < total_width; i++)
            putchar('-');
        printf(C_RST "\n");

        while(((dir = readdir(d)) != NULL))
        {
            char *pch = strstr(dir->d_name, ".im.shm");
            if(pch && (pch - dir->d_name == (int)strlen(dir->d_name) - 7))
            {
                // Found a stream candidate
                char sname[256];
                strncpy(sname, dir->d_name, sizeof(sname));
                sname[strlen(dir->d_name) - 7] = '\0'; // Remove .im.shm

                if (use_regex && regexec(&regex, sname, 0, NULL, 0) != 0) {
                    continue; // Skip if it doesn't match the regex
                }

                // Check if it's a symlink
                char fullname[STRINGMAXLEN_FULLFILENAME];
                snprintf(fullname, sizeof(fullname), "%s/%s", shmdir, dir->d_name);
                
                struct stat buf;
                if(lstat(fullname, &buf) == 0)
                {
                    if(S_ISLNK(buf.st_mode))
                    {
                        char linktarget[STRINGMAXLEN_FULLFILENAME];
                        ssize_t len = readlink(fullname,
                            linktarget,
                            sizeof(linktarget)-1);
                        if (len != -1) {
                            linktarget[len] = '\0';
                            
                            struct stat target_stat;
                            int target_exists = (stat(fullname, &target_stat) == 0);
                            
                            printf(C_LINK "%-30s" C_RST
                                   " " C_DIM "%-12s" C_RST
                                   " -> ",
                                   sname, "LINK");
                            if (!target_exists) {
                                printf(C_ERR "%s"
                                       C_RST "\n",
                                       linktarget);
                            } else {
                                printf("%s\n",
                                       linktarget);
                            }
                        } else {
                            printf(C_LINK "%-30s"
                                   C_RST " "
                                   C_ERR "%-12s"
                                   C_RST "\n",
                                   sname,
                                   "LINK (err)");
                        }
                    }
                    else
                    {
                        // Try to open image to get details
                        IMAGE image = {0};
                        
                        errno_t ret = ImageStreamIO_read_sharedmem_image_toIMAGE(
                            sname,
                            &image);
                        
                        if (ret == IMAGESTREAMIO_SUCCESS) {
                            const char *typestr = ImageStreamIO_typename(image.md->datatype);
                            char size_str[32];
                            if (image.md->naxis == 1) snprintf(size_str, 32, "%u", image.md->size[0]);
                            else if (image.md->naxis == 2) snprintf(size_str, 32, "%u x %u", image.md->size[0], image.md->size[1]);
                            else snprintf(size_str, 32, "%u x %u x %u", image.md->size[0], image.md->size[1], image.md->size[2]);

                            printf(C_NAME "%-30s" C_RST
                                   " " C_TYPE "%-12s" C_RST
                                   " " C_SIZE "%-20s" C_RST
                                   " " C_CNT "%-12lu" C_RST,
                                   sname,
                                   typestr ? typestr : "???",
                                   size_str,
                                   (unsigned long)image.md->cnt0);
                            
                            if (show_all) {
                                char sem_str[256] = "";
                                int nbsem = image.md->sem;
                                if (nbsem > 10)
                                    nbsem = 10;
                                for (int i = 0; i < nbsem; i++) {
                                    long sval =
                                        ImageStreamIO_semvalue(
                                            &image, i);
                                    char valbuf[16];
                                    snprintf(valbuf, 16,
                                             "%ld", sval);
                                    if (i > 0)
                                        strcat(sem_str, ":");
                                    strcat(sem_str, valbuf);
                                }
                                printf(" " C_SEM "%-40s"
                                       C_RST, sem_str);
                            }
                            printf("\n");

                            ImageStreamIO_closeIm(&image);
                        } else {
                            printf(C_NAME "%-30s" C_RST
                                   " " C_ERR "%-12s"
                                   C_RST "\n",
                                   sname,
                                   "ERROR_OPEN");
                        }
                    }
                }
            }
        }
        closedir(d);
        printf("\n");
    }
    else
    {
        PRINT_ERROR("Error opening directory %s", shmdir);
        return 1;
    }

    if (use_regex) {
        regfree(&regex);
    }

    return 0;
}
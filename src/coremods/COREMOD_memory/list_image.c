/**
 * @file    list_image.c
 * @brief   list images
 *
 * Uses FPS V2 framework.
 */

#ifdef USE_NCURSES
#include <ncurses.h>
#else
#define printw(...) printf(__VA_ARGS__)
#define mvprintw(y,x,...) printf(__VA_ARGS__)
#define attron(a)
#define attroff(a)
#define A_BOLD 0
#define COLOR_PAIR(c) 0
#endif

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "compute_image_memory.h"
#include "compute_nb_image.h"
#include "image_ID.h"

#define STYPESIZE 10

// MEMORY MONITOR
static FILE   *listim_scr_fpo;
static FILE   *listim_scr_fpi;
#ifdef USE_NCURSES
static SCREEN *listim_scr;
#endif

static int listim_scr_wrow;
static int listim_scr_wcol;

/* forward decls */
errno_t memory_monitor(
    const char *termttyname);
errno_t init_list_image_ID_ncurses(
    const char *termttyname);
void close_list_image_ID_ncurses();
errno_t list_image_ID_ncurses();
errno_t list_image_ID_ofp(FILE *fo);
errno_t list_image_ID_ofp_simple(FILE *fo);
errno_t list_image_ID();
errno_t list_image_ID_file(
    const char *fname);
errno_t list_variable_ID();
errno_t list_variable_ID_file(
    const char *fname);


/* ================================================================
 *  CMD 1: mmon (1 arg, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mmon",
    .cmdkey      = "mmon",
    .description =
        "monitor memory content"
};

static char p_ttyname[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "/dev/pts/4";

#define FPS_PARAMS(X) \
    X(".ttyname", p_ttyname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "terminal tty name")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS cms_mmon = {0};

static __attribute__((constructor))
void init_cms_mmon(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms_mmon;
    }
}

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    memory_monitor(p_ttyname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: listim (0 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listim = {
    .fps_name    = "listim",
    .cmdkey      = "listim",
    .description =
        "list images in memory"
};

static CLICMDDATA CLIcmddata_listim = {
    "", "", CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms_listim = {0};

static __attribute__((constructor))
void init_cms_listim(void)
{
    strncpy(CLIcmddata_listim.key,
            FPS_app_info_listim.cmdkey,
            sizeof(CLIcmddata_listim.key)
            - 1);
    strncpy(
        CLIcmddata_listim.description,
        FPS_app_info_listim.description,
        sizeof(
            CLIcmddata_listim.description
        ) - 1);
    if (CLIcmddata_listim.cmdsettings
        == NULL) {
        CLIcmddata_listim.cmdsettings =
            &cms_listim;
    }
}

static errno_t compute_listim()
{
    list_image_ID();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

static errno_t CLIfunction_listim(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_listim,
        farg, &CLIcmddata_listim,
        my_bindings, nb_bindings,
        compute_listim);
}

errno_t
CLIADDCMD_COREMOD_memory__list_image()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_listim,
            CLIfunction_listim);
        CLIcmddata_listim.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */
#ifdef USE_NCURSES
errno_t init_list_image_ID_ncurses(const char *termttyname)
{
    //    int wrow, wcol;

    listim_scr_fpi = fopen(termttyname, "r");
    listim_scr_fpo = fopen(termttyname, "w");
    listim_scr     = newterm(NULL, listim_scr_fpo, listim_scr_fpi);

    getmaxyx(stdscr, listim_scr_wrow, listim_scr_wcol);
    start_color();
    init_pair(1, COLOR_BLACK, COLOR_WHITE);
    init_pair(2, COLOR_BLACK, COLOR_RED);
    init_pair(3, COLOR_GREEN, COLOR_BLACK);
    init_pair(4, COLOR_RED, COLOR_BLACK);
    init_pair(5, COLOR_BLACK, COLOR_GREEN);
    init_pair(6, COLOR_CYAN, COLOR_BLACK);
    init_pair(7, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(8, COLOR_BLACK, COLOR_MAGENTA);
    init_pair(9, COLOR_YELLOW, COLOR_BLACK);

    return RETURN_SUCCESS;
}

errno_t list_image_ID_ncurses()
{
    int strmaxlen = 300;
    char      str[strmaxlen];
    int str1maxlen = 500;
    char      str1[500];
    int str2maxlen = 512;
    char      str2[512];
    long      i, j;
    long long tmp_long;
    char      type[STYPESIZE];
    uint8_t   datatype;
    int       n;
    uint64_t  sizeb, sizeKb, sizeMb, sizeGb;

    struct timespec timenow;
    double          timediff;

    clock_gettime(CLOCK_MILK, &timenow);

#ifdef USE_NCURSES
    set_term(listim_scr);
    clear();
#endif

    sizeb = compute_image_memory();

    printw(
        "INDEX    NAME         SIZE                    TYPE        SIZE  "
        "[percent]    LAST ACCESS\n");
    printw("\n");

    for(i = 0; i < dcnimg; i++)
    {
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            tmp_long = ((long long)(dcimg[i].md[0].nelement)) *
                       ImageStreamIO_typesize(datatype);

            if(dcimg[i].md[0].shared == 1)
            {
                printw("%4ldS", i);
            }
            else
            {
                printw("%4ld ", i);
            }

            if(dcimg[i].md[0].shared == 1)
            {
                attron(A_BOLD | COLOR_PAIR(9));
            }
            else
            {
                attron(A_BOLD | COLOR_PAIR(6));
            }
            snprintf(str, strmaxlen, "%10s ", dcimg[i].name);
            printw("%s", str);

            if(dcimg[i].md[0].shared == 1)
            {
                attroff(A_BOLD | COLOR_PAIR(9));
            }
            else
            {
                attroff(A_BOLD | COLOR_PAIR(6));
            }

            snprintf(str, strmaxlen, "[ %6ld", (long) dcimg[i].md[0].size[0]);

            for(j = 1; j < dcimg[i].md[0].naxis; j++)
            {
                snprintf(str1,
                         str1maxlen,
                         "%s x %6ld",
                         str,
                         (long) dcimg[i].md[0].size[j]);
            }
            snprintf(str2, str2maxlen, "%s]", str1);

            printw("%-28s", str2);

            attron(COLOR_PAIR(3));

            n = snprintf(type, STYPESIZE, "%s", ImageStreamIO_typename_7(datatype));

            printw("%7s ", type);

            attroff(COLOR_PAIR(3));

            if(n >= STYPESIZE)
            {
                PRINT_ERROR(
                    "Attempted to write string buffer with too "
                    "many characters");
            }

            printw("%10ld Kb %6.2f   ",
                   (long)(tmp_long / 1024),
                   (float)(100.0 * tmp_long / sizeb));

            timediff =
                (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                (1.0 * dcimg[i].md[0].lastaccesstime.tv_sec +
                 0.000000001 * dcimg[i].md[0].lastaccesstime.tv_nsec);

            if(timediff < 0.01)
            {
                attron(COLOR_PAIR(4));
                printw("%15.9f\n", timediff);
                attroff(COLOR_PAIR(4));
            }
            else
            {
                printw("%15.9f\n", timediff);
            }

        }
        else
        {
            printw("\n");
        }
    }

    sizeGb = 0;
    sizeMb = 0;
    sizeKb = 0;
    sizeb  = compute_image_memory();

    if(sizeb > 1024 - 1)
    {
        sizeKb = sizeb / 1024;
        sizeb  = sizeb - 1024 * sizeKb;
    }
    if(sizeKb > 1024 - 1)
    {
        sizeMb = sizeKb / 1024;
        sizeKb = sizeKb - 1024 * sizeMb;
    }
    if(sizeMb > 1024 - 1)
    {
        sizeGb = sizeMb / 1024;
        sizeMb = sizeMb - 1024 * sizeGb;
    }

    //attron(A_BOLD);

    snprintf(str, strmaxlen, "%ld image(s)      ", compute_nb_image());
    if(sizeGb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld GB", str, (long)(sizeGb));
        strcpy(str, str1);
    }

    if(sizeMb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld MB", str, (long)(sizeMb));
        strcpy(str, str1);
    }

    if(sizeKb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld KB", str, (long)(sizeKb));
        strcpy(str, str1);
    }

    if(sizeb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld B", str, (long)(sizeb));
        strcpy(str, str1);
    }

    mvprintw(listim_scr_wrow - 1, 0, "%s\n", str);
    //  attroff(A_BOLD);

#ifdef USE_NCURSES
    refresh();
#endif

    return RETURN_SUCCESS;
}

void close_list_image_ID_ncurses(void)
{
    printf("Closing monitor cleanly\n");
    set_term(listim_scr);
    endwin();
    fclose(listim_scr_fpo);
    fclose(listim_scr_fpi);
    dcmemmon = 0;
}
#else
errno_t init_list_image_ID_ncurses(const char *termttyname) { (void)termttyname; return 0; }
errno_t list_image_ID_ncurses() { return 0; }
void close_list_image_ID_ncurses(void) {}
#endif

errno_t list_image_ID_ofp(FILE *fo)
{
    long               i;
    long               j;
    long long          tmp_long;
    char               type[STYPESIZE];
    uint8_t            datatype;
    int                n;
    unsigned long long sizeb, sizeKb, sizeMb, sizeGb;
    int strmaxlen = 500;
    char               str[strmaxlen];
    int str1maxlen = 512;
    char               str1[str1maxlen];
    struct timespec    timenow;
    double             timediff;
    //struct mallinfo minfo;

    sizeb = compute_image_memory();
    //minfo = mallinfo();

    clock_gettime(CLOCK_MILK, &timenow);
    //fprintf(fo, "time:  %ld.%09ld\n", timenow.tv_sec % 60, timenow.tv_nsec);

    fprintf(fo, "\n");
    fprintf(fo,
            "INDEX    NAME         SIZE                    TYPE        SIZE  "
            "[percent]    LAST ACCESS\n");
    fprintf(fo, "\n");

    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            tmp_long = ((long long)(dcimg[i].md[0].nelement)) *
                       ImageStreamIO_typesize(datatype);

            if(dcimg[i].md[0].shared == 1)
            {
                fprintf(fo,
                        "%4ld %c[%d;%dm%14s%c[%d;m ",
                        i,
                        (char) 27,
                        1,
                        34,
                        dcimg[i].name,
                        (char) 27,
                        0);
            }
            else
            {
                fprintf(fo,
                        "%4ld %c[%d;%dm%14s%c[%d;m ",
                        i,
                        (char) 27,
                        1,
                        33,
                        dcimg[i].name,
                        (char) 27,
                        0);
            }
            //fprintf(fo, "%s", str);

            snprintf(str, strmaxlen, "[ %6ld", (long) dcimg[i].md[0].size[0]);

            for(j = 1; j < dcimg[i].md[0].naxis; j++)
            {
                snprintf(str1,
                         str1maxlen,
                         "%s x %6ld",
                         str,
                         (long) dcimg[i].md[0].size[j]);
                strcpy(str, str1);
            }
            snprintf(str1, str1maxlen, "%s]", str);
            strcpy(str, str1);

            fprintf(fo, "%-32s", str);

            n = snprintf(type, STYPESIZE, "%s", ImageStreamIO_typename_7(datatype));

            fprintf(fo, "%7s ", type);

            if(n >= STYPESIZE)
            {
                PRINT_ERROR(
                    "Attempted to write string buffer with too many "
                    "characters");
            }

            fprintf(fo,
                    "%10ld Kb %6.2f   ",
                    (long)(tmp_long / 1024),
                    (float)(100.0 * tmp_long / sizeb));

            timediff =
                (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                (1.0 * dcimg[i].md[0].lastaccesstime.tv_sec +
                 0.000000001 * dcimg[i].md[0].lastaccesstime.tv_nsec);

            fprintf(fo, "%15.9f\n", timediff);
        }
    fprintf(fo, "\n");

    sizeGb = 0;
    sizeMb = 0;
    sizeKb = 0;
    sizeb  = compute_image_memory();

    if(sizeb > 1024 - 1)
    {
        sizeKb = sizeb / 1024;
        sizeb  = sizeb - 1024 * sizeKb;
    }
    if(sizeKb > 1024 - 1)
    {
        sizeMb = sizeKb / 1024;
        sizeKb = sizeKb - 1024 * sizeMb;
    }
    if(sizeMb > 1024 - 1)
    {
        sizeGb = sizeMb / 1024;
        sizeMb = sizeMb - 1024 * sizeGb;
    }

    fprintf(fo, "%ld image(s)   ", compute_nb_image());
    if(sizeGb > 0)
    {
        fprintf(fo, " %ld Gb", (long)(sizeGb));
    }
    if(sizeMb > 0)
    {
        fprintf(fo, " %ld Mb", (long)(sizeMb));
    }
    if(sizeKb > 0)
    {
        fprintf(fo, " %ld Kb", (long)(sizeKb));
    }
    if(sizeb > 0)
    {
        fprintf(fo, " %ld", (long)(sizeb));
    }
    fprintf(fo, "\n");

    fflush(fo);

    return RETURN_SUCCESS;
}

errno_t list_image_ID_ofp_simple(FILE *fo)
{
    long i, j;
    //long long   tmp_long;
    uint8_t datatype;

    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            //tmp_long = ((long long) (dcimg[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            fprintf(fo,
                    "%20s %d %ld %d %4ld",
                    dcimg[i].name,
                    datatype,
                    (long) dcimg[i].md[0].naxis,
                    dcimg[i].md[0].shared,
                    (long) dcimg[i].md[0].size[0]);

            for(j = 1; j < dcimg[i].md[0].naxis; j++)
            {
                fprintf(fo, " %4ld", (long) dcimg[i].md[0].size[j]);
            }
            fprintf(fo, "\n");

        }
    fprintf(fo, "\n");

    return RETURN_SUCCESS;
}

errno_t list_image_ID()
{
    list_image_ID_ofp(stdout);
    //malloc_stats();
    return RETURN_SUCCESS;
}

/* list all images in memory
   output is written in ASCII file
   only basic info is listed
   image name
   number of axis
   size
   type
 */

errno_t list_image_ID_file(const char *fname)
{
    FILE   *fp;
    long    i, j;
    uint8_t datatype;
    char    type[STYPESIZE];
    int     n;

    fp = fopen(fname, "w");
    if(fp == NULL)
    {
        PRINT_ERROR("Cannot create file %s", fname);
        abort();
    }

    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            fprintf(fp, "%ld %s", i, dcimg[i].name);
            fprintf(fp, " %ld", (long) dcimg[i].md[0].naxis);
            for(j = 0; j < dcimg[i].md[0].naxis; j++)
            {
                fprintf(fp, " %ld", (long) dcimg[i].md[0].size[j]);
            }

            n = snprintf(type, STYPESIZE, "%s", ImageStreamIO_typename_7(datatype));

            if(n >= STYPESIZE)
            {
                PRINT_ERROR(
                    "Attempted to write string buffer with too many "
                    "characters");
            }

            fprintf(fp, " %s\n", type);
        }
    fclose(fp);

    return RETURN_SUCCESS;
}

errno_t memory_monitor(const char *termttyname)
{
    if(dcdebug > 0)
    {
        printf("starting memory_monitor on \"%s\"\n", termttyname);
    }

    dcmemmon = 1;
    init_list_image_ID_ncurses(termttyname);
    list_image_ID_ncurses();
#ifdef USE_NCURSES
    atexit(close_list_image_ID_ncurses);
#endif

    return RETURN_SUCCESS;
}

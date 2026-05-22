/**
 * @file    mmon_ui.c
 * @brief   Memory monitor standalone TUI
 *
 * Implements the "mmon" interactive ncurses interface
 * for tracking memory allocations and streams globally.
 */

#include "milk_config.h"

#ifdef USE_NCURSES
#    include <ncurses.h>
#    include <unistd.h>
#else
#    define printw(...) printf(__VA_ARGS__)
#    define mvprintw(y, x, ...) printf(__VA_ARGS__)
#    define attron(a)
#    define attroff(a)
#    define A_BOLD 0
#    define COLOR_PAIR(c) 0
#endif

// We use CLIcore standalone stubs if necessary
// But since this is a UI tool, it's typically built with CLI
#ifndef MILK_NO_CLI
#    include "CLIcore.h"
#else
#    include "CLIcore_standalone.h"
#    include "libmilkdata/milkdata.h"
#    include "libmilkdata/milkdata_macros.h"
#endif

#include "fps.h"
#include "COREMOD_memory/compute_image_memory.h"
#include "COREMOD_memory/compute_nb_image.h"

#define STYPESIZE 10

// MEMORY MONITOR globals
static FILE *listim_scr_fpo;
static FILE *listim_scr_fpi;

#ifdef USE_NCURSES
static SCREEN *listim_scr;
#endif

static int listim_scr_wrow;
static int listim_scr_wcol;

/* forward decls */
errno_t init_list_image_ID_ncurses(const char *termttyname);
/**
 * @brief Close ncurses memory monitor display.
 */
void close_list_image_ID_ncurses();
/**
 * @brief Display image list in ncurses mode.
 */
errno_t list_image_ID_ncurses();
/**
 * @brief Main memory monitor loop.
 */
errno_t memory_monitor(const char *termttyname);


/* ================================================================
 *  STANDALONE REGISTRATION
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = { .fps_name    = "mmon",
                                     .cmdkey      = "mmon",
                                     .description = "monitor memory content" };

static char p_ttyname[FUNCTION_PARAMETER_STRMAXLEN] = "/dev/pts/4";

#define FPS_PARAMS(X) \
    X(".ttyname", p_ttyname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "terminal tty name")

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int nb_bindings = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

static CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };

FPS_CMDSETTINGS_INIT(mmon, CLIcmddata, FPS_app_info)


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    memory_monitor(p_ttyname);
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  IMPLEMENTATION
 * ============================================================= */

#ifdef USE_NCURSES
errno_t init_list_image_ID_ncurses(const char *termttyname)
{
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

/**
 * @brief Display image list in ncurses mode.
 */
errno_t list_image_ID_ncurses()
{
    struct timespec timenow;

    clock_gettime(CLOCK_MILK, &timenow);

#    ifdef USE_NCURSES
    set_term(listim_scr);
    clear();
#    endif

    uint64_t sizeb = compute_image_memory();

    printw("INDEX    NAME         SIZE                    TYPE        SIZE  [percent]    LAST "
           "ACCESS\n\n");

    for (long i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            uint8_t   datatype = dcimg[i].md[0].datatype;
            long long tmp_long =
                ((long long) (dcimg[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            if (dcimg[i].md[0].shared == 1)
            {
                printw("%4ldS", i);
            }
            else
            {
                printw("%4ld ", i);
            }

            if (dcimg[i].md[0].shared == 1)
            {
                attron(A_BOLD | COLOR_PAIR(9));
            }
            else
            {
                attron(A_BOLD | COLOR_PAIR(6));
            }

            int  strmaxlen = 300;
            char str[300];
            snprintf(str, strmaxlen, "%10s ", dcimg[i].name);
            printw("%s", str);

            if (dcimg[i].md[0].shared == 1)
            {
                attroff(A_BOLD | COLOR_PAIR(9));
            }
            else
            {
                attroff(A_BOLD | COLOR_PAIR(6));
            }

            snprintf(str, strmaxlen, "[ %6ld", (long) dcimg[i].md[0].size[0]);

            int  str1maxlen = 500;
            char str1[500];
            for (long j = 1; j < dcimg[i].md[0].naxis; j++)
            {
                snprintf(str1, str1maxlen, "%s x %6ld", str, (long) dcimg[i].md[0].size[j]);
            }

            int  str2maxlen = 512;
            char str2[512];
            snprintf(str2, str2maxlen, "%s]", str1);

            printw("%-28s", str2);

            attron(COLOR_PAIR(3));
            char type[STYPESIZE];
            int  n = snprintf(type, STYPESIZE, "%s", ImageStreamIO_typename_7(datatype));
            printw("%7s ", type);
            attroff(COLOR_PAIR(3));

            if (n >= STYPESIZE)
            {
                PRINT_ERROR("Attempted to write string buffer with too many characters");
            }

            printw("%10ld Kb %6.2f   ", (long) (tmp_long / 1024),
                   (float) (100.0 * tmp_long / sizeb));

            double timediff = (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                              (1.0 * dcimg[i].md[0].lastaccesstime.tv_sec +
                               0.000000001 * dcimg[i].md[0].lastaccesstime.tv_nsec);

            if (timediff < 0.01)
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

    uint64_t sizeGb = 0;
    uint64_t sizeMb = 0;
    uint64_t sizeKb = 0;
    sizeb           = compute_image_memory();

    if (sizeb > 1024 - 1)
    {
        sizeKb = sizeb / 1024;
        sizeb  = sizeb - 1024 * sizeKb;
    }
    if (sizeKb > 1024 - 1)
    {
        sizeMb = sizeKb / 1024;
        sizeKb = sizeKb - 1024 * sizeMb;
    }
    if (sizeMb > 1024 - 1)
    {
        sizeGb = sizeMb / 1024;
        sizeMb = sizeMb - 1024 * sizeGb;
    }

    int  strmaxlen = 300;
    char str[300];
    snprintf(str, strmaxlen, "%ld image(s)      ", compute_nb_image());

    int  str1maxlen = 500;
    char str1[500];
    if (sizeGb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld GB", str, (long) (sizeGb));
        strncpy(str, str1, sizeof(str) - 1);
        str[sizeof(str) - 1] = '\0';
    }
    if (sizeMb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld MB", str, (long) (sizeMb));
        strncpy(str, str1, sizeof(str) - 1);
        str[sizeof(str) - 1] = '\0';
    }
    if (sizeKb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld KB", str, (long) (sizeKb));
        strncpy(str, str1, sizeof(str) - 1);
        str[sizeof(str) - 1] = '\0';
    }
    if (sizeb > 0)
    {
        snprintf(str1, str1maxlen, "%s %ld B", str, (long) (sizeb));
        strncpy(str, str1, sizeof(str) - 1);
        str[sizeof(str) - 1] = '\0';
    }

    mvprintw(listim_scr_wrow - 1, 0, "%s\n", str);

#    ifdef USE_NCURSES
    refresh();
#    endif

    return RETURN_SUCCESS;
}

/**
 * @brief Close ncurses memory monitor display.
 */
void close_list_image_ID_ncurses(void)
{
    printf("Closing monitor cleanly\n");
    set_term(listim_scr);
    endwin();
    if (listim_scr_fpo)
    {
        fclose(listim_scr_fpo);
    }
    if (listim_scr_fpi)
    {
        fclose(listim_scr_fpi);
    }
    dcmemmon = 0;
}
#else
errno_t init_list_image_ID_ncurses(const char *termttyname)
{
    (void) termttyname;
    return 0;
}
/**
 * @brief Display image list in ncurses mode.
 */
errno_t list_image_ID_ncurses()
{
    return 0;
}
/**
 * @brief Close ncurses memory monitor display.
 */
void close_list_image_ID_ncurses(void)
{
}
#endif

static int mmon_initialized = 0;

/**
 * @brief Main memory monitor loop.
 */
errno_t memory_monitor(const char *termttyname)
{
    if (mmon_initialized == 0)
    {
        if (dcdebug > 0)
        {
            printf("starting memory_monitor on \"%s\"\n", termttyname);
        }
        init_list_image_ID_ncurses(termttyname);
#ifdef USE_NCURSES
        atexit(close_list_image_ID_ncurses);
#endif
        mmon_initialized = 1;
    }

    list_image_ID_ncurses();

    // Small sleep to prevent 100% CPU when running free without FPS pacing
    usleep(100000);

    return RETURN_SUCCESS;
}


/* ================================================================
 *  STANDALONE V2 ENTRYPOINT / CLI REGISTRATION
 * ============================================================= */
#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_streamCTRL_mmon_ui()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    return RETURN_SUCCESS;
}
#endif

FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)

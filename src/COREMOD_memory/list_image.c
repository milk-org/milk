/**
 * @file    list_image.c
 * @brief   list images
 */


#include <ncurses.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "compute_nb_image.h"
#include "compute_image_memory.h"


#define STYPESIZE 10


// MEMORY MONITOR
static FILE *listim_scr_fpo;
static FILE *listim_scr_fpi;
static SCREEN *listim_scr; // for memory monitoring

static int listim_scr_wrow;
static int listim_scr_wcol;





// ==========================================
// Forward declaration(s)
// ==========================================

errno_t    memory_monitor(
    const char *termttyname
);


errno_t init_list_image_ID_ncurses(
    const char *termttyname
);

void close_list_image_ID_ncurses();

errno_t list_image_ID_ncurses();

errno_t list_image_ID_ofp(
    FILE *fo
);

errno_t list_image_ID_ofp_simple(
    FILE *fo
);

errno_t list_image_ID();

errno_t list_image_ID_file(
    const char *fname
);

errno_t list_variable_ID();

errno_t list_variable_ID_file(
    const char *fname
);



// ==========================================
// Command line interface wrapper function(s)
// ==========================================


static errno_t memory_monitor__cli()
{
    memory_monitor(data.cmdargtoken[1].val.string);
    return CLICMD_SUCCESS;
}




// ==========================================
// Register CLI command(s)
// ==========================================

errno_t list_image_addCLIcmd()
{
	 RegisterCLIcommand(
        "mmon",
        __FILE__,
        memory_monitor__cli,
        "Monitor memory content",
        "terminal tty name",
        "mmon /dev/pts/4",
        "int memory_monitor(const char *ttyname)");

    RegisterCLIcommand(
        "listim",
        __FILE__,
        list_image_ID,
        "list images in memory",
        "no argument",
        "listim", "errno_t list_image_ID()");

    return RETURN_SUCCESS;
}








errno_t init_list_image_ID_ncurses(
    const char *termttyname
)
{
//    int wrow, wcol;

    listim_scr_fpi = fopen(termttyname, "r");
    listim_scr_fpo = fopen(termttyname, "w");
    listim_scr = newterm(NULL, listim_scr_fpo, listim_scr_fpi);

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
    char      str[300];
    char      str1[500];
    char      str2[512];
    long      i, j;
    long long tmp_long;
    char      type[STYPESIZE];
    uint8_t   datatype;
    int       n;
    uint64_t  sizeb, sizeKb, sizeMb, sizeGb;

    struct    timespec timenow;
    double    timediff;

    clock_gettime(CLOCK_REALTIME, &timenow);

    set_term(listim_scr);

    clear();


    sizeb = compute_image_memory();


    printw("INDEX    NAME         SIZE                    TYPE        SIZE  [percent]    LAST ACCESS\n");
    printw("\n");

    for(i = 0; i < data.NB_MAX_IMAGE; i++)
    {
        if(data.image[i].used == 1)
        {
            datatype = data.image[i].md[0].datatype;
            tmp_long = ((long long)(data.image[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            if(data.image[i].md[0].shared == 1)
            {
                printw("%4ldS", i);
            }
            else
            {
                printw("%4ld ", i);
            }

            if(data.image[i].md[0].shared == 1)
            {
                attron(A_BOLD | COLOR_PAIR(9));
            }
            else
            {
                attron(A_BOLD | COLOR_PAIR(6));
            }
            sprintf(str, "%10s ", data.image[i].name);
            printw(str);

            if(data.image[i].md[0].shared == 1)
            {
                attroff(A_BOLD | COLOR_PAIR(9));
            }
            else
            {
                attroff(A_BOLD | COLOR_PAIR(6));
            }

            sprintf(str, "[ %6ld", (long) data.image[i].md[0].size[0]);

            for(j = 1; j < data.image[i].md[0].naxis; j++)
            {
                sprintf(str1, "%s x %6ld", str, (long) data.image[i].md[0].size[j]);
            }
            sprintf(str2, "%s]", str1);

            printw("%-28s", str2);

            attron(COLOR_PAIR(3));
            n = 0;

            if(datatype == _DATATYPE_UINT8)
            {
                n = snprintf(type, STYPESIZE, "UINT8  ");
            }
            if(datatype == _DATATYPE_INT8)
            {
                n = snprintf(type, STYPESIZE, "INT8   ");
            }
            if(datatype == _DATATYPE_UINT16)
            {
                n = snprintf(type, STYPESIZE, "UINT16 ");
            }
            if(datatype == _DATATYPE_INT16)
            {
                n = snprintf(type, STYPESIZE, "INT16  ");
            }
            if(datatype == _DATATYPE_UINT32)
            {
                n = snprintf(type, STYPESIZE, "UINT32 ");
            }
            if(datatype == _DATATYPE_INT32)
            {
                n = snprintf(type, STYPESIZE, "INT32  ");
            }
            if(datatype == _DATATYPE_UINT64)
            {
                n = snprintf(type, STYPESIZE, "UINT64 ");
            }
            if(datatype == _DATATYPE_INT64)
            {
                n = snprintf(type, STYPESIZE, "INT64  ");
            }
            if(datatype == _DATATYPE_FLOAT)
            {
                n = snprintf(type, STYPESIZE, "FLOAT  ");
            }
            if(datatype == _DATATYPE_DOUBLE)
            {
                n = snprintf(type, STYPESIZE, "DOUBLE ");
            }
            if(datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                n = snprintf(type, STYPESIZE, "CFLOAT ");
            }
            if(datatype == _DATATYPE_COMPLEX_DOUBLE)
            {
                n = snprintf(type, STYPESIZE, "CDOUBLE");
            }

            printw("%7s ", type);

            attroff(COLOR_PAIR(3));

            if(n >= STYPESIZE) {
                PRINT_ERROR("Attempted to write string buffer with too many characters");
			}

            printw("%10ld Kb %6.2f   ", (long)(tmp_long / 1024),
                   (float)(100.0 * tmp_long / sizeb));

            timediff = (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                       (1.0 * data.image[i].md[0].lastaccesstime.tv_sec + 0.000000001 *
                        data.image[i].md[0].lastaccesstime.tv_nsec);

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
    sizeb = compute_image_memory();

    if(sizeb > 1024 - 1)
    {
        sizeKb = sizeb / 1024;
        sizeb = sizeb - 1024 * sizeKb;
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

    sprintf(str, "%ld image(s)      ", compute_nb_image());
    if(sizeGb > 0)
    {
        sprintf(str1, "%s %ld GB", str, (long)(sizeGb));
        strcpy(str, str1);
    }

    if(sizeMb > 0)
    {
        sprintf(str1, "%s %ld MB", str, (long)(sizeMb));
        strcpy(str, str1);
    }

    if(sizeKb > 0)
    {
        sprintf(str1, "%s %ld KB", str, (long)(sizeKb));
        strcpy(str, str1);
    }

    if(sizeb > 0)
    {
        sprintf(str1, "%s %ld B", str, (long)(sizeb));
        strcpy(str, str1);
    }

    mvprintw(listim_scr_wrow - 1, 0, "%s\n", str);
    //  attroff(A_BOLD);

    refresh();


    return RETURN_SUCCESS;
}




void close_list_image_ID_ncurses(void)
{
    printf("Closing monitor cleanly\n");
    set_term(listim_scr);
    endwin();
    fclose(listim_scr_fpo);
    fclose(listim_scr_fpi);
    data.MEM_MONITOR = 0;
}







errno_t list_image_ID_ofp(
    FILE *fo
)
{
    long        i;
    long        j;
    long long   tmp_long;
    char        type[STYPESIZE];
    uint8_t     datatype;
    int         n;
    unsigned long long sizeb, sizeKb, sizeMb, sizeGb;
    char        str[500];
    char        str1[512];
    struct      timespec timenow;
    double      timediff;
    //struct mallinfo minfo;

    sizeb = compute_image_memory();
    //minfo = mallinfo();

    clock_gettime(CLOCK_REALTIME, &timenow);
    //fprintf(fo, "time:  %ld.%09ld\n", timenow.tv_sec % 60, timenow.tv_nsec);



    fprintf(fo, "\n");
    fprintf(fo,
            "INDEX    NAME         SIZE                    TYPE        SIZE  [percent]    LAST ACCESS\n");
    fprintf(fo, "\n");

    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            datatype = data.image[i].md[0].datatype;
            tmp_long = ((long long)(data.image[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            if(data.image[i].md[0].shared == 1)
            {
                fprintf(fo, "%4ld %c[%d;%dm%14s%c[%d;m ", i, (char) 27, 1, 34,
                        data.image[i].name, (char) 27, 0);
            }
            else
            {
                fprintf(fo, "%4ld %c[%d;%dm%14s%c[%d;m ", i, (char) 27, 1, 33,
                        data.image[i].name, (char) 27, 0);
            }
            //fprintf(fo, "%s", str);

            sprintf(str, "[ %6ld", (long) data.image[i].md[0].size[0]);

            for(j = 1; j < data.image[i].md[0].naxis; j++)
            {
                sprintf(str1, "%s x %6ld", str, (long) data.image[i].md[0].size[j]);
                strcpy(str, str1);
            }
            sprintf(str1, "%s]", str);
            strcpy(str, str1);

            fprintf(fo, "%-32s", str);


            n = 0;
            if(datatype == _DATATYPE_UINT8)
            {
                n = snprintf(type, STYPESIZE, "UINT8  ");
            }
            if(datatype == _DATATYPE_INT8)
            {
                n = snprintf(type, STYPESIZE, "INT8   ");
            }
            if(datatype == _DATATYPE_UINT16)
            {
                n = snprintf(type, STYPESIZE, "UINT16 ");
            }
            if(datatype == _DATATYPE_INT16)
            {
                n = snprintf(type, STYPESIZE, "INT16  ");
            }
            if(datatype == _DATATYPE_UINT32)
            {
                n = snprintf(type, STYPESIZE, "UINT32 ");
            }
            if(datatype == _DATATYPE_INT32)
            {
                n = snprintf(type, STYPESIZE, "INT32  ");
            }
            if(datatype == _DATATYPE_UINT64)
            {
                n = snprintf(type, STYPESIZE, "UINT64 ");
            }
            if(datatype == _DATATYPE_INT64)
            {
                n = snprintf(type, STYPESIZE, "INT64  ");
            }
            if(datatype == _DATATYPE_FLOAT)
            {
                n = snprintf(type, STYPESIZE, "FLOAT  ");
            }
            if(datatype == _DATATYPE_DOUBLE)
            {
                n = snprintf(type, STYPESIZE, "DOUBLE ");
            }
            if(datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                n = snprintf(type, STYPESIZE, "CFLOAT ");
            }
            if(datatype == _DATATYPE_COMPLEX_DOUBLE)
            {
                n = snprintf(type, STYPESIZE, "CDOUBLE");
            }

            fprintf(fo, "%7s ", type);


            if(n >= STYPESIZE)
            {
                PRINT_ERROR("Attempted to write string buffer with too many characters");
            }

            fprintf(fo, "%10ld Kb %6.2f   ", (long)(tmp_long / 1024),
                    (float)(100.0 * tmp_long / sizeb));

            timediff = (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                       (1.0 * data.image[i].md[0].lastaccesstime.tv_sec + 0.000000001 *
                        data.image[i].md[0].lastaccesstime.tv_nsec);

            fprintf(fo, "%15.9f\n", timediff);
        }
    fprintf(fo, "\n");


    sizeGb = 0;
    sizeMb = 0;
    sizeKb = 0;
    sizeb = compute_image_memory();

    if(sizeb > 1024 - 1)
    {
        sizeKb = sizeb / 1024;
        sizeb = sizeb - 1024 * sizeKb;
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




errno_t list_image_ID_ofp_simple(
    FILE *fo
)
{
    long        i, j;
    //long long   tmp_long;
    uint8_t     datatype;

    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            datatype = data.image[i].md[0].datatype;
            //tmp_long = ((long long) (data.image[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            fprintf(fo, "%20s %d %ld %d %4ld", data.image[i].name, datatype,
                    (long) data.image[i].md[0].naxis, data.image[i].md[0].shared,
                    (long) data.image[i].md[0].size[0]);

            for(j = 1; j < data.image[i].md[0].naxis; j++)
            {
                fprintf(fo, " %4ld", (long) data.image[i].md[0].size[j]);
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

errno_t list_image_ID_file(
    const char *fname
)
{
    FILE *fp;
    long i, j;
    uint8_t datatype;
    char type[STYPESIZE];
    int n;

    fp = fopen(fname, "w");
    if(fp == NULL)
    {
        PRINT_ERROR("Cannot create file %s", fname);
        abort();
    }

    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            datatype = data.image[i].md[0].datatype;
            fprintf(fp, "%ld %s", i, data.image[i].name);
            fprintf(fp, " %ld", (long) data.image[i].md[0].naxis);
            for(j = 0; j < data.image[i].md[0].naxis; j++)
            {
                fprintf(fp, " %ld", (long) data.image[i].md[0].size[j]);
            }

            n = 0;

            if(datatype == _DATATYPE_UINT8)
            {
                n = snprintf(type, STYPESIZE, "UINT8  ");
            }
            if(datatype == _DATATYPE_INT8)
            {
                n = snprintf(type, STYPESIZE, "INT8   ");
            }
            if(datatype == _DATATYPE_UINT16)
            {
                n = snprintf(type, STYPESIZE, "UINT16 ");
            }
            if(datatype == _DATATYPE_INT16)
            {
                n = snprintf(type, STYPESIZE, "INT16  ");
            }
            if(datatype == _DATATYPE_UINT32)
            {
                n = snprintf(type, STYPESIZE, "UINT32 ");
            }
            if(datatype == _DATATYPE_INT32)
            {
                n = snprintf(type, STYPESIZE, "INT32  ");
            }
            if(datatype == _DATATYPE_UINT64)
            {
                n = snprintf(type, STYPESIZE, "UINT64 ");
            }
            if(datatype == _DATATYPE_INT64)
            {
                n = snprintf(type, STYPESIZE, "INT64  ");
            }
            if(datatype == _DATATYPE_FLOAT)
            {
                n = snprintf(type, STYPESIZE, "FLOAT  ");
            }
            if(datatype == _DATATYPE_DOUBLE)
            {
                n = snprintf(type, STYPESIZE, "DOUBLE ");
            }
            if(datatype == _DATATYPE_COMPLEX_FLOAT)
            {
                n = snprintf(type, STYPESIZE, "CFLOAT ");
            }
            if(datatype == _DATATYPE_COMPLEX_DOUBLE)
            {
                n = snprintf(type, STYPESIZE, "CDOUBLE");
            }


            if(n >= STYPESIZE)
            {
                PRINT_ERROR("Attempted to write string buffer with too many characters");
            }

            fprintf(fp, " %s\n", type);
        }
    fclose(fp);

    return RETURN_SUCCESS;
}





errno_t memory_monitor(
    const char *termttyname
)
{
    if(data.Debug > 0)
    {
        printf("starting memory_monitor on \"%s\"\n", termttyname);
    }

    data.MEM_MONITOR = 1;
    init_list_image_ID_ncurses(termttyname);
    list_image_ID_ncurses();
    atexit(close_list_image_ID_ncurses);

    return RETURN_SUCCESS;
}




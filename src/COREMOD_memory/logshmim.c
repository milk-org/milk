/**
 * @file    logshmim.c
 * @brief   Save telemetry stream data
 */


#define _GNU_SOURCE

#include <sched.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>


#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"


#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_memory/image_keyword_addS.h"
#include "COREMOD_memory/image_keyword_addD.h"

#include "image_ID.h"
#include "list_image.h"
#include "create_image.h"
#include "delete_image.h"
#include "read_shmim.h"
#include "stream_sem.h"

#include "shmimlog_types.h"




#define likely(x)	__builtin_expect(!!(x), 1)
#define unlikely(x)	__builtin_expect(!!(x), 0)


static long tret; // thread return value









// ==========================================
// Forward declaration(s)
// ==========================================

errno_t COREMOD_MEMORY_logshim_printstatus(
    const char *IDname
);

errno_t COREMOD_MEMORY_logshim_set_on(
    const char *IDname,
    int         setv
);

errno_t COREMOD_MEMORY_logshim_set_logexit(
    const char *IDname,
    int setv
);

errno_t COREMOD_MEMORY_sharedMem_2Dim_log(
    const char  *IDname,
    uint32_t     zsize,
    const char  *logdir,
    const char  *IDlogdata_name
);



// ==========================================
// Command line interface wrapper function(s)
// ==========================================


static errno_t COREMOD_MEMORY_logshim_printstatus__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            == 0)
    {
        COREMOD_MEMORY_logshim_printstatus(
            data.cmdargtoken[1].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_logshim_set_on__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        printf("logshim_set_on ----------------------\n");
        COREMOD_MEMORY_logshim_set_on(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_logshim_set_logexit__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_logshim_set_logexit(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_sharedMem_2Dim_log__cli()
{

    if(CLI_checkarg_noerrmsg(4, CLIARG_STR_NOT_IMG) != 0)
    {
        sprintf(data.cmdargtoken[4].val.string, "null");
    }

    if(0
            + CLI_checkarg(1, 3)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, 3)
            == 0)
    {
        COREMOD_MEMORY_sharedMem_2Dim_log(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.string,
            data.cmdargtoken[4].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}






// ==========================================
// Register CLI command(s)
// ==========================================

errno_t logshmim_addCLIcmd()
{

    RegisterCLIcommand(
        "shmimstreamlog",
        __FILE__,
        COREMOD_MEMORY_sharedMem_2Dim_log__cli,
        "logs shared memory stream (run in current directory)",
        "<shm image> <cubesize [long]> <logdir>",
        "shmimstreamlog wfscamim 10000 /media/data \"\"",
        "long COREMOD_MEMORY_sharedMem_2Dim_log(const char *IDname, uint32_t zsize, const char *logdir, const char *IDlogdata_name");

    RegisterCLIcommand(
        "shmimslogstat",
        __FILE__,
        COREMOD_MEMORY_logshim_printstatus__cli,
        "print log shared memory stream status",
        "<shm image>", "shmimslogstat wfscamim",
        "int COREMOD_MEMORY_logshim_printstatus(const char *IDname)");

    RegisterCLIcommand(
        "shmimslogonset", __FILE__,
        COREMOD_MEMORY_logshim_set_on__cli,
        "set on variable in log shared memory stream",
        "<shm image> <setv [long]>",
        "shmimslogonset imwfs 1",
        "int COREMOD_MEMORY_logshim_set_on(const char *IDname, int setv)");

    RegisterCLIcommand(
        "shmimslogexitset",
        __FILE__,
        COREMOD_MEMORY_logshim_set_logexit__cli,
        "set exit variable in log shared memory stream",
        "<shm image> <setv [long]>",
        "shmimslogexitset imwfs 1",
        "int COREMOD_MEMORY_logshim_set_logexit(const char *IDname, int setv)");


    return RETURN_SUCCESS;
}


















/**
 * ## Purpose
 *
 * Save telemetry stream data
 *
 */
void *save_fits_function(
    void *ptr
)
{
    imageID  ID;


    //struct savethreadmsg *tmsg; // = malloc(sizeof(struct savethreadmsg));
    STREAMSAVE_THREAD_MESSAGE *tmsg;

    uint32_t     *imsizearray;
    uint32_t      xsize, ysize;
    uint8_t       datatype;


    imageID       IDc;
    long          framesize;  // in bytes
    char         *ptr0;       // source pointer
    char         *ptr1;       // destination pointer
    long          k;
    FILE         *fp;


    // Set save function to RT priority 0
    // This is meant to be lower priority than the data collection into buffers
    //
    int RT_priority = 0;
    struct sched_param schedpar;

    schedpar.sched_priority = RT_priority;
    if(seteuid(data.euid) != 0)     //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if(seteuid(data.ruid) != 0)     //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }




    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(imsizearray == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    //    tmsg = (struct savethreadmsg*) ptr;
    tmsg = (STREAMSAVE_THREAD_MESSAGE *) ptr;


    // Add custom keywords
    int NBcustomKW = 9;
    IMAGE_KEYWORD *imkwarray = (IMAGE_KEYWORD *) malloc(sizeof(
                                   IMAGE_KEYWORD) * NBcustomKW);





    // UT time

    strcpy(imkwarray[0].name, "UT");
    imkwarray[0].type = 'S';
    strcpy(imkwarray[0].value.valstr,
           timedouble_to_UTC_timeofdaystring(0.5 * tmsg->arraytime[0] + 0.5 *
                   tmsg->arraytime[tmsg->cubesize - 1]));
    strcpy(imkwarray[0].comment, "HH:MM:SS.SS typical UTC at exposure");

    strcpy(imkwarray[1].name, "UT-STR");
    imkwarray[1].type = 'S';
    strcpy(imkwarray[1].value.valstr,
           timedouble_to_UTC_timeofdaystring(tmsg->arraytime[0]));
    strcpy(imkwarray[1].comment, "HH:MM:SS.SS UTC at exposure start");

    strcpy(imkwarray[2].name, "UT-END");
    imkwarray[2].type = 'S';
    strcpy(imkwarray[2].value.valstr,
           timedouble_to_UTC_timeofdaystring(tmsg->arraytime[tmsg->cubesize - 1]));
    strcpy(imkwarray[2].comment, "HH:MM:SS.SS UTC at exposure start");



    // Modified Julian Date (MJD)

    strcpy(imkwarray[3].name, "MJD");
    imkwarray[3].type = 'D';
    imkwarray[3].value.numf = (0.5 * tmsg->arraytime[0] + 0.5 *
                               tmsg->arraytime[tmsg->cubesize - 1]) / 86400.0 + 40587.0;
    strcpy(imkwarray[3].comment, "Modified Julian Day at exposure");

    strcpy(imkwarray[4].name, "MJD-STR");
    imkwarray[4].type = 'D';
    imkwarray[4].value.numf = tmsg->arraytime[0] / 86400.0 + 40587.0;
    strcpy(imkwarray[4].comment, "Modified Julian Day at exposure start");

    strcpy(imkwarray[5].name, "MJD-END");
    imkwarray[5].type = 'D';
    imkwarray[5].value.numf = (tmsg->arraytime[tmsg->cubesize - 1] / 86400.0) +
                              40587.0;
    strcpy(imkwarray[5].comment, "Modified Julian Day at exposure start");


    // Local time

    // get time zone
    time_t t = time(NULL);
    struct tm lt = {0};
    localtime_r(&t, &lt);
    //printf("Offset to GMT is %lds.\n", lt.tm_gmtoff);
    //printf("The time zone is '%s'.\n", lt.tm_zone);

    sprintf(imkwarray[6].name, "%s", lt.tm_zone);
    imkwarray[6].type = 'S';
    strcpy(imkwarray[6].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               (0.5 * tmsg->arraytime[0] + 0.5 * tmsg->arraytime[tmsg->cubesize - 1]) +
               lt.tm_gmtoff
           ));
    sprintf(imkwarray[6].comment, "HH:MM:SS.SS typical %s at exposure", lt.tm_zone);

    sprintf(imkwarray[7].name, "%s-STR", lt.tm_zone);
    imkwarray[7].type = 'S';
    strcpy(imkwarray[7].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               tmsg->arraytime[0] + lt.tm_gmtoff
           ));
    sprintf(imkwarray[7].comment, "HH:MM:SS.SS typical %s at exposure", lt.tm_zone);

    sprintf(imkwarray[8].name, "%s-END", lt.tm_zone);
    imkwarray[8].type = 'S';
    strcpy(imkwarray[8].value.valstr,
           timedouble_to_UTC_timeofdaystring(
               tmsg->arraytime[tmsg->cubesize - 1] + lt.tm_gmtoff
           ));
    sprintf(imkwarray[8].comment, "HH:MM:SS.SS typical %s at exposure", lt.tm_zone);





    if(tmsg->partial == 0) // full image
    {
        printf("auxFITSheader = \"%s\"\n", tmsg->fname_auxFITSheader);

        saveFITS(
            tmsg->iname,
            tmsg->fname,
            0,
            tmsg->fname_auxFITSheader,
            imkwarray,
            NBcustomKW
        );
    }
    else
    {
        ID = image_ID(tmsg->iname);
        datatype = data.image[ID].md[0].datatype;
        xsize = data.image[ID].md[0].size[0];
        ysize = data.image[ID].md[0].size[1];


        imsizearray[0] = xsize;
        imsizearray[1] = ysize;
        imsizearray[2] = tmsg->cubesize;


        create_image_ID("tmpsavecube", 3, imsizearray, datatype, 0, 10, 0, &IDc);


        switch(datatype)
        {

            case _DATATYPE_UINT8:
                framesize = SIZEOF_DATATYPE_UINT8 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.UI8; // source
                ptr1 = (char *) data.image[IDc].array.UI8; // destination
                break;
            case _DATATYPE_INT8:
                framesize = SIZEOF_DATATYPE_INT8 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.SI8; // source
                ptr1 = (char *) data.image[IDc].array.SI8; // destination
                break;

            case _DATATYPE_UINT16:
                framesize = SIZEOF_DATATYPE_UINT16 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.UI16; // source
                ptr1 = (char *) data.image[IDc].array.UI16; // destination
                break;
            case _DATATYPE_INT16:
                framesize = SIZEOF_DATATYPE_INT16 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.SI16; // source
                ptr1 = (char *) data.image[IDc].array.SI16; // destination
                break;

            case _DATATYPE_UINT32:
                framesize = SIZEOF_DATATYPE_UINT32 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.UI32; // source
                ptr1 = (char *) data.image[IDc].array.UI32; // destination
                break;
            case _DATATYPE_INT32:
                framesize = SIZEOF_DATATYPE_INT32 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.SI32; // source
                ptr1 = (char *) data.image[IDc].array.SI32; // destination
                break;

            case _DATATYPE_UINT64:
                framesize = SIZEOF_DATATYPE_UINT64 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.UI64; // source
                ptr1 = (char *) data.image[IDc].array.UI64; // destination
                break;
            case _DATATYPE_INT64:
                framesize = SIZEOF_DATATYPE_INT64 * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.SI64; // source
                ptr1 = (char *) data.image[IDc].array.SI64; // destination
                break;

            case _DATATYPE_FLOAT:
                framesize = SIZEOF_DATATYPE_FLOAT * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.F; // source
                ptr1 = (char *) data.image[IDc].array.F; // destination
                break;
            case _DATATYPE_DOUBLE:
                framesize = SIZEOF_DATATYPE_DOUBLE * xsize * ysize;
                ptr0 = (char *) data.image[ID].array.D; // source
                ptr1 = (char *) data.image[IDc].array.D; // destination
                break;

            default:
                printf("ERROR: WRONG DATA TYPE\n");
                free(imsizearray);
                free(tmsg);
                exit(0);
                break;
        }


        memcpy((void *) ptr1, (void *) ptr0, framesize * tmsg->cubesize);

        printf("auxFITSheader = \"%s\"\n", tmsg->fname_auxFITSheader);
        saveFITS("tmpsavecube", tmsg->fname, 0, tmsg->fname_auxFITSheader, imkwarray,
                 NBcustomKW);


        delete_image_ID("tmpsavecube", DELETE_IMAGE_ERRMODE_WARNING);
    }

    free(imkwarray);

    if(tmsg->saveascii == 1)
    {
        if((fp = fopen(tmsg->fnameascii, "w")) == NULL)
        {
            printf("ERROR: cannot create file \"%s\"\n", tmsg->fnameascii);
            exit(0);
        }

        fprintf(fp, "# Telemetry stream timing data \n");
        fprintf(fp, "# File written by function %s in file %s\n", __FUNCTION__,
                __FILE__);
        fprintf(fp, "# \n");
        fprintf(fp, "# col1 : datacube frame index\n");
        fprintf(fp, "# col2 : Main index\n");
        fprintf(fp, "# col3 : Time since cube origin\n");
        fprintf(fp, "# col4 : Absolute time\n");
        fprintf(fp, "# col5 : stream cnt0 index\n");
        fprintf(fp, "# col6 : stream cnt1 index\n");
        fprintf(fp, "# \n");

        double t0; // time reference
        t0 = tmsg->arraytime[0];
        for(k = 0; k < tmsg->cubesize; k++)
        {
            //fprintf(fp, "%6ld   %10lu  %10lu   %15.9lf\n", k, tmsg->arraycnt0[k], tmsg->arraycnt1[k], tmsg->arraytime[k]);

            // entries are:
            // - index within cube
            // - loop index (if applicable)
            // - time since cube start
            // - time (absolute)
            // - cnt0
            // - cnt1

            fprintf(fp, "%10ld  %10lu  %15.9lf   %20.9lf  %10ld   %10ld\n", k,
                    tmsg->arrayindex[k], tmsg->arraytime[k] - t0, tmsg->arraytime[k],
                    tmsg->arraycnt0[k], tmsg->arraycnt1[k]);
        }
        fclose(fp);
    }


    ID = image_ID(tmsg->iname);
    tret = ID;
    free(imsizearray);
    pthread_exit(&tret);
}



/** @brief creates logshimconf shared memory and loads it
 *
 */
LOGSHIM_CONF *COREMOD_MEMORY_logshim_create_SHMconf(
    const char *logshimname
)
{
    int             SM_fd;
    size_t          sharedsize = 0; // shared memory size in bytes
    char            SM_fname[STRINGMAXLEN_FULLFILENAME];
    int             result;
    LOGSHIM_CONF   *map;

    sharedsize = sizeof(LOGSHIM_CONF);

    WRITE_FULLFILENAME(SM_fname, "%s/%s.logshimconf.shm", data.shmdir, logshimname);

    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if(SM_fd == -1)
    {
        printf("File \"%s\"\n", SM_fname);
        fflush(stdout);
        perror("Error opening file for writing");
        exit(0);
    }

    result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
    if(result == -1)
    {
        close(SM_fd);
        PRINT_ERROR("Error calling lseek() to 'stretch' the file");
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if(result != 1)
    {
        close(SM_fd);
        perror("Error writing last byte of the file");
        exit(0);
    }

    map = (LOGSHIM_CONF *) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED,
                                SM_fd, 0);
    if(map == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        exit(0);
    }

    map[0].on = 0;
    map[0].cnt = 0;
    map[0].filecnt = 0;
    map[0].interval = 1;
    map[0].logexit = 0;
    strcpy(map[0].fname, SM_fname);

    return map;
}






// IDname is name of image logged
errno_t COREMOD_MEMORY_logshim_printstatus(
    const char *IDname
)
{
    LOGSHIM_CONF *map;
    char          SM_fname[STRINGMAXLEN_FULLFILENAME];
    int           SM_fd;
    struct        stat file_stat;

    // read shared mem
    WRITE_FULLFILENAME(SM_fname, "%s/%s.logshimconf.shm", data.shmdir, IDname);
    printf("Importing mmap file \"%s\"\n", SM_fname);

    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd == -1)
    {
        printf("Cannot import file - continuing\n");
        exit(0);
    }
    else
    {
        fstat(SM_fd, &file_stat);
        printf("File %s size: %zd\n", SM_fname, file_stat.st_size);

        map = (LOGSHIM_CONF *) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, SM_fd, 0);
        if(map == MAP_FAILED)
        {
            close(SM_fd);
            perror("Error mmapping the file");
            exit(0);
        }

        printf("LOG   on = %d\n", map[0].on);
        printf("    cnt  = %lld\n", map[0].cnt);
        printf(" filecnt = %lld\n", map[0].filecnt);
        printf("interval = %ld\n", map[0].interval);
        printf("logexit  = %d\n", map[0].logexit);

        if(munmap(map, sizeof(LOGSHIM_CONF)) == -1)
        {
            printf("unmapping %s\n", SM_fname);
            perror("Error un-mmapping the file");
        }
        close(SM_fd);
    }
    return RETURN_SUCCESS;
}






// set the on field in logshim
// IDname is name of image logged
errno_t COREMOD_MEMORY_logshim_set_on(
    const char *IDname,
    int         setv
)
{
    LOGSHIM_CONF  *map;
    char           SM_fname[STRINGMAXLEN_FULLFILENAME];
    int            SM_fd;
    struct stat    file_stat;

    // read shared mem
    WRITE_FULLFILENAME(SM_fname, "%s/%s.logshimconf.shm", data.shmdir, IDname);
    printf("Importing mmap file \"%s\"\n", SM_fname);

    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd == -1)
    {
        printf("Cannot import file - continuing\n");
        exit(0);
    }
    else
    {
        fstat(SM_fd, &file_stat);
        printf("File %s size: %zd\n", SM_fname, file_stat.st_size);

        map = (LOGSHIM_CONF *) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, SM_fd, 0);
        if(map == MAP_FAILED)
        {
            close(SM_fd);
            perror("Error mmapping the file");
            exit(0);
        }

        map[0].on = setv;

        if(munmap(map, sizeof(LOGSHIM_CONF)) == -1)
        {
            printf("unmapping %s\n", SM_fname);
            perror("Error un-mmapping the file");
        }
        close(SM_fd);
    }
    return RETURN_SUCCESS;
}





// set the on field in logshim
// IDname is name of image logged
errno_t COREMOD_MEMORY_logshim_set_logexit(
    const char *IDname,
    int         setv
)
{
    LOGSHIM_CONF  *map;
    char           SM_fname[STRINGMAXLEN_FULLFILENAME];
    int            SM_fd;
    struct stat    file_stat;

    // read shared mem
    WRITE_FULLFILENAME(SM_fname, "%s/%s.logshimconf.shm", data.shmdir, IDname);
    printf("Importing mmap file \"%s\"\n", SM_fname);

    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd == -1)
    {
        printf("Cannot import file - continuing\n");
        exit(0);
    }
    else
    {
        fstat(SM_fd, &file_stat);
        printf("File %s size: %zd\n", SM_fname, file_stat.st_size);

        map = (LOGSHIM_CONF *) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, SM_fd, 0);
        if(map == MAP_FAILED)
        {
            close(SM_fd);
            perror("Error mmapping the file");
            exit(0);
        }

        map[0].logexit = setv;

        if(munmap(map, sizeof(LOGSHIM_CONF)) == -1)
        {
            printf("unmapping %s\n", SM_fname);
            perror("Error un-mmapping the file");
        }
        close(SM_fd);
    }
    return RETURN_SUCCESS;
}






/** @brief Logs a shared memory stream onto disk
 *
 * uses semlog semaphore
 *
 * uses data cube buffer to store frames
 * if an image name logdata exists (should ideally be in shared mem),
 * then this will be included in the timing txt file
 */
errno_t __attribute__((hot)) COREMOD_MEMORY_sharedMem_2Dim_log(
    const char *IDname,
    uint32_t    zsize,
    const char *logdir,
    const char *IDlogdata_name
)
{
    // WAIT time. If no new frame during this time, save existing cube
    int        WaitSec = 5;

    imageID    ID;
    uint32_t   xsize;
    uint32_t   ysize;
    imageID    IDb;
    imageID    IDb0;
    imageID    IDb1;
    long       index = 0;
    unsigned long long       cnt = 0;
    int        buffer;
    uint8_t    datatype;
    uint32_t  *imsizearray;
    char       fname[STRINGMAXLEN_FILENAME];
    char       iname[STRINGMAXLEN_IMGNAME];

    time_t          t;
    struct tm      *uttimeStart;
    struct timespec ts;
    struct timespec timenow;
    struct timespec timenowStart;
    int             ret;
    imageID         IDlogdata;

    char *ptr0_0; // source image data
    char *ptr1_0; // destination image data
    char *ptr0; // source image data, after offset
    char *ptr1; // destination image data, after offset

    long framesize; // in bytes

    char fnameascii[200];

    pthread_t  thread_savefits;
    int        tOK = 0;
    int        iret_savefits;
    STREAMSAVE_THREAD_MESSAGE *tmsg = malloc(sizeof(STREAMSAVE_THREAD_MESSAGE));


    long NBfiles = -1; // run forever

    long long cntwait;
    long waitdelayus = 50;  // max speed = 20 kHz
    long long cntwaitlim = 10000; // 5 sec
    int wOK;
    int noframe;


    char logb0name[500];
    char logb1name[500];

    int is3Dcube = 0; // this is a rolling buffer

    LOGSHIM_CONF *logshimconf;

    // recording time for each frame
    double *array_time;
    double *array_time_cp;

    // counters
    uint64_t *array_cnt0;
    uint64_t *array_cnt0_cp;
    uint64_t *array_cnt1;
    uint64_t *array_cnt1_cp;

    int RT_priority = 80; //any number from 0-99
    struct sched_param schedpar;

    int use_semlog;
    int semval;


    int VERBOSE = 0;
    // 0: don't print
    // 1: print statements outside fast loop
    // 2: print everything

    // convert wait time into number of couunter steps (counter mode only)
    cntwaitlim = (long)(WaitSec * 1000000 / waitdelayus);



    schedpar.sched_priority = RT_priority;

    if(seteuid(data.euid) != 0)     //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if(seteuid(data.ruid) != 0)     //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }



    IDlogdata = image_ID(IDlogdata_name);
    if(IDlogdata != -1)
    {
        if(data.image[IDlogdata].md[0].datatype != _DATATYPE_FLOAT)
        {
            IDlogdata = -1;
        }
    }
    printf("log data name = %s\n", IDlogdata_name);


    logshimconf = COREMOD_MEMORY_logshim_create_SHMconf(IDname);


    logshimconf[0].on = 1;
    logshimconf[0].cnt = 0;
    logshimconf[0].filecnt = 0;
    logshimconf[0].logexit = 0;
    logshimconf[0].interval = 1;



    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 3);



    read_sharedmem_image(IDname);
    ID = image_ID(IDname);
    datatype = data.image[ID].md[0].datatype;
    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];

    if(data.image[ID].md[0].naxis == 3)
    {
        is3Dcube = 1;
    }

    /** create the 2 buffers */

    imsizearray[0] = xsize;
    imsizearray[1] = ysize;
    imsizearray[2] = zsize;

    sprintf(logb0name, "%s_logbuff0", IDname);
    sprintf(logb1name, "%s_logbuff1", IDname);


    create_image_ID(logb0name, 3, imsizearray, datatype, 1,
                    data.image[ID].md[0].NBkw, 0, &IDb0);
    create_image_ID(logb1name, 3, imsizearray, datatype, 1,
                    data.image[ID].md[0].NBkw, 0, &IDb1);

    // copy keywords
    {
        memcpy(data.image[IDb0].kw, data.image[ID].kw,
               sizeof(IMAGE_KEYWORD)*data.image[ID].md[0].NBkw);
        memcpy(data.image[IDb1].kw, data.image[ID].kw,
               sizeof(IMAGE_KEYWORD)*data.image[ID].md[0].NBkw);
    }


    COREMOD_MEMORY_image_set_semflush(logb0name, -1);
    COREMOD_MEMORY_image_set_semflush(logb1name, -1);


    array_time = (double *) malloc(sizeof(double) * zsize);
    array_cnt0 = (uint64_t *) malloc(sizeof(uint64_t) * zsize);
    array_cnt1 = (uint64_t *) malloc(sizeof(uint64_t) * zsize);

    array_time_cp = (double *) malloc(sizeof(double) * zsize);
    array_cnt0_cp = (uint64_t *) malloc(sizeof(uint64_t) * zsize);
    array_cnt1_cp = (uint64_t *) malloc(sizeof(uint64_t) * zsize);


    IDb = IDb0;

    switch(datatype)
    {

        case _DATATYPE_FLOAT:
            framesize = SIZEOF_DATATYPE_FLOAT * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.F;
            break;

        case _DATATYPE_INT8:
            framesize = SIZEOF_DATATYPE_INT8 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.SI8;
            break;

        case _DATATYPE_UINT8:
            framesize = SIZEOF_DATATYPE_UINT8 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.UI8;
            break;

        case _DATATYPE_INT16:
            framesize = SIZEOF_DATATYPE_INT16 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.SI16;
            break;

        case _DATATYPE_UINT16:
            framesize = SIZEOF_DATATYPE_UINT16 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.UI16;
            break;

        case _DATATYPE_INT32:
            framesize = SIZEOF_DATATYPE_INT32 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.SI32;
            break;

        case _DATATYPE_UINT32:
            framesize = SIZEOF_DATATYPE_UINT32 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.UI32;
            break;

        case _DATATYPE_INT64:
            framesize = SIZEOF_DATATYPE_INT64 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.SI64;
            break;

        case _DATATYPE_UINT64:
            framesize = SIZEOF_DATATYPE_UINT64 * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.UI64;
            break;


        case _DATATYPE_DOUBLE:
            framesize = SIZEOF_DATATYPE_DOUBLE * xsize * ysize;
            ptr0_0 = (char *) data.image[ID].array.D;
            break;

        default:
            printf("ERROR: WRONG DATA TYPE\n");
            exit(0);
            break;
    }



    switch(datatype)
    {

        case _DATATYPE_FLOAT:
            ptr1_0 = (char *) data.image[IDb].array.F;
            break;

        case _DATATYPE_INT8:
            ptr1_0 = (char *) data.image[IDb].array.SI8;
            break;

        case _DATATYPE_UINT8:
            ptr1_0 = (char *) data.image[IDb].array.UI8;
            break;

        case _DATATYPE_INT16:
            ptr1_0 = (char *) data.image[IDb].array.SI16;
            break;

        case _DATATYPE_UINT16:
            ptr1_0 = (char *) data.image[IDb].array.UI16;
            break;

        case _DATATYPE_INT32:
            ptr1_0 = (char *) data.image[IDb].array.SI32;
            break;

        case _DATATYPE_UINT32:
            ptr1_0 = (char *) data.image[IDb].array.UI32;
            break;

        case _DATATYPE_INT64:
            ptr1_0 = (char *) data.image[IDb].array.SI64;
            break;

        case _DATATYPE_UINT64:
            ptr1_0 = (char *) data.image[IDb].array.UI64;
            break;

        case _DATATYPE_DOUBLE:
            ptr1_0 = (char *) data.image[IDb].array.D;
            break;

    }




    cnt = data.image[ID].md[0].cnt0 - 1;

    buffer = 0;
    index = 0;

    printf("logdata ID = %ld\n", IDlogdata);
    list_image_ID();



    // using semlog ?
    use_semlog = 0;
    if(data.image[ID].semlog != NULL)
    {
        use_semlog = 1;
        sem_getvalue(data.image[ID].semlog, &semval);

        // bring semaphore value to 1 to only save 1 frame
        while(semval > 1)
        {
            sem_wait(data.image[ID].semlog);
            sem_getvalue(data.image[ID].semlog, &semval);
        }
        if(semval == 0)
        {
            sem_post(data.image[ID].semlog);
        }
    }



    while((logshimconf[0].filecnt != NBfiles) && (logshimconf[0].logexit == 0))
    {
        int timeout; // 1 if timeout has occurred

        cntwait = 0;
        noframe = 0;
        wOK = 1;

        if(VERBOSE > 1)
        {
            printf("%5d  Entering wait loop   index = %ld %d\n", __LINE__, index, noframe);
        }

        timeout = 0;
        if(likely(use_semlog == 1))
        {
            if(VERBOSE > 1)
            {
                printf("%5d  Waiting for semaphore\n", __LINE__);
            }

            if(clock_gettime(CLOCK_REALTIME, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += WaitSec;

            ret = sem_timedwait(data.image[ID].semlog, &ts);
            if(ret == -1)
            {
                if(errno == ETIMEDOUT)
                {
                    printf("%5d  sem_timedwait() timed out (%d sec) -[index %ld]\n", __LINE__,
                           WaitSec, index);
                    if(VERBOSE > 0)
                    {
                        printf("%5d  sem time elapsed -> Save current cube [index %ld]\n", __LINE__,
                               index);
                    }

                    strcpy(tmsg->iname, iname);
                    strcpy(tmsg->fname, fname);
                    tmsg->partial = 1; // partial cube
                    tmsg->cubesize = index;

                    memcpy(array_time_cp, array_time, sizeof(double)*index);
                    memcpy(array_cnt0_cp, array_cnt0, sizeof(uint64_t)*index);
                    memcpy(array_cnt1_cp, array_cnt1, sizeof(uint64_t)*index);

                    tmsg->arrayindex = array_cnt0_cp;
                    tmsg->arraycnt0 = array_cnt0_cp;
                    tmsg->arraycnt1 = array_cnt1_cp;
                    tmsg->arraytime = array_time_cp;

                    timeout = 1;
                }
                if(errno == EINTR)
                {
                    printf("%5d  sem_timedwait [index %ld]: The call was interrupted by a signal handler\n",
                           __LINE__, index);
                }

                if(errno == EINVAL)
                {
                    printf("%5d  sem_timedwait [index %ld]: Not a valid semaphore\n", __LINE__,
                           index);
                    printf("               The value of abs_timeout.tv_nsecs is less than 0, or greater than or equal to 1000 million\n");
                }

                if(errno == EAGAIN)
                {
                    printf("%5d  sem_timedwait [index %ld]: The operation could not be performed without blocking (i.e., the semaphore currently has the value zero)\n",
                           __LINE__, index);
                }


                wOK = 0;
                if(index == 0)
                {
                    noframe = 1;
                }
                else
                {
                    noframe = 0;
                }
            }

        }
        else
        {
            if(VERBOSE > 1)
            {
                printf("%5d  Not using semaphore, watching counter\n", __LINE__);
            }

            while(((cnt == data.image[ID].md[0].cnt0) || (logshimconf[0].on == 0))
                    && (wOK == 1))
            {
                if(VERBOSE > 1)
                {
                    printf("%5d  waiting time step\n", __LINE__);
                }

                usleep(waitdelayus);
                cntwait++;

                if(VERBOSE > 1)
                {
                    printf("%5d  cntwait = %lld\n", __LINE__, cntwait);
                    fflush(stdout);
                }

                if(cntwait > cntwaitlim) // save current cube
                {
                    if(VERBOSE > 0)
                    {
                        printf("%5d  cnt time elapsed -> Save current cube\n", __LINE__);
                    }


                    strcpy(tmsg->iname, iname);
                    strcpy(tmsg->fname, fname);
                    tmsg->partial = 1; // partial cube
                    tmsg->cubesize = index;

                    memcpy(array_time_cp, array_time, sizeof(double)*index);
                    memcpy(array_cnt0_cp, array_cnt0, sizeof(uint64_t)*index);
                    memcpy(array_cnt1_cp, array_cnt1, sizeof(uint64_t)*index);

                    tmsg->arrayindex = array_cnt0_cp;
                    tmsg->arraycnt0 = array_cnt0_cp;
                    tmsg->arraycnt1 = array_cnt1_cp;
                    tmsg->arraytime = array_time_cp;

                    wOK = 0;
                    if(index == 0)
                    {
                        noframe = 1;
                    }
                    else
                    {
                        noframe = 0;
                    }
                }
            }
        }



        if(index == 0)
        {
            if(VERBOSE > 0)
            {
                printf("%5d  Setting cube start time [index %ld]\n", __LINE__, index);
            }

            /// measure time
            t = time(NULL);
            uttimeStart = gmtime(&t);
            clock_gettime(CLOCK_REALTIME, &timenowStart);

            //     sprintf(fname,"!%s/%s_%02d:%02d:%02ld.%09ld.fits", logdir, IDname, uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
            //            sprintf(fnameascii,"%s/%s_%02d:%02d:%02ld.%09ld.txt", logdir, IDname, uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec);
        }


        if(VERBOSE > 1)
        {
            printf("%5d  logshimconf[0].on = %d\n", __LINE__, logshimconf[0].on);
        }


        if(likely(logshimconf[0].on == 1))
        {
            if(likely(wOK == 1)) // normal step: a frame has arrived
            {
                if(VERBOSE > 1)
                {
                    printf("%5d  Frame has arrived [index %ld]\n", __LINE__, index);
                }

                /// measure time
                //   t = time(NULL);
                //   uttime = gmtime(&t);

                clock_gettime(CLOCK_REALTIME, &timenow);


                if(is3Dcube == 1)
                {
                    ptr0 = ptr0_0 + framesize * data.image[ID].md[0].cnt1;
                }
                else
                {
                    ptr0 = ptr0_0;
                }

                ptr1 = ptr1_0 + framesize * index;

                if(VERBOSE > 1)
                {
                    printf("%5d  memcpy framesize = %ld\n", __LINE__, framesize);
                }

                memcpy((void *) ptr1, (void *) ptr0, framesize);

                if(VERBOSE > 1)
                {
                    printf("%5d  memcpy done\n", __LINE__);
                }

                array_cnt0[index] = data.image[ID].md[0].cnt0;
                array_cnt1[index] = data.image[ID].md[0].cnt1;
                //array_time[index] = uttime->tm_hour*3600.0 + uttime->tm_min*60.0 + timenow.tv_sec % 60 + 1.0e-9*timenow.tv_nsec;
                array_time[index] = timenow.tv_sec + 1.0e-9 * timenow.tv_nsec;

                index++;
            }
        }
        else
        {
            // save partial if possible
            //if(index>0)
            wOK = 0;

        }


        if(VERBOSE > 1)
        {
            printf("%5d  index = %ld  wOK = %d\n", __LINE__, index, wOK);
        }


        // SAVE CUBE TO DISK
        /// cases:
        /// index>zsize-1  buffer full
        /// timeout==1 && index>0  : partial
        if((index > zsize - 1)  || ((timeout == 1) && (index > 0)))
        {
            long NBframemissing;

            /// save image
            if(VERBOSE > 0)
            {
                printf("%5d  Save image   [index  %ld]  [timeout %d] [zsize %ld]\n", __LINE__,
                       index, timeout, (long) zsize);
            }

            sprintf(iname, "%s_logbuff%d", IDname, buffer);
            if(buffer == 0)
            {
                IDb = IDb0;
            }
            else
            {
                IDb = IDb1;
            }

            // update buffer content
            memcpy(data.image[IDb].kw, data.image[ID].kw,
                   sizeof(IMAGE_KEYWORD)*data.image[ID].md[0].NBkw);


            if(VERBOSE > 0)
            {
                printf("%5d  Building file name: ascii\n", __LINE__);
                fflush(stdout);
            }

            sprintf(fnameascii, "%s/%s_%02d:%02d:%02ld.%09ld.txt", logdir, IDname,
                    uttimeStart->tm_hour, uttimeStart->tm_min, timenowStart.tv_sec % 60,
                    timenowStart.tv_nsec);


            if(VERBOSE > 0)
            {
                printf("%5d  Building file name: fits\n", __LINE__);
                fflush(stdout);
            }
            sprintf(fname, "%s/%s_%02d:%02d:%02ld.%09ld.fits", logdir, IDname,
                    uttimeStart->tm_hour, uttimeStart->tm_min, timenowStart.tv_sec % 60,
                    timenowStart.tv_nsec);



            strcpy(tmsg->iname, iname);
            strcpy(tmsg->fname, fname);
            strcpy(tmsg->fnameascii, fnameascii);
            tmsg->saveascii = 1;


            if(wOK == 1) // full cube
            {
                tmsg->partial = 0; // full cube
                if(VERBOSE > 0)
                {
                    printf("%5d  SAVING FULL CUBE\n", __LINE__);
                    fflush(stdout);
                }

            }
            else // partial cube
            {
                tmsg->partial = 1; // partial cube
                if(VERBOSE > 0)
                {
                    printf("%5d  SAVING PARTIAL CUBE\n", __LINE__);
                    fflush(stdout);
                }
            }

            {
                long cnt0start = data.image[ID].md[0].cnt0;

                // Wait for save thread to complete to launch next one
                if(tOK == 1)
                {
                    if(pthread_tryjoin_np(thread_savefits, NULL) == EBUSY)
                    {
                        if(VERBOSE > 0)
                        {
                            printf("%5d  PREVIOUS SAVE THREAD NOT TERMINATED -> waiting\n", __LINE__);
                        }
                        pthread_join(thread_savefits, NULL);
                        if(VERBOSE > 0)
                        {
                            printf("%5d  PREVIOUS SAVE THREAD NOW COMPLETED -> continuing\n", __LINE__);
                        }
                    }
                    else
                    {
                        if(VERBOSE > 0)
                        {
                            printf("%5d  PREVIOUS SAVE THREAD ALREADY COMPLETED -> OK\n", __LINE__);
                        }
                    }
                }

                printf("\n ************** MISSED = %ld\n",
                       data.image[ID].md[0].cnt0 - cnt0start);
            }


            COREMOD_MEMORY_image_set_sempost_byID(IDb, -1);
            data.image[IDb].md[0].cnt0++;
            data.image[IDb].md[0].write = 0;



            tmsg->cubesize = index;
            strcpy(tmsg->iname, iname);


            memcpy(array_time_cp, array_time, sizeof(double)*index);


            memcpy(array_cnt0_cp, array_cnt0, sizeof(uint64_t)*index);


            memcpy(array_cnt1_cp, array_cnt1, sizeof(uint64_t)*index);

            NBframemissing = (array_cnt0[index - 1] - array_cnt0[0]) - (index - 1);


            printf("=>=>=>=>= CUBE %8lld   Number of missed frames = %8ld  / %ld  / %8ld ====\n",
                   logshimconf[0].filecnt, NBframemissing, index, (long) zsize);

            if(VERBOSE > 0)
            {
                printf("%5d  Starting image save thread\n", __LINE__);
                fflush(stdout);
            }



            tmsg->arrayindex = array_cnt0_cp;
            tmsg->arraycnt0 = array_cnt0_cp;
            tmsg->arraycnt1 = array_cnt1_cp;
            tmsg->arraytime = array_time_cp;
            WRITE_FILENAME(tmsg->fname_auxFITSheader,
                           "%s/%s.auxFITSheader.shm",
                           data.shmdir,
                           IDname);


            iret_savefits = pthread_create(&thread_savefits, NULL, save_fits_function,
                                           tmsg);


            logshimconf[0].cnt ++;

            tOK = 1;
            if(iret_savefits)
            {
                fprintf(stderr, "Error - pthread_create() return code: %d\n", iret_savefits);
                exit(EXIT_FAILURE);
            }


            index = 0;
            buffer++;
            if(buffer == 2)
            {
                buffer = 0;
            }
            //            printf("[%ld -> %d]", cnt, buffer);
            //           fflush(stdout);
            if(buffer == 0)
            {
                IDb = IDb0;
            }
            else
            {
                IDb = IDb1;
            }

            switch(datatype)
            {

                case _DATATYPE_FLOAT:
                    ptr1_0 = (char *) data.image[IDb].array.F;
                    break;

                case _DATATYPE_INT8:
                    ptr1_0 = (char *) data.image[IDb].array.SI8;
                    break;

                case _DATATYPE_UINT8:
                    ptr1_0 = (char *) data.image[IDb].array.UI8;
                    break;

                case _DATATYPE_INT16:
                    ptr1_0 = (char *) data.image[IDb].array.SI16;
                    break;

                case _DATATYPE_UINT16:
                    ptr1_0 = (char *) data.image[IDb].array.UI16;
                    break;

                case _DATATYPE_INT32:
                    ptr1_0 = (char *) data.image[IDb].array.SI32;
                    break;

                case _DATATYPE_UINT32:
                    ptr1_0 = (char *) data.image[IDb].array.UI32;
                    break;

                case _DATATYPE_INT64:
                    ptr1_0 = (char *) data.image[IDb].array.SI64;
                    break;

                case _DATATYPE_UINT64:
                    ptr1_0 = (char *) data.image[IDb].array.UI64;
                    break;

                case _DATATYPE_DOUBLE:
                    ptr1_0 = (char *) data.image[IDb].array.D;
                    break;

            }

            data.image[IDb].md[0].write = 1;
            logshimconf[0].filecnt ++;
        }


        cnt = data.image[ID].md[0].cnt0;
    }

    free(imsizearray);
    free(tmsg);

    free(array_time);
    free(array_cnt0);
    free(array_cnt1);

    free(array_time_cp);
    free(array_cnt0_cp);
    free(array_cnt1_cp);

    return RETURN_SUCCESS;
}









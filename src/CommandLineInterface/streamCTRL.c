
/**
 * @file streamCTRL.c
 * @brief Data streams control panel
 * 
 * Manages data streams
 * 
 * 
 */




#define _GNU_SOURCE

#define STREAMCTRL_LOGDEBUG 1





/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/file.h>
#include <malloc.h>
#include <sys/mman.h> // mmap()

#include <time.h>
#include <signal.h>

#include <unistd.h>    // getpid() access()
#include <sys/types.h>

#include <sys/stat.h>
#include <sys/types.h>


#include <ncurses.h>
#include <fcntl.h> 
#include <ctype.h>

#include <dirent.h>

#include <wchar.h>
#include <locale.h>
#include <errno.h>

#include <pthread.h>

#ifdef STANDALONE
#include "standalone_dependencies.h"
#else
#include "CLIcore.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "COREMOD_memory/COREMOD_memory.h"
#define SHAREDSHMDIR    data.shmdir  /**< default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR */
#endif

#include "streamCTRL.h"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */


#define STRINGLENMAX  32

#define streamOpenNBpid_MAX 100
#define nameNBchar 100
#define PIDnameStringLen 12


#define DISPLAY_MODE_HELP    1
#define DISPLAY_MODE_SEMVAL  2
#define DISPLAY_MODE_WRITE   3
#define DISPLAY_MODE_READ    4
#define DISPLAY_MODE_SPTRACE 5
#define DISPLAY_MODE_FUSER   6

#define PRINT_PID_DEFAULT          0
#define PRINT_PID_FORCE_NOUPSTREAM 1

#define NO_DOWNSTREAM_INDEX    100

/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */


static int wrow, wcol;


/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */







/**
 * INITIALIZE ncurses
 *
 */
static int initncurses()
{
    if ( initscr() == NULL ) {
        fprintf(stderr, "Error initialising ncurses.\n");
        exit(EXIT_FAILURE);
    }
    getmaxyx(stdscr, wrow, wcol);		/* get the number of rows and columns */
    cbreak();
    keypad(stdscr, TRUE);		/* We get F1, F2 etc..		*/
    nodelay(stdscr, TRUE);
    curs_set(0);
    noecho();			/* Don't echo() while we do getch */
	nonl();             // Do not translates newline into return and line-feed on output

    init_color(COLOR_GREEN, 700, 1000, 700);
    init_color(COLOR_YELLOW, 1000, 1000, 700);
    start_color();

    //  colored background
    
    init_pair(  1, COLOR_BLACK,  COLOR_WHITE  );
    init_pair(  2, COLOR_BLACK,  COLOR_GREEN  );  // all good
    init_pair(  3, COLOR_BLACK,  COLOR_YELLOW );  // parameter out of sync
    init_pair(  4, COLOR_WHITE,  COLOR_RED    );
    init_pair(  5, COLOR_WHITE,  COLOR_BLUE   ); // DIRECTORY
    init_pair(  6, COLOR_GREEN,  COLOR_BLACK  );
    init_pair(  7, COLOR_YELLOW, COLOR_BLACK  );
    init_pair(  8, COLOR_RED,    COLOR_BLACK  );
    init_pair(  9, COLOR_BLACK,  COLOR_RED    );
    init_pair( 10, COLOR_BLACK,  COLOR_CYAN   );
    init_pair( 12, COLOR_GREEN,  COLOR_WHITE  ); // highlighted version of #2
    

    return 0;
}




static errno_t streamCTRL__print_header(const char *str, char c)
{
    long n;
    long i;

    attron(A_BOLD);
    n = strlen(str);
    for(i = 0; i < (wcol - n) / 2; i++)
    {
        printw("%c", c);
    }
    printw("%s", str);
    for(i = 0; i < (wcol - n) / 2 - 1; i++)
    {
        printw("%c", c);
    }
    printw("\n");
    attroff(A_BOLD);

    return RETURN_SUCCESS;
}







/*
 * returns ID number corresponding to a name
 */
imageID image_ID_from_images(
    IMAGE* images,
    const char* restrict name
)
{
    imageID i;

    i = 0;
    do {
        if(images[i].used == 1)
        {
            if((strncmp(name, images[i].name, strlen(name))==0)
                    && (images[i].name[strlen(name)]=='\0'))
            {
                clock_gettime(CLOCK_REALTIME, &images[i].md[0].lastaccesstime);
                return i;
            }
        }
        i++;
    } while(i != streamNBID_MAX);

    return -1;
}




imageID image_get_first_ID_available_from_images(
    IMAGE* images
)
{
    imageID i;

    i = 0;
    do {
        if(images[i].used == 0) {
            images[i].used = 1;
            return i;
        }
        i++;
    } while(i != streamNBID_MAX);
    printf("ERROR: ran out of image IDs - cannot allocate new ID\n");
    printf("NB_MAX_IMAGE should be increased above current value (%d)\n", streamNBID_MAX);
    return -1;
}



int get_process_name_by_pid(const int pid, char *pname)
{	
    char* fname = (char*) calloc(STRINGMAXLEN_FULLFILENAME, sizeof(char));

	WRITE_FULLFILENAME(fname, "/proc/%d/cmdline", pid);
//    sprintf(fname, "/proc/%d/cmdline", pid);
    FILE* fp = fopen(fname,"r");
    if(fp) {
        size_t size;
        size = fread(pname, sizeof(char), 1024, fp);
        if(size>0) {
            if('\n'==pname[size-1])
                pname[size-1]='\0';
        }
        fclose(fp);
    }
    
    free(fname);

    return 0;
}









int streamCTRL_CatchSignals()
{

#ifndef STANDALONE
    if (sigaction(SIGTERM, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGTERM\n");

    if (sigaction(SIGINT, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGINT\n");

    if (sigaction(SIGABRT, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGABRT\n");

    if (sigaction(SIGBUS, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGBUS\n");

    if (sigaction(SIGSEGV, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGSEGV\n");

    if (sigaction(SIGHUP, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGHUP\n");

    if (sigaction(SIGPIPE, &data.sigact, NULL) == -1)
        printf("\ncan't catch SIGPIPE\n");
#endif

    return 0;
}





static int get_PIDmax() {
    FILE *fp;
    int PIDmax;
    int fscanfcnt;

    fp = fopen("/proc/sys/kernel/pid_max", "r");



    fscanfcnt = fscanf(fp, "%d", &PIDmax);
    if(fscanfcnt == EOF) {
        if(ferror(fp)) {
            perror("fscanf");
        } else {
            fprintf(stderr, "Error: fscanf reached end of file, no matching characters, no matching failure\n");
        }
        exit(EXIT_FAILURE);
    } else if(fscanfcnt != 1) {
        fprintf(stderr, "Error: fscanf successfully matched and assigned %i input items, 1 expected\n", fscanfcnt);
        exit(EXIT_FAILURE);
    }


    fclose(fp);

    return PIDmax;
}






struct streamCTRLarg_struct {
    STREAMINFOPROC* streaminfoproc;
    IMAGE *images;
};




/** @brief find shared memory streams on system
 *
 * If filter is set to 1, require stream name to contain namefilter string
 * streaminfo needs to be pre-allocated
 *
 */

int find_streams(
    STREAMINFO *streaminfo,
    int filter,
    const char *namefilter
)
{
    int NBstream = 0;
    DIR *d;
    struct dirent *dir;


    d = opendir(SHAREDSHMDIR);
    if(d)
    {
        int sindex = 0;
        while(((dir = readdir(d)) != NULL))
        {
            int scanentryOK = 1;
            char *pch = strstr(dir->d_name, ".im.shm");

            int matchOK = 0;

            // name filtering (first pass, not exclusive to stream name, includes path and extension
            if(filter == 1)
            {
                if(strstr(dir->d_name, namefilter) != NULL)
                {
                    matchOK = 1;
                }
            }
            else
            {
                matchOK = 1;
            }


            if((pch) && (matchOK == 1))
            {
                // is file sym link ?
                struct stat buf;
                int retv;
                char fullname[STRINGMAXLEN_FULLFILENAME];


                snprintf(fullname, STRINGMAXLEN_FULLFILENAME, "%s/%s", SHAREDSHMDIR,
                         dir->d_name);
                retv = lstat(fullname, &buf);
                if(retv == -1)
                {
                    endwin();
                    printf("File \"%s\"", dir->d_name);
                    perror("Error running lstat on file ");
                    exit(EXIT_FAILURE);
                }

                if(S_ISLNK(buf.st_mode))  // resolve link name
                {
                    char fullname[STRINGMAXLEN_FULLFILENAME];
                    char *linknamefull;
                    char linkname[STRINGMAXLEN_FULLFILENAME];
                    int pathOK = 1;


                    streaminfo[sindex].SymLink = 1;
                    snprintf(fullname, STRINGMAXLEN_FULLFILENAME, "%s/%s", SHAREDSHMDIR,
                             dir->d_name);
                    linknamefull = realpath(fullname, NULL);

                    if(linknamefull == NULL)
                    {
                        pathOK = 0;
                    }
                    else if(access(linknamefull, R_OK))     // file cannot be read
                    {
                        pathOK = 0;
                    }

                    if(pathOK == 0)   // file cannot be read
                    {
                        scanentryOK = 0;
                    }
                    else
                    {
                        strcpy(linkname, basename(linknamefull));

                        int lOK = 1;
                        unsigned int ii = 0;
                        while((lOK == 1) && (ii < strlen(linkname)))
                        {
                            if(linkname[ii] == '.')
                            {
                                linkname[ii] = '\0';
                                lOK = 0;
                            }
                            ii++;
                        }
                        strncpy(streaminfo[sindex].linkname, linkname, nameNBchar);
                    }

                    if(linknamefull != NULL)
                    {
                        free(linknamefull);
                    }

                }
                else
                {
                    streaminfo[sindex].SymLink = 0;
                }


                // get stream name
                if(scanentryOK == 1)
                {
                    int strlencp1 = STRINGMAXLEN_STREAMINFO_NAME;
                    int strlencp = strlen(dir->d_name) - strlen(".im.shm");
                    if(strlencp < strlencp1)
                    {
                        strlencp1 = strlencp;
                    }
                    strncpy(streaminfo[sindex].sname, dir->d_name, strlencp1);
                    streaminfo[sindex].sname[strlen(dir->d_name) - strlen(".im.shm")] = '\0';


                    if(filter == 1)
                    {
                        if(strstr(streaminfo[sindex].sname, namefilter) != NULL)
                        {
                            sindex++;
                        }
                    }
                    else
                    {
                        sindex++;
                    }

                }
            }
        }

        NBstream = sindex;
    }
    closedir(d);


    return NBstream;
}







void *streamCTRL_scan(
    void *argptr
)
{
    long NBsindex = 0;
    long sindex = 0;
    long scancnt = 0;

    STREAMINFO *streaminfo;
    char **PIDname_array;


    // timing
    static int firstIter = 1;
    static struct timespec t0;
    struct timespec t1;
    double tdiffv;
    struct timespec tdiff;

    struct streamCTRLarg_struct *streamCTRLdata        = (struct
            streamCTRLarg_struct *)argptr;
    STREAMINFOPROC *streaminfoproc = streamCTRLdata->streaminfoproc;
    IMAGE *images                  = streamCTRLdata->images;

    streaminfo = streaminfoproc->sinfo;
    PIDname_array = streaminfoproc->PIDtable;

    streaminfoproc->loopcnt = 0;


    // if set, write file list to file on first scan
    //int WriteFlistToFile = 1;


    FILE *fpfscan;

    while(streaminfoproc->loop == 1)
    {

        // timing measurement
        clock_gettime(CLOCK_REALTIME, &t1);
        if(firstIter == 1)
        {
            tdiffv = 0.1;
        }
        else
        {
            tdiff = timespec_diff(t0, t1);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        }
        clock_gettime(CLOCK_REALTIME, &t0);
        streaminfoproc->dtscan = tdiffv;



        int mode = 0;
        if(mode == 0) // preferred mode
        {
            NBsindex = find_streams(
                           streaminfo,
                           streaminfoproc->filter,
                           streaminfoproc->namefilter);



            // write stream list to file if applicable
            // ususally used for debugging only
            //
            if(streaminfoproc->WriteFlistToFile == 1)
            {
                fpfscan = fopen("streamCTRL_filescan.dat", "w");
                fprintf(fpfscan, "# stream scan result\n");
                fprintf(fpfscan, "filter: %d %s\n", streaminfoproc->filter,
                        streaminfoproc->namefilter);
                fprintf(fpfscan, "NBsindex = %ld\n", NBsindex);

                for(sindex = 0; sindex < NBsindex; sindex++)
                {
                    //fprintf(fpfscan, "%4ld  %20s ", sindex, dir->d_name);

                    if(streaminfo[sindex].SymLink == 1)
                    {
                        fprintf(fpfscan, "| %12s -> [ %12s ] ", streaminfo[sindex].sname,
                                streaminfo[sindex].linkname);
                    }
                    else
                    {
                        fprintf(fpfscan, "| %12s -> [ %12s ] ", streaminfo[sindex].sname, " ");
                    }
                    fprintf(fpfscan, "\n");
                }
                fclose(fpfscan);
            }


            // Load into memory
            for(sindex = 0; sindex < NBsindex; sindex++)
            {
                imageID ID;

                ID = image_ID_from_images(images, streaminfo[sindex].sname);

                // connect to stream
                if(ID == -1)
                {
                    ID = image_get_first_ID_available_from_images(images);
                    if(ID < 0)
                    {
                        return NULL;
                    }
                    ImageStreamIO_read_sharedmem_image_toIMAGE(streaminfo[sindex].sname,
                            &images[ID]);
                    streaminfo[sindex].deltacnt0 = 1;
                    streaminfo[sindex].updatevalue = 1.0;
                    streaminfo[sindex].updatevalue_frozen = 1.0;
                }
                else
                {
                    float gainv = 1.0;
                    if(firstIter == 0)
                    {
                        streaminfo[sindex].deltacnt0 = images[ID].md[0].cnt0 - streaminfo[sindex].cnt0;
                        streaminfo[sindex].updatevalue = (1.0 - gainv) * streaminfo[sindex].updatevalue
                                                         + gainv * (1.0 * streaminfo[sindex].deltacnt0 / tdiffv);
                    }

                    streaminfo[sindex].cnt0 = images[ID].md[0].cnt0; // keep memory of cnt0
                    streaminfo[sindex].ID = ID;
                    streaminfo[sindex].datatype = images[ID].md[0].datatype;
                }

            }
        }
        else // candidate for removal
        {
		//---------------------------------------
            DIR *d;
            struct dirent *dir;

            d = opendir(SHAREDSHMDIR);
            if(d)
            {
                sindex = 0;

                while(((dir = readdir(d)) != NULL))
                {
                    int scanentryOK = 1;
                    char *pch = strstr(dir->d_name, ".im.shm");

                    int matchOK = 0;

                    // name filtering
                    if(streaminfoproc->filter == 1)
                    {
                        if(strstr(dir->d_name, streaminfoproc->namefilter) != NULL)
                        {
                            matchOK = 1;
                        }
                    }
                    else
                    {
                        matchOK = 1;
                    }



                    if((pch) && (matchOK == 1))
                    {
                        imageID ID;

                        // is file sym link ?
                        struct stat buf;
                        int retv;
                        char fullname[STRINGMAXLEN_FULLFILENAME];


                        if(streaminfoproc->WriteFlistToFile == 1)
                        {
                            fprintf(fpfscan, "%4ld  %20s ", sindex, dir->d_name);
                        }

                        snprintf(fullname, STRINGMAXLEN_FULLFILENAME, "%s/%s", SHAREDSHMDIR,
                                 dir->d_name);
                        retv = lstat(fullname, &buf);
                        if(retv == -1)
                        {
                            endwin();
                            printf("File \"%s\"", dir->d_name);
                            perror("Error running lstat on file ");
                            exit(EXIT_FAILURE);
                        }



                        if(S_ISLNK(buf.st_mode))  // resolve link name
                        {
                            char fullname[STRINGMAXLEN_FULLFILENAME];
                            char *linknamefull;
                            char linkname[STRINGMAXLEN_FULLFILENAME];
                            int pathOK = 1;


                            streaminfo[sindex].SymLink = 1;
                            snprintf(fullname, STRINGMAXLEN_FULLFILENAME, "%s/%s", SHAREDSHMDIR,
                                     dir->d_name);
                            //                        readlink (fullname, linknamefull, 200-1);
                            linknamefull = realpath(fullname, NULL);

                            if(linknamefull == NULL)
                            {
                                pathOK = 0;
                            }
                            else if(access(linknamefull, R_OK))     // file cannot be read
                            {
                                pathOK = 0;
                            }

                            if(pathOK == 0)   // file cannot be read
                            {
                                if(streaminfoproc->WriteFlistToFile == 1)
                                {
                                    fprintf(fpfscan, " %s <-> LINK %s CANNOT BE READ -> off", fullname,
                                            linknamefull);
                                }
                                scanentryOK = 0;
                            }
                            else
                            {
                                strcpy(linkname, basename(linknamefull));

                                int lOK = 1;
                                unsigned int ii = 0;
                                while((lOK == 1) && (ii < strlen(linkname)))
                                {
                                    if(linkname[ii] == '.')
                                    {
                                        linkname[ii] = '\0';
                                        lOK = 0;
                                    }
                                    ii++;
                                }
                                strncpy(streaminfo[sindex].linkname, linkname, nameNBchar);
                            }

                            if(linknamefull != NULL)
                            {
                                free(linknamefull);
                            }

                        }
                        else
                        {
                            streaminfo[sindex].SymLink = 0;
                        }



                        // get stream name and ID
                        if(scanentryOK == 1)
                        {
                            int strlencp1 = STRINGMAXLEN_STREAMINFO_NAME;
                            int strlencp = strlen(dir->d_name) - strlen(".im.shm");
                            if(strlencp < strlencp1)
                            {
                                strlencp1 = strlencp;
                            }
                            strncpy(streaminfo[sindex].sname, dir->d_name, strlencp1);
                            streaminfo[sindex].sname[strlen(dir->d_name) - strlen(".im.shm")] = '\0';

                            if(streaminfoproc->WriteFlistToFile == 1)
                            {
                                if(streaminfo[sindex].SymLink == 1)
                                {
                                    fprintf(fpfscan, "| %12s -> [ %12s ] ", streaminfo[sindex].sname,
                                            streaminfo[sindex].linkname);
                                }
                                else
                                {
                                    fprintf(fpfscan, "| %12s -> [ %12s ] ", streaminfo[sindex].sname, " ");
                                }
                            }
                        }


                        if(scanentryOK == 1)
                        {

                            ID = image_ID_from_images(images, streaminfo[sindex].sname);

                            // connect to stream
                            if(ID == -1)
                            {
                                ID = image_get_first_ID_available_from_images(images);
                                if(ID < 0)
                                {
                                    return NULL;
                                }
                                ImageStreamIO_read_sharedmem_image_toIMAGE(streaminfo[sindex].sname,
                                        &images[ID]);
                                streaminfo[sindex].deltacnt0 = 1;
                                streaminfo[sindex].updatevalue = 1.0;
                                streaminfo[sindex].updatevalue_frozen = 1.0;
                            }
                            else
                            {
                                float gainv = 1.0;
                                if(firstIter == 0)
                                {
                                    streaminfo[sindex].deltacnt0 = images[ID].md[0].cnt0 - streaminfo[sindex].cnt0;
                                    streaminfo[sindex].updatevalue = (1.0 - gainv) * streaminfo[sindex].updatevalue
                                                                     + gainv * (1.0 * streaminfo[sindex].deltacnt0 / tdiffv);
                                }

                                streaminfo[sindex].cnt0 = images[ID].md[0].cnt0; // keep memory of cnt0
                                streaminfo[sindex].ID = ID;
                                streaminfo[sindex].datatype = images[ID].md[0].datatype;

                                sindex++;
                            }
                        }
                        if(streaminfoproc->WriteFlistToFile == 1)
                        {
                            fprintf(fpfscan, "\n");
                        }
                    }
                }

                NBsindex = sindex;
            }
            closedir(d);

            if(streaminfoproc->WriteFlistToFile == 1)
            {
                fclose(fpfscan);
            }
		//---------------------------------------
        }

        streaminfoproc->WriteFlistToFile = 0;


        firstIter = 0;





        if(streaminfoproc->fuserUpdate == 1)
        {
            FILE *fp;
            int STRINGMAXLEN_LINE = 2000;
            char plistoutline[STRINGMAXLEN_LINE];
            char command[STRINGMAXLEN_COMMAND];

            int NBpid = 0;

            //            sindexscan1 = ssindex[sindexscan];
            int sindexscan1 = streaminfoproc->sindexscan;


            if(streaminfoproc->sindexscan > NBsindex - 1)
            {
                streaminfoproc->fuserUpdate = 0;
            }
            else
            {
                int PReadMode = 1;

                if(PReadMode == 0)
                {
                    // popen option
                    {
                        int slen = snprintf(command, STRINGMAXLEN_COMMAND,
                                            "/bin/fuser %s/%s.im.shm 2>/dev/null", SHAREDSHMDIR,
                                            streaminfo[sindexscan1].sname);
                        if(slen < 1)
                        {
                            PRINT_ERROR("snprintf wrote <1 char");
                            abort(); // can't handle this error any other way
                        }
                        if(slen >= STRINGMAXLEN_COMMAND)
                        {
                            PRINT_ERROR("snprintf string truncation");
                            abort(); // can't handle this error any other way
                        }
                    }

                    fp = popen(command, "r");
                    if(fp == NULL)
                    {
                        streaminfo[sindexscan1].streamOpenPID_status = 2; // failed
                    }
                    else
                    {
                        streaminfo[sindexscan1].streamOpenPID_status = 1;

                        if(fgets(plistoutline, STRINGMAXLEN_LINE - 1, fp) == NULL)
                        {
                            sprintf(plistoutline, " ");
                        }
                        pclose(fp);
                    }
                }
                else
                {
                    // filesystem option
                    char plistfname[STRINGMAXLEN_FULLFILENAME];
                    WRITE_FULLFILENAME(plistfname, "%s/%s.shmplist", SHAREDSHMDIR,
                                       streaminfo[sindexscan1].sname);

                    {
                        int slen = snprintf(command, STRINGMAXLEN_COMMAND,
                                            "/bin/fuser %s/%s.im.shm 2>/dev/null > %s", SHAREDSHMDIR,
                                            streaminfo[sindexscan1].sname, plistfname);
                        if(slen < 1)
                        {
                            PRINT_ERROR("snprintf wrote <1 char");
                            abort(); // can't handle this error any other way
                        }
                        if(slen >= STRINGMAXLEN_COMMAND)
                        {
                            PRINT_ERROR("snprintf string truncation");
                            abort(); // can't handle this error any other way
                        }
                    }

                    if(system(command) == -1)
                    {
                        perror("Command system() failed");
                        exit(EXIT_FAILURE);
                    }

                    fp = fopen(plistfname, "r");
                    if(fp == NULL)
                    {
                        streaminfo[sindexscan1].streamOpenPID_status = 2;
                    }
                    else
                    {
                        if(fgets(plistoutline, STRINGMAXLEN_LINE - 1, fp) == NULL)
                        {
                            sprintf(plistoutline, " ");
                        }

                        fclose(fp);
                    }
                }


                if(streaminfo[sindexscan1].streamOpenPID_status != 2)
                {
                    char *pch;

                    pch = strtok(plistoutline, " ");

                    while(pch != NULL)
                    {
                        if(NBpid < streamOpenNBpid_MAX)
                        {
                            streaminfo[sindexscan1].streamOpenPID[NBpid] = atoi(pch);
                            if(getpgid(streaminfo[sindexscan1].streamOpenPID[NBpid]) >= 0)
                            {
                                NBpid++;
                            }
                        }
                        pch = strtok(NULL, " ");
                    }
                    streaminfo[sindexscan1].streamOpenPID_status = 1; // success
                }

                streaminfo[sindexscan1].streamOpenPID_cnt = NBpid;
                // Get PID names
                int pidIndex;
                int cnt1 = 0;
                for(pidIndex = 0; pidIndex < streaminfo[sindexscan1].streamOpenPID_cnt;
                        pidIndex++)
                {
                    pid_t pid = streaminfo[sindexscan1].streamOpenPID[pidIndex];
                    if((getpgid(pid) >= 0) && (pid != getpid()))
                    {
                        char *pname = (char *) calloc(1024, sizeof(char));
                        get_process_name_by_pid(pid, pname);

                        if(PIDname_array[pid] == NULL)
                        {
                            PIDname_array[pid] = (char *) malloc(sizeof(char) * (PIDnameStringLen + 1));
                        }
                        strncpy(PIDname_array[pid], pname, PIDnameStringLen);
                        free(pname);
                        cnt1++;
                    }
                }
                streaminfo[sindexscan1].streamOpenPID_cnt1 = cnt1;

                streaminfoproc->sindexscan++;
            }
        }

        streaminfoproc->fuserUpdate0 = 0;

        streaminfoproc->NBstream = NBsindex;
        streaminfoproc->loopcnt++;


        usleep(streaminfoproc->twaitus);

        scancnt ++;
    }



    return NULL;
}








static int streamCTRL_print_inode(
    ino_t inode,
    ino_t *upstreaminode,
    int NBupstreaminode,
    int downstreamindex
)
{
    int Dispinode_NBchar = 8;
    int is_upstream = 0;
    int is_downstream = 0;
    int upstreamindex = 0;

    for(int i=0; i<NBupstreaminode; i++)
    {
        if(inode == upstreaminode[i])
        {
            is_upstream = 1;
            upstreamindex = i;
        }
    }


    if (downstreamindex < NO_DOWNSTREAM_INDEX)
    {
        is_downstream = 1;
    }


    if( is_upstream || is_downstream )
    {
        int colorcode = 3;
        if(upstreamindex > 0)
        {
            colorcode = 7;
        }

        if(is_upstream)
        {
            attron(COLOR_PAIR(colorcode));
            printw("%02d >", upstreamindex);
            attroff(COLOR_PAIR(colorcode));
        }
        else
        {
            printw("    ");
        }

        if(is_downstream)
        {
            int colorcode = 3;
            if(downstreamindex > 0)
            {
                colorcode = 7;
            }

            attron(COLOR_PAIR(colorcode));
            printw("> %02d", downstreamindex);
            attroff(COLOR_PAIR(colorcode));
        }
        else
        {
            printw("    ");
        }
    }
    else
    {
        printw("%*d", Dispinode_NBchar, (int) inode);
    }

    return Dispinode_NBchar;
}






/** @brief print PID with highlighting
 *
 */
static int streamCTRL_print_procpid(
    pid_t procpid,
    pid_t *upstreamproc,
    int NBupstreamproc,
    uint32_t mode
)
{
    int DispPID_NBchar = 8;
    int activitycolorcode = 0;
    int is_upstream = 0;
	int upstreamindex = 0;


    if ( mode & PRINT_PID_FORCE_NOUPSTREAM )
    {
        is_upstream = 0;
    }
    else
    {
        for(int i=0; i<NBupstreamproc; i++)
        {
            if(procpid == upstreamproc[i])
            {
                is_upstream = 1;
                upstreamindex = i;
            }
        }

    }


    if(procpid>0)
    {
        if(getpgid(procpid) >= 0)   // check if pid active
        {
            activitycolorcode = 2;
        }
        else
        {
            if(procpid > 0)
            {
                activitycolorcode = 4;
            }
        }
    }
    
    if ( is_upstream == 1 )
    {
		if( activitycolorcode !=2 )
		{
			attron(A_REVERSE);
		}
		else
		{
			activitycolorcode = 12;
		}
	}
	
    
    

    if(activitycolorcode > 0)
    {
        attron(COLOR_PAIR(activitycolorcode));
    }

    
    if(is_upstream)
    {
		char upstreamstring[DispPID_NBchar+1];
		sprintf(upstreamstring, "%2d >>", upstreamindex);
		printw("%*s", DispPID_NBchar, upstreamstring);
	}
	else
	{
		printw("%*d", DispPID_NBchar, (int) procpid);
    }
    

    if(activitycolorcode > 0)
    {
        attroff(COLOR_PAIR(activitycolorcode));
    }


    if( (activitycolorcode !=2 ) && (is_upstream == 1) )
    {
		attroff(A_REVERSE);
	}


    return DispPID_NBchar;
}





static errno_t streamCTRL_print_SPTRACE_details(
    IMAGE *streamCTRLimages,
    imageID ID,
    pid_t *upstreamproc,
    int NBupstreamproc,
    uint32_t print_pid_mode
)
{
    /*
       	int             triggermode;
    	pid_t           procwrite_PID;
    	ino_t           trigger_inode;
    	struct timespec ts_procstart;
    	struct timespec ts_streamupdate;
    	int             trigsemindex;
    	uint64_t        cnt0;

    */

    char string[100];

    int Disp_inode_NBchar = 8;

    int Disp_sname_NBchar = 16;

    int Disp_cnt0_NBchar = 12;
    int Disp_PID_NBchar = 8;
    int Disp_type_NBchar = 8;
    int Disp_trigstat_NBchar = 12;

    printw("\n");
    printw("   %*s %*s %*s\n",
           Disp_inode_NBchar, "inode",
           Disp_cnt0_NBchar, "cnt0",
           Disp_PID_NBchar, "PID",
           Disp_type_NBchar, "type"
          );


    for(int spti = 0; spti < streamCTRLimages[ID].md[0].NBproctrace; spti++)
    {
        ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
        int   sem   = streamCTRLimages[ID].streamproctrace[spti].trigsemindex;
        pid_t pid   = streamCTRLimages[ID].streamproctrace[spti].procwrite_PID;

        uint64_t cnt0 = streamCTRLimages[ID].streamproctrace[spti].cnt0;

        printw("%02d", spti);

        printw(" %*lu", Disp_inode_NBchar, inode);


        // look for ID corresponding to inode
        int IDscan = 0;
        int IDfound = -1;
        while ((IDfound == -1) && (IDscan < streamNBID_MAX))
        {
            if(streamCTRLimages[IDscan].used == 1)
            {
                if(streamCTRLimages[IDscan].md[0].inode == inode)
                {
                    IDfound = IDscan;
                }
            }
            IDscan++;
        }
        if(IDfound == -1)
        {
            printw(" %*s", Disp_sname_NBchar, "???");
        }
        else
        {
            printw(" %*s", Disp_sname_NBchar, streamCTRLimages[IDfound].name);
        }

        printw(" %*llu", Disp_cnt0_NBchar, cnt0);

        Disp_PID_NBchar = streamCTRL_print_procpid(pid, upstreamproc, NBupstreamproc, PRINT_PID_FORCE_NOUPSTREAM);



        switch (streamCTRLimages[ID].streamproctrace[spti].triggermode)
        {
        case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
            printw(" %*s", Disp_type_NBchar, "IMMEDIATE");
            break;

        case PROCESSINFO_TRIGGERMODE_CNT0:
            printw(" %*s", Disp_type_NBchar, "CNT0");
            break;

        case PROCESSINFO_TRIGGERMODE_CNT1:
            printw(" %*s", Disp_type_NBchar, "CNT1");
            break;

        case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
            printw(" %*s", Disp_type_NBchar, "SEMAPHORE");
            printw(" %2d", sem);
            break;

        case PROCESSINFO_TRIGGERMODE_DELAY:
            printw(" %*s", Disp_type_NBchar, "DELAY");
            break;

        default:
            printw(" %*s", Disp_type_NBchar, "UNKNOWN");
            break;
        }
        
        
        switch (streamCTRLimages[ID].streamproctrace[spti].triggerstatus)
        {
			case PROCESSINFO_TRIGGERSTATUS_WAITING:
			printw(" %*s", Disp_trigstat_NBchar, "WAITING");
			break;
			
			case PROCESSINFO_TRIGGERSTATUS_RECEIVED:
			printw(" %*s", Disp_trigstat_NBchar, "RECEIVED");
			break;
			
			case PROCESSINFO_TRIGGERSTATUS_TIMEDOUT:
			attron(COLOR_PAIR(2));
			printw(" %*s", Disp_trigstat_NBchar, "TIMEOUT");
			attroff(COLOR_PAIR(2));
			break;
		}
		
		// trigger time
		printw(" at %ld.%09ld s", streamCTRLimages[ID].streamproctrace[spti].ts_procstart.tv_sec, streamCTRLimages[ID].streamproctrace[spti].ts_procstart.tv_nsec);
		
		struct timespec tnow;
		clock_gettime(CLOCK_REALTIME, &tnow);
		struct timespec tdiff;
		
		tdiff = timespec_diff(streamCTRLimages[ID].streamproctrace[spti].ts_procstart, tnow);
        double tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

		printw("  %12.3f us ago", tdiffv*1.0e6);


        printw("\n");
    }


    return RETURN_SUCCESS;
}




/**
 * ## Purpose
 *
 * Control screen for stream structures
 *
 * ## Description
 *
 * Relies on ncurses for display\n
 *
 *
 */

errno_t streamCTRL_CTRLscreen()
{

    int stringmaxlen = 300;

    // Display fields
    STREAMINFO *streaminfo;
    STREAMINFOPROC streaminfoproc;

    long sindex;  // scan index
    long dindex;  // display index
    long doffsetindex = 0; // offset index if more entries than can be displayed

    long ssindex[streamNBID_MAX]; // sorted index array

    float frequ = 32.0; // Hz
    char  monstring[200];





    int SORTING = 0;
    int SORT_TOGGLE = 0;


    DEBUG_TRACEPOINT(" ");

    pthread_t threadscan;


    // display
    int DispName_NBchar = 36;
    int DispSize_NBchar = 20;
    int Dispcnt0_NBchar = 10;
    int Dispfreq_NBchar = 8;

    int DispPID_NBchar = 8;

    // create PID name table
    char **PIDname_array;
    int PIDmax;



    PIDmax = get_PIDmax();

    PIDname_array = (char **)malloc(sizeof(char *)*PIDmax);
    for(int pidi = 0; pidi < PIDmax; pidi++)
    {
        PIDname_array[pidi] = NULL;
    }

    streaminfoproc.WriteFlistToFile = 0;
    streaminfoproc.loopcnt = 0;
    streaminfoproc.fuserUpdate = 0;

    streaminfo = (STREAMINFO *) malloc(sizeof(STREAMINFO) * streamNBID_MAX);
    streaminfoproc.sinfo = streaminfo;
    for(int sindex = 0; sindex < streamNBID_MAX; sindex++)
    {
        streaminfo[sindex].updatevalue = 0.0;
        streaminfo[sindex].updatevalue_frozen = 0.0;
        streaminfo[sindex].cnt0 = 0;
        streaminfo[sindex].streamOpenPID_status = 0;
    }
    streaminfoproc.PIDtable = PIDname_array;


    IMAGE *streamCTRLimages = (IMAGE *) malloc(sizeof(IMAGE) * streamNBID_MAX);
    for(imageID imID = 0; imID < streamNBID_MAX; imID++)
    {
        streamCTRLimages[imID].used    = 0;
        streamCTRLimages[imID].shmfd   = -1;
        streamCTRLimages[imID].memsize = 0;
        streamCTRLimages[imID].semptr  = NULL;
        streamCTRLimages[imID].semlog  = NULL;
    }

    struct streamCTRLarg_struct streamCTRLdata;
    streamCTRLdata.streaminfoproc = &streaminfoproc;
    streamCTRLdata.images         = streamCTRLimages;

    setlocale(LC_ALL, "");


    streamCTRL_CatchSignals();

    // INITIALIZE ncurses
    initncurses();

    int NBsinfodisp = wrow - 7;
    int NBsindex = 0;
    int loopOK = 1;
    long long loopcnt = 0;


    int dindexSelected = 0;

    int DisplayMode = DISPLAY_MODE_SEMVAL;


    struct tm *uttime_lastScan;
    time_t rawtime;
    int fuserScan = 0;

    streaminfoproc.filter = 0;
    streaminfoproc.NBstream = 0;
    streaminfoproc.twaitus = 50000; // 20 Hz
    streaminfoproc.fuserUpdate0 = 1; //update on first instance

    // inodes that are upstream of current selection
    int NBupstreaminodeMAX = 100;
    ino_t *upstreaminode;
    int NBupstreaminode = 0;
    upstreaminode = (ino_t*) malloc(sizeof(ino_t)*NBupstreaminodeMAX);


    // processes that are upstream of current selection
    int NBupstreamprocMAX = 100;
    pid_t *upstreamproc;
    int NBupstreamproc = 0;
    upstreamproc = (pid_t*) malloc(sizeof(pid_t)*NBupstreamprocMAX);


    clear();
    DEBUG_TRACEPOINT(" ");

    // redirect stderr to /dev/null

    int backstderr;
    int newstderr;
    char newstderrfname[STRINGMAXLEN_FULLFILENAME];


    fflush(stderr);
    backstderr = dup(STDERR_FILENO);
    WRITE_FULLFILENAME(newstderrfname, "%s/stderr.cli.%d.txt", SHAREDSHMDIR,
                       CLIPID);
    //sprintf(newstderrfname, "%s/stderr.cli.%d.txt", SHAREDSHMDIR, CLIPID);

#ifndef STANDALONE  // TODO(sevin): why it fails in STANDALONE mode?
    newstderr = open(newstderrfname, O_WRONLY | O_CREAT, 0644);
    dup2(newstderr, STDERR_FILENO);
    close(newstderr);
#endif


    DEBUG_TRACEPOINT(" ");



    // Start scan thread
    streaminfoproc.loop = 1;
    pthread_create(&threadscan, NULL, streamCTRL_scan, (void *) &streamCTRLdata);


    DEBUG_TRACEPOINT(" ");

    char c; // for user input
    int stringindex;

    loopcnt = 0;


	ino_t inodeselected = 0;
    int DisplayDetailLevel = 0;
    while(loopOK == 1)
    {
        //int pid;
        //char command[200];

        DEBUG_TRACEPOINT(" ");

        if(streaminfoproc.loopcnt == 1)
        {
            SORTING = 2;
            SORT_TOGGLE = 1;
        }

        //if(fuserUpdate != 1) // don't wait if ongoing fuser scan

        usleep((long)(1000000.0 / frequ));
        int ch = getch();


        NBsindex = streaminfoproc.NBstream;

        DEBUG_TRACEPOINT(" ");


        //int selectedOK = 0; // goes to 1 if at least one process is selected
        switch(ch)
        {
        case 'x':     // Exit control screen
            loopOK = 0;
            break;


        case KEY_UP:
            dindexSelected --;
            if(dindexSelected < 0)
            {
                dindexSelected = 0;
            }
            break;

        case KEY_DOWN:
            dindexSelected ++;
            if(dindexSelected > NBsindex - 1)
            {
                dindexSelected = NBsindex - 1;
            }
            break;

        case KEY_PPAGE:
            dindexSelected -= 10;
            if(dindexSelected < 0)
            {
                dindexSelected = 0;
            }
            break;

        case KEY_LEFT:
            DisplayDetailLevel = 0;
            break;

        case KEY_RIGHT:
            DisplayDetailLevel = 1;
            break;


        case KEY_NPAGE:
            dindexSelected += 10;
            if(dindexSelected > NBsindex - 1)
            {
                dindexSelected = NBsindex - 1;
            }
            break;



        // ============ SCREENS

        case 'h': // help
            DisplayMode = DISPLAY_MODE_HELP;
            break;

        case KEY_F(2): // semvals
            DisplayMode = DISPLAY_MODE_SEMVAL;
            break;

        case KEY_F(3): // write PIDs
            DisplayMode = DISPLAY_MODE_WRITE;
            break;

        case KEY_F(4): // read PIDs
            DisplayMode = DISPLAY_MODE_READ;
            break;

        case KEY_F(5): // read PIDs
            DisplayMode = DISPLAY_MODE_SPTRACE;
            break;

        case KEY_F(6): // open files
            if((DisplayMode == DISPLAY_MODE_FUSER) || (streaminfoproc.fuserUpdate0 == 1))
            {
                streaminfoproc.fuserUpdate = 1;
                time(&rawtime);
                uttime_lastScan = gmtime(&rawtime);
                fuserScan = 1;
                streaminfoproc.sindexscan = 0;
            }

            DisplayMode = DISPLAY_MODE_FUSER;
            //erase();
            //printw("SCANNING PROCESSES AND FILESYSTEM: PLEASE WAIT ...\n");
            //refresh();
            break;



        // ============ ACTIONS

        case 'R': // remove stream
            DEBUG_TRACEPOINT(" ");
            sindex = ssindex[dindexSelected];

            ImageStreamIO_destroyIm(&streamCTRLimages[streaminfo[sindex].ID]);


            DEBUG_TRACEPOINT("%d", dindexSelected);
            break;



        // ============ SCANNING

        case '{': // slower scan update
            streaminfoproc.twaitus = (int)(1.2 * streaminfoproc.twaitus);
            if(streaminfoproc.twaitus > 1000000)
            {
                streaminfoproc.twaitus = 1000000;
            }
            break;

        case '}': // faster scan update
            streaminfoproc.twaitus = (int)(0.83333333333333333333 * streaminfoproc.twaitus);
            if(streaminfoproc.twaitus < 1000)
            {
                streaminfoproc.twaitus = 1000;
            }
            break;

        case 'o': // output next scan to file
            streaminfoproc.WriteFlistToFile = 1;
            break;

        // ============ DISPLAY

        case '-': // slower display update
            frequ *= 0.5;
            if(frequ < 1.0)
            {
                frequ = 1.0;
            }
            if(frequ > 64.0)
            {
                frequ = 64.0;
            }
            break;


        case '+': // faster display update
            frequ *= 2.0;
            if(frequ < 1.0)
            {
                frequ = 1.0;
            }
            if(frequ > 64.0)
            {
                frequ = 64.0;
            }
            break;

        case '1': // sorting by stream name
            SORTING = 1;
            break;

        case '2': // sorting by update freq (default)
            SORTING = 2;
            SORT_TOGGLE = 1;
            break;

        case '3': // sort by number of processes accessing
            SORTING = 3;
            SORT_TOGGLE = 1;
            break;


        case 'f': // stream name filter toggle
            if(streaminfoproc.filter == 0)
            {
                streaminfoproc.filter = 1;
            }
            else
            {
                streaminfoproc.filter = 0;
            }
            break;

        case 'F': // set stream name filter string
            endwin();
            EXECUTE_SYSTEM_COMMAND("clear");
            printf("Enter string: ");
            fflush(stdout);
            stringindex = 0;
            while(((c = getchar()) != 13) && (stringindex < STRINGLENMAX - 2))
            {
                streaminfoproc.namefilter[stringindex] = c;
                if(c == 127)   // delete key
                {
                    putchar(0x8);
                    putchar(' ');
                    putchar(0x8);
                    stringindex --;
                }
                else
                {
                    putchar(c);  // echo on screen
                    stringindex++;
                }
            }
            streaminfoproc.namefilter[stringindex] = '\0';
            initncurses();
            break;



        }

        DEBUG_TRACEPOINT(" ");

        if(dindexSelected < 0)
        {
            dindexSelected = 0;
        }
        if(dindexSelected > NBsindex - 1)
        {
            dindexSelected = NBsindex - 1;
        }

        DEBUG_TRACEPOINT(" ");

        erase();

        attron(A_BOLD);
        sprintf(monstring, "[PID %d] STREAM MONITOR: PRESS (x) TO STOP, (h) FOR HELP",
                getpid());
        streamCTRL__print_header(monstring, '-');
        attroff(A_BOLD);




        if(DisplayMode == DISPLAY_MODE_HELP)   // help
        {
            int attrval = A_BOLD;

            DEBUG_TRACEPOINT(" ");

            attron(attrval);
            printw("    x");
            attroff(attrval);
            printw("    Exit\n");


            printw("\n");
            printw("============ SCREENS \n");

            attron(attrval);
            printw("     h");
            attroff(attrval);
            printw("   Help screen\n");

            attron(attrval);
            printw("    F2");
            attroff(attrval);
            printw("   Display semaphore values\n");

            attron(attrval);
            printw("    F3");
            attroff(attrval);
            printw("   Display semaphore write PIDs\n");

            attron(attrval);
            printw("    F4");
            attroff(attrval);
            printw("   Display semaphore read PIDs\n");

            attron(attrval);
            printw("    F5");
            attroff(attrval);
            printw("   stream process trace\n");

            attron(attrval);
            printw("    F6");
            attroff(attrval);
            printw("   stream open by processes ...\n");

            printw("\n");
            printw("============ ACTIONS \n");

            attron(attrval);
            printw("    R");
            attroff(attrval);
            printw("    Remove stream\n");

            printw("\n");
            printw("============ SCANNING \n");

            attron(attrval);
            printw("    }");
            attroff(attrval);
            printw("    Increase scan frequency\n");

            attron(attrval);
            printw("    {");
            attroff(attrval);
            printw("    Decrease scan frequency\n");

            attron(attrval);
            printw("    o");
            attroff(attrval);
            printw("    output next scan to file\n");


            printw("\n");
            printw("============ DISPLAY \n");

            attron(attrval);
            printw("    +");
            attroff(attrval);
            printw("    Increase display frequency\n");

            attron(attrval);
            printw("    -");
            attroff(attrval);
            printw("    Decrease display frequency\n");

            attron(attrval);
            printw("    1");
            attroff(attrval);
            printw("    Sort by stream name (alphabetical)\n");

            attron(attrval);
            printw("    2");
            attroff(attrval);
            printw("    Sort by recently updated\n");

            attron(attrval);
            printw("    3");
            attroff(attrval);
            printw("    Sort by processes access\n");

            attron(attrval);
            printw("    F");
            attroff(attrval);
            printw("    Set match string pattern\n");

            attron(attrval);
            printw("    f");
            attroff(attrval);
            printw("    Toggle apply match string to stream\n");


            printw("\n\n");
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            if(DisplayMode == DISPLAY_MODE_HELP)
            {
                attron(A_REVERSE);
                printw("[h] Help");
                attroff(A_REVERSE);
            }
            else
            {
                printw("[h] Help");
            }
            printw("   ");

            if(DisplayMode == DISPLAY_MODE_SEMVAL)
            {
                attron(A_REVERSE);
                printw("[F2] sem values");
                attroff(A_REVERSE);
            }
            else
            {
                printw("[F2] sem values");
            }
            printw("   ");

            if(DisplayMode == DISPLAY_MODE_WRITE)
            {
                attron(A_REVERSE);
                printw("[F3] write PIDs");
                attroff(A_REVERSE);
            }
            else
            {
                printw("[F3] write PIDs");
            }
            printw("   ");

            if(DisplayMode == DISPLAY_MODE_READ)
            {
                attron(A_REVERSE);
                printw("[F4] read PIDs");
                attroff(A_REVERSE);
            }
            else
            {
                printw("[F4] read PIDs");
            }
            printw("   ");

            if(DisplayMode == DISPLAY_MODE_SPTRACE)
            {
                attron(A_REVERSE);
                printw("[F5] process traces");
                attroff(A_REVERSE);
            }
            else
            {
                printw("[F5] process traces");
            }
            printw("   ");

            if(DisplayMode == DISPLAY_MODE_FUSER)
            {
                attron(A_REVERSE);
                printw("[F6] access");
                attroff(A_REVERSE);
            }
            else
            {
                printw("[F6] access");
            }
            printw("   ");
            printw("\n");


            printw("PIDmax = %d    Update frequ = %2d Hz  fscan=%5.2f Hz ( %5.2f Hz %5.2f %% busy ) ",
                   PIDmax,
                   (int)(frequ + 0.5),
                   1.0 / streaminfoproc.dtscan,
                   1000000.0 / streaminfoproc.twaitus,
                   100.0 * (streaminfoproc.dtscan - 1.0e-6 * streaminfoproc.twaitus) /
                   streaminfoproc.dtscan
                  );

            if(streaminfoproc.fuserUpdate == 1)
            {
                attron(COLOR_PAIR(9));
                printw("fuser scan ongoing  %4d  / %4d   ", streaminfoproc.sindexscan,
                       NBsindex);
                attroff(COLOR_PAIR(9));
            }
            if(DisplayMode == DISPLAY_MODE_FUSER)
            {
                if(fuserScan == 1)
                {
                    printw("Last scan on  %02d:%02d:%02d  - Press F6 again to re-scan    C-c to stop scan\n",
                           uttime_lastScan->tm_hour, uttime_lastScan->tm_min,  uttime_lastScan->tm_sec);
                }
                else
                {
                    printw("Last scan on  XX:XX:XX  - Press F6 again to scan             C-c to stop scan\n");
                }
            }
            else
            {
                printw("\n");
            }

            int lastindex;
            lastindex = doffsetindex + NBsinfodisp;
            if(lastindex > NBsindex - 1)
            {
                lastindex = NBsindex - 1;
            }

            if(lastindex < 0)
            {
                lastindex = 0;
            }

            printw("%4d streams    Currently displaying %4d-%4d   Selected %d  ID = %d  inode = %d", NBsindex,
                   doffsetindex, lastindex, dindexSelected, ssindex[dindexSelected], (int) inodeselected);

            if(streaminfoproc.filter == 1)
            {
                attron(COLOR_PAIR(9));
                printw("  Filter = \"%s\"", streaminfoproc.namefilter);
                attroff(COLOR_PAIR(9));
            }

            printw("\n");



            attron(A_BOLD);

            printw("%8s  %-*s  %-*s  %*s   %*s %*s %*s %8s",
                   "inode",
                   DispName_NBchar, "name",
                   DispSize_NBchar, "type",
                   Dispcnt0_NBchar, "cnt0",
                   DispPID_NBchar,  "creaPID",
                   DispPID_NBchar,  "ownPID",
                   Dispfreq_NBchar, "   frequ ",
                   "#sem"
                  );

            switch (DisplayMode) {
            case DISPLAY_MODE_SEMVAL:
                printw("     Semaphore values ....\n");
                break;

            case DISPLAY_MODE_WRITE:
                printw("     write PIDs ....\n");
                break;

            case DISPLAY_MODE_READ:
                printw("     read PIDs ....\n");
                break;

            case DISPLAY_MODE_SPTRACE:
                printw("     stream process traces:   \"(INODE TYPE/SEM PID)>\"\n");
                break;

            case DISPLAY_MODE_FUSER:
                printw("     connected processes\n");
                break;

            default:
                printw("\n");
                break;
            }



            attroff(A_BOLD);



            DEBUG_TRACEPOINT(" ");





            // SORT

            // default : no sorting
            for(dindex = 0; dindex < NBsindex; dindex++)
            {
                ssindex[dindex] = dindex;
            }


            DEBUG_TRACEPOINT(" ");

            if(SORTING == 1)   // alphabetical sorting
            {
                long *larray;
                larray = (long *) malloc(sizeof(long) * NBsindex);
                for(sindex = 0; sindex < NBsindex; sindex++)
                {
                    larray[sindex] = sindex;
                }

                int sindex0, sindex1;
                for(sindex0 = 0; sindex0 < NBsindex - 1; sindex0++)
                {
                    for(sindex1 = sindex0 + 1; sindex1 < NBsindex; sindex1++)
                    {
                        if(strcmp(streaminfo[larray[sindex0]].sname,
                                  streaminfo[larray[sindex1]].sname) > 0)
                        {
                            int tmpindex = larray[sindex0];
                            larray[sindex0] = larray[sindex1];
                            larray[sindex1] = tmpindex;
                        }
                    }
                }

                for(dindex = 0; dindex < NBsindex; dindex++)
                {
                    ssindex[dindex] = larray[dindex];
                }
                free(larray);
            }

            DEBUG_TRACEPOINT(" ");


            if((SORTING == 2) || (SORTING == 3))   // recent update and process access
            {
                long *larray;
                double *varray;
                larray = (long *) malloc(sizeof(long) * NBsindex);
                varray = (double *) malloc(sizeof(double) * NBsindex);

                if(SORT_TOGGLE == 1)
                {
                    for(sindex = 0; sindex < NBsindex; sindex++)
                    {
                        streaminfo[sindex].updatevalue_frozen = streaminfo[sindex].updatevalue;
                    }

                    if(SORTING == 3)
                    {
                        for(sindex = 0; sindex < NBsindex; sindex++)
                        {
                            streaminfo[sindex].updatevalue_frozen += 10000.0 *
                                    streaminfo[sindex].streamOpenPID_cnt1;
                        }
                    }

                    SORT_TOGGLE = 0;
                }



                for(sindex = 0; sindex < NBsindex; sindex++)
                {
                    larray[sindex] = sindex;
                    varray[sindex] = streaminfo[sindex].updatevalue_frozen;
                }

                if(NBsindex > 1)
                {
                    quick_sort2l(varray, larray, NBsindex);
                }

                for(dindex = 0; dindex < NBsindex; dindex++)
                {
                    ssindex[NBsindex - dindex - 1] = larray[dindex];
                }

                free(larray);
                free(varray);
            }



            DEBUG_TRACEPOINT(" ");

            // compute doffsetindex

            while(dindexSelected - doffsetindex > NBsinfodisp - 5)   // scroll down
            {
                doffsetindex ++;
            }

            while(dindexSelected - doffsetindex < NBsinfodisp - 10)   // scroll up
            {
                doffsetindex --;
            }

            if(doffsetindex < 0)
            {
                doffsetindex = 0;
            }





            // DISPLAY

            int DisplayFlag = 0;

            int print_pid_mode = PRINT_PID_DEFAULT;
            for(dindex = 0; dindex < NBsindex; dindex++)
            {
                long ID;
                sindex = ssindex[dindex];
                ID = streaminfo[sindex].ID;
	
				int downstreammin = NO_DOWNSTREAM_INDEX; // minumum downstream index
				// looks for inodeselected in the list of upstream inodes
				// picks the smallest corresponding index
				// for example, if equal to 3, the current inode is a 3-rd gen children of selected inode
				// default initial value 100 is a placeholder indicating it is not a child

                if((dindex > doffsetindex - 1) && (dindex < NBsinfodisp - 1 + doffsetindex))
                {
                    DisplayFlag = 1;
                }
                else
                {
                    DisplayFlag = 0;
                }

				if( DisplayDetailLevel == 1 )
				{
					if ( dindex == dindexSelected )
					{
						DisplayFlag = 1;
					}
					else
					{
						DisplayFlag = 0;
					}
				}



                if(dindex == dindexSelected)
                {
					// currently selected inode
					inodeselected = streamCTRLimages[streaminfo[sindex].ID].md[0].inode;
					
                    // identify upstream inodes
                    NBupstreaminode = 0;
                    for(int spti = 0; spti < streamCTRLimages[ID].md[0].NBproctrace; spti++)
                    {
                        if(NBupstreaminode < NBupstreaminodeMAX)
                        {
                            ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
                            if( inode != 0 )
                            {
                                upstreaminode[NBupstreaminode] = inode;
                                NBupstreaminode ++;
                            }
                        }
                    }

                    // identify upstream processes
                    print_pid_mode = PRINT_PID_FORCE_NOUPSTREAM;                    
                    NBupstreamproc = 0;
                    for(int spti = 0; spti < streamCTRLimages[ID].md[0].NBproctrace; spti++)
                    {
                        if(NBupstreamproc < NBupstreamprocMAX)
                        {
                            ino_t procpid = streamCTRLimages[ID].streamproctrace[spti].procwrite_PID;
                            if( procpid > 0 )
                            {
                                upstreamproc[NBupstreamproc] = procpid;
                                NBupstreamproc ++;
                            }
                        }
                    }                   
                    
                }
                else
                {
                    print_pid_mode = PRINT_PID_DEFAULT;
                                       
                    for(int spti = 0; spti < streamCTRLimages[ID].md[0].NBproctrace; spti++)
                      {
                            ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
                            if(inode == inodeselected)
                            {
								if(spti < downstreammin)
								{
									downstreammin = spti;
								}
							}
						}
                    
                }





                //char line[200];
                char string[200];
                int charcnt = 0;        // how many chars are about to be printed
                int linecharcnt = 0;    // keeping track of number of characters in line


                if(DisplayFlag == 1)
                {
                    // print file inode
                    linecharcnt += 1+streamCTRL_print_inode(streamCTRLimages[ID].md[0].inode, upstreaminode, NBupstreaminode, downstreammin);
                    printw(" ");
                }


                charcnt = DispName_NBchar + 1;
                if((dindex == dindexSelected) && ( DisplayDetailLevel == 0 ))
                {
                    attron(A_REVERSE);
                }



                if(DisplayFlag == 1)
                {
                    if(streaminfo[sindex].SymLink == 1)
                    {
                        char namestring[stringmaxlen];

                        snprintf(namestring, stringmaxlen, "%s->%s", streaminfo[sindex].sname,
                                 streaminfo[sindex].linkname);

                        attron(COLOR_PAIR(5));
                        printw("%-*.*s", DispName_NBchar, DispName_NBchar, namestring);
                        attroff(COLOR_PAIR(5));
                    }
                    else
                    {
                        printw("%-*.*s", DispName_NBchar, DispName_NBchar, streaminfo[sindex].sname);
                    }

                    if((int) strlen(streaminfo[sindex].sname) > DispName_NBchar)
                    {
                        attron(COLOR_PAIR(9));
                        printw("+");
                        attroff(COLOR_PAIR(9));
                    }
                    else
                    {
                        printw(" ");
                    }
                    linecharcnt += charcnt;
                }



                DEBUG_TRACEPOINT(" ");



                if((DisplayMode < DISPLAY_MODE_FUSER) && (DisplayFlag == 1))
                {
                    char str[STRINGMAXLEN_DEFAULT];
                    char str1[STRINGMAXLEN_DEFAULT];
                    int j;

                    if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                    {
                        charcnt = sprintf(string, " ???");
                    }
                    else
                    {
                        if(streaminfo[sindex].datatype == _DATATYPE_UINT8)
                        {
                            charcnt = sprintf(string, " UI8");
                        }
                        if(streaminfo[sindex].datatype == _DATATYPE_INT8)
                        {
                            charcnt = sprintf(string, "  I8");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_UINT16)
                        {
                            charcnt = sprintf(string, "UI16");
                        }
                        if(streaminfo[sindex].datatype == _DATATYPE_INT16)
                        {
                            charcnt = sprintf(string, " I16");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_UINT32)
                        {
                            charcnt = sprintf(string, "UI32");
                        }
                        if(streaminfo[sindex].datatype == _DATATYPE_INT32)
                        {
                            charcnt = sprintf(string, " I32");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_UINT64)
                        {
                            charcnt = sprintf(string, "UI64");
                        }
                        if(streaminfo[sindex].datatype == _DATATYPE_INT64)
                        {
                            charcnt = sprintf(string, " I64");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_HALF)
                        {
                            charcnt = sprintf(string, " HLF");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_FLOAT)
                        {
                            charcnt = sprintf(string, " FLT");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_DOUBLE)
                        {
                            charcnt = sprintf(string, " DBL");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_COMPLEX_FLOAT)
                        {
                            charcnt = sprintf(string, "CFLT");
                        }

                        if(streaminfo[sindex].datatype == _DATATYPE_COMPLEX_DOUBLE)
                        {
                            charcnt = sprintf(string, "CDBL");
                        }
                    }
                    linecharcnt += charcnt;
                    if(linecharcnt < wcol)
                    {
                        printw(string);
                    }

                    DEBUG_TRACEPOINT(" ");
                    if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                    {
                        sprintf(str, "???");
                    }
                    else
                    {
                        sprintf(str, " [%3ld", (long) streamCTRLimages[ID].md[0].size[0]);

                        for(j = 1; j < streamCTRLimages[ID].md[0].naxis; j++)
                        {
                            {
                                int slen = snprintf(str1, STRINGMAXLEN_DEFAULT, "%sx%3ld", str,
                                                    (long) streamCTRLimages[ID].md[0].size[j]);
                                if(slen < 1)
                                {
                                    PRINT_ERROR("snprintf wrote <1 char");
                                    abort(); // can't handle this error any other way
                                }
                                if(slen >= STRINGMAXLEN_DEFAULT)
                                {
                                    PRINT_ERROR("snprintf string truncation");
                                    abort(); // can't handle this error any other way
                                }
                            }
                            strcpy(str, str1);
                        }
                        {
                            int slen = snprintf(str1, STRINGMAXLEN_DEFAULT, "%s]", str);
                            if(slen < 1)
                            {
                                PRINT_ERROR("snprintf wrote <1 char");
                                abort(); // can't handle this error any other way
                            }
                            if(slen >= STRINGMAXLEN_DEFAULT)
                            {
                                PRINT_ERROR("snprintf string truncation");
                                abort(); // can't handle this error any other way
                            }
                        }

                        strcpy(str, str1);
                    }

                    DEBUG_TRACEPOINT(" ");


                    charcnt = sprintf(string, "%-*.*s ", DispSize_NBchar, DispSize_NBchar, str);
                    linecharcnt += charcnt;
                    if(linecharcnt < wcol)
                    {
                        printw(string);
                    }

                    if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                    {
                        charcnt = sprintf(string, "???");
                    }
                    else
                    {

                        charcnt = sprintf(string, " %*ld ", Dispcnt0_NBchar, streamCTRLimages[ID].md[0].cnt0);
                    }
                    linecharcnt += charcnt;
                    if(linecharcnt < wcol)
                    {
                        if(streaminfo[sindex].deltacnt0 == 0)
                        {
                            printw(string);
                        }
                        else
                        {
                            attron(COLOR_PAIR(2));
                            printw(string);
                            attroff(COLOR_PAIR(2));
                        }
                    }


                    // creatorPID
                    // ownerPID
                    if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                    {
                        charcnt = sprintf(string, "???");
                    }
                    else
                    {
                        pid_t cpid; // creator PID
                        pid_t opid; // owner PID

                        cpid = streamCTRLimages[ID].md[0].creatorPID;
                        opid = streamCTRLimages[ID].md[0].ownerPID;

                        linecharcnt += streamCTRL_print_procpid(cpid, upstreamproc, NBupstreamproc, print_pid_mode);
                        printw(" ");
                        linecharcnt += streamCTRL_print_procpid(opid, upstreamproc, NBupstreamproc, print_pid_mode);
                        printw(" ");
                        linecharcnt += 2;
                    }



                    // stream update frequency
                    //
                    if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                    {
                        charcnt = sprintf(string, "???");
                    }
                    else
                    {
                        charcnt = sprintf(string, " %*.2f Hz", Dispfreq_NBchar, streaminfo[sindex].updatevalue);
                    }
                    linecharcnt += charcnt;
                    if(linecharcnt < wcol)
                    {
                        printw(string);
                    }

                }

                DEBUG_TRACEPOINT(" ");

                if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                {
                    if((DisplayMode == DISPLAY_MODE_SEMVAL) && (DisplayFlag == 1))   // sem vals
                    {

                        charcnt = sprintf(string, " %3d sems ", streamCTRLimages[ID].md[0].sem);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
                        {
                            printw(string);
                        }

                        int s;
                        for(s = 0; s < streamCTRLimages[ID].md[0].sem; s++)
                        {
                            int semval;
                            sem_getvalue(streamCTRLimages[ID].semptr[s], &semval);
                            charcnt = sprintf(string, " %7d", semval);
                            linecharcnt += charcnt;
                            if(linecharcnt < wcol)
                            {
                                printw(string);
                            }
                        }
                    }
                }

                DEBUG_TRACEPOINT(" ");
                if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                {
                    if((DisplayMode == DISPLAY_MODE_WRITE) && (DisplayFlag == 1))   // sem write PIDs
                    {
                        charcnt = sprintf(string, " %3d sems ", streamCTRLimages[ID].md[0].sem);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
                        {
                            printw(string);
                        }

                        int s;
                        for(s = 0; s < streamCTRLimages[ID].md[0].sem; s++)
                        {
                            pid_t pid = streamCTRLimages[ID].semWritePID[s];
                            linecharcnt += 1 + streamCTRL_print_procpid(pid, upstreamproc, NBupstreamproc, print_pid_mode);
                            printw(" ");

                        }
                    }
                }

                DEBUG_TRACEPOINT(" ");


                if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                {
                    if((DisplayMode == DISPLAY_MODE_READ) && (DisplayFlag == 1))   // sem read PIDs
                    {
                        charcnt = sprintf(string, " %3d sems ", streamCTRLimages[ID].md[0].sem);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
                        {
                            printw(string);
                        }

                        int s;
                        for(s = 0; s < streamCTRLimages[ID].md[0].sem; s++)
                        {
                            pid_t pid = streamCTRLimages[ID].semReadPID[s];
                            linecharcnt += 1 + streamCTRL_print_procpid(pid, upstreamproc, NBupstreamproc, print_pid_mode);
                            printw(" ");
                        }
                    }
                }



                if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                {
                    if((DisplayMode == DISPLAY_MODE_SPTRACE) && (DisplayFlag == 1))   // sem read PIDs
                    {
                        charcnt = sprintf(string, " %2d ", streamCTRLimages[ID].md[0].NBproctrace);
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
                        {
                            printw(string);
                        }

                        for(int spti = 0; spti < streamCTRLimages[ID].md[0].NBproctrace; spti++)
                        {
                            ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
                            int   sem   = streamCTRLimages[ID].streamproctrace[spti].trigsemindex;
                            pid_t pid   = streamCTRLimages[ID].streamproctrace[spti].procwrite_PID;
                            
                            


                            switch( streamCTRLimages[ID].streamproctrace[spti].triggermode)
                            {
                            case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
                                charcnt = sprintf(string, "(%7lu IM ", inode);
                                break;

                            case PROCESSINFO_TRIGGERMODE_CNT0:
                                charcnt = sprintf(string, "(%7lu C0 ", inode);
                                break;

                            case PROCESSINFO_TRIGGERMODE_CNT1:
                                charcnt = sprintf(string, "(%7lu C1 ", inode);
                                break;

                            case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
                                charcnt = sprintf(string, "(%7lu %02d ", inode, sem);
                                break;

                            case PROCESSINFO_TRIGGERMODE_DELAY:
                                charcnt = sprintf(string, "(%7lu DL ", inode);
                                break;

                            default:
                                charcnt = sprintf(string, "(%7lu ?? ", inode);
                                break;
                            }
                            linecharcnt += charcnt + 2;
                            if(linecharcnt < wcol)
                            {
                                printw(string);
                            }

                            linecharcnt += 3 + streamCTRL_print_procpid(pid, upstreamproc, NBupstreamproc, print_pid_mode);
                            printw(")> ");
                        }
                    
						if(DisplayDetailLevel == 1)
						{
							printw("\n");
							streamCTRL_print_SPTRACE_details(streamCTRLimages, ID, upstreamproc, NBupstreamproc, PRINT_PID_DEFAULT);
						}                    
                    }
                }



                if((DisplayMode == DISPLAY_MODE_FUSER)
                        && (DisplayFlag == 1))   // list processes that are accessing streams
                {
                    if(streaminfoproc.fuserUpdate == 2)
                    {
                        streaminfo[sindex].streamOpenPID_status = 0; // not scanned
                    }


                    DEBUG_TRACEPOINT(" ");

                    int pidIndex;

                    switch(streaminfo[sindex].streamOpenPID_status)
                    {

                    case 1:
                        streaminfo[sindex].streamOpenPID_cnt1 = 0;
                        for(pidIndex = 0; pidIndex < streaminfo[sindex].streamOpenPID_cnt ; pidIndex++)
                        {
                            pid_t pid = streaminfo[sindex].streamOpenPID[pidIndex];
                            linecharcnt += streamCTRL_print_procpid(pid, upstreamproc, NBupstreamproc, print_pid_mode);

                            if((getpgid(pid) >= 0) && (pid != getpid()))
                            {

                                charcnt = sprintf(string, ":%-*.*s",
                                                  PIDnameStringLen, PIDnameStringLen, PIDname_array[pid]);
                                linecharcnt += charcnt;
                                if(linecharcnt < wcol)
                                {
                                    printw(string);
                                }

                                streaminfo[sindex].streamOpenPID_cnt1 ++;
                            }
                        }

                        //const chtype * lstring1 = "This is a test";
                        //addchstr(lstring1);

                        break;

                    case 2:
                        charcnt = sprintf(string, "FAILED");
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
                        {
                            printw(string);
                        }
                        break;

                    default:
                        charcnt = sprintf(string, "NOT SCANNED");
                        linecharcnt += charcnt;
                        if(linecharcnt < wcol)
                        {
                            printw(string);
                        }
                        break;

                    }

                }

                DEBUG_TRACEPOINT(" ");

                if(DisplayFlag == 1)
                {
                    if(dindex == dindexSelected)
                    {
                        attroff(A_REVERSE);
                    }

                    if(linecharcnt > wcol)
                    {
                        attron(COLOR_PAIR(9));
                        printw("+");
                        attroff(COLOR_PAIR(9));
                    }
                    printw("\n");
                }


                DEBUG_TRACEPOINT(" ");

#ifndef STANDALONE
                if(streaminfoproc.fuserUpdate == 1)
                {
                    //      refresh();
                    if(data.signal_INT == 1)   // stop scan
                    {
                        streaminfoproc.fuserUpdate = 2;     // complete loop without scan
                        data.signal_INT = 0; // reset
                    }
                }
#endif
            }



        }

        DEBUG_TRACEPOINT(" ");


        refresh();

        DEBUG_TRACEPOINT(" ");

        loopcnt++;
#ifndef STANDALONE
        if((data.signal_TERM == 1) || (data.signal_INT == 1) || (data.signal_ABRT == 1)
                || (data.signal_BUS == 1) || (data.signal_SEGV == 1) || (data.signal_HUP == 1)
                || (data.signal_PIPE == 1))
        {
            loopOK = 0;
        }
#endif

        DEBUG_TRACEPOINT(" ");
    }


    endwin();

    streaminfoproc.loop = 0;
    pthread_join(threadscan, NULL);

    for(int pidi = 0; pidi < PIDmax; pidi++)
    {
        if(PIDname_array[pidi] != NULL)
        {
            free(PIDname_array[pidi]);
        }
    }
    free(PIDname_array);

    for(imageID ID = 0; ID < streamNBID_MAX; ID++)
    {
        if(streamCTRLimages[ID].used == 1)
        {
            ImageStreamIO_closeIm(&streamCTRLimages[ID]);
        }
    }

    free(streamCTRLimages);
    free(streaminfo);
    free(upstreaminode);
    free(upstreamproc);


    fflush(stderr);
    dup2(backstderr, STDERR_FILENO);
    close(backstderr);

    remove(newstderrfname);

    DEBUG_TRACEPOINT(" ");



    return EXIT_SUCCESS;
}

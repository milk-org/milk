#include <sys/file.h>
#include <sys/mman.h> // mmap()
#include <sys/types.h>
#include <sys/stat.h>

#include "CLIcore.h"
#include <processtools.h>

#include "processinfo_shm_list_create.h"
#include "processinfo_procdirname.h"


#define FILEMODE 0666

extern PROCESSINFOLIST *pinfolist;

/**
 * Create PROCESSINFO structure in shared memory
 *
 * The structure holds real-time information about a process, so its status can be monitored and controlled
 * See structure PROCESSINFO in CLLIcore.h for details
 *
*/

PROCESSINFO *processinfo_shm_create(const char *pname, int CTRLval)
{
    DEBUG_TRACE_FSTART();

    size_t       sharedsize = 0; // shared memory size in bytes
    int          SM_fd;          // shared memory file descriptor
    PROCESSINFO *pinfo;

    static int LogFileCreated = 0;
    // toggles to 1 when created. To avoid re-creating file on same process

    sharedsize = sizeof(PROCESSINFO);

    char  SM_fname[STRINGMAXLEN_FULLFILENAME];
    pid_t PID;

    PID = getpid();

    DEBUG_TRACEPOINT("create/update pinfolist");
    long pindex;
    pindex = processinfo_shm_list_create();

    DEBUG_TRACEPOINT("index = %ld", pindex);

    pinfolist->PIDarray[pindex] = PID;

    DEBUG_TRACEPOINT(" ");

    strncpy(pinfolist->pnamearray[pindex],
            pname,
            STRINGMAXLEN_PROCESSINFO_NAME - 1);

    DEBUG_TRACEPOINT("getting procdname");
    char procdname[STRINGMAXLEN_FULLFILENAME];
    processinfo_procdirname(procdname);


    WRITE_FULLFILENAME(SM_fname,
                       "%s/proc.%s.%06d.shm",
                       procdname,
                       pname,
                       (int) PID);

    DEBUG_TRACEPOINT("SM_fname = %s", SM_fname);

    umask(0);
    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t) FILEMODE);
    if (SM_fd == -1)
    {
        perror("Error opening file for writing");
        exit(0);
    }

    int result;
    result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
    if (result == -1)
    {
        close(SM_fd);
        fprintf(stderr, "Error calling lseek() to 'stretch' the file");
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if (result != 1)
    {
        close(SM_fd);
        perror("Error writing last byte of the file");
        exit(0);
    }

    pinfo = (PROCESSINFO *)
        mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (pinfo == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        exit(0);
    }

    DEBUG_TRACEPOINT("created processinfo entry at %s\n", SM_fname);
    DEBUG_TRACEPOINT("shared memory space = %ld bytes\n", sharedsize);

    clock_gettime(CLOCK_REALTIME, &pinfo->createtime);
    pinfolist->createtime[pindex] =
        1.0 * pinfo->createtime.tv_sec + 1.0e-9 * pinfo->createtime.tv_nsec;

    strcpy(pinfo->name, pname);

    pinfolist->active[pindex] = 1;

    char  tmuxname[100];
    FILE *fpout;
    int   notmux = 0;

    fpout = popen("tmuxsessionname", "r");
    if (fpout == NULL)
    {
        printf("WARNING: cannot run command \"tmuxsessionname\"\n");
    }
    else
    {
        if (fgets(tmuxname, 100, fpout) == NULL)
        {
            //printf("WARNING: fgets error\n");
            notmux = 1;
        }
        pclose(fpout);
    }
    // remove line feed
    if (strlen(tmuxname) > 0)
    {
        //  printf("tmux name : %s\n", tmuxname);
        //  printf("len: %d\n", (int) strlen(tmuxname));
        fflush(stdout);

        if (tmuxname[strlen(tmuxname) - 1] == '\n')
        {
            tmuxname[strlen(tmuxname) - 1] = '\0';
        }
        else
        {
            printf("tmux name empty\n");
        }
    }
    else
    {
        notmux = 1;
    }

    if (notmux == 1)
    {
        sprintf(tmuxname, " ");
    }

    // force last char to be term, just in case
    tmuxname[99] = '\0';

    DEBUG_TRACEPOINT("tmux name : %s\n", tmuxname);

    strncpy(pinfo->tmuxname, tmuxname, 100);

    // set control value (default 0)
    // 1 : pause
    // 2 : increment single step (will go back to 1)
    // 3 : exit loop
    pinfo->CTRLval = CTRLval;

    pinfo->MeasureTiming = 1;

    // initialize timer indexes and counters
    pinfo->timerindex      = 0;
    pinfo->timingbuffercnt = 0;

    // disable timer limit feature
    pinfo->dtiter_limit_enable = 0;
    pinfo->dtexec_limit_enable = 0;

    data.pinfo = pinfo;
    pinfo->PID = PID;

    // create logfile
    //char logfilename[300];
    struct timespec tnow;

    clock_gettime(CLOCK_REALTIME, &tnow);

    {
        int slen = snprintf(pinfo->logfilename,
                            STRINGMAXLEN_PROCESSINFO_LOGFILENAME,
                            "%s/proc.%s.%06d.%09ld.logfile",
                            procdname,
                            pinfo->name,
                            (int) pinfo->PID,
                            tnow.tv_sec);
        if (slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort();
        }
        if (slen >= STRINGMAXLEN_PROCESSINFO_LOGFILENAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort();
        }
    }

    if (LogFileCreated == 0)
    {
        pinfo->logFile = fopen(pinfo->logfilename, "w");
        LogFileCreated = 1;
    }

    char msgstring[300];
    sprintf(msgstring, "LOG START %s", pinfo->logfilename);
    processinfo_WriteMessage(pinfo, msgstring);

    DEBUG_TRACE_FEXIT();

    return pinfo;
}

/**
 * @file function_parameters.c
 * @brief Tools to help expose and control function parameters
 * 
 * @see @ref page_FunctionParameterStructure
 * 
 * 
 */


#define _GNU_SOURCE


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
#include <malloc.h>
#include <math.h>
#include <limits.h>
#include <sys/syscall.h> // needed for tid = syscall(SYS_gettid);
#include <errno.h>

#include <time.h>

#include <sys/types.h>

#include <ncurses.h>
#include <dirent.h>

#include <pthread.h>
#include <fcntl.h> // for open
#include <unistd.h> // for close
#include <sys/mman.h> // mmap
#include <sys/stat.h> // fstat
#include <signal.h>
#include <unistd.h> // usleep


#ifndef STANDALONE
#include <00CORE/00CORE.h>
#include <CommandLineInterface/CLIcore.h>
#include "info/info.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "COREMOD_memory/COREMOD_memory.h"
#define SHAREDSHMDIR data.shmdir
#else
#include "standalone_dependencies.h"
#endif

#include "function_parameters.h"


/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */



#define NB_FPS_MAX 100
#define NB_KEYWNODE_MAX 10000

#define MAXNBLEVELS 20


// regular color codes
#define RESET   "\033[0m"
#define BLACK   "\033[0;30m"      /* Black */
#define RED     "\033[0;31m"      /* Red */
#define GREEN   "\033[0;32m"      /* Green */
#define YELLOW  "\033[0;33m"      /* Yellow */
#define BLUE    "\033[0;34m"      /* Blue */
#define MAGENTA "\033[0;35m"      /* Magenta */
#define CYAN    "\033[0;36m"      /* Cyan */
#define WHITE   "\033[0;37m"      /* White */

// Bold
#define BOLDBLACK   "\033[1;30m"      /* Bold Black */
#define BOLDRED     "\033[1;31m"      /* Bold Red */
#define BOLDGREEN   "\033[1;32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1;33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1;34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1;35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1;36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1;37m"      /* Bold White */

// Blink
#define BLINKBLACK   "\033[5;30m"      /* Blink Black */
#define BLINKRED     "\033[5;31m"      /* Blink  Red */
#define BLINKGREEN   "\033[5;32m"      /* Blink  Green */
#define BLINKYELLOW  "\033[5;33m"      /* Blink  Yellow */
#define BLINKBLUE    "\033[5;34m"      /* Blink  Blue */
#define BLINKMAGENTA "\033[5;35m"      /* Blink  Magenta */
#define BLINKCYAN    "\033[5;36m"      /* Blink  Cyan */
#define BLINKWHITE   "\033[5;37m"      /* Blink  White */


// Blink High Intensity
#define BLINKHIBLACK   "\033[5;90m"      /* Blink Black */
#define BLINKHIRED     "\033[5;91m"      /* Blink  Red */
#define BLINKHIGREEN   "\033[5;92m"      /* Blink  Green */
#define BLINKHIYELLOW  "\033[5;93m"      /* Blink  Yellow */
#define BLINKHIBLUE    "\033[5;94m"      /* Blink  Blue */
#define BLINKHIMAGENTA "\033[5;95m"      /* Blink  Magenta */
#define BLINKHICYAN    "\033[5;96m"      /* Blink  Cyan */
#define BLINKHIWHITE   "\033[5;97m"      /* Blink  White */


// Underline
#define UNDERLINEBLACK   "\033[4;30m"      /* Bold Black */
#define UNDERLINERED     "\033[4;31m"      /* Bold Red */
#define UNDERLINEGREEN   "\033[4;32m"      /* Bold Green */
#define UNDERLINEYELLOW  "\033[4;33m"      /* Bold Yellow */
#define UNDERLINEBLUE    "\033[4;34m"      /* Bold Blue */
#define UNDERLINEMAGENTA "\033[4;35m"      /* Bold Magenta */
#define UNDERLINECYAN    "\033[4;36m"      /* Bold Cyan */
#define UNDERLINEWHITE   "\033[4;37m"      /* Bold White */

// High Intensity
#define HIBLACK   "\033[0;90m"      /* Black */
#define HIRED     "\033[0;91m"      /* Red */
#define HIGREEN   "\033[0;92m"      /* Green */
#define HIYELLOW  "\033[0;93m"      /* Yellow */
#define HIBLUE    "\033[0;94m"      /* Blue */
#define HIMAGENTA "\033[0;95m"      /* Magenta */
#define HICYAN    "\033[0;96m"      /* Cyan */
#define HIWHITE   "\033[0;97m"      /* White */

// Bold High Intensity
#define BOLDHIBLACK   "\033[1;90m"      /* Black */
#define BOLDHIRED     "\033[1;91m"      /* Red */
#define BOLDHIGREEN   "\033[1;92m"      /* Green */
#define BOLDHIYELLOW  "\033[1;93m"      /* Yellow */
#define BOLDHIBLUE    "\033[1;94m"      /* Blue */
#define BOLDHIMAGENTA "\033[1;95m"      /* Magenta */
#define BOLDHICYAN    "\033[1;96m"      /* Cyan */
#define BOLDHIWHITE   "\033[1;97m"      /* White */





// Background
#define BACKGROUNDBLACK   "\033[40m"      /* Black */
#define BACKGROUNDRED     "\033[41m"      /* Red */
#define BACKGROUNDGREEN   "\033[42m"      /* Green */
#define BACKGROUNDYELLOW  "\033[43m"      /* Yellow */
#define BACKGROUNDBLUE    "\033[44m"      /* Blue */
#define BACKGROUNDMAGENTA "\033[45m"      /* Magenta */
#define BACKGROUNDCYAN    "\033[46m"      /* Cyan */
#define BACKGROUNDWHITE   "\033[47m"      /* White */

// High Intensity background
#define BACKGROUNDHIBLACK   "\033[0;100m"      /* Black */
#define BACKGROUNDHIRED     "\033[0;101m"      /* Red */
#define BACKGROUNDHIGREEN   "\033[0;102m"      /* Green */
#define BACKGROUNDHIYELLOW  "\033[0;103m"      /* Yellow */
#define BACKGROUNDHIBLUE    "\033[0;104m"      /* Blue */
#define BACKGROUNDHIMAGENTA "\033[0;105m"      /* Magenta */
#define BACKGROUNDHICYAN    "\033[0;106m"      /* Cyan */
#define BACKGROUNDHIWHITE   "\033[0;107m"      /* White */




/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */


static int wrow, wcol;

#define MAX_NB_CHILD 500

typedef struct
{
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int  keywordlevel;

    int parent_index;

    int NBchild;
    int child[MAX_NB_CHILD];

    int leaf; // 1 if this is a leaf (no child)
    int fpsindex;
    int pindex;
    

} KEYWORD_TREE_NODE;




/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */



errno_t function_parameter_struct_shmdirname(char *shmdname)
{
    int shmdirOK = 0;
    DIR *tmpdir;
    static unsigned long functioncnt = 0;
    static char shmdname_static[STRINGMAXLEN_SHMDIRNAME];

    if(functioncnt == 0)
    {
        functioncnt++; // ensure we only run this once, and then retrieve stored result from shmdname_static

        // first, we try the env variable if it exists
        char* MILK_SHM_DIR = getenv("MILK_SHM_DIR");
        if(MILK_SHM_DIR != NULL) {
            printf(" [ MILK_SHM_DIR ] is '%s'\n", MILK_SHM_DIR);

            {
                int slen = snprintf(shmdname, STRINGMAXLEN_SHMDIRNAME, "%s", MILK_SHM_DIR);
                if(slen<1) {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if(slen >= STRINGMAXLEN_SHMDIRNAME) {
                    PRINT_ERROR("snprintf string truncation");
                    abort(); // can't handle this error any other way
                }
            }

            // does this direcory exist ?
            tmpdir = opendir(shmdname);
            if(tmpdir) // directory exits
            {
                shmdirOK = 1;
                closedir(tmpdir);
            }
            else
            {
                abort();
            }
        }

        // second, we try SHAREDSHMDIR default
        if(shmdirOK == 0)
        {
            tmpdir = opendir(SHAREDSHMDIR);
            if(tmpdir) // directory exits
            {
                {
                    int slen = snprintf(shmdname, STRINGMAXLEN_SHMDIRNAME, "%s", SHAREDSHMDIR);
                    if(slen<1) {
                        PRINT_ERROR("snprintf wrote <1 char");
                        abort(); // can't handle this error any other way
                    }
                    if(slen >= STRINGMAXLEN_SHMDIRNAME) {
                        PRINT_ERROR("snprintf string truncation");
                        abort(); // can't handle this error any other way
                    }
                }


                shmdirOK = 1;
                closedir(tmpdir);
            }
        }

        // if all above fails, set to /tmp
        if(shmdirOK == 0)
        {
            tmpdir = opendir("/tmp");
            if ( !tmpdir )
                exit(EXIT_FAILURE);
            else
            {
                sprintf(shmdname, "/tmp");
                shmdirOK = 1;
                closedir(tmpdir);
            }
        }


        {
            int slen = snprintf(shmdname_static, STRINGMAXLEN_SHMDIRNAME, "%s", shmdname); // keep it memory
            if(slen<1) {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_SHMDIRNAME) {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }
    }
    else {
        {
            int slen = snprintf(shmdname, STRINGMAXLEN_SHMDIRNAME, "%s", shmdname_static);
            if(slen<1) {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_SHMDIRNAME) {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

    }

    return RETURN_SUCCESS;
}








/**
 *
 * ## Purpose
 *
 * Construct FPS name and set FPSCMDCODE from command line function call
 *
 *
 */

errno_t function_parameter_getFPSname_from_CLIfunc(
    char     *fpsname_default
)
{
    // Check if function will be executed through FPS interface
    // set to 0 as default (no FPS)
    data.FPS_CMDCODE = 0;

    // if using FPS implementation, FPSCMDCODE will be set to != 0
    if(CLI_checkarg(1, 5) == 0) {
        // check that first arg is a string
        // if it isn't, the non-FPS implementation should be called

        // check if recognized FPSCMDCODE
        if(strcmp(data.cmdargtoken[1].val.string, "_FPSINIT_") == 0) {  // Initialize FPS
            data.FPS_CMDCODE = FPSCMDCODE_FPSINIT;
        }
        else if (strcmp(data.cmdargtoken[1].val.string, "_CONFSTART_") == 0) {  // Start conf process
            data.FPS_CMDCODE = FPSCMDCODE_CONFSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string, "_CONFSTOP_") == 0) { // Stop conf process
            data.FPS_CMDCODE = FPSCMDCODE_CONFSTOP;
        }
        else if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTART_") == 0) { // Run process
            data.FPS_CMDCODE = FPSCMDCODE_RUNSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTOP_") == 0) { // Stop process
            data.FPS_CMDCODE = FPSCMDCODE_RUNSTOP;
        }
    }


    // if recognized FPSCMDCODE, use FPS implementation
    if(data.FPS_CMDCODE != 0) {
        // ===============================
        //     SET FPS INTERFACE NAME
        // ===============================

        // if main CLI process has been named with -n option, than use the process name to construct fpsname
        if(data.processnameflag == 1) {
            // Automatically set fps name to be process name up to first instance of character '.'
            strcpy(data.FPS_name, data.processname0);
        }
        else { // otherwise, construct name as follows

            // Adopt default name for fpsname
            int slen = snprintf(data.FPS_name, FUNCTION_PARAMETER_STRMAXLEN, "%s", fpsname_default);
            if(slen < 1) {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= FUNCTION_PARAMETER_STRMAXLEN) {
                PRINT_ERROR("snprintf string truncation.\n"
                            "Full string  : %s\n"
                            "Truncated to : %s",
                            fpsname_default,
                            data.FPS_name);
                abort(); // can't handle this error any other way
            }


            // By convention, if there are optional arguments,
            // they should be appended to the default fps name
            //
            int argindex = 2; // start at arg #2
            while(strlen(data.cmdargtoken[argindex].val.string) > 0) {
                char fpsname1[FUNCTION_PARAMETER_STRMAXLEN];

                int slen = snprintf(fpsname1, FUNCTION_PARAMETER_STRMAXLEN,
                                    "%s-%s", data.FPS_name, data.cmdargtoken[2].val.string);
                if(slen < 1) {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if(slen >= FUNCTION_PARAMETER_STRMAXLEN) {
                    PRINT_ERROR("snprintf string truncation.\n"
                                "Full string  : %s-%s\n"
                                "Truncated to : %s",
                                data.FPS_name,
                                data.cmdargtoken[argindex].val.string,
                                fpsname1);
                    abort(); // can't handle this error any other way
                }

                strncpy(data.FPS_name, fpsname1, FUNCTION_PARAMETER_STRMAXLEN);
                argindex ++;
            }
        }

    }
    
    return RETURN_SUCCESS;
}




errno_t function_parameter_execFPScmd()
{
    if(data.FPS_CMDCODE == FPSCMDCODE_FPSINIT) { // Initialize FPS
        data.FPS_CONFfunc();
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_CONFSTART) {  // Start CONF process
        data.FPS_CONFfunc();
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_CONFSTOP) { // Stop CONF process
        data.FPS_CONFfunc();
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_RUNSTART) { // Start RUN process
        data.FPS_RUNfunc();
        return RETURN_SUCCESS;
    }

    if(data.FPS_CMDCODE == FPSCMDCODE_RUNSTOP) { // Stop RUN process
        data.FPS_CONFfunc();
        return RETURN_SUCCESS;
    }

    return RETURN_SUCCESS;
}











errno_t function_parameter_struct_create(
    int NBparamMAX,
    const char *name
)
{
    int index;
    char *mapv;
    FUNCTION_PARAMETER_STRUCT fps;

    //  FUNCTION_PARAMETER_STRUCT_MD *funcparammd;
    //  FUNCTION_PARAMETER *funcparamarray;

    char SM_fname[200];
    size_t sharedsize = 0; // shared memory size in bytes
    int SM_fd; // shared memory file descriptor

    char shmdname[200];
    function_parameter_struct_shmdirname(shmdname);

    if(snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", shmdname, name)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    remove(SM_fname);

    printf("Creating file %s, holding NBparamMAX = %d\n", SM_fname, NBparamMAX);
    fflush(stdout);

    sharedsize = sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    sharedsize += sizeof(FUNCTION_PARAMETER)*NBparamMAX;

    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (SM_fd == -1) {
        perror("Error opening file for writing");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    fps.SMfd = SM_fd;

    int result;
    result = lseek(SM_fd, sharedsize-1, SEEK_SET);
    if (result == -1) {
        close(SM_fd);
        printf("ERROR [%s %s %d]: Error calling lseek() to 'stretch' the file\n", __FILE__, __func__, __LINE__);
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if (result != 1) {
        close(SM_fd);
        perror("Error writing last byte of the file");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    fps.md = (FUNCTION_PARAMETER_STRUCT_MD*) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (fps.md == MAP_FAILED) {
        close(SM_fd);
        perror("Error mmapping the file");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }
    //funcparamstruct->md = funcparammd;

    mapv = (char*) fps.md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps.parray = (FUNCTION_PARAMETER*) mapv;



    printf("shared memory space = %ld bytes\n", sharedsize); //TEST


    fps.md->NBparamMAX = NBparamMAX;

    for(index=0; index<NBparamMAX; index++)
    {
        fps.parray[index].fpflag = 0; // not active
        fps.parray[index].cnt0 = 0;   // update counter
    }


    strcpy(fps.md->name, name);



    char cwd[FPS_CWD_STRLENMAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        strncpy(fps.md->fpsdirectory, cwd, FPS_CWD_STRLENMAX);
    } else {
        perror("getcwd() error");
        return 1;
    }

    strncpy(fps.md->sourcefname, "NULL", FPS_SRCDIR_STRLENMAX);
    fps.md->sourceline = 0;



    fps.md->signal     = (uint64_t) FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
    fps.md->confwaitus = (uint64_t) 1000; // 1 kHz default
    fps.md->msgcnt = 0;

    munmap(fps.md, sharedsize);


    return EXIT_SUCCESS;
}





/**
 *
 * ## Purpose
 *
 * Connect to function parameter structure
 *
 *
 * ## Arguments
 *
 * fpsconnectmode can take following value
 *
 * FPSCONNECT_SIMPLE : simple connect, don't try load streams
 * FPSCONNECT_CONF   : connect as CONF process
 * FPSCONNECT_RUN    : connect as RUN process
 *
 */


long function_parameter_struct_connect(
    const char *name,
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsconnectmode
) {
    int stringmaxlen = 500;
    char SM_fname[stringmaxlen];
    int SM_fd; // shared memory file descriptor
    long NBparamMAX;
    //    long NBparamActive;
    char *mapv;

    char shmdname[stringmaxlen];

    if(fps->SMfd > -1) {
        printf("[%s %s %d] ERROR: file descriptor already allocated : %d\n", __FILE__, __func__, __LINE__, fps->SMfd);
        //	exit(0);
    }


    function_parameter_struct_shmdirname(shmdname);

    if(snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", shmdname, name)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    printf("File : %s\n", SM_fname);
    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd == -1) {
        printf("ERROR [%s %s %d]: cannot connect to %s\n", __FILE__, __func__, __LINE__, SM_fname);
        return(-1);
    }
    else
    {
        fps->SMfd = SM_fd;
    }


    struct stat file_stat;
    fstat(SM_fd, &file_stat);


    fps->md = (FUNCTION_PARAMETER_STRUCT_MD *) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if(fps->md == MAP_FAILED) {
        close(SM_fd);
        perror("Error mmapping the file");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(EXIT_FAILURE);
    }

    if(fpsconnectmode == FPSCONNECT_CONF) {
        fps->md->confpid = getpid();    // write process PID into FPS
    }

    if(fpsconnectmode == FPSCONNECT_RUN) {
        fps->md->runpid = getpid();    // write process PID into FPS
    }

    mapv = (char *) fps->md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps->parray = (FUNCTION_PARAMETER *) mapv;

    //	NBparam = (int) (file_stat.st_size / sizeof(FUNCTION_PARAMETER));
    NBparamMAX = fps->md->NBparamMAX;
    printf("[%s %5d] Connected to %s, %ld entries\n", __FILE__, __LINE__, SM_fname, NBparamMAX);
    fflush(stdout);


    // decompose full name into pname and indices
    int NBi = 0;
    char tmpstring[stringmaxlen];
    char tmpstring1[stringmaxlen];
    char *pch;


    strncpy(tmpstring, name, stringmaxlen);
    NBi = -1;
    pch = strtok(tmpstring, "-");
    while(pch != NULL) {
        strncpy(tmpstring1, pch, stringmaxlen);

        if(NBi == -1) {
            //            strncpy(fps->md->pname, tmpstring1, stringmaxlen);
            if(snprintf(fps->md->pname, FPS_PNAME_STRMAXLEN, "%s", tmpstring1)< 0 ) {
                PRINT_ERROR("snprintf error");
            }
        }

        if((NBi >= 0) && (NBi < 10)) {
            if(snprintf(fps->md->nameindexW[NBi], 16, "%s", tmpstring1)< 0 ) {
                PRINT_ERROR("snprintf error");
            }
            //strncpy(fps->md->nameindexW[NBi], tmpstring1, 16);
        }

        NBi++;
        pch = strtok(NULL, "-");
    }


    fps->md->NBnameindex = NBi;
    function_parameter_printlist(fps->parray, NBparamMAX);


    if((fpsconnectmode == FPSCONNECT_CONF) || (fpsconnectmode == FPSCONNECT_RUN)) {
        // load streams
        int pindex;
        for(pindex = 0; pindex < NBparamMAX; pindex++) {
            if( (fps->parray[pindex].fpflag & FPFLAG_ACTIVE) && (fps->parray[pindex].fpflag & FPFLAG_USED) && (fps->parray[pindex].type & FPTYPE_STREAMNAME)) {
                functionparameter_LoadStream(fps, pindex, fpsconnectmode);
            }
        }
    }

    return(NBparamMAX);
}





int function_parameter_struct_disconnect(FUNCTION_PARAMETER_STRUCT *funcparamstruct)
{
    int NBparamMAX;

    NBparamMAX = funcparamstruct->md->NBparamMAX;
    //funcparamstruct->md->NBparam = 0;
    funcparamstruct->parray = NULL;
    munmap(funcparamstruct->md, sizeof(FUNCTION_PARAMETER_STRUCT_MD)+sizeof(FUNCTION_PARAMETER)*NBparamMAX);
    close(funcparamstruct->SMfd);
    funcparamstruct->SMfd = -1;

    return(0);
}






//
// stand-alone function to set parameter value
//
int function_parameter_SetValue_int64(char *keywordfull, long val)
{
    FUNCTION_PARAMETER_STRUCT fps;
    char tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel = 0;
    char *pch;


    // break full keyword into keywords
    strncpy(tmpstring, keywordfull, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL);
    keywordlevel = 0;
    pch = strtok (tmpstring, ".");
    while (pch != NULL)
    {
        strncpy(keyword[keywordlevel], pch, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN);
        keywordlevel++;
        pch = strtok (NULL, ".");
    }

    function_parameter_struct_connect(keyword[9], &fps, FPSCONNECT_SIMPLE);

    int pindex = functionparameter_GetParamIndex(&fps, keywordfull);


    fps.parray[pindex].val.l[0] = val;

    function_parameter_struct_disconnect(&fps);

    return EXIT_SUCCESS;
}









int function_parameter_printlist(
    FUNCTION_PARAMETER  *funcparamarray,
    long NBparamMAX
)
{
    long pindex = 0;
    long pcnt = 0;

    printf("\n");
    for(pindex=0; pindex<NBparamMAX; pindex++)
    {
        if(funcparamarray[pindex].fpflag & FPFLAG_ACTIVE)
        {
            printf("Parameter %4ld : %s\n", pindex, funcparamarray[pindex].keywordfull);
            /*for(int kl=0; kl< funcparamarray[pindex].keywordlevel; kl++)
            	printf("  %s", funcparamarray[pindex].keyword[kl]);
            printf("\n");*/
            printf("    %s\n", funcparamarray[pindex].description);

            // STATUS FLAGS
            printf("    STATUS FLAGS (0x%02hhx) :", (int) funcparamarray[pindex].fpflag);
            if(funcparamarray[pindex].fpflag & FPFLAG_ACTIVE)
                printf(" ACTIVE");
            if(funcparamarray[pindex].fpflag & FPFLAG_USED)
                printf(" USED");
            if(funcparamarray[pindex].fpflag & FPFLAG_VISIBLE)
                printf(" VISIBLE");
            if(funcparamarray[pindex].fpflag & FPFLAG_WRITE)
                printf(" WRITE");
            if(funcparamarray[pindex].fpflag & FPFLAG_WRITECONF)
                printf(" WRITECONF");
            if(funcparamarray[pindex].fpflag & FPFLAG_WRITERUN)
                printf(" WRITERUN");
            if(funcparamarray[pindex].fpflag & FPFLAG_LOG)
                printf(" LOG");
            if(funcparamarray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
                printf(" SAVEONCHANGE");
            if(funcparamarray[pindex].fpflag & FPFLAG_SAVEONCLOSE)
                printf(" SAVEONCLOSE");
            if(funcparamarray[pindex].fpflag & FPFLAG_MINLIMIT)
                printf(" MINLIMIT");
            if(funcparamarray[pindex].fpflag & FPFLAG_MAXLIMIT)
                printf(" MAXLIMIT");
            if(funcparamarray[pindex].fpflag & FPFLAG_CHECKSTREAM)
                printf(" CHECKSTREAM");
            if(funcparamarray[pindex].fpflag & FPFLAG_IMPORTED)
                printf(" IMPORTED");
            if(funcparamarray[pindex].fpflag & FPFLAG_FEEDBACK)
                printf(" FEEDBACK");
            if(funcparamarray[pindex].fpflag & FPFLAG_ERROR)
                printf(" ERROR");
            if(funcparamarray[pindex].fpflag & FPFLAG_ONOFF)
                printf(" ONOFF");
            printf("\n");

            // DATA TYPE
            //			printf("    TYPE : 0x%02hhx\n", (int) funcparamarray[pindex].type);
            if(funcparamarray[pindex].type & FPTYPE_UNDEF)
                printf("    TYPE = UNDEF\n");
            if(funcparamarray[pindex].type & FPTYPE_INT64)
            {
                printf("    TYPE  = INT64\n");
                printf("    VALUE = %ld\n", (long) funcparamarray[pindex].val.l[0]);
            }
            if(funcparamarray[pindex].type & FPTYPE_FLOAT64)
                printf("    TYPE = FLOAT64\n");
            if(funcparamarray[pindex].type & FPTYPE_PID)
                printf("    TYPE = PID\n");
            if(funcparamarray[pindex].type & FPTYPE_TIMESPEC)
                printf("    TYPE = TIMESPEC\n");
            if(funcparamarray[pindex].type & FPTYPE_FILENAME)
                printf("    TYPE = FILENAME\n");
            if(funcparamarray[pindex].type & FPTYPE_DIRNAME)
                printf("    TYPE = DIRNAME\n");
            if(funcparamarray[pindex].type & FPTYPE_STREAMNAME)
                printf("    TYPE = STREAMNAME\n");
            if(funcparamarray[pindex].type & FPTYPE_STRING)
                printf("    TYPE = STRING\n");
            if(funcparamarray[pindex].type & FPTYPE_ONOFF)
                printf("    TYPE = ONOFF\n");
            if(funcparamarray[pindex].type & FPTYPE_FPSNAME)
                printf("    TYPE = FPSNAME\n");

            pcnt ++;
        }
    }
    printf("\n");
    printf("%ld/%ld active parameters\n", pcnt, NBparamMAX);
    printf("\n");

    return 0;
}






int functionparameter_GetFileName(
    FUNCTION_PARAMETER_STRUCT *fps,
    FUNCTION_PARAMETER *fparam,
    char *outfname,
    char *tagname
) {
    int stringmaxlen = 500;
    char fname[stringmaxlen];
    char fname1[stringmaxlen];
    char command[stringmaxlen];
    int l;

    if(snprintf(fname, stringmaxlen, "%s/fpsconf", fps->md->fpsdirectory)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    if(snprintf(command, stringmaxlen, "mkdir -p %s", fname)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    if(system(command) != 0) {
        PRINT_ERROR("system() returns non-zero value");
    }

    for(l = 0; l < fparam->keywordlevel - 1; l++) {
        if(snprintf(fname1, stringmaxlen, "/%s", fparam->keyword[l])< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        strcat(fname, fname1);
        if(snprintf(command, stringmaxlen, "mkdir -p %s", fname)< 0 ) {
            PRINT_ERROR("snprintf error");
        }

        if(system(command) != 0) {
            PRINT_ERROR("system() returns non-zero value");
        }
    }

    if(snprintf(fname1, stringmaxlen, "/%s.%s.txt", fparam->keyword[l], tagname)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    strcat(fname, fname1);
    strcpy(outfname, fname);

    return 0;
}





int functionparameter_GetParamIndex(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *paramname
)
{
    long index = -1;
    long pindex = 0;

    long NBparamMAX = fps->md->NBparamMAX;

    int found = 0;
    for(pindex=0; pindex<NBparamMAX; pindex++)
    {
        if(found==0)
        {
            if(fps->parray[pindex].fpflag & FPFLAG_ACTIVE)
            {
                if(strstr(fps->parray[pindex].keywordfull, paramname) != NULL)
                {
                    index = pindex;
                    found = 1;
                }
            }
        }
    }

    if (index == -1)
    {
        printf("ERROR: cannot find parameter \"%s\" in structure\n", paramname);
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    return index;
}




long functionparameter_GetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    long value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.l[0];
    fps->parray[fpsi].val.l[3] = value;

    return value;
}


int functionparameter_SetParamValue_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    long value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.l[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}


long *functionparameter_GetParamPtr_INT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    long *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.l[0];

    return ptr;
}





double functionparameter_GetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    double value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.f[0];
    fps->parray[fpsi].val.f[3] = value;

    return value;
}

int functionparameter_SetParamValue_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    double value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.f[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}

double *functionparameter_GetParamPtr_FLOAT64(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    double *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.f[0];

    return ptr;
}


float functionparameter_GetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    float value;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    value = fps->parray[fpsi].val.s[0];
    fps->parray[fpsi].val.s[3] = value;

    return value;
}

int functionparameter_SetParamValue_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    float value
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    fps->parray[fpsi].val.s[0] = value;
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}

float *functionparameter_GetParamPtr_FLOAT32(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    float *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].val.s[0];

    return ptr;
}



char *functionparameter_GetParamPtr_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    return fps->parray[fpsi].val.string[0];
}

int functionparameter_SetParamValue_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    const char *stringvalue
)
{
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    strncpy(fps->parray[fpsi].val.string[0], stringvalue, FUNCTION_PARAMETER_STRMAXLEN);
    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}




int functionparameter_GetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
) {
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    if(fps->parray[fpsi].fpflag & FPFLAG_ONOFF) {
        return 1;
    } else {
        return 0;
    }
}



int functionparameter_SetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int ONOFFvalue
) {
    int fpsi = functionparameter_GetParamIndex(fps, paramname);

    if(ONOFFvalue == 1) {
        fps->parray[fpsi].fpflag |= FPFLAG_ONOFF;
        fps->parray[fpsi].val.l[0] = 1;
    } else {
        fps->parray[fpsi].fpflag &= ~FPFLAG_ONOFF;
        fps->parray[fpsi].val.l[0] = 0;
    }

    fps->parray[fpsi].cnt0++;

    return EXIT_SUCCESS;
}



uint64_t *functionparameter_GetParamPtr_fpflag(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    uint64_t *ptr;

    int fpsi = functionparameter_GetParamIndex(fps, paramname);
    ptr = &fps->parray[fpsi].fpflag;

    return ptr;
}








imageID functionparameter_LoadStream(
    FUNCTION_PARAMETER_STRUCT *fps,
    int                        pindex,
    int                        fpsconnectmode
) {
    imageID ID = -1;
    uint32_t     imLOC;


#ifdef STANDALONE
    printf("====================== Not working in standalone mode \n");
#else
    printf("====================== Loading stream \"%s\" = %s\n", fps->parray[pindex].keywordfull, fps->parray[pindex].val.string[0]);
    ID = COREMOD_IOFITS_LoadMemStream(fps->parray[pindex].val.string[0], &(fps->parray[pindex].fpflag), &imLOC);


    if(fpsconnectmode == FPSCONNECT_CONF) {
        if(fps->parray[pindex].fpflag & FPFLAG_STREAM_CONF_REQUIRED) {
            printf("    FPFLAG_STREAM_CONF_REQUIRED\n");
            if(ID == -1) {
                printf("FAILURE: Required stream %s could not be loaded\n", fps->parray[pindex].val.string[0]);
                exit(EXIT_FAILURE);
            }
        }
    }

    if(fpsconnectmode == FPSCONNECT_RUN) {
        if(fps->parray[pindex].fpflag & FPFLAG_STREAM_RUN_REQUIRED) {
            printf("    FPFLAG_STREAM_RUN_REQUIRED\n");
            if(ID == -1) {
                printf("FAILURE: Required stream %s could not be loaded\n", fps->parray[pindex].val.string[0]);
                exit(EXIT_FAILURE);
            }
        }
    }
#endif


    // TODO: Add testing for fps



    return ID;
}





/**
 * ## Purpose
 *
 * Add parameter to database with default settings
 *
 * If entry already exists, do not modify it
 *
 */

int function_parameter_add_entry(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *keywordstring,
    const char                *descriptionstring,
    uint64_t             type,
    uint64_t             fpflag,
    void *               valueptr
)
{
    int RVAL = 0;
    // 0: parameter initialized to default value
    // 1: initialized using file value
    // 2: initialized to function argument value

    long pindex = 0;
    char *pch;
    char tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    FUNCTION_PARAMETER *funcparamarray;

    funcparamarray = fps->parray;

    long NBparamMAX = -1;

    NBparamMAX = fps->md->NBparamMAX;





    // process keywordstring
    // if string starts with ".", insert fps name
    char keywordstringC[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    if(keywordstring[0] == '.')
    {
        sprintf(keywordstringC, "%s%s", fps->md->name, keywordstring);
    }
    else {
        strcpy(keywordstringC, keywordstring);
    }



    // scan for existing keyword
    int scanOK = 0;
    long pindexscan;
    for(pindexscan=0; pindexscan<NBparamMAX; pindexscan++)
    {
        if(strcmp(keywordstringC, funcparamarray[pindexscan].keywordfull)==0)
        {
            pindex = pindexscan;
            scanOK = 1;
        }
    }

    if(scanOK==0) // not found
    {
        // scan for first available entry
        pindex = 0;
        while( (funcparamarray[pindex].fpflag & FPFLAG_ACTIVE) && (pindex<NBparamMAX) )
            pindex++;

        if(pindex == NBparamMAX)
        {
            printf("ERROR [%s line %d]: NBparamMAX %ld limit reached\n", __FILE__, __LINE__, NBparamMAX);
            fflush(stdout);
            printf("STEP %s %d\n", __FILE__, __LINE__);
            fflush(stdout);
            exit(0);
        }
    }
    else
    {
        printf("Found matching keyword: applying values to existing entry\n");
    }

    funcparamarray[pindex].fpflag = fpflag;



    // break full keyword into keywords
    strncpy(funcparamarray[pindex].keywordfull, keywordstringC, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL);
    strncpy(tmpstring, keywordstringC, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL);
    funcparamarray[pindex].keywordlevel = 0;
    pch = strtok (tmpstring, ".");
    while (pch != NULL)
    {
        strncpy(funcparamarray[pindex].keyword[funcparamarray[pindex].keywordlevel], pch, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN);
        funcparamarray[pindex].keywordlevel++;
        pch = strtok (NULL, ".");
    }


    // Write description
    strncpy(funcparamarray[pindex].description, descriptionstring, FUNCTION_PARAMETER_DESCR_STRMAXLEN);

    // type
    funcparamarray[pindex].type = type;



    // Allocate value
    funcparamarray[pindex].cnt0 = 0; // not allocated

    // Default values
    switch (funcparamarray[pindex].type) {
    case FPTYPE_INT64 :
        funcparamarray[pindex].val.l[0] = 0;
        funcparamarray[pindex].val.l[1] = 0;
        funcparamarray[pindex].val.l[2] = 0;
        funcparamarray[pindex].val.l[3] = 0;
        break;
    case FPTYPE_FLOAT64 :
        funcparamarray[pindex].val.f[0] = 0.0;
        funcparamarray[pindex].val.f[1] = 0.0;
        funcparamarray[pindex].val.f[2] = 0.0;
        funcparamarray[pindex].val.f[3] = 0.0;
        break;
    case FPTYPE_FLOAT32 :
        funcparamarray[pindex].val.s[0] = 0.0;
        funcparamarray[pindex].val.s[1] = 0.0;
        funcparamarray[pindex].val.s[2] = 0.0;
        funcparamarray[pindex].val.s[3] = 0.0;
        break;
    case FPTYPE_PID :
        funcparamarray[pindex].val.pid[0] = 0;
        funcparamarray[pindex].val.pid[1] = 0;
        break;
    case FPTYPE_TIMESPEC :
        funcparamarray[pindex].val.ts[0].tv_sec = 0;
        funcparamarray[pindex].val.ts[0].tv_nsec = 0;
        funcparamarray[pindex].val.ts[1].tv_sec = 0;
        funcparamarray[pindex].val.ts[1].tv_nsec = 0;
        break;
    case FPTYPE_FILENAME :
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    case FPTYPE_FITSFILENAME :
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    case FPTYPE_EXECFILENAME :
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    case FPTYPE_DIRNAME :
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    case FPTYPE_STREAMNAME :
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    case FPTYPE_STRING :
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    case FPTYPE_ONOFF :
        funcparamarray[pindex].fpflag &= ~FPFLAG_ONOFF; // initialize state to OFF
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "OFF state")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, " ON state")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    case FPTYPE_FPSNAME :
        if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN, "NULL")< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        break;
    }



    if( valueptr != NULL )// allocate value requested by function call
    {
        int64_t *valueptr_INT64;
        double *valueptr_FLOAT64;
        float *valueptr_FLOAT32;
        struct timespec *valueptr_ts;

        switch (funcparamarray[pindex].type) {

        case FPTYPE_INT64 :
            valueptr_INT64 = (int64_t *) valueptr;
            funcparamarray[pindex].val.l[0] = valueptr_INT64[0];
            funcparamarray[pindex].val.l[1] = valueptr_INT64[1];
            funcparamarray[pindex].val.l[2] = valueptr_INT64[2];
            funcparamarray[pindex].val.l[3] = valueptr_INT64[3];
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_FLOAT64 :
            valueptr_FLOAT64 = (double *) valueptr;
            funcparamarray[pindex].val.f[0] = valueptr_FLOAT64[0];
            funcparamarray[pindex].val.f[1] = valueptr_FLOAT64[1];
            funcparamarray[pindex].val.f[2] = valueptr_FLOAT64[2];
            funcparamarray[pindex].val.f[3] = valueptr_FLOAT64[3];
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_FLOAT32 :
            valueptr_FLOAT32 = (float *) valueptr;
            funcparamarray[pindex].val.s[0] = valueptr_FLOAT32[0];
            funcparamarray[pindex].val.s[1] = valueptr_FLOAT32[1];
            funcparamarray[pindex].val.s[2] = valueptr_FLOAT32[2];
            funcparamarray[pindex].val.s[3] = valueptr_FLOAT32[3];
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_PID :
            valueptr_INT64 = (int64_t *) valueptr;
            funcparamarray[pindex].val.pid[0] = (pid_t) (*valueptr_INT64);
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_TIMESPEC:
            valueptr_ts = (struct timespec *) valueptr;
            funcparamarray[pindex].val.ts[0] = *valueptr_ts;
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_FILENAME :
            strncpy(funcparamarray[pindex].val.string[0], (char*) valueptr,  FUNCTION_PARAMETER_STRMAXLEN);
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_FITSFILENAME :
            strncpy(funcparamarray[pindex].val.string[0], (char*) valueptr,  FUNCTION_PARAMETER_STRMAXLEN);
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_EXECFILENAME :
            strncpy(funcparamarray[pindex].val.string[0], (char*) valueptr,  FUNCTION_PARAMETER_STRMAXLEN);
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_DIRNAME :
            strncpy(funcparamarray[pindex].val.string[0], (char*) valueptr,  FUNCTION_PARAMETER_STRMAXLEN);
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_STREAMNAME :
            strncpy(funcparamarray[pindex].val.string[0], (char*) valueptr,  FUNCTION_PARAMETER_STRMAXLEN);
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_STRING :
            strncpy(funcparamarray[pindex].val.string[0], (char*) valueptr,  FUNCTION_PARAMETER_STRMAXLEN);
            funcparamarray[pindex].cnt0++;
            break;

        case FPTYPE_ONOFF : // already allocated through the status flag
            break;

        case FPTYPE_FPSNAME :
            strncpy(funcparamarray[pindex].val.string[0], (char*) valueptr,  FUNCTION_PARAMETER_STRMAXLEN);
            funcparamarray[pindex].cnt0++;
            break;
        }

        RVAL = 2;  // default value entered
    }



    // attempt to read value for filesystem
    char fname[200];
    FILE *fp;
    long tmpl;




    int index;
    // index = 0  : setval
    // index = 1  : minval
    // index = 2  : maxval


    for(index=0; index<3; index++)
    {
        char systemcmd[500];

        switch (index) {

        case 0 :
            functionparameter_GetFileName(fps, &funcparamarray[pindex], fname, "setval");
            break;

        case 1 :
            functionparameter_GetFileName(fps, &funcparamarray[pindex], fname, "minval");
            break;

        case 2 :
            functionparameter_GetFileName(fps, &funcparamarray[pindex], fname, "maxval");
            break;

        }


        if ( (fp = fopen(fname, "r")) != NULL)
        {

            sprintf(systemcmd, "echo  \"-------- FILE FOUND: %s \" >> tmplog.txt", fname);
            if ( system(systemcmd) == -1) {
                printERROR(__FILE__,__func__,__LINE__, "system() error");
            }

            switch (funcparamarray[pindex].type) {

            case FPTYPE_INT64 :
                if ( fscanf(fp, "%ld", &funcparamarray[pindex].val.l[index]) == 1)
                    if ( index == 0 )  // return value is set by setval, cnt0 tracks updates to setval, not to minval or maxval
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                break;

            case FPTYPE_FLOAT64 :
                if ( fscanf(fp, "%lf", &funcparamarray[pindex].val.f[index]) == 1)
                    if ( index == 0 )
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                break;

            case FPTYPE_FLOAT32 :
                if ( fscanf(fp, "%f", &funcparamarray[pindex].val.s[index]) == 1)
                    if ( index == 0 )
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                break;

            case FPTYPE_PID :
                if(index==0)  // PID does not have min / max
                {
                    if ( fscanf(fp, "%d", &funcparamarray[pindex].val.pid[index]) == 1)
                        RVAL = 1;
                    funcparamarray[pindex].cnt0++;
                }
                break;

            case FPTYPE_TIMESPEC :
                if ( fscanf(fp, "%ld %ld", &funcparamarray[pindex].val.ts[index].tv_sec, &funcparamarray[pindex].val.ts[index].tv_nsec) == 2)
                    if ( index == 0 )
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                break;

            case FPTYPE_FILENAME :
                if ( index == 0 ) // FILENAME does not have min / max
                {
                    if ( fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;

            case FPTYPE_FITSFILENAME :
                if ( index == 0 ) // FITSFILENAME does not have min / max
                {
                    if ( fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;

            case FPTYPE_EXECFILENAME :
                if ( index == 0 ) // EXECFILENAME does not have min / max
                {
                    if ( fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;


            case FPTYPE_DIRNAME :
                if ( index == 0 ) // DIRNAME does not have min / max
                {
                    if ( fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;

            case FPTYPE_STREAMNAME :
                if ( index == 0 ) // STREAMNAME does not have min / max
                {
                    if ( fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;

            case FPTYPE_STRING :
                if ( index == 0 ) // STRING does not have min / max
                {
                    if ( fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;

            case FPTYPE_ONOFF :
                if(index==0)
                {
                    if ( fscanf(fp, "%ld", &tmpl) == 1)
                    {
                        if(tmpl == 1)
                            funcparamarray[pindex].fpflag |= FPFLAG_ONOFF;
                        else
                            funcparamarray[pindex].fpflag &= ~FPFLAG_ONOFF;

                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;


            case FPTYPE_FPSNAME :
                if ( index == 0 ) // FPSNAME does not have min / max
                {
                    if ( fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                    {
                        RVAL = 1;
                        funcparamarray[pindex].cnt0++;
                    }
                }
                break;

            }
            fclose(fp);


        }
        else
        {
            sprintf(systemcmd, "echo  \"-------- FILE NOT FOUND: %s \" >> tmplog.txt", fname);
            if ( system(systemcmd) == -1) {
                printERROR(__FILE__,__func__,__LINE__, "system() error");
            }

        }
    }



    if(RVAL == 0) {
        functionparameter_WriteParameterToDisk(fps, pindex, "setval", "AddEntry created");
        if(funcparamarray[pindex].fpflag |= FPFLAG_MINLIMIT)
            functionparameter_WriteParameterToDisk(fps, pindex, "minval", "AddEntry created");
        if(funcparamarray[pindex].fpflag |= FPFLAG_MAXLIMIT)
            functionparameter_WriteParameterToDisk(fps, pindex, "maxval", "AddEntry created");
    }

    if(RVAL == 2) {
        functionparameter_WriteParameterToDisk(fps, pindex, "setval", "AddEntry argument");
        if(funcparamarray[pindex].fpflag |= FPFLAG_MINLIMIT)
            functionparameter_WriteParameterToDisk(fps, pindex, "minval", "AddEntry argument");
        if(funcparamarray[pindex].fpflag |= FPFLAG_MAXLIMIT)
            functionparameter_WriteParameterToDisk(fps, pindex, "maxval", "AddEntry argument");
    }

    if(RVAL != 0)
    {
        functionparameter_WriteParameterToDisk(fps, pindex, "fpsname", "AddEntry");
        functionparameter_WriteParameterToDisk(fps, pindex, "fpsdir", "AddEntry");
        functionparameter_WriteParameterToDisk(fps, pindex, "status", "AddEntry");
    }

    return pindex;
}









// ======================================== LOOP MANAGEMENT FUNCTIONS =======================================


FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(
    const char *fpsname,
    uint32_t CMDmode
) {
    long NBparamMAX = FUNCTION_PARAMETER_NBPARAM_DEFAULT;
    uint32_t FPSCONNECTFLAG;

    FUNCTION_PARAMETER_STRUCT fps;

    fps.CMDmode = CMDmode;
    fps.SMfd = -1;


    if(CMDmode & FPSCMDCODE_FPSINITCREATE) { // (re-)create fps even if it exists
        printf("=== FPSINITCREATE NBparamMAX = %ld\n", NBparamMAX);
        function_parameter_struct_create(NBparamMAX, fpsname);
        function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
    } else { // load existing fps if exists
        printf("=== CHECK IF FPS EXISTS\n");


        FPSCONNECTFLAG = FPSCONNECT_SIMPLE;
        if(CMDmode & FPSCMDCODE_CONFSTART) {
            FPSCONNECTFLAG = FPSCONNECT_CONF;
        }

        if(function_parameter_struct_connect(fpsname, &fps, FPSCONNECTFLAG) == -1) {
            printf("=== FPS DOES NOT EXISTS -> CREATE\n");
            function_parameter_struct_create(NBparamMAX, fpsname);
            function_parameter_struct_connect(fpsname, &fps, FPSCONNECTFLAG);
        }
        else
        {
            printf("=== FPS EXISTS\n");
        }
    }

    if(CMDmode & FPSCMDCODE_CONFSTOP) { // stop fps
        fps.md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
        function_parameter_struct_disconnect(&fps);
        fps.loopstatus = 0; // stop loop
    } else {
        fps.loopstatus = 1;
    }



    if( (CMDmode & FPSCMDCODE_FPSINITCREATE) || (CMDmode & FPSCMDCODE_FPSINIT) || (CMDmode & FPSCMDCODE_CONFSTOP) ) {
        fps.loopstatus = 0; // do not start conf
    }

    if ( CMDmode & FPSCMDCODE_CONFSTART ) {
        fps.loopstatus = 1;
    }


    return fps;
}





uint16_t function_parameter_FPCONFloopstep(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    static int loopINIT = 0;
    uint16_t updateFLAG = 0;

    static uint32_t prev_status;
    //static uint32_t statuschanged = 0;


    if(loopINIT == 0) {
        loopINIT = 1; // update on first loop iteration
        fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

        if(fps->CMDmode & FPSCMDCODE_CONFSTART) {  // parameter configuration loop
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
            fps->md->confpid = getpid();
            fps->loopstatus = 1;
        } else {
            fps->loopstatus = 0;
        }
    }


    if(fps->md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN) {
        // Test if CONF process is running
        if((getpgid(fps->md->confpid) >= 0) && (fps->md->confpid > 0)) {
            fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CONF;    // running
        } else {
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CONF;    // not running
        }

        // Test if RUN process is running
        if((getpgid(fps->md->runpid) >= 0) && (fps->md->runpid > 0)) {
            fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_RUN;    // running
        } else {
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_RUN;    // not running
        }


        if(prev_status != fps->md->status) {
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // request an update
        }



        if(fps->md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE) { // update is required
            updateFLAG = 1;
            fps->md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // disable update (should be moved to conf process)
        }
        usleep(fps->md->confwaitus);
    } else {
        fps->loopstatus = 0;
    }



    prev_status = fps->md->status;


    return updateFLAG;
}





uint16_t function_parameter_FPCONFexit( FUNCTION_PARAMETER_STRUCT *fps)
{
    //fps->md->confpid = 0;


    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    function_parameter_struct_disconnect(fps);

    return 0;
}



uint16_t function_parameter_RUNexit( FUNCTION_PARAMETER_STRUCT *fps)
{
    //fps->md->confpid = 0;


    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    function_parameter_struct_disconnect(fps);

    return 0;
}






// ======================================== GUI FUNCTIONS =======================================



/**
 * INITIALIZE ncurses
 *
 */
static errno_t initncurses()
{
    if ( initscr() == NULL ) {
        fprintf(stderr, "Error initialising ncurses.\n");
        exit(EXIT_FAILURE);
    }
    getmaxyx(stdscr, wrow, wcol);		/* get the number of rows and columns */

    cbreak();
    // disables line buffering and erase/kill character-processing (interrupt and flow control characters are unaffected),
    // making characters typed by the user immediately available to the program

    keypad(stdscr, TRUE);
    // enable F1, F2 etc..

    nodelay(stdscr, TRUE);
    curs_set(0);

    noecho();
    // Don't echo() while we do getch



    //nonl();
    // Do not translates newline into return and line-feed on output


    init_color(COLOR_GREEN, 400, 1000, 400);
    start_color();

    //  colored background
    init_pair(1, COLOR_BLACK, COLOR_WHITE);
    init_pair(2, COLOR_BLACK, COLOR_GREEN);  // all good
    init_pair(3, COLOR_BLACK, COLOR_YELLOW); // parameter out of sync
    init_pair(4, COLOR_WHITE, COLOR_RED);
    init_pair(5, COLOR_WHITE, COLOR_BLUE); // DIRECTORY
    init_pair(6, COLOR_GREEN, COLOR_BLACK);
    init_pair(7, COLOR_YELLOW, COLOR_BLACK);
    init_pair(8, COLOR_RED, COLOR_BLACK);
    init_pair(9, COLOR_BLACK, COLOR_RED);
    init_pair(10, COLOR_BLACK, COLOR_CYAN);


    return RETURN_SUCCESS;
}




/**
 * ## Purpose
 *
 * Write parameter to disk
 *
 * ## TAG names
 *
 * One of the following:
 * - "setval"  Set value
 * - "fpsname" Name of FPS to which parameter belongs
 * - "fpsdir"  FPS directory
 * - "minval"  Minimum value (if applicable)
 * - "maxval"  Maximum value (if applicable)
 * - "currval" Current value (if applicable)
 *
 *
 */
int functionparameter_WriteParameterToDisk(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int pindex,
    char *tagname,
    char *commentstr
)
{
    char fname[200];
    FILE *fp;


    // create time change tag
    pid_t tid;
    tid = syscall(SYS_gettid);

    // Get GMT time
    char timestring[200];
    struct timespec tnow;
    time_t now;

    clock_gettime(CLOCK_REALTIME, &tnow);
    now = tnow.tv_sec;
    struct tm *uttime;
    uttime = gmtime(&now);

    sprintf(timestring, "%04d%02d%02d%02d%02d%02d.%09ld %8ld [%6d %6d] %s",
            1900+uttime->tm_year, 1+uttime->tm_mon, uttime->tm_mday, uttime->tm_hour, uttime->tm_min,  uttime->tm_sec, tnow.tv_nsec,
            fpsentry->parray[pindex].cnt0, getpid(), (int) tid, commentstr);



    if ( strcmp(tagname, "setval") == 0) // VALUE
    {
        functionparameter_GetFileName(fpsentry, &(fpsentry->parray[pindex]), fname, tagname);
        fp = fopen(fname, "w");
        switch (fpsentry->parray[pindex].type) {

        case FPTYPE_INT64:
            fprintf(fp, "%10ld  # %s\n", fpsentry->parray[pindex].val.l[0], timestring);
            break;

        case FPTYPE_FLOAT64:
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.f[0], timestring);
            break;

        case FPTYPE_FLOAT32:
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.s[0], timestring);
            break;

        case FPTYPE_PID:
            fprintf(fp, "%18ld  # %s\n", (long) fpsentry->parray[pindex].val.pid[0], timestring);
            break;

        case FPTYPE_TIMESPEC:
            fprintf(fp, "%15ld %09ld  # %s\n", (long) fpsentry->parray[pindex].val.ts[0].tv_sec, (long) fpsentry->parray[pindex].val.ts[0].tv_nsec, timestring);
            break;

        case FPTYPE_FILENAME:
            fprintf(fp, "%s  # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        case FPTYPE_FITSFILENAME:
            fprintf(fp, "%s  # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        case FPTYPE_EXECFILENAME:
            fprintf(fp, "%s  # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        case FPTYPE_DIRNAME:
            fprintf(fp, "%s  # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        case FPTYPE_STREAMNAME:
            fprintf(fp, "%s  # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        case FPTYPE_STRING:
            fprintf(fp, "%s  # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        case FPTYPE_ONOFF:
            if( fpsentry->parray[pindex].fpflag & FPFLAG_ONOFF )
                fprintf(fp, "1  %10s # %s\n", fpsentry->parray[pindex].val.string[1], timestring);
            else
                fprintf(fp, "0  %10s # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        case FPTYPE_FPSNAME:
            fprintf(fp, "%s  # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        }
        fclose(fp);
    }



    if ( strcmp(tagname, "minval") == 0) // MIN VALUE
    {
        functionparameter_GetFileName(fpsentry, &(fpsentry->parray[pindex]), fname, tagname);

        switch (fpsentry->parray[pindex].type) {

        case FPTYPE_INT64:
            fp = fopen(fname, "w");
            fprintf(fp, "%10ld  # %s\n", fpsentry->parray[pindex].val.l[1], timestring);
            fclose(fp);
            break;

        case FPTYPE_FLOAT64:
            fp = fopen(fname, "w");
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.f[1], timestring);
            fclose(fp);
            break;

        case FPTYPE_FLOAT32:
            fp = fopen(fname, "w");
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.s[1], timestring);
            fclose(fp);
            break;
        }
    }


    if ( strcmp(tagname, "maxval") == 0) // MAX VALUE
    {
        functionparameter_GetFileName(fpsentry, &(fpsentry->parray[pindex]), fname, tagname);

        switch (fpsentry->parray[pindex].type) {

        case FPTYPE_INT64:
            fp = fopen(fname, "w");
            fprintf(fp, "%10ld  # %s\n", fpsentry->parray[pindex].val.l[2], timestring);
            fclose(fp);
            break;

        case FPTYPE_FLOAT64:
            fp = fopen(fname, "w");
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.f[2], timestring);
            fclose(fp);
            break;

        case FPTYPE_FLOAT32:
            fp = fopen(fname, "w");
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.s[2], timestring);
            fclose(fp);
            break;
        }
    }


    if ( strcmp(tagname, "currval") == 0) // CURRENT VALUE
    {
        functionparameter_GetFileName(fpsentry, &(fpsentry->parray[pindex]), fname, tagname);

        switch (fpsentry->parray[pindex].type) {

        case FPTYPE_INT64:
            fp = fopen(fname, "w");
            fprintf(fp, "%10ld  # %s\n", fpsentry->parray[pindex].val.l[3], timestring);
            fclose(fp);
            break;

        case FPTYPE_FLOAT64:
            fp = fopen(fname, "w");
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.f[3], timestring);
            fclose(fp);
            break;

        case FPTYPE_FLOAT32:
            fp = fopen(fname, "w");
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.s[3], timestring);
            fclose(fp);
            break;
        }
    }




    if ( strcmp(tagname, "fpsname") == 0) // FPS name
    {
        functionparameter_GetFileName(fpsentry, &(fpsentry->parray[pindex]), fname, tagname);
        fp = fopen(fname, "w");
        fprintf(fp, "%10s    # %s\n", fpsentry->md->name, timestring);
        fclose(fp);
    }

    if ( strcmp(tagname, "fpsdir") == 0) // FPS name
    {
        functionparameter_GetFileName(fpsentry, &(fpsentry->parray[pindex]), fname, tagname);
        fp = fopen(fname, "w");
        fprintf(fp, "%10s    # %s\n", fpsentry->md->fpsdirectory, timestring);
        fclose(fp);
    }

    if ( strcmp(tagname, "status") == 0) // FPS name
    {
        functionparameter_GetFileName(fpsentry, &(fpsentry->parray[pindex]), fname, tagname);
        fp = fopen(fname, "w");
        fprintf(fp, "%10ld    # %s\n", fpsentry->parray[pindex].fpflag, timestring);
        fclose(fp);
    }






    return 0;
}










int functionparameter_CheckParameter(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int pindex
) {
    int err = 0;

    // if entry is not active or not used, no error reported
    if(  (!(fpsentry->parray[pindex].fpflag & FPFLAG_ACTIVE))  ) {
        return 0;
    }
    else
    {
        char msg[STRINGMAXLEN_FPS_LOGMSG];
        SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG, "%s", fpsentry->parray[pindex].keywordfull);
        functionparameter_outlog("CHECKPARAM", msg);
    }

    // if entry is not used, no error reported
    if(!(fpsentry->parray[pindex].fpflag & FPFLAG_USED)) {
        return 0;
    }


    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKINIT)
        if(fpsentry->parray[pindex].cnt0 == 0) {
            fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
            fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_NOTINITIALIZED | FPS_MSG_FLAG_ERROR;
            if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "Not initialized")< 0 ) {
                PRINT_ERROR("snprintf error");
            }
            fpsentry->md->msgcnt++;
            fpsentry->md->conferrcnt++;
            err = 1;
        }

    if(err == 0) {
        // Check min value
        if(fpsentry->parray[pindex].type == FPTYPE_INT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
                if(fpsentry->parray[pindex].val.l[0] < fpsentry->parray[pindex].val.l[1]) {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "int64 value %ld below min %ld",
                                fpsentry->parray[pindex].val.l[0],
                                fpsentry->parray[pindex].val.l[1])< 0 ) {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }

        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
                if(fpsentry->parray[pindex].val.f[0] < fpsentry->parray[pindex].val.f[1]) {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float64 value %lf below min %lf",
                                fpsentry->parray[pindex].val.f[0],
                                fpsentry->parray[pindex].val.f[1])< 0 ) {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }

        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT32)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
                if(fpsentry->parray[pindex].val.s[0] < fpsentry->parray[pindex].val.s[1]) {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float32 value %f below min %f",
                                fpsentry->parray[pindex].val.s[0],
                                fpsentry->parray[pindex].val.s[1])< 0 ) {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }
    }

    if(err == 0) {
        // Check max value
        if(fpsentry->parray[pindex].type == FPTYPE_INT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
                if(fpsentry->parray[pindex].val.l[0] > fpsentry->parray[pindex].val.l[2]) {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "int64 value %ld above max %ld",
                                fpsentry->parray[pindex].val.l[0],
                                fpsentry->parray[pindex].val.l[2])< 0 ) {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }


        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
                if(fpsentry->parray[pindex].val.f[0] > fpsentry->parray[pindex].val.f[2]) {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float64 value %lf above max %lf",
                                fpsentry->parray[pindex].val.f[0],
                                fpsentry->parray[pindex].val.f[2])< 0 ) {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }



        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT32)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
                if(fpsentry->parray[pindex].val.s[0] > fpsentry->parray[pindex].val.s[2]) {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float32 value %f above max %f",
                                fpsentry->parray[pindex].val.s[0],
                                fpsentry->parray[pindex].val.s[2])< 0 ) {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }
    }


#ifndef STANDALONE
    if(fpsentry->parray[pindex].type == FPTYPE_FILENAME) {
        if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED) {
            if(file_exists(fpsentry->parray[pindex].val.string[0])==0) {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "File %s does not exist",
                            fpsentry->parray[pindex].val.string[0])< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                fpsentry->md->conferrcnt++;
                err = 1;
            }
        }
    }



    if(fpsentry->parray[pindex].type == FPTYPE_FITSFILENAME) {
        if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED) {
            if(is_fits_file(fpsentry->parray[pindex].val.string[0])==0) {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "FITS file %s does not exist",
                            fpsentry->parray[pindex].val.string[0])< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                fpsentry->md->conferrcnt++;
                err = 1;
            }
        }
    }
#endif


    if(fpsentry->parray[pindex].type == FPTYPE_EXECFILENAME) {
        if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED) {
            struct stat sb;
            if (!(stat(fpsentry->parray[pindex].val.string[0], &sb) == 0 && sb.st_mode & S_IXUSR)) {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "File %s cannot be executed",
                            fpsentry->parray[pindex].val.string[0])< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                fpsentry->md->conferrcnt++;
                err = 1;
            }
        }
    }



    FUNCTION_PARAMETER_STRUCT fpstest;
    if(fpsentry->parray[pindex].type == FPTYPE_FPSNAME) {
        if(fpsentry->parray[pindex].fpflag & FPFLAG_FPS_RUN_REQUIRED) {
            long NBparamMAX = function_parameter_struct_connect(fpsentry->parray[pindex].val.string[0], &fpstest, FPSCONNECT_SIMPLE);
            if(NBparamMAX < 1) {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "FPS %s: no connection",
                            fpsentry->parray[pindex].val.string[0])< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                fpsentry->md->conferrcnt++;
                err = 1;
            }
            else {
                function_parameter_struct_disconnect(&fpstest);
            }
        }
    }





#ifdef STANDALONE
    printf("====================== Not working in standalone mode \n");
#else
    // STREAM CHECK

    if( (fpsentry->parray[pindex].type & FPTYPE_STREAMNAME) ) {

        uint32_t imLOC;
        long ID = COREMOD_IOFITS_LoadMemStream(fpsentry->parray[pindex].val.string[0], &(fpsentry->parray[pindex].fpflag), &imLOC);
        fpsentry->parray[pindex].info.stream.streamID = ID;

        if(ID>-1)
        {
            fpsentry->parray[pindex].info.stream.stream_sourceLocation = imLOC;
            fpsentry->parray[pindex].info.stream.stream_atype = data.image[ID].md[0].datatype;

            fpsentry->parray[pindex].info.stream.stream_naxis[0] = data.image[ID].md[0].naxis;
            fpsentry->parray[pindex].info.stream.stream_xsize[0] = data.image[ID].md[0].size[0];

            if(fpsentry->parray[pindex].info.stream.stream_naxis[0]>1)
                fpsentry->parray[pindex].info.stream.stream_ysize[0] = data.image[ID].md[0].size[1];
            else
                fpsentry->parray[pindex].info.stream.stream_ysize[0] = 1;

            if(fpsentry->parray[pindex].info.stream.stream_naxis[0]>2)
                fpsentry->parray[pindex].info.stream.stream_zsize[0] = data.image[ID].md[0].size[2];
            else
                fpsentry->parray[pindex].info.stream.stream_zsize[0] = 1;
        }



        if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_RUN_REQUIRED) {
            char msg[200];
            sprintf(msg, "Loading stream %s", fpsentry->parray[pindex].val.string[0]);
            functionparameter_outlog("LOADMEMSTREAM", msg);

            if ( imLOC == STREAM_LOAD_SOURCE_NOTFOUND ) {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "cannot load stream %s", fpsentry->parray[pindex].val.string[0])< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                fpsentry->md->conferrcnt++;
                err = 1;
            }
        }
    }
#endif


    if(err == 1) {
        fpsentry->parray[pindex].fpflag |= FPFLAG_ERROR;
    } else {
        fpsentry->parray[pindex].fpflag &= ~FPFLAG_ERROR;
    }



    return err;
}








int functionparameter_CheckParametersAll(
    FUNCTION_PARAMETER_STRUCT *fpsentry
) {
    long NBparamMAX;
    long pindex;
    int errcnt = 0;

    char msg[200];
    sprintf(msg, "%s", fpsentry->md->name);
    functionparameter_outlog("CHECKPARAMALL", msg);



    strcpy(fpsentry->md->message[0], "\0");
    NBparamMAX = fpsentry->md->NBparamMAX;

    // Check if Value is OK
    fpsentry->md->msgcnt = 0;
    fpsentry->md->conferrcnt = 0;
    //    printf("Checking %d parameter entries\n", NBparam);
    for(pindex = 0; pindex < NBparamMAX; pindex++) {
        errcnt += functionparameter_CheckParameter(fpsentry, pindex);
    }


    // number of configuration errors - should be zero for run process to start
    fpsentry->md->conferrcnt = errcnt;


    if(errcnt == 0) {
        fpsentry->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK;
    } else {
        fpsentry->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK;
    }


    // compute write status

    for(pindex = 0; pindex < NBparamMAX; pindex++) {
        int writeOK; // do we have write permission ?

        // by default, adopt FPFLAG_WRITE flag
        if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITE) {
            writeOK = 1;
        } else {
            writeOK = 0;
        }

        // if CONF running
        if(fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CONF) {
            if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITECONF) {
                writeOK = 1;
            } else {
                writeOK = 0;
            }
        }

        // if RUN running
        if(fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_RUN) {
            if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITERUN) {
                writeOK = 1;
            } else {
                writeOK = 0;
            }
        }

        if(writeOK == 0) {
            fpsentry->parray[pindex].fpflag &= ~FPFLAG_WRITESTATUS;
        } else {
            fpsentry->parray[pindex].fpflag |= FPFLAG_WRITESTATUS;
        }
    }

    fpsentry->md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED;


    return 0;
}








int functionparameter_ConnectExternalFPS(
    FUNCTION_PARAMETER_STRUCT *FPS,
    int pindex,
    FUNCTION_PARAMETER_STRUCT *FPSext
)
{
    FPS->parray[pindex].info.fps.FPSNBparamMAX = function_parameter_struct_connect(FPS->parray[pindex].val.string[0], FPSext, FPSCONNECT_SIMPLE);

    FPS->parray[pindex].info.fps.FPSNBparamActive = 0;
    FPS->parray[pindex].info.fps.FPSNBparamUsed = 0;
    int pindexext;
    for(pindexext=0; pindexext<FPS->parray[pindex].info.fps.FPSNBparamMAX; pindexext++) {
        if(FPSext->parray[pindexext].fpflag & FPFLAG_ACTIVE) {
            FPS->parray[pindex].info.fps.FPSNBparamActive++;
        }
        if(FPSext->parray[pindexext].fpflag & FPFLAG_USED) {
            FPS->parray[pindex].info.fps.FPSNBparamUsed++;
        }
    }

    return 0;
}




errno_t functionparameter_GetTypeString(
    uint32_t type,
    char *typestring
) {

    sprintf(typestring, " ");

    // using if statements (not switch) to allow for multiple types
    if(type & FPTYPE_UNDEF) {
        strcat(typestring, "UNDEF ");
    }
    if(type & FPTYPE_INT64) {
        strcat(typestring, "INT64 ");
    }
    if(type & FPTYPE_FLOAT64) {
        strcat(typestring, "FLOAT64 ");
    }
    if(type & FPTYPE_FLOAT32) {
        strcat(typestring, "FLOAT32 ");
    }
    if(type & FPTYPE_PID) {
        strcat(typestring, "PID ");
    }
    if(type & FPTYPE_TIMESPEC) {
        strcat(typestring, "TIMESPEC ");
    }
    if(type & FPTYPE_FILENAME) {
        strcat(typestring, "FILENAME ");
    }
    if(type & FPTYPE_FITSFILENAME) {
        strcat(typestring, "FITSFILENAME ");
    }
    if(type & FPTYPE_EXECFILENAME) {
        strcat(typestring, "EXECFILENAME");
    }
    if(type & FPTYPE_DIRNAME) {
        strcat(typestring, "DIRNAME");
    }
    if(type & FPTYPE_STREAMNAME) {
        strcat(typestring, "STREAMNAME");
    }
    if(type & FPTYPE_STRING) {
        strcat(typestring, "STRING ");
    }
    if(type & FPTYPE_ONOFF) {
        strcat(typestring, "ONOFF ");
    }
    if(type & FPTYPE_FPSNAME) {
        strcat(typestring, "FPSNAME ");
    }

    return RETURN_SUCCESS;
}



errno_t functionparameter_PrintParameterInfo(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int pindex
) {
    printf("%s\n", fpsentry->parray[pindex].description);
    printf("\n");


    printf("------------- FUNCTION PARAMETER STRUCTURE\n");
    printf("FPS name       : %s\n", fpsentry->md->name);
    printf("   %s ", fpsentry->md->pname);
    int i;
    for(i = 0; i < fpsentry->md->NBnameindex; i++) {
        printf(" [%s]", fpsentry->md->nameindexW[i]);
    }
    printf("\n\n");

    if(fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK) {
        printf("[%ld] %sScan OK%s\n", fpsentry->md->msgcnt, BLINKHIGREEN, RESET);
    } else {
        int msgi;

        printf("%s [%ld] %s%d ERROR(s)%s\n", fpsentry->md->name, fpsentry->md->msgcnt, BLINKHIRED, fpsentry->md->conferrcnt, RESET);
        for(msgi = 0; msgi < fpsentry->md->msgcnt; msgi++) {
            printf("%s [%3d] %s%s%s\n", fpsentry->md->name, fpsentry->md->msgpindex[msgi], BOLDHIRED, fpsentry->md->message[msgi], RESET);
        }
    }


    //snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "cannot load stream");
    //			fpsentry->md->msgcnt++;

    printf("\n");



    printf("------------- FUNCTION PARAMETER \n");
    printf("[%d] Parameter name : %s\n", pindex, fpsentry->parray[pindex].keywordfull);

    char typestring[100];
    functionparameter_GetTypeString(fpsentry->parray[pindex].type, typestring);
    printf("type: %s\n", typestring);


    printf("\n");
    printf("-- FLAG: ");



    // print binary flag
    printw("FLAG : ");
    uint64_t mask = (uint64_t) 1 << (sizeof (uint64_t) * CHAR_BIT - 1);
    while(mask) {
        int digit = fpsentry->parray[pindex].fpflag&mask ? 1 : 0;
        if(digit==1) {
            attron(COLOR_PAIR(2));
            printf("%s%d%s", BOLDHIGREEN, digit, RESET);
            attroff(COLOR_PAIR(2));
        } else {
            printf("%d", digit);
        }
        mask >>= 1;
    }
    printf("\n");


    int flagstringlen = 32;

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ACTIVE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "ACTIVE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "ACTIVE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_USED) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "USED");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "USED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_VISIBLE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "VISIBLE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "VISIBLE");
    }

    printf("%*s", flagstringlen, "---");

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "WRITE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITECONF) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITECONF");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "WRITECONF");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITERUN) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITERUN");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "WRITERUN");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITESTATUS) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITESTATUS");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "WRITESTATUS");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_LOG) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "LOG");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "LOG");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCHANGE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "SAVEONCHANGE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "SAVEONCHANGE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCLOSE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "SAVEONCLOSE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "SAVEONCLOSE");
    }


    printf("%*s", flagstringlen, "---");


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_IMPORTED) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "IMPORTED");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "IMPORTED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FEEDBACK) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "FEEDBACK");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "FEEDBACK");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ONOFF) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "ONOFF");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "ONOFF");
    }


    printf("%*s", flagstringlen, "---");


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKINIT) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "CHECKINIT");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "CHECKINIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "MINLIMIT");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "MINLIMIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "MAXLIMIT");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "MAXLIMIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ERROR) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "ERROR");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "ERROR");
    }


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_LOCALMEM) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_LOCALMEM");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_LOCALMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_SHAREMEM) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_SHAREMEM");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_CONFFITS) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFFITS");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_CONFNAME) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFNAME");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFNAME");
    }


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_LOCALMEM");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_LOCALMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_SHAREMEM");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFFITS");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFNAME");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFNAME");
    }


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_SHAREMEM");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_UPDATE_CONFFITS) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_CONFFITS");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_CONF_REQUIRED) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_CONF_REQUIRED");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_CONF_REQUIRED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_RUN_REQUIRED");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_RUN_REQUIRED");
    }


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_DATATYPE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_DATATYPE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_DATATYPE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT8) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT8");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT8");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT8) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT8");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT8");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT16) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT16");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT16");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT16) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT16");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT16");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT32) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT32");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT32");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT32) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT32");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT32");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT64) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT64");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT64");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT64) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT64");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT64");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_HALF) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_HALF");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_HALF");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_FLOAT) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_FLOAT");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_FLOAT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_DOUBLE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_DOUBLE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_DOUBLE");
    }


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_1D) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_1D");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_1D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_2D) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_2D");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_2D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_3D) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_3D");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_3D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_XSIZE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_XSIZE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_XSIZE");
    }


    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_YSIZE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_YSIZE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_YSIZE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_ZSIZE) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_ZSIZE");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_ZSIZE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKSTREAM) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "CHECKSTREAM");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "CHECKSTREAM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_MEMLOADREPORT) {
        printf("%s", BOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_MEMLOADREPORT");
        printf("%s", RESET);
    } else {
        printf("%*s", flagstringlen, "STREAM_MEMLOADREPORT");
    }









    printf("\n");
    printf("\n");
    printf("cnt0 = %ld\n", fpsentry->parray[pindex].cnt0);

    printf("\n");

    printf("Current value : ");

    if(fpsentry->parray[pindex].type == FPTYPE_UNDEF) {
        printf("  %s", "-undef-");
    }

    if(fpsentry->parray[pindex].type == FPTYPE_INT64) {
        printf("  %10d", (int) fpsentry->parray[pindex].val.l[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FLOAT64) {
        printf("  %10f", (float) fpsentry->parray[pindex].val.f[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FLOAT32) {
        printf("  %10f", (float) fpsentry->parray[pindex].val.s[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_PID) {
        printf("  %10d", (int) fpsentry->parray[pindex].val.pid[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_TIMESPEC) {
        printf("  %10s", "-timespec-");
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FILENAME) {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FITSFILENAME) {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_EXECFILENAME) {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_DIRNAME) {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_STREAMNAME) {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_STRING) {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_ONOFF) {
        /*if ( fpsentry->parray[pindex].status & FPFLAG_ONOFF )
        	printf("    ON  %s\n", fpsentry->parray[pindex].val.string[1]);
        else
        	printf("   OFF  %s\n", fpsentry->parray[pindex].val.string[0]);*/
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FPSNAME) {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    printf("\n");
    printf("\n");

    return RETURN_SUCCESS;

}









int functionparameter_SaveParam2disk(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    const char *paramname
) {
    int pindex;

    pindex = functionparameter_GetParamIndex(fpsentry, paramname);
    functionparameter_WriteParameterToDisk(fpsentry, pindex, "setval", "SaveParam2disk");

    return RETURN_SUCCESS;
}








/**
 *
 * ## PURPOSE
 *
 * Enter new value for parameter
 *
 *
 */


int functionparameter_UserInputSetParamValue(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int pindex
) {
    int inputOK;
    int strlenmax = 20;
    char buff[100];
    char c;

    functionparameter_PrintParameterInfo(fpsentry, pindex);


    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITESTATUS) {
        inputOK = 0;
        fflush(stdout);

        while(inputOK == 0) {
            printf("\nESC or update value : ");
            fflush(stdout);

            int stringindex = 0;
            c = getchar();
            while((c != 27) && (c != 10) && (c != 13) && (stringindex < strlenmax - 1)) {
                buff[stringindex] = c;
                if(c == 127) { // delete key
                    putchar(0x8);
                    putchar(' ');
                    putchar(0x8);
                    stringindex --;
                } else {
                    putchar(c);  // echo on screen
                    // printf("[%d]", (int) c);
                    stringindex++;
                }
                if(stringindex < 0) {
                    stringindex = 0;
                }
                c = getchar();
            }
            buff[stringindex] = '\0';
            inputOK = 1;
        }

        if(c != 27) { // do not update value if escape key

            long lval = 0;
            double fval = 0.0;
            char *endptr;
            int vOK = 1;

            switch(fpsentry->parray[pindex].type) {

            case FPTYPE_INT64:
                errno = 0;    /* To distinguish success/failure after call */
                lval = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if((errno == ERANGE && (lval == LONG_MAX || lval == LONG_MIN))
                        || (errno != 0 && lval == 0)) {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff) {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1) {
                    fpsentry->parray[pindex].val.l[0] = lval;
                }
                break;

            case FPTYPE_FLOAT64:
                errno = 0;    /* To distinguish success/failure after call */
                fval = strtod(buff, &endptr);

                /* Check for various possible errors */
                if((errno == ERANGE)
                        || (errno != 0 && fval == 0.0)) {
                    perror("strtod");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff) {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1) {
                    fpsentry->parray[pindex].val.f[0] = fval;
                }
                break;


            case FPTYPE_FLOAT32:
                errno = 0;    /* To distinguish success/failure after call */
                fval = strtod(buff, &endptr);

                /* Check for various possible errors */
                if((errno == ERANGE)
                        || (errno != 0 && fval == 0.0)) {
                    perror("strtod");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff) {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1) {
                    fpsentry->parray[pindex].val.s[0] = fval;
                }
                break;


            case FPTYPE_PID :
                errno = 0;    /* To distinguish success/failure after call */
                lval = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if((errno == ERANGE && (lval == LONG_MAX || lval == LONG_MIN))
                        || (errno != 0 && lval == 0)) {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff) {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1) {
                    fpsentry->parray[pindex].val.pid[0] = (pid_t) lval;
                }
                break;


            case FPTYPE_FILENAME :
                if(snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff)< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                break;

            case FPTYPE_FITSFILENAME :
                if(snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff)< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                break;

            case FPTYPE_EXECFILENAME :
                if(snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff)< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                break;

            case FPTYPE_DIRNAME :
                if(snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff)< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                break;

            case FPTYPE_STREAMNAME :
                if(snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff)< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                break;

            case FPTYPE_STRING :
                if(snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff)< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                break;

            case FPTYPE_FPSNAME :
                if(snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff)< 0 ) {
                    PRINT_ERROR("snprintf error");
                }
                break;

            }

            fpsentry->parray[pindex].cnt0++;

            // notify GUI
            fpsentry->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;


            // Save to disk
            if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCHANGE) {
                functionparameter_WriteParameterToDisk(fpsentry, pindex, "setval", "UserInputSetParamValue");
            }
        }
    } else {
        printf("%s Value cannot be modified %s\n", BOLDHIRED, RESET);
        c = getchar();
    }



    return 0;
}







/**
 * ## Purpose
 *
 * Process command line.
 *
 * ## Commands
 *
 * - logsymlink  : create log sym link
 * - setval      : set parameter value
 * - getval      : get value, write to output log
 * - fwrval      : get value, write to file or fifo
 * - confupdate  : update configuration
 * - confwupdate : update configuration, wait for completion to proceed
 * - runstart    : start RUN process associated with parameter
 * - runstop     : stop RUN process associated with parameter
 * - fpsrm       : remove fps
 *
 * - queueprio   : change queue priority
 *
 *
 */


int functionparameter_FPSprocess_cmdline(
    char *FPScmdline,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    KEYWORD_TREE_NODE *keywnode,
    int NBkwn,
    FUNCTION_PARAMETER_STRUCT *fps
) {
    int fpsindex;
    long pindex;

    // break FPScmdline in words
    // [FPScommand] [FPSentryname]
    //
    char *pch;
    int nbword = 0;
    char FPScommand[50];

    int cmdOK = 2;    // 0 : failed, 1: OK
    int cmdFOUND = 0; // toggles to 1 when command has been found

    char FPSentryname[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];  // first arg is always an FPS entry name
    char FPScmdarg1[FUNCTION_PARAMETER_STRMAXLEN];

    char FPSarg0[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char FPSarg1[FUNCTION_PARAMETER_STRMAXLEN];
    char FPSarg2[FUNCTION_PARAMETER_STRMAXLEN];
    char FPSarg3[FUNCTION_PARAMETER_STRMAXLEN];




    char msgstring[STRINGMAXLEN_FPS_LOGMSG];
    char inputcmd[STRINGMAXLEN_FPS_CMDLINE];
	
	if(strlen(FPScmdline) > 0) { // only send command if non-empty
		SNPRINTF_CHECK(inputcmd, STRINGMAXLEN_FPS_CMDLINE, "%s", FPScmdline);
	}
	
	SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "\"%s\"", inputcmd);
    
    functionparameter_outlog("CMDRCV", msgstring);


    DEBUG_TRACEPOINT(" ");

    if(strlen(inputcmd)>1)
    {
        pch = strtok(inputcmd, " \t");
        sprintf(FPScommand, "%s", pch);
    }
    else {
        pch = NULL;
    }


    DEBUG_TRACEPOINT(" ");



    while(pch != NULL) {

        nbword++;
        pch = strtok(NULL, " \t");

        if(nbword == 1) { // first arg (0)
            char *pos;
            sprintf(FPSarg0, "%s", pch);
            if((pos = strchr(FPSarg0, '\n')) != NULL) {
                *pos = '\0';
            }

        }

        if(nbword == 2) {
            char *pos;
            if(snprintf(FPSarg1, FUNCTION_PARAMETER_STRMAXLEN, "%s", pch) >= FUNCTION_PARAMETER_STRMAXLEN) {
                printf("WARNING: string truncated\n");
                printf("STRING: %s\n", pch);
            }
            if((pos = strchr(FPSarg1, '\n')) != NULL) {
                *pos = '\0';
            }
        }

        if(nbword == 3) {
            char *pos;
            if(snprintf(FPSarg2, FUNCTION_PARAMETER_STRMAXLEN, "%s", pch) >= FUNCTION_PARAMETER_STRMAXLEN) {
                printf("WARNING: string truncated\n");
                printf("STRING: %s\n", pch);
            }
            if((pos = strchr(FPSarg2, '\n')) != NULL) {
                *pos = '\0';
            }
        }

        if(nbword == 4) {
            char *pos;
            if(snprintf(FPSarg3, FUNCTION_PARAMETER_STRMAXLEN, "%s", pch) >= FUNCTION_PARAMETER_STRMAXLEN) {
                printf("WARNING: string truncated\n");
                printf("STRING: %s\n", pch);
            }
            if((pos = strchr(FPSarg3, '\n')) != NULL) {
                *pos = '\0';
            }
        }

    }



    DEBUG_TRACEPOINT(" ");


    if(nbword==0) {
        cmdFOUND = 1;   // do nothing, proceed
        cmdOK = 2;
    }




    // Handle commands for which FPSarg0 is NOT an FPS entry

    // logsymlink
    if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "logsymlink") == 0))
    {
        cmdFOUND = 1;
        if(nbword != 2) {
			SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND logsymlink NBARGS = 1");            
            functionparameter_outlog("ERROR", msgstring);
            cmdOK = 0;
        }
        else
        {
            char logfname[STRINGMAXLEN_FULLFILENAME];
            char shmdname[STRINGMAXLEN_SHMDIRNAME];
            function_parameter_struct_shmdirname(shmdname);


            //sprintf(logfname, "%s/fpslog.%06d", shmdname, getpid());
            WRITE_FULLFILENAME(logfname, "%s/fpslog.%06d", shmdname, getpid());

			SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "CREATE SYM LINK %s <- %s", FPSarg0, logfname);           
            functionparameter_outlog("INFO", msgstring);

            if( symlink(logfname, FPSarg0) != 0) {
                PRINT_ERROR("symlink error");
            }

        }
    }




    // queueprio
    if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "queueprio") == 0))
    {
        cmdFOUND = 1;
        if(nbword != 3) {
			SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND queueprio NBARGS = 2");
            functionparameter_outlog("ERROR", msgstring);
            cmdOK = 0;
        }
        else
        {
            int queue = atoi(FPSarg0);
            int prio = atoi(FPSarg1);

            if((queue>=0) && (queue<FPSTASK_MAX_NBQUEUE))
            {
                fpsctrlqueuelist[queue].priority = prio;
				
				SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "QUEUE %d PRIO = %d", queue, prio);               
                functionparameter_outlog("INFO", msgstring);
            }
        }
    }









    // From this point on, FPSarg0 is expected to be a FPS entry
    // so we resolve it and look for fps
    int kwnindex = -1;
    if(cmdFOUND == 0)
    {
        strcpy(FPSentryname, FPSarg0);
        strcpy(FPScmdarg1, FPSarg1);


        // look for entry, if found, kwnindex points to it
        if((nbword > 1) && (FPScommand[0] != '#')) {
            //                printf("Looking for entry for %s\n", FPSentryname);

            int kwnindexscan = 0;
            while((kwnindex == -1) && (kwnindexscan < NBkwn)) {
                if(strcmp(keywnode[kwnindexscan].keywordfull, FPSentryname) == 0) {
                    kwnindex = kwnindexscan;
                }
                kwnindexscan ++;
            }
        }

        //            sprintf(msgstring, "nbword = %d  cmdOK = %d   kwnindex = %d",  nbword, cmdOK, kwnindex);
        //            functionparameter_outlog("INFO", msgstring);
    }






    if(kwnindex != -1) {
        fpsindex = keywnode[kwnindex].fpsindex;
        pindex = keywnode[kwnindex].pindex;
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "FPS ENTRY FOUND : %-40s  %d %ld", FPSentryname, fpsindex, pindex);
        functionparameter_outlog("INFO", msgstring);
    }
    else
    {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "FPS ENTRY NOT FOUND : %-40s", FPSentryname);
        functionparameter_outlog("ERROR", msgstring);
        cmdOK = 0;
    }



    if(kwnindex != -1) { // if FPS has been found

        // confstart
        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "confstart") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG,"COMMAND confstart NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_CONFstart(fps, fpsindex);

                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "start CONF process %d %s", fpsindex, fps[fpsindex].md->name);
                functionparameter_outlog("CONFSTART", msgstring);
                cmdOK = 1;
            }
        }


        // confstop
        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "confstop") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND confstop NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_CONFstop(fps, fpsindex);

                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "stop CONF process %d %s", fpsindex, fps[fpsindex].md->name);
                functionparameter_outlog("CONFSTOP", msgstring);
                cmdOK = 1;
            }
        }










        // confupdate

        DEBUG_TRACEPOINT(" ");
        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "confupdate") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND confupdate NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED; // update status: check waiting to be done
                fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // request an update

                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "update CONF process %d %s", fpsindex, fps[fpsindex].md->name);
                functionparameter_outlog("CONFUPDATE", msgstring);
                cmdOK = 1;
            }
        }





        // confwupdate
        // Wait until update is cleared
        // if not successful, retry until time lapsed

        DEBUG_TRACEPOINT(" ");
        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "confwupdate") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND confwupdate NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                int looptry = 1;
                int looptrycnt = 0;
                unsigned int timercnt = 0;
                useconds_t dt = 100;
                unsigned int timercntmax = 10000; // 1 sec max

                while ( looptry == 1)
                {

                    DEBUG_TRACEPOINT(" ");
                    fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED; // update status: check waiting to be done
                    fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // request an update

                    while(  (( fps[fpsindex].md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED )) && (timercnt<timercntmax)) {
                        usleep(dt);
                        timercnt++;
                    }
                    usleep(dt);
                    timercnt++;

                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "[%d] waited %d us on FPS %d %s. conferrcnt = %d",
                        looptrycnt,
                        dt*timercnt,
                        fpsindex,
                        fps[fpsindex].md->name,
                        fps[fpsindex].md->conferrcnt);
                    functionparameter_outlog("CONFWUPDATE", msgstring);
                    looptrycnt++;

                    if(fps[fpsindex].md->conferrcnt == 0) { // no error ! we can proceed
                        looptry = 0;
                    }

                    if (timercnt > timercntmax) { // ran out of time ... giving up
                        looptry = 0;
                    }


                }

                cmdOK = 1;
            }
        }




        // runstart
        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "runstart") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND runstart NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_RUNstart(fps, fpsindex);

                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "start RUN process %d %s", fpsindex, fps[fpsindex].md->name);
                functionparameter_outlog("RUNSTART", msgstring);
                cmdOK = 1;

            }
        }



        // runwait
        // wait until run process is completed

        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "runwait") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND runwait NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");

                unsigned int timercnt = 0;
                useconds_t dt = 10000;
                unsigned int timercntmax = 100000; // 10000 sec max

                while(  (( fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN )) && (timercnt<timercntmax)) {
                    usleep(dt);
                    timercnt++;
                }

                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "waited %d us on FPS %d %s", dt*timercnt, fpsindex, fps[fpsindex].md->name);
                functionparameter_outlog("RUNWAIT", msgstring);
                cmdOK = 1;
            }
        }



        // runstop

        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "runstop") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND runstop NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_RUNstop(fps, fpsindex);

                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "stop RUN process %d %s", fpsindex, fps[fpsindex].md->name);
                functionparameter_outlog("RUNSTOP", msgstring);
                cmdOK = 1;
            }
        }




        // fpsrm

        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "fpsrm") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND fpsrm NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_FPSremove(fps, fpsindex);

                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "FPS remove %d %s", fpsindex, fps[fpsindex].md->name);
                functionparameter_outlog("FPSRM", msgstring);
                cmdOK = 1;
            }
        }








        DEBUG_TRACEPOINT(" ");





        // setval
        if((FPScommand[0] != '#') && (cmdFOUND == 0) && (strcmp(FPScommand, "setval") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 3) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND setval NBARGS = 2");
                functionparameter_outlog("ERROR", msgstring);
            }
            else
            {
                int updated = 0;

                switch(fps[fpsindex].parray[pindex].type) {

                case FPTYPE_INT64:
                    if(functionparameter_SetParamValue_INT64(&fps[fpsindex], FPSentryname, atol(FPScmdarg1)) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s INT64      %ld", FPSentryname, atol(FPScmdarg1));
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_FLOAT64:
                    if(functionparameter_SetParamValue_FLOAT64(&fps[fpsindex], FPSentryname, atof(FPScmdarg1)) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s FLOAT64    %f", FPSentryname, atof(FPScmdarg1));
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_FLOAT32:
                    if(functionparameter_SetParamValue_FLOAT32(&fps[fpsindex], FPSentryname, atof(FPScmdarg1)) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s FLOAT32    %f", FPSentryname, atof(FPScmdarg1));
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_PID:
                    if(functionparameter_SetParamValue_INT64(&fps[fpsindex], FPSentryname, atol(FPScmdarg1)) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s PID        %ld", FPSentryname, atol(FPScmdarg1));
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_TIMESPEC:
                    //
                    break;

                case FPTYPE_FILENAME:
                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPScmdarg1) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s FILENAME   %s", FPSentryname, FPScmdarg1);
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_FITSFILENAME:
                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPScmdarg1) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s FITSFILENAME   %s", FPSentryname, FPScmdarg1);
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_EXECFILENAME:
                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPScmdarg1) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s EXECFILENAME   %s", FPSentryname, FPScmdarg1);
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_DIRNAME:
                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPScmdarg1) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s DIRNAME    %s", FPSentryname, FPScmdarg1);
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_STREAMNAME:
                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPScmdarg1) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s STREAMNAME %s", FPSentryname, FPScmdarg1);
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_STRING:
                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPScmdarg1) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s STRING     %s", FPSentryname, FPScmdarg1);
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                case FPTYPE_ONOFF:
                    if(strncmp(FPScmdarg1, "ON", 2) == 0) {
                        if(functionparameter_SetParamValue_ONOFF(&fps[fpsindex], FPSentryname, 1) == EXIT_SUCCESS) {
                            updated = 1;
                        }
                        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s ONOFF      ON", FPSentryname);
                        functionparameter_outlog("SETVAL", msgstring);
                    }
                    if(strncmp(FPScmdarg1, "OFF", 3) == 0) {
                        if(functionparameter_SetParamValue_ONOFF(&fps[fpsindex], FPSentryname, 0) == EXIT_SUCCESS) {
                            updated = 1;
                        }
                        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s ONOFF      OFF", FPSentryname);
                        functionparameter_outlog("SETVAL", msgstring);
                    }
                    break;


                case FPTYPE_FPSNAME:
                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPScmdarg1) == EXIT_SUCCESS) {
                        updated = 1;
                    }
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s FPSNAME   %s", FPSentryname, FPScmdarg1);
                    functionparameter_outlog("SETVAL", msgstring);
                    break;

                }

                // notify fpsCTRL that parameter has been updated
                if(updated == 1) {
                    cmdOK = 1;
                    functionparameter_WriteParameterToDisk(&fps[fpsindex], pindex, "setval", "input command file");
                    fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                }
                else
                    cmdOK = 0;

            }
        }





        // getval or fwrval
        if((FPScommand[0] != '#') && (cmdFOUND == 0) && ((strcmp(FPScommand, "getval")==0) || (strcmp(FPScommand, "fwrval")== 0)) )
        {
            cmdFOUND = 1;
            cmdOK = 0;

            if((strcmp(FPScommand, "getval")==0)&&(nbword != 2)) {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND getval NBARGS = 1");
                functionparameter_outlog("ERROR", msgstring);
            }
            else if ((strcmp(FPScommand, "fwrval")==0)&&(nbword != 3))
            {
                SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND fwrval NBARGS = 2");
                functionparameter_outlog("ERROR", msgstring);
            }
            else
            {
                switch(fps[fpsindex].parray[pindex].type) {

                case FPTYPE_INT64:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s INT64      %ld %ld %ld %ld",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.l[0],
                        fps[fpsindex].parray[pindex].val.l[1],
                        fps[fpsindex].parray[pindex].val.l[2],
                        fps[fpsindex].parray[pindex].val.l[3]);
                    cmdOK = 1;
                    break;

                case FPTYPE_FLOAT64:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s FLOAT64    %f %f %f %f",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.f[0],
                        fps[fpsindex].parray[pindex].val.f[1],
                        fps[fpsindex].parray[pindex].val.f[2],
                        fps[fpsindex].parray[pindex].val.f[3]);
                    cmdOK = 1;
                    break;

                case FPTYPE_FLOAT32:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s FLOAT32    %f %f %f %f",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.s[0],
                        fps[fpsindex].parray[pindex].val.s[1],
                        fps[fpsindex].parray[pindex].val.s[2],
                        fps[fpsindex].parray[pindex].val.s[3]);
                    cmdOK = 1;
                    break;

                case FPTYPE_PID:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s PID        %ld",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.l[0]);
                    cmdOK = 1;
                    break;

                case FPTYPE_TIMESPEC:
                    //
                    break;

                case FPTYPE_FILENAME:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s FILENAME   %s",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.string[0]);
                    cmdOK = 1;
                    break;

                case FPTYPE_FITSFILENAME:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s FITSFILENAME   %s",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.string[0]);
                    cmdOK = 1;
                    break;

                case FPTYPE_EXECFILENAME:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s EXECFILENAME   %s",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.string[0]);
                    cmdOK = 1;
                    break;

                case FPTYPE_DIRNAME:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s DIRNAME    %s",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.string[0]);
                    cmdOK = 1;
                    break;

                case FPTYPE_STREAMNAME:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s STREAMNAME %s",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.string[0]);
                    cmdOK = 1;
                    break;

                case FPTYPE_STRING:
                    SNPRINTF_CHECK(
                        msgstring,
                        STRINGMAXLEN_FPS_LOGMSG,
                        "%-40s STRING     %s",
                        FPSentryname,
                        fps[fpsindex].parray[pindex].val.string[0]);
                    cmdOK = 1;
                    break;

                case FPTYPE_ONOFF:
                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ONOFF) {
                        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s ONOFF      ON", FPSentryname);
                    }
                    else {
                        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s ONOFF      OFF", FPSentryname);
                    }
                    cmdOK = 1;
                    break;


                case FPTYPE_FPSNAME:
                    SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s FPSNAME   %s", FPSentryname, fps[fpsindex].parray[pindex].val.string[0]);
                    cmdOK = 1;
                    break;

                }

                if(cmdOK==1) {
                    if(strcmp(FPScommand, "getval")==0) {
                        functionparameter_outlog("GETVAL", msgstring);
                    }
                    if(strcmp(FPScommand, "fwrval")==0) {

                        FILE *fpouttmp = fopen(FPScmdarg1, "a");
                        functionparameter_outlog_file("FWRVAL", msgstring, fpouttmp);
                        fclose(fpouttmp);

                        functionparameter_outlog("FWRVAL", msgstring);
                        char msgstring1[STRINGMAXLEN_FPS_LOGMSG];
                        SNPRINTF_CHECK(msgstring1, STRINGMAXLEN_FPS_LOGMSG, "WROTE to file %s", FPScmdarg1);
                        functionparameter_outlog("FWRVAL", msgstring1);
                    }
                }

            }
        }


    }


    if(cmdOK == 0) {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "\"%s\"", FPScmdline);
        functionparameter_outlog("CMDFAIL", msgstring);
    }

    if(cmdOK == 1) {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "\"%s\"", FPScmdline);
        functionparameter_outlog("CMDOK", msgstring);
    }

    if(cmdFOUND == 0) {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND NOT FOUND: %s", FPScommand);
        functionparameter_outlog("ERROR", msgstring);
    }


    DEBUG_TRACEPOINT(" ");


    return (fpsindex);
}







// fill up task list from fifo submissions

int functionparameter_read_fpsCMD_fifo(
    int fpsCTRLfifofd,
    FPSCTRL_TASK_ENTRY *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist
) {
    int cmdcnt = 0;
    char *FPScmdline = NULL;
    char buff[200];
    int total_bytes = 0;
    int bytes;
    char buf0[1];

    // toggles
    static uint32_t queue = 0;
    static int waitonrun = 0;
    static int waitonconf = 0;



    static uint16_t	cmdinputcnt = 0;

    int lineOK = 1; // keep reading


    DEBUG_TRACEPOINT(" ");

    while(lineOK == 1) {
        total_bytes = 0;
        lineOK = 0;
        for(;;) {
            bytes = read(fpsCTRLfifofd, buf0, 1);  // read one char at a time
            DEBUG_TRACEPOINT("ERRROR: BUFFER OVERFLOW %d %d\n", bytes, total_bytes);
            if(bytes > 0) {
                buff[total_bytes] = buf0[0];
                total_bytes += (size_t)bytes;

            } else {
                if(errno == EWOULDBLOCK) {
                    break;
                } else { // read 0 byte
                    //perror("read 0 byte");
                    return cmdcnt;
                }
            }


            DEBUG_TRACEPOINT(" ");


            if(buf0[0] == '\n') {  // reached end of line
                buff[total_bytes - 1] = '\0';
                FPScmdline = buff;



                // find next index
                int cmdindex = 0;
                int cmdindexOK = 0;
                while((cmdindexOK==0)&&(cmdindex<NB_FPSCTRL_TASK_MAX)) {
                    if(fpsctrltasklist[cmdindex].status == 0) {
                        cmdindexOK = 1;
                    }
                    else
                        cmdindex ++;
                }


                if(cmdindex==NB_FPSCTRL_TASK_MAX) {
                    printf("ERROR: fpscmdarray is full\n");
                    exit(0);
                }


                DEBUG_TRACEPOINT(" ");

                // Some commands affect how the task list is configured instead of being inserted as entries
                int cmdFOUND = 0;


                if ( (FPScmdline[0] == '#') ||  (FPScmdline[0] == ' ') ) { // disregard line
                    cmdFOUND = 1;
                }

                // set wait on run ON
                if((FPScmdline[0] != '#') && (cmdFOUND == 0) && (strncmp(FPScmdline, "taskcntzero", strlen("taskcntzero")) == 0)) {
                    cmdFOUND = 1;
                    cmdinputcnt = 0;
                }

                // Set queue index
                // entries will now be placed in queue specified by this command
                if((FPScmdline[0] != '#') && (cmdFOUND == 0) && (strncmp(FPScmdline, "setqindex", strlen("setqindex")) == 0)) {
                    cmdFOUND = 1;
                    char stringtmp[200];
                    int queue_index;
                    sscanf(FPScmdline, "%s %d", stringtmp, &queue_index);

                    if((queue_index > -1)&&(queue_index<FPSTASK_MAX_NBQUEUE))
                        queue = queue_index;
                }

                // Set queue priority
                if((FPScmdline[0] != '#') && (cmdFOUND == 0) && (strncmp(FPScmdline, "setqprio", strlen("setqprio")) == 0)) {
                    cmdFOUND = 1;
                    char stringtmp[200];
                    int queue_priority;
                    sscanf(FPScmdline, "%s %d", stringtmp, &queue_priority);

                    if(queue_priority < 0)
                        queue_priority = 0;

                    fpsctrlqueuelist[queue].priority = queue_priority;
                }



                // set wait on run ON
                if((FPScmdline[0] != '#') && (cmdFOUND == 0) && (strncmp(FPScmdline, "waitonrunON", strlen("waitonrunON")) == 0)) {
                    cmdFOUND = 1;
                    waitonrun = 1;
                }

                // set wait on run OFF
                if((FPScmdline[0] != '#') && (cmdFOUND == 0) && (strncmp(FPScmdline, "waitonrunOFF", strlen("waitonrunOFF")) == 0)) {
                    cmdFOUND = 1;
                    waitonrun = 0;
                }

                // set wait on conf ON
                if((FPScmdline[0] != '#') && (cmdFOUND == 0) && (strncmp(FPScmdline, "waitonconfON", strlen("waitonconfON")) == 0)) {
                    cmdFOUND = 1;
                    waitonconf = 1;
                }

                // set wait on conf OFF
                if((FPScmdline[0] != '#') && (cmdFOUND == 0) && (strncmp(FPScmdline, "waitonconfOFF", strlen("waitonconfOFF")) == 0)) {
                    cmdFOUND = 1;
                    waitonconf = 0;
                }


                // set wait point for arbitrary FPS run to have finished

                DEBUG_TRACEPOINT(" ");

                // for all other commands, put in task list
                if(cmdFOUND == 0) {
                    
                    strncpy(fpsctrltasklist[cmdindex].cmdstring, FPScmdline, STRINGMAXLEN_FPS_CMDLINE);
                    
                    fpsctrltasklist[cmdindex].status = FPSTASK_STATUS_ACTIVE | FPSTASK_STATUS_SHOW;
                    fpsctrltasklist[cmdindex].inputindex = cmdinputcnt;
                    fpsctrltasklist[cmdindex].queue = queue;
                    clock_gettime(CLOCK_REALTIME, &fpsctrltasklist[cmdindex].creationtime);

                    if(waitonrun==1) {
                        fpsctrltasklist[cmdindex].flag |= FPSTASK_FLAG_WAITONRUN;
                    } else {
                        fpsctrltasklist[cmdindex].flag &= ~FPSTASK_FLAG_WAITONRUN;
                    }

                    if(waitonconf==1) {
                        fpsctrltasklist[cmdindex].flag |= FPSTASK_FLAG_WAITONCONF;
                    } else {
                        fpsctrltasklist[cmdindex].flag &= ~FPSTASK_FLAG_WAITONCONF;
                    }

                    cmdinputcnt++;

                    cmdcnt ++;
                }
                lineOK = 1;
                break;
            }
        }

    }

    DEBUG_TRACEPOINT(" ");

    return cmdcnt;
}




// find next command to execute
//
static errno_t function_parameter_process_fpsCMDarray(
    FPSCTRL_TASK_ENTRY         *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE         *fpsctrlqueuelist,
    KEYWORD_TREE_NODE          *keywnode,
    int                         NBkwn,
    FUNCTION_PARAMETER_STRUCT  *fps
) {
    // the scheduler handles multiple queues
    // in each queue, we look for a task to run, and run it if conditions are met


    // sort priorities
    long *queuepriolist = (long*) malloc(sizeof(long)*NB_FPSCTRL_TASKQUEUE_MAX);
    for( int queue = 0; queue<NB_FPSCTRL_TASKQUEUE_MAX; queue++) {
        queuepriolist[queue] = fpsctrlqueuelist[queue].priority;
    }
    quick_sort_long(queuepriolist, NB_FPSCTRL_TASKQUEUE_MAX);

    for( int qi = NB_FPSCTRL_TASKQUEUE_MAX-1; qi > 0; qi--)
    {
        int priority = queuepriolist[qi];
        if(priority > 0)
        {

            for( unsigned int queue = 0; queue<NB_FPSCTRL_TASKQUEUE_MAX; queue++)
            {
                if(priority == fpsctrlqueuelist[queue].priority) {

                    // find next command to execute
                    uint64_t inputindexmin = UINT_MAX;
                    int cmdindexExec;
                    int cmdOK = 0;


                    for(int cmdindex=0; cmdindex < NB_FPSCTRL_TASK_MAX; cmdindex++) {
                        if((fpsctrltasklist[cmdindex].status & FPSTASK_STATUS_ACTIVE) && (fpsctrltasklist[cmdindex].queue == queue) ) {
                            if(fpsctrltasklist[cmdindex].inputindex < inputindexmin) {
                                inputindexmin = fpsctrltasklist[cmdindex].inputindex;
                                cmdindexExec = cmdindex;
                                cmdOK = 1;
                            }
                        }
                    }


                    if(cmdOK == 1) {
                        if( !(fpsctrltasklist[cmdindexExec].status & FPSTASK_STATUS_RUNNING) ) { // if not running, launch it
                            fpsctrltasklist[cmdindexExec].fpsindex =
                                functionparameter_FPSprocess_cmdline(fpsctrltasklist[cmdindexExec].cmdstring, fpsctrlqueuelist, keywnode, NBkwn, fps);
                            clock_gettime(CLOCK_REALTIME, &fpsctrltasklist[cmdindexExec].activationtime);
                            fpsctrltasklist[cmdindexExec].status |= FPSTASK_STATUS_RUNNING; // update status to running
                        }
                        else
                        {   // if it's already running, lets check if it is completed
                            int task_completed = 1; // default

                            if(fpsctrltasklist[cmdindexExec].flag & FPSTASK_FLAG_WAITONRUN) { // are we waiting for run to be completed ?
                                if ((fps[fpsctrltasklist[cmdindexExec].fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)) {
                                    task_completed = 0;
                                }
                            }
                            if(fpsctrltasklist[cmdindexExec].flag & FPSTASK_FLAG_WAITONCONF) { // are we waiting for conf update to be completed ?
                                if (fps[fpsctrltasklist[cmdindexExec].fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED) {
                                    task_completed = 0;
                                }
                            }

                            if(task_completed == 1) {
                                fpsctrltasklist[cmdindexExec].status &= ~FPSTASK_STATUS_RUNNING; // update status - no longer running
                                fpsctrltasklist[cmdindexExec].status &= ~FPSTASK_STATUS_ACTIVE; //no longer active, remove it from list
                                //   fpsctrltasklist[cmdindexExec].status &= ~FPSTASK_STATUS_SHOW; // and stop displaying

                                clock_gettime(CLOCK_REALTIME, &fpsctrltasklist[cmdindexExec].completiontime);
                            }
                        }
                    } // end cmdOK


                }
            }
        }
    }
    free(queuepriolist);

    return RETURN_SUCCESS;
}







errno_t functionparameter_RUNstart(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
) {
    int  stringmaxlen = 500;
    char command[stringmaxlen];

    if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK) {

        if ( snprintf(command, stringmaxlen, "tmux new-session -d -s %s-run > /dev/null 2>&1", fps[fpsindex].md->name) < 0 ) {
            PRINT_ERROR("snprintf error");
        }

        if(system(command) != 0) {
            // this is probably OK - duplicate session
            //printf("command: \"%s\"\n", command);
            //PRINT_ERROR("system() returns non-zero value");
            //printf("This error message may be due to pre-existing session\n");
        }


        // Move to correct launch directory
        if (snprintf(command, stringmaxlen, "tmux send-keys -t %s-run \"cd %s\" C-m", fps[fpsindex].md->name, fps[fpsindex].md->fpsdirectory) < 0 ) {
            PRINT_ERROR("snprintf error");
        }


        if(system(command) != 0) {
            PRINT_ERROR("system() returns non-zero value");
        }

        if (snprintf(command, stringmaxlen, "tmux send-keys -t %s-run \"./fpscmd/%s-runstart", fps[fpsindex].md->name, fps[fpsindex].md->pname) < 0 ) {
            PRINT_ERROR("snprintf error");
        }

        for(int nameindexlevel = 0; nameindexlevel < fps[fpsindex].md->NBnameindex; nameindexlevel++) {
            int tmpstrlen = 20;
            char tmpstring[tmpstrlen];

            if (snprintf(tmpstring, tmpstrlen, " %s", fps[fpsindex].md->nameindexW[nameindexlevel]) < 0 ) {
                PRINT_ERROR("snprintf error");
            }

            strcat(command, tmpstring);
        }
        strcat(command, "\" C-m");
        if(system(command) != 0) {
            PRINT_ERROR("system() returns non-zero value");
        }
        fps[fpsindex].md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
        fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
    }
    return RETURN_SUCCESS;
}






errno_t functionparameter_RUNstop(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
) {
    int stringmaxlen = 500;
    char command[stringmaxlen];


    // First, run the runstop command
    if (snprintf(command, stringmaxlen, "%s/fpscmd/%s-runstop", fps[fpsindex].md->fpsdirectory, fps[fpsindex].md->pname) < 0 ) {
        PRINT_ERROR("snprintf error");
    }

    for(int nameindexlevel = 0; nameindexlevel < fps[fpsindex].md->NBnameindex; nameindexlevel++) {
        int tmpstrlen = 20;
        char tmpstring[tmpstrlen];

        snprintf(tmpstring, tmpstrlen, " %s", fps[fpsindex].md->nameindexW[nameindexlevel]);
        strcat(command, tmpstring);
    }
    if(system(command) != 0) {
        //PRINT_ERROR("system() returns non-zero value");
    }
    fps[fpsindex].md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update



    // Send C-c in case runstop command is not implemented
    if (snprintf(command, stringmaxlen, "tmux send-keys -t %s-run C-c &> /dev/null", fps[fpsindex].md->name) < 0 ) {
        PRINT_ERROR("snprintf error");
    }

    if(system(command) != 0) {
        //PRINT_ERROR("system() returns non-zero value");
    }

    return RETURN_SUCCESS;
}





errno_t functionparameter_CONFstart(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
) {
    int stringmaxlen = 500;
    char command[stringmaxlen];


    if ( snprintf(command, stringmaxlen, "tmux new-session -d -s %s-conf > /dev/null 2>&1", fps[fpsindex].md->name) < 0 ) {
        PRINT_ERROR("snprintf error");
    }

    if(system(command) != 0) {
        // this is probably OK - duplicate session warning
        //PRINT_ERROR("system() returns non-zero value");
    }

    // Move to correct launch directory
    if (snprintf(command, stringmaxlen, "tmux send-keys -t %s-conf \"cd %s\" C-m", fps[fpsindex].md->name, fps[fpsindex].md->fpsdirectory)< 0 ) {
        PRINT_ERROR("snprintf error");
    }

    if(system(command) != 0) {
        PRINT_ERROR("system() returns non-zero value");
    }


    if (snprintf(command, stringmaxlen, "tmux send-keys -t %s-conf \"./fpscmd/%s-confstart", fps[fpsindex].md->name, fps[fpsindex].md->pname) < 0 ) {
        PRINT_ERROR("snprintf error");
    }

    for(int nameindexlevel = 0; nameindexlevel < fps[fpsindex].md->NBnameindex; nameindexlevel++) {
        int tmpstrlen = 20;
        char tmpstring[tmpstrlen];

        if (snprintf(tmpstring, tmpstrlen, " %s", fps[fpsindex].md->nameindexW[nameindexlevel])< 0 ) {
            PRINT_ERROR("snprintf error");
        }

        strcat(command, tmpstring);
    }
    strcat(command, "\" C-m");
    if(system(command) != 0) {
        PRINT_ERROR("system() returns non-zero value");
    }
    fps[fpsindex].md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update

    return RETURN_SUCCESS;
}





errno_t functionparameter_CONFstop(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
) {
    int stringmaxlen = 500;
    char command[stringmaxlen];

    fps[fpsindex].md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
    if (snprintf(command, stringmaxlen, "tmux send-keys -t %s-conf C-c &> /dev/null", fps[fpsindex].md->name)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    if(system(command) != 0) {
        PRINT_ERROR("system() returns non-zero value");
    }
    fps[fpsindex].md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update

    return RETURN_SUCCESS;
}





errno_t functionparameter_FPSremove(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex
) {
    int stringmaxlen = 500;
    char command[stringmaxlen];


    functionparameter_RUNstop(fps, fpsindex);
    functionparameter_CONFstop(fps, fpsindex);


    char shmdname[stringmaxlen];
    function_parameter_struct_shmdirname(shmdname);

    // conf log
    char conflogfname[stringmaxlen];
    if(snprintf(conflogfname, stringmaxlen, "%s/fpslog.%06d", shmdname, fps[fpsindex].md->confpid)< 0 ) {
        PRINT_ERROR("snprintf error");
    }

    // FPS shm
    char fpsfname[stringmaxlen];
    if(snprintf(fpsfname, stringmaxlen, "%s/%s.fps.shm", shmdname, fps[fpsindex].md->name)< 0 ) {
        PRINT_ERROR("snprintf error");
    }


    fps[fpsindex].SMfd = -1;
    close(fps[fpsindex].SMfd);

    remove(conflogfname);
    remove(fpsfname);



    if(snprintf(command, stringmaxlen, "tmux send-keys -t %s-run \"exit\" C-m", fps[fpsindex].md->name)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    if(system(command) != 0) {
        PRINT_ERROR("system() returns non-zero value");
    }
    if (snprintf(command, stringmaxlen, "tmux send-keys -t %s-conf \"exit\" C-m", fps[fpsindex].md->name)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    if(system(command) != 0) {
        PRINT_ERROR("system() returns non-zero value");
    }



    return RETURN_SUCCESS;
}














errno_t functionparameter_outlog_file(
    char *keyw,
    char *msgstring,
    FILE *fpout
) {
    // Get GMT time
    struct timespec tnow;
    time_t now;

    clock_gettime(CLOCK_REALTIME, &tnow);
    now = tnow.tv_sec;
    struct tm *uttime;
    uttime = gmtime(&now);

    char timestring[30];
    sprintf(
        timestring,        
        "%04d%02d%02dT%02d%02d%02d.%09ld",
        1900+uttime->tm_year,
        1+uttime->tm_mon,
        uttime->tm_mday,
        uttime->tm_hour,
        uttime->tm_min,
        uttime->tm_sec,
        tnow.tv_nsec);

    fprintf(fpout, "%s %-12s %s\n", timestring, keyw, msgstring);
    fflush(fpout);

    return RETURN_SUCCESS;
}






errno_t functionparameter_outlog(
    char *keyw,
    char *msgstring
) {
    static int LogOutOpen = 0;
    static FILE *fpout;


    if(LogOutOpen == 0) {
        char logfname[STRINGMAXLEN_FULLFILENAME];
        char shmdname[STRINGMAXLEN_SHMDIRNAME];
        function_parameter_struct_shmdirname(shmdname);

        WRITE_FULLFILENAME(logfname, "%s/fpslog.%06d", shmdname, getpid());
        
        fpout = fopen(logfname, "a");
        if(fpout == NULL) {
            printf("ERROR: cannot open file\n");
            exit(EXIT_FAILURE);
        }
        LogOutOpen = 1;
    }

    functionparameter_outlog_file(keyw, msgstring, fpout);

    if(strcmp(keyw, "LOGFILECLOSE") == 0) {
        fclose(fpout);
        LogOutOpen = 1;
    }

    return RETURN_SUCCESS;
}















static errno_t functionparameter_scan_fps(
    uint32_t mode,
    char *fpsnamemask,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    int *ptr_NBkwn,
    int *ptr_fpsindex,
    long *ptr_pindex,
    int verbose
) {
    int stringmaxlen = 500;
    int fpsindex;
    int pindex;
    //int fps_symlink[NB_FPS_MAX];
    int kwnindex;
    int NBkwn;
    int l;


    // int nodechain[MAXNBLEVELS];
    //int GUIlineSelected[MAXNBLEVELS];

    // FPS list file
    FILE *fpfpslist;
    int fpslistcnt = 0;
    char FPSlist[200][100];



    // Static variables
    static int shmdirname_init = 0;
    static char shmdname[200];




    // scan filesystem for fps entries

    if(verbose > 0) {
        printf("\n\n\n====================== SCANNING FPS ON SYSTEM ==============================\n\n");
        fflush(stdout);
    }


    if(shmdirname_init == 0)   {
        function_parameter_struct_shmdirname(shmdname);
        shmdirname_init = 1;
    }





    // disconnect previous fps
    for(fpsindex=0; fpsindex<NB_FPS_MAX; fpsindex++) {
        if(fps[fpsindex].SMfd > -1) { // connected
            function_parameter_struct_disconnect(&fps[fpsindex]);
        }
    }



    // request match to file ./fpscomd/fpslist.txt
    if(mode & 0x0001) {
        if((fpfpslist = fopen("fpscmd/fpslist.txt", "r")) != NULL) {
            char *FPSlistline = NULL;
            size_t len = 0;
            ssize_t read;

            while((read = getline(&FPSlistline, &len, fpfpslist)) != -1) {
                if(FPSlistline[0] != '#') {
                    char *pch;

                    pch = strtok(FPSlistline, " \t\n\r");
                    if(pch != NULL) {
                        sprintf(FPSlist[fpslistcnt], "%s", pch);
                        fpslistcnt++;
                    }
                }
            }
            fclose(fpfpslist);
        } else {
            if(verbose > 0) {
                printf("Cannot open file fpscmd/fpslist.txt\n");
            }
        }

        int fpsi;
        for(fpsi = 0; fpsi < fpslistcnt; fpsi++) {
            if(verbose > 0) {
                printf("FPSname must match %s\n", FPSlist[fpsi]);
            }
        }
    }




    //  for(l = 0; l < MAXNBLEVELS; l++) {
    // nodechain[l] = 0;
    // GUIlineSelected[l] = 0;
    //}

    for( int kindex = 0; kindex <NB_KEYWNODE_MAX; kindex++ ) {
        keywnode[kindex].NBchild = 0;
    }




    //    NBparam = function_parameter_struct_connect(fpsname, &fps[fpsindex]);


    // create ROOT node (invisible)
    keywnode[0].keywordlevel = 0;
    sprintf(keywnode[0].keyword[0], "ROOT");
    keywnode[0].leaf = 0;
    keywnode[0].NBchild = 0;
    NBkwn = 1;









    DIR *d;
    struct dirent *dir;
    d = opendir(shmdname);
    if(d) {
        fpsindex = 0;
        pindex = 0;
        while(((dir = readdir(d)) != NULL)) {
            char *pch = strstr(dir->d_name, ".fps.shm");

            int matchOK = 0;
            // name filtering
            if(strcmp(fpsnamemask, "_ALL") == 0) {
                matchOK = 1;
            } else {
                if(strncmp(dir->d_name, fpsnamemask, strlen(fpsnamemask)) == 0) {
                    matchOK = 1;
                }
            }


            if(mode & 0x0001) { // enforce match to list
                int matchOKlist = 0;
                int fpsi;

                for(fpsi = 0; fpsi < fpslistcnt; fpsi++)
                    if(strncmp(dir->d_name, FPSlist[fpsi], strlen(FPSlist[fpsi])) == 0) {
                        matchOKlist = 1;
                    }

                matchOK *= matchOKlist;
            }




            if((pch) && (matchOK == 1)) {

                // is file sym link ?
                struct stat buf;
                int retv;
                char fullname[stringmaxlen];
                char shmdname[stringmaxlen];
                function_parameter_struct_shmdirname(shmdname);

                sprintf(fullname, "%s/%s", shmdname, dir->d_name);

                retv = lstat(fullname, &buf);
                if(retv == -1) {
                    endwin();
                    printf("File \"%s\"", dir->d_name);
                    perror("Error running lstat on file ");
                    printf("File %s line %d\n", __FILE__, __LINE__);
                    fflush(stdout);
                    exit(EXIT_FAILURE);
                }

                if(S_ISLNK(buf.st_mode)) { // resolve link name
                    char fullname[stringmaxlen];
                    char linknamefull[stringmaxlen];
                    char linkname[stringmaxlen];

                    char shmdname[stringmaxlen];
                    function_parameter_struct_shmdirname(shmdname);

                    //fps_symlink[fpsindex] = 1;
                    if (snprintf(fullname, stringmaxlen, "%s/%s", shmdname, dir->d_name)< 0 ) {
                        PRINT_ERROR("snprintf error");
                    }

                    if ( readlink(fullname, linknamefull, 200 - 1) == -1 ) {
                        // todo: replace with realpath()
                        PRINT_ERROR("readlink() error");
                    }
                    strcpy(linkname, basename(linknamefull));

                    int lOK = 1;
                    unsigned int ii = 0;
                    while((lOK == 1) && (ii < strlen(linkname))) {
                        if(linkname[ii] == '.') {
                            linkname[ii] = '\0';
                            lOK = 0;
                        }
                        ii++;
                    }

                    //                        strncpy(streaminfo[sindex].linkname, linkname, nameNBchar);
                }
                //else {
                //  fps_symlink[fpsindex] = 0;
                //}


                //fps_symlink[fpsindex] = 0;


                char fpsname[stringmaxlen];
                long strcplen = strlen(dir->d_name) - strlen(".fps.shm");
                strncpy(fpsname, dir->d_name, strcplen);
                fpsname[strcplen] = '\0';

                if(verbose > 0) {
                    printf("FOUND FPS %s - (RE)-CONNECTING  [%d]\n", fpsname, fpsindex);
                    fflush(stdout);
                }


                long NBparamMAX = function_parameter_struct_connect(fpsname, &fps[fpsindex], FPSCONNECT_SIMPLE);


                long pindex0;
                for(pindex0 = 0; pindex0 < NBparamMAX; pindex0++) {
                    if(fps[fpsindex].parray[pindex0].fpflag & FPFLAG_ACTIVE) { // if entry is active
                        // find or allocate keyword node
                        int level;
                        for(level = 1; level < fps[fpsindex].parray[pindex0].keywordlevel + 1; level++) {

                            // does node already exist ?
                            int scanOK = 0;
                            for(kwnindex = 0; kwnindex < NBkwn; kwnindex++) { // scan existing nodes looking for match
                                if(keywnode[kwnindex].keywordlevel == level) { // levels have to match
                                    int match = 1;
                                    for(l = 0; l < level; l++) { // keywords at all levels need to match
                                        if(strcmp(fps[fpsindex].parray[pindex0].keyword[l], keywnode[kwnindex].keyword[l]) != 0) {
                                            match = 0;
                                        }
                                        //                        printf("TEST MATCH : %16s %16s  %d\n", fps[fpsindex].parray[i].keyword[l], keywnode[kwnindex].keyword[l], match);
                                    }
                                    if(match == 1) { // we have a match
                                        scanOK = 1;
                                    }
                                    //             printf("   -> %d\n", scanOK);
                                }
                            }



                            if(scanOK == 0) { // node does not exit -> create it

                                // look for parent
                                int scanparentOK = 0;
                                int kwnindexp = 0;
                                keywnode[kwnindex].parent_index = 0; // default value, not found -> assigned to ROOT

                                while((kwnindexp < NBkwn) && (scanparentOK == 0)) {
                                    if(keywnode[kwnindexp].keywordlevel == level - 1) { // check parent has level-1
                                        int match = 1;

                                        for(l = 0; l < level - 1; l++) { // keywords at all levels need to match
                                            if(strcmp(fps[fpsindex].parray[pindex0].keyword[l], keywnode[kwnindexp].keyword[l]) != 0) {
                                                match = 0;
                                            }
                                        }
                                        if(match == 1) { // we have a match
                                            scanparentOK = 1;
                                        }
                                    }
                                    kwnindexp++;
                                }

                                if(scanparentOK == 1) {
                                    keywnode[kwnindex].parent_index = kwnindexp - 1;
                                    int cindex;
                                    cindex = keywnode[keywnode[kwnindex].parent_index].NBchild;
                                    keywnode[keywnode[kwnindex].parent_index].child[cindex] = kwnindex;
                                    keywnode[keywnode[kwnindex].parent_index].NBchild++;
                                }

                                if(verbose > 0) {
                                    printf("CREATING NODE %d ", kwnindex);
                                }
                                keywnode[kwnindex].keywordlevel = level;

                                for(l = 0; l < level; l++) {
                                    char tmpstring[200];
                                    strcpy(keywnode[kwnindex].keyword[l], fps[fpsindex].parray[pindex0].keyword[l]);
                                    printf(" %s", keywnode[kwnindex].keyword[l]);
                                    if(l == 0) {
                                        strcpy(keywnode[kwnindex].keywordfull, keywnode[kwnindex].keyword[l]);
                                    } else {
                                        sprintf(tmpstring, ".%s", keywnode[kwnindex].keyword[l]);
                                        strcat(keywnode[kwnindex].keywordfull, tmpstring);
                                    }
                                }
                                if(verbose > 0) {
                                    printf("   %d %d\n", keywnode[kwnindex].keywordlevel, fps[fpsindex].parray[pindex0].keywordlevel);
                                }

                                if(keywnode[kwnindex].keywordlevel == fps[fpsindex].parray[pindex0].keywordlevel) {
                                    //									strcpy(keywnode[kwnindex].keywordfull, fps[fpsindex].parray[i].keywordfull);

                                    keywnode[kwnindex].leaf = 1;
                                    keywnode[kwnindex].fpsindex = fpsindex;
                                    keywnode[kwnindex].pindex = pindex0;
                                } else {


                                    keywnode[kwnindex].leaf = 0;
                                    keywnode[kwnindex].fpsindex = fpsindex;
                                    keywnode[kwnindex].pindex = 0;
                                }

                                kwnindex ++;
                                NBkwn = kwnindex;
                            }
                        }
                        pindex++;
                    }
                }

                if(verbose > 0) {
                    printf("--- FPS %4d  %-20s %ld parameters\n", fpsindex, fpsname, fps[fpsindex].md->NBparamMAX);
                }


                fpsindex ++;
            }
        }
        closedir(d);
    } else {
        char shmdname[200];
        function_parameter_struct_shmdirname(shmdname);
        printf("ERROR: missing %s directory\n", shmdname);
        printf("File %s line %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(EXIT_FAILURE);
    }


    if(verbose > 0) {
        printf("\n\n=================[END] SCANNING FPS ON SYSTEM [END]=  %d  ========================\n\n\n", fpsindex);
        fflush(stdout);
    }

    *ptr_NBkwn = NBkwn;
    *ptr_fpsindex = fpsindex;
    *ptr_pindex = pindex;



    return RETURN_SUCCESS;
}










void functionparameter_CTRLscreen_atexit()
{
    //printf("exiting CTRLscreen\n");

    // endwin();
}





inline static void print_help_entry(char *key, char *descr) {
    int attrval = A_BOLD;

    attron(attrval);
    printw("    %4s", key);
    attroff(attrval);
    printw("   %s\n", descr);
}





inline static void fpsCTRLscreen_print_DisplayMode_status(int fpsCTRL_DisplayMode, int NBfps) {

    int stringmaxlen = 500;
    char  monstring[stringmaxlen];

    attron(A_BOLD);
    if (snprintf(monstring, stringmaxlen, "[%d %d] FUNCTION PARAMETER MONITOR: PRESS (x) TO STOP, (h) FOR HELP   PID %d  [%d FPS]", wrow, wcol, (int) getpid(), NBfps)< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    print_header(monstring, '-');
    attroff(A_BOLD);
    printw("\n");

    if(fpsCTRL_DisplayMode==1)    {
        attron(A_REVERSE);
        printw("[h] Help");
        attroff(A_REVERSE);
    }            else {
        printw("[h] Help");
    }
    printw("   ");

    if(fpsCTRL_DisplayMode==2)            {
        attron(A_REVERSE);
        printw("[F2] FPS CTRL");
        attroff(A_REVERSE);
    }            else {
        printw("[F2] FPS CTRL");
    }
    printw("   ");

    if(fpsCTRL_DisplayMode==3)            {
        attron(A_REVERSE);
        printw("[F3] Sequencer");
        attroff(A_REVERSE);
    }            else {
        printw("[F3] Sequencer");
    }
    printw("\n");
}



inline static void fpsCTRLscreen_print_help() {
    // int attrval = A_BOLD;

    printw("\n");
    print_help_entry("x", "Exit");

    printw("\n============ SCREENS \n");
    print_help_entry("h", "Help screen");
    print_help_entry("F2", "FPS control screen");
    print_help_entry("F3", "FPS command list (Sequencer)");

    printw("\n============ OTHER \n");
    print_help_entry("s", "rescan");
    print_help_entry("e", "erase FPS");
    print_help_entry("E", "erase FPS and tmux sessions");
    print_help_entry("u", "update CONF process");
    print_help_entry("C/c", "start/stop CONF process");
    print_help_entry("R/r", "start/stop RUN process");
    print_help_entry("l", "list all entries");
    print_help_entry("P", "(P)rocess input file \"confscript\"");
    printw("        format: setval <paramfulname> <value>\n");
}





inline static void fpsCTRLscreen_print_nodeinfo(
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    int nodeSelected,
    int fpsindexSelected,
    int pindexSelected) {

    DEBUG_TRACEPOINT("Selected node %d in fps %d",
                     nodeSelected,
                     keywnode[nodeSelected].fpsindex);

    printw("========= FPS info node %d %d ============\n",
           nodeSelected,
           keywnode[nodeSelected].fpsindex);


    char teststring[200];
    sprintf(teststring, "%s", fps[keywnode[nodeSelected].fpsindex].md->sourcefname);
    DEBUG_TRACEPOINT("TEST STRING : %s", teststring);


    DEBUG_TRACEPOINT("TEST LINE : %d", fps[keywnode[nodeSelected].fpsindex].md->sourceline);

    printw("Source  : %s %d\n",
           fps[keywnode[nodeSelected].fpsindex].md->sourcefname,
           fps[keywnode[nodeSelected].fpsindex].md->sourceline);

    DEBUG_TRACEPOINT(" ");
    printw("Root directory    : %s\n",
           fps[keywnode[nodeSelected].fpsindex].md->fpsdirectory);

    DEBUG_TRACEPOINT(" ");
    printw("tmux sessions     :  %s-conf  %s-run\n",
           fps[keywnode[nodeSelected].fpsindex].md->name,
           fps[keywnode[nodeSelected].fpsindex].md->name);

    DEBUG_TRACEPOINT(" ");
    printw("========= NODE info ============\n");
    printw("%-30s ", keywnode[nodeSelected].keywordfull);

    if(keywnode[nodeSelected].leaf > 0) { // If this is not a directory
        char typestring[100];
        functionparameter_GetTypeString(
            fps[fpsindexSelected].parray[pindexSelected].type,
            typestring);
        printw("type %s\n", typestring);

        // print binary flag
        printw("FLAG : ");
        uint64_t mask = (uint64_t) 1 << (sizeof (uint64_t) * CHAR_BIT - 1);
        while(mask) {
            int digit = fps[fpsindexSelected].parray[pindexSelected].fpflag&mask ? 1 : 0;
            if(digit==1) {
                attron(COLOR_PAIR(2));
                printw("%d", digit);
                attroff(COLOR_PAIR(2));
            } else {
                printw("%d", digit);
            }
            mask >>= 1;
        }
    }
    else
    {
        printw("-DIRECTORY-\n");
    }
    printw("\n\n");
}







inline static void fpsCTRLscreen_level0node_summary(
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsindex)
{
    pid_t pid;

    pid = fps[fpsindex].md->confpid;
    if((getpgid(pid) >= 0) && (pid > 0)) {
        attron(COLOR_PAIR(2));
        printw("%06d ", (int) pid);
        attroff(COLOR_PAIR(2));
    } else { // PID not active
        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
        {   // not clean exit
            attron(COLOR_PAIR(4));
            printw("%06d ", (int) pid);
            attroff(COLOR_PAIR(4));
        }
        else
        {   // All OK
            printw("%06d ", (int) pid);
        }
    }


    if(fps[fpsindex].md->conferrcnt>99)
    {
        attron(COLOR_PAIR(4));
        printw("[XX]");
        attroff(COLOR_PAIR(4));
    }
    if(fps[fpsindex].md->conferrcnt>0)
    {
        attron(COLOR_PAIR(4));
        printw("[%02d]", fps[fpsindex].md->conferrcnt);
        attroff(COLOR_PAIR(4));
    }
    if(fps[fpsindex].md->conferrcnt == 0)
    {
        attron(COLOR_PAIR(2));
        printw("[%02d]", fps[fpsindex].md->conferrcnt);
        attroff(COLOR_PAIR(2));
    }

    pid = fps[fpsindex].md->runpid;
    if((getpgid(pid) >= 0) && (pid > 0)) {
        attron(COLOR_PAIR(2));
        printw("%06d ", (int) pid);
        attroff(COLOR_PAIR(2));
    } else {
        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
        {   // not clean exit
            attron(COLOR_PAIR(4));
            printw("%06d ", (int) pid);
            attroff(COLOR_PAIR(4));
        }
        else
        {   // All OK
            printw("%06d ", (int) pid);
        }
    }

}










inline static int fpsCTRLscreen_process_user_key(
    int ch,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_TASK_ENTRY *fpsctrltasklist,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    FPSCTRL_GUIVARS *fpsCTRLgui
) {
    int stringmaxlen = 500;
    int loopOK = 1;
    int fpsindex;
    int pindex;
    FILE *fpinputcmd;

    char command[stringmaxlen];
    char msg[stringmaxlen];


    switch(ch) {
    case 'x':     // Exit control screen
        loopOK = 0;
        break;

    // ============ SCREENS

    case 'h': // help
        fpsCTRLgui->fpsCTRL_DisplayMode = 1;
        break;

    case KEY_F(2): // control
        fpsCTRLgui->fpsCTRL_DisplayMode = 2;
        break;

    case KEY_F(3): // scheduler
        fpsCTRLgui->fpsCTRL_DisplayMode = 3;
        break;

    case 's' : // (re)scan
        functionparameter_scan_fps(
            fpsCTRLgui->mode,
            fpsCTRLgui->fpsnamemask,
            fps,
            keywnode,
            &fpsCTRLgui->NBkwn,
            &fpsCTRLgui->NBfps,
            &fpsCTRLgui->NBindex,
            0);
        clear();
        break;

    case 'e' : // erase FPS
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;
        functionparameter_FPSremove(fps, fpsindex);

        functionparameter_scan_fps(
            fpsCTRLgui->mode,
            fpsCTRLgui->fpsnamemask,
            fps,
            keywnode,
            &fpsCTRLgui->NBkwn,
            &(fpsCTRLgui->NBfps),
            &fpsCTRLgui->NBindex,
            0);
        clear();
        //DEBUG_TRACEPOINT("fpsCTRLgui->NBfps = %d\n", fpsCTRLgui->NBfps);
        // abort();
        fpsCTRLgui->run_display = 0; // skip next display
        fpsCTRLgui->fpsindexSelected = 0; // safeguard in case current selection disappears
        break;

    case 'E' : // Erase FPS and close tmux sessions
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;

        functionparameter_FPSremove(fps, fpsindex);
        functionparameter_scan_fps(
            fpsCTRLgui->mode,
            fpsCTRLgui->fpsnamemask,
            fps,
            keywnode,
            &fpsCTRLgui->NBkwn,
            &fpsCTRLgui->NBfps,
            &fpsCTRLgui->NBindex, 0);
        clear();
        DEBUG_TRACEPOINT(" ");
        fpsCTRLgui->fpsindexSelected = 0; // safeguard in case current selection disappears
        break;

    case KEY_UP:
        fpsCTRLgui->direction = -1;
        fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] --;
        if(fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] < 0) {
            fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] = 0;
        }
        break;


    case KEY_DOWN:
        fpsCTRLgui->direction = 1;
        fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] ++;
        if(fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] > fpsCTRLgui->NBindex - 1) {
            fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] = fpsCTRLgui->NBindex - 1;
        }
        if(fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] > keywnode[fpsCTRLgui->directorynodeSelected].NBchild-1) {
            fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] = keywnode[fpsCTRLgui->directorynodeSelected].NBchild-1;
        }
        break;

    case KEY_PPAGE:
        fpsCTRLgui->direction = -1;
        fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] -= 10;
        if(fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] < 0) {
            fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] = 0;
        }
        break;

    case KEY_NPAGE:
        fpsCTRLgui->direction = 1;
        fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] += 10;
        while(fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] > fpsCTRLgui->NBindex - 1) {
            fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] = fpsCTRLgui->NBindex - 1;
        }
        while(fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] > keywnode[fpsCTRLgui->directorynodeSelected].NBchild-1) {
            fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel] = keywnode[fpsCTRLgui->directorynodeSelected].NBchild-1;
        }
        break;


    case KEY_LEFT:
        if(fpsCTRLgui->directorynodeSelected != 0) { // ROOT has no parent
            fpsCTRLgui->directorynodeSelected = keywnode[fpsCTRLgui->directorynodeSelected].parent_index;
            fpsCTRLgui->nodeSelected = fpsCTRLgui->directorynodeSelected;
        }
        break;


    case KEY_RIGHT :
        if(keywnode[fpsCTRLgui->nodeSelected].leaf == 0) { // this is a directory
            if(keywnode[keywnode[fpsCTRLgui->directorynodeSelected].child[fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel]]].leaf == 0) {
                fpsCTRLgui->directorynodeSelected = keywnode[fpsCTRLgui->directorynodeSelected].child[fpsCTRLgui->GUIlineSelected[fpsCTRLgui->currentlevel]];
                fpsCTRLgui->nodeSelected = fpsCTRLgui->directorynodeSelected;
            }
        }
        break;

    case 10 : // enter key
        if(keywnode[fpsCTRLgui->nodeSelected].leaf == 1) { // this is a leaf
            endwin();
            if(system("clear") != 0) { // clear screen
                PRINT_ERROR("system() returns non-zero value");
            }
            functionparameter_UserInputSetParamValue(&fps[fpsCTRLgui->fpsindexSelected], fpsCTRLgui->pindexSelected);
            initncurses();
        }
        break;

    case ' ' :
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;
        pindex = keywnode[fpsCTRLgui->nodeSelected].pindex;

        // toggles ON / OFF - this is a special case not using function functionparameter_UserInputSetParamValue
        if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_WRITESTATUS) {
            if(fps[fpsindex].parray[pindex].type == FPTYPE_ONOFF) {

                if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ONOFF) {  // ON -> OFF
                    fps[fpsindex].parray[pindex].fpflag &= ~FPFLAG_ONOFF;
                } else { // OFF -> ON
                    fps[fpsindex].parray[pindex].fpflag |= FPFLAG_ONOFF;
                }

                // Save to disk
                if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_SAVEONCHANGE) {
                    functionparameter_WriteParameterToDisk(&fps[fpsindex], pindex, "setval", "UserInputSetParamValue");
                }
                fps[fpsindex].parray[pindex].cnt0 ++;
                fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            }
        }

        if(fps[fpsindex].parray[pindex].type == FPTYPE_EXECFILENAME) {
            if(snprintf(command, stringmaxlen, "tmux send-keys -t %s-run \"cd %s\" C-m", fps[fpsindex].md->name, fps[fpsindex].md->fpsdirectory)< 0 ) {
                PRINT_ERROR("snprintf error");
            }
            if(system(command) != 0) {
                PRINT_ERROR("system() returns non-zero value");
            }
            if(snprintf(command, stringmaxlen, "tmux send-keys -t %s-run \"%s %s\" C-m", fps[fpsindex].md->name, fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].md->name)< 0 ) {
                PRINT_ERROR("snprintf error");
            }
            if(system(command) != 0) {
                PRINT_ERROR("system() returns non-zero value");
            }
        }

        break;


    case 'u' : // update conf process
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;
        fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
        if(snprintf(msg, stringmaxlen, "UPDATE %s", fps[fpsindex].md->name)< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        functionparameter_outlog("FPSCTRL", msg);
        //functionparameter_CONFupdate(fps, fpsindex);
        break;

    case 'R' : // start run process if possible
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;
        if(snprintf(msg, stringmaxlen,"RUNSTART %s", fps[fpsindex].md->name)< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        functionparameter_outlog("FPSCTRL", msg);
        functionparameter_RUNstart(fps, fpsindex);
        break;

    case 'r' : // stop run process
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;
        if(snprintf(msg, stringmaxlen, "RUNSTOP %s", fps[fpsindex].md->name)< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        functionparameter_outlog("FPSCTRL", msg);
        functionparameter_RUNstop(fps, fpsindex);
        break;


    case 'C' : // start conf process
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;
        if(snprintf(msg, stringmaxlen, "CONFSTART %s", fps[fpsindex].md->name)< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        functionparameter_outlog("FPSCTRL", msg);
        functionparameter_CONFstart(fps, fpsindex);
        break;

    case 'c': // kill conf process
        fpsindex = keywnode[fpsCTRLgui->nodeSelected].fpsindex;
        if(snprintf(msg, stringmaxlen, "CONFSTOP %s", fps[fpsindex].md->name)< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        functionparameter_outlog("FPSCTRL", msg);
        functionparameter_CONFstop(fps, fpsindex);
        break;

    case 'l': // list all parameters
        endwin();
        if(system("clear") != 0) {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("FPS entries - Full list \n");
        printf("\n");
        for(int kwnindex = 0; kwnindex < fpsCTRLgui->NBkwn; kwnindex++) {
            if(keywnode[kwnindex].leaf == 1) {
                printf("%4d  %4d  %s\n", keywnode[kwnindex].fpsindex, keywnode[kwnindex].pindex, keywnode[kwnindex].keywordfull);
            }
        }
        printf("  TOTAL :  %d nodes\n", fpsCTRLgui->NBkwn);
        printf("\n");
        printf("Press Any Key to Continue\n");
        getchar();
        initncurses();
        break;


    case 'F': // process FIFO
        endwin();
        if(system("clear") != 0) {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("Reading FIFO file \"%s\"  fd=%d\n", fpsCTRLgui->fpsCTRLfifoname, fpsCTRLgui->fpsCTRLfifofd);

        if(fpsCTRLgui->fpsCTRLfifofd > 0) {
            // int verbose = 1;
            functionparameter_read_fpsCMD_fifo(fpsCTRLgui->fpsCTRLfifofd, fpsctrltasklist, fpsctrlqueuelist);
        }

        printf("\n");
        printf("Press Any Key to Continue\n");
        getchar();
        initncurses();
        break;


    case 'P': // process input command file
        endwin();
        if(system("clear") != 0) {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("Reading file confscript\n");
        fpinputcmd = fopen("confscript", "r");
        if(fpinputcmd != NULL) {
            char *FPScmdline = NULL;
            size_t len = 0;
            ssize_t read;

            while((read = getline(&FPScmdline, &len, fpinputcmd)) != -1) {
                printf("Processing line : %s\n", FPScmdline);
                functionparameter_FPSprocess_cmdline(FPScmdline, fpsctrlqueuelist, keywnode, fpsCTRLgui->NBkwn, fps);
            }
            fclose(fpinputcmd);
        }

        printf("\n");
        printf("Press Any Key to Continue\n");
        getchar();
        initncurses();
        break;
    }


    return(loopOK);
}








/**
 * ## Purpose
 *
 * Automatically build simple ASCII GUI from function parameter structure (fps) name mask
 *
 *
 *
 */
errno_t functionparameter_CTRLscreen(
    uint32_t mode,
    char *fpsnamemask,
    char *fpsCTRLfifoname
) {
    int stringmaxlen = 500;

    // function parameter structure(s)
    int fpsindex;

    FPSCTRL_GUIVARS fpsCTRLgui;

    FUNCTION_PARAMETER_STRUCT *fps;


    // function parameters
    long NBpindex = 0;
    long pindex;
    //int *p_fpsindex; // fps index for parameter
    //int *p_pindex;   // index within fps

    // keyword tree
    //int kwnindex;
    KEYWORD_TREE_NODE *keywnode;

    int level;

    int loopOK = 1;
    long long loopcnt = 0;


    int nodechain[MAXNBLEVELS];


    // What to run ?
    // disable for testing
    int run_display = 1;
    loopOK = 1;


    functionparameter_outlog("FPSCTRL", "START\n");

    DEBUG_TRACEPOINT("function start");



    // initialize fpsCTRLgui
    fpsCTRLgui.mode                  = mode;
    fpsCTRLgui.nodeSelected          = 1;
    fpsCTRLgui.run_display           = run_display;
    fpsCTRLgui.fpsindexSelected      = 0;
    fpsCTRLgui.pindexSelected        = 0;
    fpsCTRLgui.directorynodeSelected = 0;
    fpsCTRLgui.currentlevel          = 0;
    fpsCTRLgui.direction             = 1;
    strcpy(fpsCTRLgui.fpsnamemask, fpsnamemask);
    strcpy(fpsCTRLgui.fpsCTRLfifoname, fpsCTRLfifoname);


    fpsCTRLgui.fpsCTRL_DisplayMode = 2;
    // 1: [h]  help
    // 2: [F2] list of conf and run
    // 3: [F3] fpscmdarray





    // allocate memory


    // Array holding fps structures
    //
    fps = (FUNCTION_PARAMETER_STRUCT *) malloc(sizeof(FUNCTION_PARAMETER_STRUCT) * NB_FPS_MAX);


    // Initialize file descriptors to -1
    //
    for(fpsindex = 0; fpsindex<NB_FPS_MAX; fpsindex++) {
        fps[fpsindex].SMfd = -1;
    }

    // All parameters held in this array
    //
    keywnode = (KEYWORD_TREE_NODE *) malloc(sizeof(KEYWORD_TREE_NODE) * NB_KEYWNODE_MAX);
    for(int kn=0; kn<NB_KEYWNODE_MAX; kn++) {
        strcpy(keywnode[kn].keywordfull,"");
        for(int ch=0; ch<MAX_NB_CHILD; ch++) {
            keywnode[kn].child[ch] = 0;
        }
    }



    // initialize nodechain
    for(int l=0; l<MAXNBLEVELS; l++) {
        nodechain[l] = 0;
    }



    // Set up instruction buffer to sequence commands
    //
    FPSCTRL_TASK_ENTRY *fpsctrltasklist;
    fpsctrltasklist = (FPSCTRL_TASK_ENTRY*) malloc(sizeof(FPSCTRL_TASK_ENTRY) * NB_FPSCTRL_TASK_MAX);
    for(int cmdindex = 0; cmdindex < NB_FPSCTRL_TASK_MAX; cmdindex++) {
        fpsctrltasklist[cmdindex].status = 0;
        fpsctrltasklist[cmdindex].queue = 0;
    }

    // Set up task queue list
    //
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist;
    fpsctrlqueuelist = (FPSCTRL_TASK_QUEUE*) malloc(sizeof(FPSCTRL_TASK_QUEUE) * FPSTASK_MAX_NBQUEUE);
    for(int queueindex = 0; queueindex < FPSTASK_MAX_NBQUEUE; queueindex++) {
        fpsctrlqueuelist[queueindex].priority = 1; // 0 = not active
    }


#ifndef STANDALONE
    set_signal_catch();
#endif




    // fifo
    fpsCTRLgui.fpsCTRLfifofd = open(fpsCTRLgui.fpsCTRLfifoname, O_RDWR | O_NONBLOCK);
    long fifocmdcnt = 0;


    for(level=0; level<MAXNBLEVELS; level++)
        fpsCTRLgui.GUIlineSelected[level] = 0;




    functionparameter_scan_fps(
        fpsCTRLgui.mode,
        fpsCTRLgui.fpsnamemask,
        fps,
        keywnode,
        &fpsCTRLgui.NBkwn,
        &fpsCTRLgui.NBfps,
        &NBpindex, 1);
    printf("%d function parameter structure(s) imported, %ld parameters\n",
           fpsCTRLgui.NBfps, NBpindex);
    fflush(stdout);
    DEBUG_TRACEPOINT(" ");



    if(fpsCTRLgui.NBfps == 0) {
        printf("No function parameter structure found\n");
        printf("File %s line %d\n", __FILE__, __LINE__);
        fflush(stdout);

        char logfname[stringmaxlen];
        char shmdname[stringmaxlen];
        function_parameter_struct_shmdirname(shmdname);
        if(snprintf(logfname, stringmaxlen, "%s/fpslog.%06d", shmdname, getpid())< 0 ) {
            PRINT_ERROR("snprintf error");
        }
        remove(logfname);

        return RETURN_SUCCESS;
    }

    fpsCTRLgui.nodeSelected = 1;
    fpsindex = 0;












    // INITIALIZE ncurses

    if(run_display == 1) {
        initncurses();
        atexit(functionparameter_CTRLscreen_atexit);
        clear();
    }



    fpsCTRLgui.NBindex = 0;
    char shmdname[200];
    function_parameter_struct_shmdirname(shmdname);



    if(run_display == 0) {
        loopOK = 0;
    }

    while(loopOK == 1) {

        long icnt = 0;

        usleep(10000); // 100 Hz display



        // ==================
        // = GET USER INPUT =
        // ==================

        int ch = getch();

        loopOK = fpsCTRLscreen_process_user_key(
                     ch,
                     fps,
                     keywnode,
                     fpsctrltasklist,
                     fpsctrlqueuelist,
                     &fpsCTRLgui
                 );


        if(fpsCTRLgui.NBfps == 0) {
            endwin();

            printf("\n fpsCTRLgui.NBfps = %d ->  No FPS on system - nothing to display\n", fpsCTRLgui.NBfps);
            return RETURN_FAILURE;
        }







        if( fpsCTRLgui.run_display == 1)	{

            erase();
            fpsCTRLscreen_print_DisplayMode_status(fpsCTRLgui.fpsCTRL_DisplayMode, fpsCTRLgui.NBfps);



            DEBUG_TRACEPOINT(" ");

            printw("INPUT FIFO:  %s (fd=%d)    fifocmdcnt = %ld\n", fpsCTRLgui.fpsCTRLfifoname, fpsCTRLgui.fpsCTRLfifofd, fifocmdcnt);

            int fcnt = functionparameter_read_fpsCMD_fifo(fpsCTRLgui.fpsCTRLfifofd, fpsctrltasklist, fpsctrlqueuelist);

            DEBUG_TRACEPOINT(" ");

            function_parameter_process_fpsCMDarray(fpsctrltasklist, fpsctrlqueuelist, keywnode, fpsCTRLgui.NBkwn, fps);

            fifocmdcnt += fcnt;

            DEBUG_TRACEPOINT(" ");

            printw("OUTPUT LOG:  %s/fpslog.%06d\n", shmdname, getpid());

            DEBUG_TRACEPOINT(" ");


            if(fpsCTRLgui.fpsCTRL_DisplayMode == 1) { // help
                fpsCTRLscreen_print_help();
            }


            if(fpsCTRLgui.fpsCTRL_DisplayMode == 2) { // FPS content


                DEBUG_TRACEPOINT("Check that selected node is OK");
                /* printw("node selected : %d\n", fpsCTRLgui.nodeSelected);
                 printw("full keyword :  %s\n", keywnode[fpsCTRLgui.nodeSelected].keywordfull);*/
                if(strlen(keywnode[fpsCTRLgui.nodeSelected].keywordfull) < 1) { // if not OK, set to last valid entry
                    fpsCTRLgui.nodeSelected = 1;
                    while((strlen(keywnode[fpsCTRLgui.nodeSelected].keywordfull)<1) && (fpsCTRLgui.nodeSelected < NB_KEYWNODE_MAX))
                        fpsCTRLgui.nodeSelected ++;
                }

                DEBUG_TRACEPOINT("Get info from selected node");
                fpsCTRLgui.fpsindexSelected = keywnode[fpsCTRLgui.nodeSelected].fpsindex;
                fpsCTRLgui.pindexSelected = keywnode[fpsCTRLgui.nodeSelected].pindex;
                fpsCTRLscreen_print_nodeinfo(
                    fps,
                    keywnode,
                    fpsCTRLgui.nodeSelected,
                    fpsCTRLgui.fpsindexSelected,
                    fpsCTRLgui.pindexSelected);



                DEBUG_TRACEPOINT("trace back node chain");
                nodechain[fpsCTRLgui.currentlevel] = fpsCTRLgui.directorynodeSelected;

                printw("[level %d %d] ", fpsCTRLgui.currentlevel+1, nodechain[fpsCTRLgui.currentlevel + 1]);

                if(fpsCTRLgui.currentlevel>0) {
                    printw("[level %d %d] ", fpsCTRLgui.currentlevel, nodechain[fpsCTRLgui.currentlevel]);
                }
                level = fpsCTRLgui.currentlevel - 1;
                while(level > 0) {
                    nodechain[level] = keywnode[nodechain[level + 1]].parent_index;
                    printw("[level %d %d] ", level, nodechain[level]);
                    level --;
                }
                printw("[level 0 0]\n");
                nodechain[0] = 0; // root

                DEBUG_TRACEPOINT("Get number of lines to be displayed");
                fpsCTRLgui.currentlevel = keywnode[fpsCTRLgui.directorynodeSelected].keywordlevel;
                int GUIlineMax = keywnode[fpsCTRLgui.directorynodeSelected].NBchild;
                for(level = 0; level < fpsCTRLgui.currentlevel; level ++) {
                    DEBUG_TRACEPOINT("update GUIlineMax, the maximum number of lines");
                    if(keywnode[nodechain[level]].NBchild > GUIlineMax) {
                        GUIlineMax = keywnode[nodechain[level]].NBchild;
                    }
                }


                printw("[node %d] level = %d   [%d] NB child = %d",
                       fpsCTRLgui.nodeSelected,
                       fpsCTRLgui.currentlevel,
                       fpsCTRLgui.directorynodeSelected,
                       keywnode[fpsCTRLgui.directorynodeSelected].NBchild
                      );

                printw("   fps %d",
                       fpsCTRLgui.fpsindexSelected
                      );

                printw("   pindex %d ",
                       keywnode[fpsCTRLgui.nodeSelected].pindex
                      );

                printw("\n");

                /*      printw("SELECTED DIR = %3d    SELECTED = %3d   GUIlineMax= %3d\n\n",
                             fpsCTRLgui.directorynodeSelected,
                             fpsCTRLgui.nodeSelected,
                             GUIlineMax);
                      printw("LINE: %d / %d\n\n",
                             fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel],
                             keywnode[fpsCTRLgui.directorynodeSelected].NBchild);
                	*/


                //while(!(fps[fpsindexSelected].parray[pindexSelected].fpflag & FPFLAG_VISIBLE)) { // if invisible
                //		fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel]++;
                //}

                //if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_VISIBLE)) { // if invisible


                //              if( !(  fps[keywnode[fpsCTRLgui.nodeSelected].fpsindex].parray[keywnode[fpsCTRLgui.nodeSelected].pindex].fpflag & FPFLAG_VISIBLE)) { // if invisible
                //				if( !(  fps[fpsCTRLgui.fpsindexSelected].parray[fpsCTRLgui.pindexSelected].fpflag & FPFLAG_VISIBLE)) { // if invisible
                if( !(  fps[fpsCTRLgui.fpsindexSelected].parray[0].fpflag & FPFLAG_VISIBLE)) { // if invisible
                    if(fpsCTRLgui.direction > 0) {
                        fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel] ++;
                    }
                    else
                    {
                        fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel] --;
                    }
                }



                while(fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel] >
                        keywnode[fpsCTRLgui.directorynodeSelected].NBchild-1) {
                    fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel]--;
                }



                int child_index[MAXNBLEVELS];
                for(level = 0; level <MAXNBLEVELS ; level ++) {
                    child_index[level] = 0;
                }




                for(int GUIline = 0; GUIline < GUIlineMax; GUIline++) { // GUIline is the line number on GUI display


                    for(level = 0; level < fpsCTRLgui.currentlevel; level ++) {

                        if(GUIline < keywnode[nodechain[level]].NBchild) {
                            int snode = 0; // selected node
                            int knodeindex;

                            knodeindex = keywnode[nodechain[level]].child[GUIline];


                            //TODO: adjust len to string
                            char pword[100];


                            if(level==0) {
                                DEBUG_TRACEPOINT("provide a fps status summary if at root");
                                fpsindex = keywnode[knodeindex].fpsindex;
                                fpsCTRLscreen_level0node_summary(fps, fpsindex);
                            }

                            // toggle highlight if node is in the chain
                            int v1 = keywnode[nodechain[level]].child[GUIline];
                            int v2 = nodechain[level + 1];
                            if(v1 == v2) {
                                snode = 1;
                                attron(A_REVERSE);
                            }

                            // color node if directory
                            if(keywnode[knodeindex].leaf == 0) {
                                attron(COLOR_PAIR(5));
                            }

                            // print keyword
                            if(snprintf(pword, 10, "%s", keywnode[keywnode[nodechain[level]].child[GUIline]].keyword[level])< 0 ) {
                                PRINT_ERROR("snprintf error");
                            }
                            printw("%-10s ", pword);

                            if(keywnode[knodeindex].leaf == 0) { // directory
                                attroff(COLOR_PAIR(5));
                            }

                            attron(A_REVERSE);
                            if(snode == 1)
                                printw(">");
                            else
                                printw(" ");
                            attroff(A_REVERSE);

                            if(snode == 1) {
                                attroff(A_REVERSE);
                            }


                        } else { // blank space
                            if(level==0) {
                                printw("                  ");
                            }
                            printw("            ");
                        }
                    }






                    int knodeindex;
                    knodeindex = keywnode[fpsCTRLgui.directorynodeSelected].child[child_index[level]];
                    if(knodeindex < fpsCTRLgui.NBkwn )
                    {
                        fpsindex = keywnode[knodeindex].fpsindex;
                        pindex = keywnode[knodeindex].pindex;

                        if(child_index[level] > keywnode[fpsCTRLgui.directorynodeSelected].NBchild - 1) {
                            child_index[level] = keywnode[fpsCTRLgui.directorynodeSelected].NBchild - 1;
                        }

                        /*
                                                if(fpsCTRLgui.currentlevel != 0) { // this does not apply to root menu
                                                    while((!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_VISIBLE)) && // if not visible, advance to next one
                                                            (child_index[level] < keywnode[fpsCTRLgui.directorynodeSelected].NBchild-1)) {
                                                        child_index[level] ++;
                                                        DEBUG_TRACEPOINT("knodeindex = %d  child %d / %d",
                                                                  knodeindex,
                                                                  child_index[level],
                                                                  keywnode[fpsCTRLgui.directorynodeSelected].NBchild);
                                                        knodeindex = keywnode[fpsCTRLgui.directorynodeSelected].child[child_index[level]];
                                                        fpsindex = keywnode[knodeindex].fpsindex;
                                                        pindex = keywnode[knodeindex].pindex;
                                                    }
                                                }
                        */

                        DEBUG_TRACEPOINT(" ");

                        if(child_index[level] < keywnode[fpsCTRLgui.directorynodeSelected].NBchild) {

                            if(fpsCTRLgui.currentlevel > 0)
                            {
                                attron(A_REVERSE);
                                printw(" ");
                                attroff(A_REVERSE);
                            }

                            DEBUG_TRACEPOINT(" ");

                            if(keywnode[knodeindex].leaf == 0) { // If this is a directory
                                DEBUG_TRACEPOINT(" ");
                                if(fpsCTRLgui.currentlevel == 0) { // provide a status summary if at root
                                    DEBUG_TRACEPOINT(" ");

                                    fpsindex = keywnode[knodeindex].fpsindex;
                                    pid_t pid;

                                    pid = fps[fpsindex].md->confpid;
                                    if((getpgid(pid) >= 0) && (pid > 0)) {
                                        attron(COLOR_PAIR(2));
                                        printw("%06d ", (int) pid);
                                        attroff(COLOR_PAIR(2));
                                    } else { // PID not active
                                        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
                                        {   // not clean exit
                                            attron(COLOR_PAIR(4));
                                            printw("%06d ", (int) pid);
                                            attroff(COLOR_PAIR(4));
                                        }
                                        else
                                        {   // All OK
                                            printw("%06d ", (int) pid);
                                        }
                                    }

                                    if(fps[fpsindex].md->conferrcnt>99)
                                    {
                                        attron(COLOR_PAIR(4));
                                        printw("[XX]");
                                        attroff(COLOR_PAIR(4));
                                    }
                                    if(fps[fpsindex].md->conferrcnt>0)
                                    {
                                        attron(COLOR_PAIR(4));
                                        printw("[%02d]", fps[fpsindex].md->conferrcnt);
                                        attroff(COLOR_PAIR(4));
                                    }
                                    if(fps[fpsindex].md->conferrcnt == 0)
                                    {
                                        attron(COLOR_PAIR(2));
                                        printw("[%02d]", fps[fpsindex].md->conferrcnt);
                                        attroff(COLOR_PAIR(2));
                                    }

                                    pid = fps[fpsindex].md->runpid;
                                    if((getpgid(pid) >= 0) && (pid > 0)) {
                                        attron(COLOR_PAIR(2));
                                        printw("%06d ", (int) pid);
                                        attroff(COLOR_PAIR(2));
                                    } else {
                                        if(fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
                                        {   // not clean exit
                                            attron(COLOR_PAIR(4));
                                            printw("%06d ", (int) pid);
                                            attroff(COLOR_PAIR(4));
                                        }
                                        else
                                        {   // All OK
                                            printw("%06d ", (int) pid);
                                        }
                                    }
                                }





                                if(GUIline == fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel]) {
                                    attron(A_REVERSE);
                                    fpsCTRLgui.nodeSelected = knodeindex;
                                    fpsCTRLgui.fpsindexSelected = keywnode[knodeindex].fpsindex;
                                }


                                if(child_index[level+1] < keywnode[fpsCTRLgui.directorynodeSelected].NBchild)
                                {
                                    attron(COLOR_PAIR(5));
                                    level = keywnode[knodeindex].keywordlevel;
                                    printw("%-16s", keywnode[knodeindex].keyword[level - 1]);
                                    attroff(COLOR_PAIR(5));

                                    if(GUIline == fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel]) {
                                        attroff(A_REVERSE);
                                    }
                                }
                                else {
                                    printw("%-16s", " ");
                                }


                                DEBUG_TRACEPOINT(" ");

                            }
                            else { // If this is a parameter
                                DEBUG_TRACEPOINT(" ");
                                fpsindex = keywnode[knodeindex].fpsindex;
                                pindex = keywnode[knodeindex].pindex;




                                DEBUG_TRACEPOINT(" ");
                                int isVISIBLE = 1;
                                if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_VISIBLE)) { // if invisible
                                    isVISIBLE = 0;
                                    attron(A_DIM|A_BLINK);
                                }



                                //int kl;

                                if(GUIline == fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel]) {
                                    fpsCTRLgui.pindexSelected = keywnode[knodeindex].pindex;
                                    fpsCTRLgui.fpsindexSelected = keywnode[knodeindex].fpsindex;
                                    fpsCTRLgui.nodeSelected = knodeindex;

                                    if(isVISIBLE == 1) {
                                        attron(COLOR_PAIR(10) | A_BOLD);
                                    }
                                }
                                DEBUG_TRACEPOINT(" ");

                                if(isVISIBLE == 1) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_WRITESTATUS) {
                                        attron(COLOR_PAIR(10) | A_BLINK);
                                        printw("W "); // writable
                                        attroff(COLOR_PAIR(10) | A_BLINK);
                                    } else {
                                        attron(COLOR_PAIR(4) | A_BLINK);
                                        printw("NW"); // non writable
                                        attroff(COLOR_PAIR(4) | A_BLINK);
                                    }
                                } else {
                                    printw("  ");
                                }

                                DEBUG_TRACEPOINT(" ");
                                level = keywnode[knodeindex].keywordlevel;
                                printw(" %-20s", fps[fpsindex].parray[pindex].keyword[level - 1]);

                                if(GUIline == fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel]) {
                                    attroff(COLOR_PAIR(10));
                                }
                                DEBUG_TRACEPOINT(" ");
                                printw("   ");

                                // VALUE

                                int paramsync = 1; // parameter is synchronized

                                if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR) { // parameter setting error
                                    if(isVISIBLE == 1) {
                                        attron(COLOR_PAIR(4));
                                    }
                                }

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_UNDEF) {
                                    printw("  %s", "-undef-");
                                }

                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_INT64) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(fps[fpsindex].parray[pindex].val.l[0] != fps[fpsindex].parray[pindex].val.l[3]) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10d", (int) fps[fpsindex].parray[pindex].val.l[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }
                                }

                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_FLOAT64) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR)) {
                                            double absdiff;
                                            double abssum;
                                            double epsrel = 1.0e-6;
                                            double epsabs = 1.0e-10;

                                            absdiff = fabs(fps[fpsindex].parray[pindex].val.f[0] - fps[fpsindex].parray[pindex].val.f[3]);
                                            abssum = fabs(fps[fpsindex].parray[pindex].val.f[0]) + fabs(fps[fpsindex].parray[pindex].val.f[3]);


                                            if((absdiff < epsrel * abssum) || (absdiff < epsabs)) {
                                                paramsync = 1;
                                            } else {
                                                paramsync = 0;
                                            }
                                        }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10f", (float) fps[fpsindex].parray[pindex].val.f[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }
                                }

                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_FLOAT32) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR)) {
                                            double absdiff;
                                            double abssum;
                                            double epsrel = 1.0e-6;
                                            double epsabs = 1.0e-10;

                                            absdiff = fabs(fps[fpsindex].parray[pindex].val.s[0] - fps[fpsindex].parray[pindex].val.s[3]);
                                            abssum = fabs(fps[fpsindex].parray[pindex].val.s[0]) + fabs(fps[fpsindex].parray[pindex].val.s[3]);


                                            if((absdiff < epsrel * abssum) || (absdiff < epsabs)) {
                                                paramsync = 1;
                                            } else {
                                                paramsync = 0;
                                            }
                                        }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10f", (float) fps[fpsindex].parray[pindex].val.s[0]);

                                    if(paramsync == 0) {
                                        attroff(COLOR_PAIR(3));
                                    }
                                }


                                DEBUG_TRACEPOINT(" ");
                                if(fps[fpsindex].parray[pindex].type == FPTYPE_PID) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(fps[fpsindex].parray[pindex].val.pid[0] !=
                                                    fps[fpsindex].parray[pindex].val.pid[1]) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10d", (float) fps[fpsindex].parray[pindex].val.pid[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10d", (int) fps[fpsindex].parray[pindex].val.pid[0]);
                                }


                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_TIMESPEC) {
                                    printw("  %10s", "-timespec-");
                                }


                                if(fps[fpsindex].parray[pindex].type == FPTYPE_FILENAME) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(strcmp(fps[fpsindex].parray[pindex].val.string[0],
                                                      fps[fpsindex].parray[pindex].val.string[1])) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }
                                }
                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_FITSFILENAME) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(strcmp(fps[fpsindex].parray[pindex].val.string[0],
                                                      fps[fpsindex].parray[pindex].val.string[1])) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }
                                }
                                DEBUG_TRACEPOINT(" ");
                                if(fps[fpsindex].parray[pindex].type == FPTYPE_EXECFILENAME) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(strcmp(fps[fpsindex].parray[pindex].val.string[0],
                                                      fps[fpsindex].parray[pindex].val.string[1])) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }
                                }
                                DEBUG_TRACEPOINT(" ");
                                if(fps[fpsindex].parray[pindex].type == FPTYPE_DIRNAME) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(strcmp(fps[fpsindex].parray[pindex].val.string[0],
                                                      fps[fpsindex].parray[pindex].val.string[1])) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }
                                }

                                DEBUG_TRACEPOINT(" ");
                                if(fps[fpsindex].parray[pindex].type == FPTYPE_STREAMNAME) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            //  if(strcmp(fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].parray[pindex].val.string[1])) {
                                            //      paramsync = 0;
                                            //  }

                                            if(fps[fpsindex].parray[pindex].info.stream.streamID > -1) {
                                                if(isVISIBLE == 1) {
                                                    attron(COLOR_PAIR(2));
                                                }
                                            }

                                    printw("[%d]  %10s",
                                           fps[fpsindex].parray[pindex].info.stream.stream_sourceLocation,
                                           fps[fpsindex].parray[pindex].val.string[0]);

                                    if(fps[fpsindex].parray[pindex].info.stream.streamID > -1) {

                                        printw(" [ %d", fps[fpsindex].parray[pindex].info.stream.stream_xsize[0]);
                                        if(fps[fpsindex].parray[pindex].info.stream.stream_naxis[0]>1)
                                            printw("x%d", fps[fpsindex].parray[pindex].info.stream.stream_ysize[0]);
                                        if(fps[fpsindex].parray[pindex].info.stream.stream_naxis[0]>2)
                                            printw("x%d", fps[fpsindex].parray[pindex].info.stream.stream_zsize[0]);

                                        printw(" ]");
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(2));
                                        }
                                    }

                                }
                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_STRING) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(strcmp(fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].parray[pindex].val.string[1])) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(3));
                                        }
                                    }

                                    printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(3));
                                        }
                                    }
                                }
                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].type == FPTYPE_ONOFF) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ONOFF) {
                                        attron(COLOR_PAIR(2));
                                        printw("  ON ", fps[fpsindex].parray[pindex].val.string[1]);
                                        attroff(COLOR_PAIR(2));
                                        printw(" [%15s]", fps[fpsindex].parray[pindex].val.string[1]);
                                    } else {
                                        attron(COLOR_PAIR(1));
                                        printw(" OFF ", fps[fpsindex].parray[pindex].val.string[0]);
                                        attroff(COLOR_PAIR(1));
                                        printw(" [%15s]", fps[fpsindex].parray[pindex].val.string[0]);
                                    }
                                }


                                if(fps[fpsindex].parray[pindex].type == FPTYPE_FPSNAME) {
                                    if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_FEEDBACK)   // Check value feedback if available
                                        if(!(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR))
                                            if(strcmp(fps[fpsindex].parray[pindex].val.string[0],
                                                      fps[fpsindex].parray[pindex].val.string[1])) {
                                                paramsync = 0;
                                            }

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(2));
                                        }
                                    }
                                    else {
                                        if(isVISIBLE == 1) {
                                            attron(COLOR_PAIR(4));
                                        }
                                    }

                                    printw(" %10s [%ld %ld %ld]",
                                           fps[fpsindex].parray[pindex].val.string[0],
                                           fps[fpsindex].parray[pindex].info.fps.FPSNBparamMAX,
                                           fps[fpsindex].parray[pindex].info.fps.FPSNBparamActive,
                                           fps[fpsindex].parray[pindex].info.fps.FPSNBparamUsed);

                                    if(paramsync == 0) {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(2));
                                        }
                                    }
                                    else {
                                        if(isVISIBLE == 1) {
                                            attroff(COLOR_PAIR(4));
                                        }
                                    }

                                }

                                DEBUG_TRACEPOINT(" ");

                                if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ERROR) { // parameter setting error
                                    if(isVISIBLE == 1) {
                                        attroff(COLOR_PAIR(4));
                                    }
                                }

                                printw("    %s", fps[fpsindex].parray[pindex].description);



                                if(GUIline == fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel]) {
                                    if(isVISIBLE == 1) {
                                        attroff(A_BOLD);
                                    }
                                }


                                if(isVISIBLE==0) {
                                    attroff(A_DIM|A_BLINK);
                                }
                                // END LOOP


                            }


                            DEBUG_TRACEPOINT(" ");
                            icnt++;


                            for(level = 0; level <MAXNBLEVELS ; level ++) {
                                child_index[level] ++;
                            }
                        }
                    }

                    printw("\n");
                }

                DEBUG_TRACEPOINT(" ");

                fpsCTRLgui.NBindex = icnt;

                if(fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel] > fpsCTRLgui.NBindex - 1) {
                    fpsCTRLgui.GUIlineSelected[fpsCTRLgui.currentlevel] = fpsCTRLgui.NBindex - 1;
                }

                DEBUG_TRACEPOINT(" ");

                printw("\n");

                if(fps[fpsCTRLgui.fpsindexSelected].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK) {
                    attron(COLOR_PAIR(2));
                    printw("[%ld] PARAMETERS OK - RUN function good to go\n", fps[fpsCTRLgui.fpsindexSelected].md->msgcnt);
                    attroff(COLOR_PAIR(2));
                } else {
                    int msgi;

                    attron(COLOR_PAIR(4));
                    printw("[%ld] %d PARAMETER SETTINGS ERROR(s) :\n",
                           fps[fpsCTRLgui.fpsindexSelected].md->msgcnt,
                           fps[fpsCTRLgui.fpsindexSelected].md->conferrcnt);
                    attroff(COLOR_PAIR(4));

                    attron(A_BOLD);

                    for(msgi = 0; msgi < fps[fpsCTRLgui.fpsindexSelected].md->msgcnt; msgi++) {
                        pindex = fps[fpsCTRLgui.fpsindexSelected].md->msgpindex[msgi];
                        printw("%-40s %s\n",
                               fps[fpsCTRLgui.fpsindexSelected].parray[pindex].keywordfull,
                               fps[fpsCTRLgui.fpsindexSelected].md->message[msgi]);
                    }

                    attroff(A_BOLD);
                }


                DEBUG_TRACEPOINT(" ");

            }

            DEBUG_TRACEPOINT(" ");

            if(fpsCTRLgui.fpsCTRL_DisplayMode == 3) { // Task scheduler status
                struct timespec tnow;
                struct timespec tdiff;

                clock_gettime(CLOCK_REALTIME, &tnow);

                printw(" \n");

                //int dispcnt = 0;


                // Sort entries from most recent to most ancient, using inputindex
                DEBUG_TRACEPOINT(" ");
                double * sort_evalarray;
                sort_evalarray = (double*) malloc(sizeof(double)*NB_FPSCTRL_TASK_MAX);
                long * sort_indexarray;
                sort_indexarray = (long*) malloc(sizeof(long)*NB_FPSCTRL_TASK_MAX);

                long sortcnt = 0;
                for(int fpscmdindex=0; fpscmdindex<NB_FPSCTRL_TASK_MAX; fpscmdindex++) {
                    if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_SHOW) {
                        sort_evalarray[sortcnt] = -1.0*fpsctrltasklist[fpscmdindex].inputindex;
                        sort_indexarray[sortcnt] = fpscmdindex;
                        sortcnt++;
                    }
                }
                DEBUG_TRACEPOINT(" ");
                if(sortcnt>0) {
                    quick_sort2l(sort_evalarray, sort_indexarray, sortcnt);
                }
                free(sort_evalarray);

                DEBUG_TRACEPOINT(" ");

                for(int sortindex=0; sortindex<sortcnt; sortindex++) {


                    DEBUG_TRACEPOINT("iteration %d / %ld", sortindex, sortcnt);

                    int fpscmdindex = sort_indexarray[sortindex];

                    DEBUG_TRACEPOINT("fpscmdindex = %d", fpscmdindex);

                    if(sortindex > wrow-8) {// remove oldest
                        fpsctrltasklist[fpscmdindex].status &= ~FPSTASK_STATUS_SHOW;
                    } else { // display

                        int attron2 = 0;
                        int attrbold = 0;


                        if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RUNNING) { // task is running
                            attron2 = 1;
                            attron(COLOR_PAIR(2));
                        } else if (fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE) { // task is queued to run
                            attrbold = 1;
                            attron(A_BOLD);
                        }



                        // measure age since submission
                        tdiff =  info_time_diff(fpsctrltasklist[fpscmdindex].creationtime, tnow);
                        double tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
                        printw("%6.2f s ", tdiffv);

                        if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_RUNNING) { // run time (ongoing)
                            tdiff =  info_time_diff(fpsctrltasklist[fpscmdindex].activationtime, tnow);
                            tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
                            printw(" %6.2f s ", tdiffv);
                        } else if (!(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE)) { // run time (past)
                            tdiff =  info_time_diff(fpsctrltasklist[fpscmdindex].activationtime, fpsctrltasklist[fpscmdindex].completiontime);
                            tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
                            attron(COLOR_PAIR(3));
                            printw(" %6.2f s ", tdiffv);
                            attroff(COLOR_PAIR(3));
                            // age since completion
                            tdiff =  info_time_diff(fpsctrltasklist[fpscmdindex].completiontime, tnow);
                            double tdiffv = tdiffv = 1.0*tdiff.tv_sec + 1.0e-9*tdiff.tv_nsec;
                            //printw("<%6.2f s>      ", tdiffv);

                            //if(tdiffv > 30.0)
                            //fpsctrltasklist[fpscmdindex].status &= ~FPSTASK_STATUS_SHOW;

                        } else {
                            printw("          ", tdiffv);
                        }


                        if(fpsctrltasklist[fpscmdindex].status & FPSTASK_STATUS_ACTIVE) {
                            printw(">>");
                        } else {
                            printw("  ");
                        }

                        if(fpsctrltasklist[fpscmdindex].flag & FPSTASK_FLAG_WAITONRUN) {
                            printw("WR ");
                        } else {
                            printw("   ");
                        }

                        if(fpsctrltasklist[fpscmdindex].flag & FPSTASK_FLAG_WAITONCONF) {
                            printw("WC ");
                        } else {
                            printw("   ");
                        }

                        printw("[Q %02d %02d] %4d  %s\n",
                               fpsctrltasklist[fpscmdindex].queue,
                               fpsctrlqueuelist[fpsctrltasklist[fpscmdindex].queue].priority,
                               fpscmdindex,
                               fpsctrltasklist[fpscmdindex].cmdstring);

                        if ( attron2 == 1 )
                            attroff(COLOR_PAIR(2));
                        if ( attrbold == 1 )
                            attroff(A_BOLD);

                    }
                }
                free(sort_indexarray);



            }



            DEBUG_TRACEPOINT(" ");

            refresh();

            DEBUG_TRACEPOINT(" ");

        } // end run_display

        DEBUG_TRACEPOINT("exit from if( fpsCTRLgui.run_display == 1)");

        fpsCTRLgui.run_display = run_display;

        loopcnt++;

#ifndef STANDALONE
        if((data.signal_TERM == 1)
                || (data.signal_INT == 1)
                || (data.signal_ABRT == 1)
                || (data.signal_BUS == 1)
                || (data.signal_SEGV == 1)
                || (data.signal_HUP == 1)
                || (data.signal_PIPE == 1)) {
            printf("Exit condition met\n");
            loopOK = 0;
        }
#endif
    }


    if(run_display == 1) {
        endwin();
    }

    functionparameter_outlog("FPSCTRL", "STOP");

    DEBUG_TRACEPOINT("Disconnect from FPS entries");
    for(fpsindex = 0; fpsindex < fpsCTRLgui.NBfps; fpsindex++) {
        function_parameter_struct_disconnect(&fps[fpsindex]);
    }

    free(fps);
    free(keywnode);


    char logfname[500];
    if(snprintf(logfname, stringmaxlen, "%s/fpslog.%06d", shmdname, getpid())< 0 ) {
        PRINT_ERROR("snprintf error");
    }
    remove(logfname);

    free(fpsctrltasklist);
    free(fpsctrlqueuelist);
    functionparameter_outlog("LOGFILECLOSE", "close log file");

    DEBUG_TRACEPOINT("exit from function");

    return RETURN_SUCCESS;
}




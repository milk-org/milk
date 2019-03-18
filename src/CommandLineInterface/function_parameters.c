
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

#include <00CORE/00CORE.h>
#include <CommandLineInterface/CLIcore.h>
#include "info/info.h"

/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */


#define NB_FPS_MAX 100

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


typedef struct
{
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char keyword[FUNCTION_PARAMETER_KEYWORD_MAXLEVEL][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN];
    int keywordlevel;

    int parent_index;

    int NBchild;
    int child[500];

    int leaf; // 1 if this is a leaf (no child)
    int fpsindex;
    int pindex;

} KEYWORD_TREE_NODE;




/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */





errno_t function_parameter_struct_create(
    int NBparam,
    const char *name
)
{
    int index;
    char *mapv;
    FUNCTION_PARAMETER_STRUCT  fps;

    //  FUNCTION_PARAMETER_STRUCT_MD *funcparammd;
    //  FUNCTION_PARAMETER *funcparamarray;

    char SM_fname[200];
    size_t sharedsize = 0; // shared memory size in bytes
    int SM_fd; // shared memory file descriptor

    snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", data.shmdir, name);
    remove(SM_fname);

    printf("Creating file %s\n", SM_fname);
    fflush(stdout);

    sharedsize = sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    sharedsize += sizeof(FUNCTION_PARAMETER)*NBparam;

    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (SM_fd == -1) {
        perror("Error opening file for writing");
        printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);	
        exit(0);
    }

    int result;
    result = lseek(SM_fd, sharedsize-1, SEEK_SET);
    if (result == -1) {
        close(SM_fd);
        printf("ERROR [%s %s %d]: Error calling lseek() to 'stretch' the file\n", __FILE__, __func__, __LINE__);
        printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);	
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if (result != 1) {
        close(SM_fd);
        perror("Error writing last byte of the file");
        printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);	
        exit(0);
    }

    fps.md = (FUNCTION_PARAMETER_STRUCT_MD*) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (fps.md == MAP_FAILED) {
        close(SM_fd);
        perror("Error mmapping the file");
        printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);	
        exit(0);
    }
    //funcparamstruct->md = funcparammd;

    mapv = (char*) fps.md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps.parray = (FUNCTION_PARAMETER*) mapv;



    printf("shared memory space = %ld bytes\n", sharedsize); //TEST


    fps.md->NBparam = NBparam;

    for(index=0; index<NBparam; index++)
    {
        fps.parray[index].status = 0; // not active
        fps.parray[index].cnt0 = 0;   // update counter
    }

	
    strcpy(fps.md->name, name);
    
    
    
    char cwd[FPS_CWD_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
       strcpy(fps.md->fpsdirectory, cwd);
	} else {
		perror("getcwd() error");
		return 1;
	}
    
 
	

    fps.md->signal     = (uint64_t) FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
    fps.md->confwaitus = (uint64_t) 1000; // 1 kHz default
    fps.md->msgcnt = 0;

    munmap(fps.md, sharedsize);


    return EXIT_SUCCESS;
}







long function_parameter_struct_connect(
    const char *name,
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    char SM_fname[200];
    int SM_fd; // shared memory file descriptor
    int NBparam;
    char *mapv;

    snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", data.shmdir, name);
    printf("File : %s\n", SM_fname);
    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd==-1)
    {
        printf("ERROR [%s %s %d]: cannot connect to %s\n", __FILE__, __func__, __LINE__, SM_fname);
        return(-1);
    }


    struct stat file_stat;
    fstat(SM_fd, &file_stat);


    fps->md = (FUNCTION_PARAMETER_STRUCT_MD*) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (fps->md == MAP_FAILED) {
        close(SM_fd);
        perror("Error mmapping the file");
        printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);	
        exit(0);
    }

    mapv = (char*) fps->md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps->parray = (FUNCTION_PARAMETER*) mapv;

    //	NBparam = (int) (file_stat.st_size / sizeof(FUNCTION_PARAMETER));
    NBparam = fps->md->NBparam;
    printf("Connected to %s, %d entries\n", SM_fname, NBparam);
    fflush(stdout);


    // decompose full name into pname and indices
    int NBi = 0;
    char tmpstring[200];
    char tmpstring1[100];
    char *pch;

    strncpy(tmpstring, name, 200);
    NBi = -1;
    pch = strtok (tmpstring, "-");
    while (pch != NULL)
    {
        strncpy(tmpstring1, pch, 100);

        if(NBi==-1)
            strncpy(fps->md->pname, tmpstring1, 100);

        if((NBi>=0)&&(NBi<10))
            fps->md->nameindex[NBi] = atoi(tmpstring1);

        NBi++;
        pch = strtok (NULL, "-");
    }
    fps->md->NBnameindex = NBi;

    function_parameter_printlist(fps->parray, NBparam);

    return(NBparam);
}





int function_parameter_struct_disconnect(FUNCTION_PARAMETER_STRUCT *funcparamstruct)
{
    int NBparam;

    NBparam = funcparamstruct->md->NBparam;
    munmap(funcparamstruct, sizeof(FUNCTION_PARAMETER_STRUCT_MD)+sizeof(FUNCTION_PARAMETER)*NBparam);

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

    function_parameter_struct_connect(keyword[9], &fps);

    int pindex = functionparameter_GetParamIndex(&fps, keywordfull);


    fps.parray[pindex].val.l[0] = val;

    function_parameter_struct_disconnect(&fps);

    return EXIT_SUCCESS;
}









int function_parameter_printlist(
    FUNCTION_PARAMETER  *funcparamarray,
    int NBparam
)
{
    int pindex = 0;
    int pcnt = 0;

    printf("\n");
    for(pindex=0; pindex<NBparam; pindex++)
    {
        if(funcparamarray[pindex].status & FPFLAG_ACTIVE)
        {
            int kl;

            printf("Parameter %4d : %s\n", pindex, funcparamarray[pindex].keywordfull);
            /*for(kl=0; kl< funcparamarray[pindex].keywordlevel; kl++)
            	printf("  %s", funcparamarray[pindex].keyword[kl]);
            printf("\n");*/
            printf("    %s\n", funcparamarray[pindex].description);

            // STATUS FLAGS
            printf("    STATUS FLAGS (0x%02hhx) :", (int) funcparamarray[pindex].status);
            if(funcparamarray[pindex].status & FPFLAG_ACTIVE)
                printf(" ACTIVE");
            if(funcparamarray[pindex].status & FPFLAG_USED)
                printf(" USED");
            if(funcparamarray[pindex].status & FPFLAG_VISIBLE)
                printf(" VISIBLE");
            if(funcparamarray[pindex].status & FPFLAG_WRITE)
                printf(" WRITE");
            if(funcparamarray[pindex].status & FPFLAG_WRITECONF)
                printf(" WRITECONF");
            if(funcparamarray[pindex].status & FPFLAG_WRITERUN)
                printf(" WRITERUN");
            if(funcparamarray[pindex].status & FPFLAG_LOG)
                printf(" LOG");
            if(funcparamarray[pindex].status & FPFLAG_SAVEONCHANGE)
                printf(" SAVEONCHANGE");
            if(funcparamarray[pindex].status & FPFLAG_SAVEONCLOSE)
                printf(" SAVEONCLOSE");
            if(funcparamarray[pindex].status & FPFLAG_MINLIMIT)
                printf(" MINLIMIT");
            if(funcparamarray[pindex].status & FPFLAG_MAXLIMIT)
                printf(" MAXLIMIT");
            if(funcparamarray[pindex].status & FPFLAG_CHECKSTREAM)
                printf(" CHECKSTREAM");
            if(funcparamarray[pindex].status & FPFLAG_IMPORTED)
                printf(" IMPORTED");
            if(funcparamarray[pindex].status & FPFLAG_FEEDBACK)
                printf(" FEEDBACK");
            if(funcparamarray[pindex].status & FPFLAG_ERROR)
                printf(" ERROR");
            if(funcparamarray[pindex].status & FPFLAG_ONOFF)
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

            pcnt ++;
        }
    }
    printf("\n");
    printf("%d parameters\n", pcnt);
    printf("\n");

    return 0;
}






int functionparameter_GetFileName(FUNCTION_PARAMETER *fparam, char *outfname, char *tagname)
{
    char fname[500];
    char fname1[500];
    char command[1000];
    int ret;
    int l;


    sprintf(fname, "./fpsconf");
    sprintf(command, "mkdir -p %s", fname);
    ret = system(command);

    for(l=0; l<fparam->keywordlevel-1; l++)
    {
        sprintf(fname1, "/%s", fparam->keyword[l]);
        strcat(fname, fname1);
        sprintf(command, "mkdir -p %s", fname);
        ret = system(command);
    }

    sprintf(fname1, "/%s.%s.txt", fparam->keyword[l], tagname);
    strcat(fname, fname1);
    strcpy(outfname, fname);

    return 0;
}





int functionparameter_GetParamIndex(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{
    int index = -1;
    int pindex = 0;
    int pcnt = 0;

    int NBparam = fps->md->NBparam;

    int found = 0;
    for(pindex=0; pindex<NBparam; pindex++)
    {
        if(found==0)
        {
            if(fps->parray[pindex].status & FPFLAG_ACTIVE)
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
        printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);	
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




char *functionparameter_GetParamPtr_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{	
	int fpsi = functionparameter_GetParamIndex(fps, paramname);

    return fps->parray[fpsi].val.string[0];
}

char *functionparameter_SetParamValue_STRING(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    const char *stringvalue
)
{	
	int fpsi = functionparameter_GetParamIndex(fps, paramname);

    strncpy(fps->parray[fpsi].val.string[0], stringvalue, FUNCTION_PARAMETER_STRMAXLEN);

	return EXIT_SUCCESS;
}




int functionparameter_GetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname
)
{	
	int fpsi = functionparameter_GetParamIndex(fps, paramname);

	if( fps->parray[fpsi].status & FPFLAG_ONOFF )
		return 1;
	else
		return 0;
}

int functionparameter_SetParamValue_ONOFF(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char *paramname,
    int ONOFFvalue
)
{	
	int fpsi = functionparameter_GetParamIndex(fps, paramname);

	if( ONOFFvalue == 1)
		fps->parray[fpsi].status |= FPFLAG_ONOFF;
	else
		fps->parray[fpsi].status &= ~FPFLAG_ONOFF;
	
	return EXIT_SUCCESS;
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
    char                *keywordstring,
    char                *descriptionstring,
    uint64_t             type,
    uint64_t             status,
    void *               valueptr
)
{
    int RVAL = 0;
    // 0: parameter initialized to default value
    // 1: initialized using file value
    // 2: initialized to function argument value

    int pindex = 0;
    char *pch;
    char tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    FUNCTION_PARAMETER *funcparamarray;

    funcparamarray = fps->parray;

    int NBparam = fps->md->NBparam;





    // process keywordstring
    // if string starts with ".", insert fps name
    char keywordstringC[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    if(keywordstring[0] == '.')
        sprintf(keywordstringC, "%s%s", fps->md->name, keywordstring);
    else
        strcpy(keywordstringC, keywordstring);



    // scan for existing keyword
    int scanOK = 0;
    int pindexscan;
    for(pindexscan=0; pindexscan<NBparam; pindexscan++)
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
        while((funcparamarray[pindex].status & FPFLAG_ACTIVE)&&(pindex<NBparam))
            pindex++;

        if(pindex == NBparam)
        {
            printf("ERROR [%s line %d]: NBparam limit reached\n", __FILE__, __LINE__);
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

    funcparamarray[pindex].status = status;



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
        sprintf(funcparamarray[pindex].val.string[0], "NULL");
        sprintf(funcparamarray[pindex].val.string[1], "NULL");
        break;
    case FPTYPE_DIRNAME :
        sprintf(funcparamarray[pindex].val.string[0], "NULL");
        sprintf(funcparamarray[pindex].val.string[1], "NULL");
        break;
    case FPTYPE_STREAMNAME :
        sprintf(funcparamarray[pindex].val.string[0], "NULL");
        sprintf(funcparamarray[pindex].val.string[1], "NULL");
        break;
    case FPTYPE_STRING :
        sprintf(funcparamarray[pindex].val.string[0], "NULL");
        sprintf(funcparamarray[pindex].val.string[1], "NULL");
        break;
    case FPTYPE_ONOFF :
        funcparamarray[pindex].status &= ~FPFLAG_ONOFF; // initialize state to OFF
        sprintf(funcparamarray[pindex].val.string[0], "OFF state");
        sprintf(funcparamarray[pindex].val.string[1], " ON state");
        break;
    }



    if( valueptr != NULL )// allocate value requested by function call
    {
        int64_t *valueptr_INT64;
        double *valueptr_FLOAT64;
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
			functionparameter_GetFileName(&funcparamarray[pindex], fname, "setval");
			break;
		
			case 1 :
			functionparameter_GetFileName(&funcparamarray[pindex], fname, "minval");
			break;		

			case 2 :
			functionparameter_GetFileName(&funcparamarray[pindex], fname, "maxval");
			break;

		}
		
		
        if ( (fp = fopen(fname, "r")) != NULL)
        {
			
			sprintf(systemcmd, "echo  \"-------- FILE FOUND: %s \" >> tmplog.txt", fname);
			system(systemcmd);
			
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
							funcparamarray[pindex].status |= FPFLAG_ONOFF;
						else
							funcparamarray[pindex].status &= ~FPFLAG_ONOFF;

						funcparamarray[pindex].cnt0++;
					}
                }
            }
            fclose(fp);
        }
		else
		{
			sprintf(systemcmd, "echo  \"-------- FILE NOT FOUND: %s \" >> tmplog.txt", fname);
			system(systemcmd);		
		}
    }



    if(RVAL == 0) {
        functionparameter_WriteParameterToDisk(fps, pindex, "setval", "AddEntry created");
        if(funcparamarray[pindex].status |= FPFLAG_MINLIMIT)
            functionparameter_WriteParameterToDisk(fps, pindex, "minval", "AddEntry created");
        if(funcparamarray[pindex].status |= FPFLAG_MAXLIMIT)
            functionparameter_WriteParameterToDisk(fps, pindex, "maxval", "AddEntry created");
    }

    if(RVAL == 2) {
        functionparameter_WriteParameterToDisk(fps, pindex, "setval", "AddEntry argument");
        if(funcparamarray[pindex].status |= FPFLAG_MINLIMIT)
            functionparameter_WriteParameterToDisk(fps, pindex, "minval", "AddEntry argument");
        if(funcparamarray[pindex].status |= FPFLAG_MAXLIMIT)
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


FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(const char *fpsname, uint32_t CMDmode, uint16_t *loopstatus)
{
	int NBparam = FUNCTION_PARAMETER_NBPARAM_DEFAULT;

    FUNCTION_PARAMETER_STRUCT fps;


    if(CMDmode & CMDCODE_CONFINIT) // (re-)create fps even if it exists
    {
        function_parameter_struct_create(NBparam, fpsname);
        function_parameter_struct_connect(fpsname, &fps);
    }
    else // load existing fps if exists
    {
        if(function_parameter_struct_connect(fpsname, &fps) == -1)
        {
            function_parameter_struct_create(NBparam, fpsname);
            function_parameter_struct_connect(fpsname, &fps);
        }
    }
    
    if( CMDmode & CMDCODE_CONFSTOP ) // stop fps
    {
        fps.md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
        function_parameter_struct_disconnect(&fps);
        *loopstatus = 0; // stop loop
    }
    else
		*loopstatus = 1;
    

    return fps;
}





uint16_t function_parameter_FPCONFloopstep( FUNCTION_PARAMETER_STRUCT *fps, uint32_t CMDmode, uint16_t *loopstatus )
{
    static int loopINIT = 0;
    uint16_t updateFLAG = 0;


    if(loopINIT == 0)
    {
        loopINIT = 1; // update on first loop iteration
        fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

        if( CMDmode & CMDCODE_CONFSTART )  // parameter configuration loop
        {
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
            fps->md->confpid = getpid();
            *loopstatus = 1;
        }
        else
            *loopstatus = 0;
    }


    if(fps->md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN)
    {
        // Test if CONF process is running
        if((getpgid(fps->md->confpid) >= 0)&&(fps->md->confpid>0))
            fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CONF;
        else
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CONF;

        // Test if RUN process is running
        if((getpgid(fps->md->runpid) >= 0)&&(fps->md->runpid>0))
            fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_RUN;
        else
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_RUN;


        if( fps->md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE ) // update is required
        {            
            updateFLAG = 1;
            
            fps->md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
        }
        usleep(fps->md->confwaitus);
    }
    else
        *loopstatus = 0;


    return updateFLAG;
}





uint16_t function_parameter_FPCONFexit( FUNCTION_PARAMETER_STRUCT *fps )
{
	fps->md->confpid = 0;
	fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CONF;
    function_parameter_struct_disconnect(fps);
    
    return 0;	
}




// ======================================== GUI FUNCTIONS =======================================



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

    return 0;
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
int functionparameter_WriteParameterToDisk(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex, char *tagname, char *commentstr)
{
    char fname[200];
    FILE *fp;


    // create time change tag
    char timestring[200];

    pid_t tid;
    time_t tnow;
    struct tm *uttime;
    struct timespec timenow;

    tnow = time(NULL);
    uttime = gmtime(&tnow);
    clock_gettime(CLOCK_REALTIME, &timenow);
    tid = syscall(SYS_gettid);

    sprintf(timestring, "%04d%02d%02d_%02d%02d%02ld.%09ld %8ld [%6d %6d] %s",
            1900+uttime->tm_year, 1+uttime->tm_mon, uttime->tm_mday, uttime->tm_hour, uttime->tm_min, timenow.tv_sec % 60, timenow.tv_nsec, fpsentry->parray[pindex].cnt0, getpid(), (int) tid, commentstr);


    if ( strcmp(tagname, "setval") == 0) // VALUE
    {
        functionparameter_GetFileName(&(fpsentry->parray[pindex]), fname, tagname);
        fp = fopen(fname, "w");
        switch (fpsentry->parray[pindex].type) {

        case FPTYPE_INT64:
            fprintf(fp, "%10ld  # %s\n", fpsentry->parray[pindex].val.l[0], timestring);
            break;

        case FPTYPE_FLOAT64:
            fprintf(fp, "%18f  # %s\n", fpsentry->parray[pindex].val.f[0], timestring);
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
            if( fpsentry->parray[pindex].status & FPFLAG_ONOFF )
                fprintf(fp, "1  %10s # %s\n", fpsentry->parray[pindex].val.string[1], timestring);
            else
                fprintf(fp, "0  %10s # %s\n", fpsentry->parray[pindex].val.string[0], timestring);
            break;

        }
        fclose(fp);
    }

	
	
    if ( strcmp(tagname, "minval") == 0) // MIN VALUE
    {
        functionparameter_GetFileName(&(fpsentry->parray[pindex]), fname, tagname);
        
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
        }       
    }


    if ( strcmp(tagname, "maxval") == 0) // MAX VALUE
    {
        functionparameter_GetFileName(&(fpsentry->parray[pindex]), fname, tagname);
        
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
        }       
    }
	

    if ( strcmp(tagname, "currval") == 0) // CURRENT VALUE
    {
        functionparameter_GetFileName(&(fpsentry->parray[pindex]), fname, tagname);
        
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
        }       
    }
	



	if ( strcmp(tagname, "fpsname") == 0) // FPS name
	{
		functionparameter_GetFileName(&(fpsentry->parray[pindex]), fname, tagname);
		fp = fopen(fname, "w");
		fprintf(fp, "%10s    # %s\n", fpsentry->md->name, timestring);
		fclose(fp);
	}

	if ( strcmp(tagname, "fpsdir") == 0) // FPS name
	{
		functionparameter_GetFileName(&(fpsentry->parray[pindex]), fname, tagname);
		fp = fopen(fname, "w");
		fprintf(fp, "%10s    # %s\n", fpsentry->md->fpsdirectory, timestring);
		fclose(fp);
	}	

	if ( strcmp(tagname, "status") == 0) // FPS name
	{
		functionparameter_GetFileName(&(fpsentry->parray[pindex]), fname, tagname);
		fp = fopen(fname, "w");
		fprintf(fp, "%10ld    # %s\n", fpsentry->parray[pindex].status, timestring);
		fclose(fp);
	}	






    return 0;
}










int functionparameter_CheckParameter(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex)
{
    int err = 0;
    int msglen;
    char msgadd[200];

    // if entry is not active, no error reported
    if( !(fpsentry->parray[pindex].status & FPFLAG_ACTIVE ) )
    {
        return 0;
    }

    // if entry is not used, no error reported
    if( !(fpsentry->parray[pindex].status & FPFLAG_USED ) )
    {
        return 0;
    }

    if( fpsentry->parray[pindex].status & FPFLAG_CHECKINIT )
        if(fpsentry->parray[pindex].cnt0 == 0)
        {
            fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
            fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_NOTINITIALIZED | FPS_MSG_FLAG_ERROR;
            snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "Not initialized");
            fpsentry->md->msgcnt++;
            fpsentry->md->errcnt++;
            err = 1;
        }

    if(err == 0)
    {
        // Check min value
        if( fpsentry->parray[pindex].type & FPTYPE_INT64 )
            if( fpsentry->parray[pindex].status & FPFLAG_MINLIMIT )
                if ( fpsentry->parray[pindex].val.l[0] < fpsentry->parray[pindex].val.l[1] )
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "int64 value below min");
                    fpsentry->md->msgcnt++;
                    fpsentry->md->errcnt++;

                    //msglen = strlen(fpsentry->md->message);
                    //sprintf(msgadd, "%s value below min limit\n", fpsentry->parray[pindex].keywordfull);
                    //strncat(fpsentry->md->message, msgadd, FUNCTION_PARAMETER_STRUCT_MSG_SIZE-msglen-1);
                    err = 1;
                }

        if( fpsentry->parray[pindex].type & FPTYPE_FLOAT64 )
            if( fpsentry->parray[pindex].status & FPFLAG_MINLIMIT )
                if ( fpsentry->parray[pindex].val.f[0] < fpsentry->parray[pindex].val.f[1] )
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "float value below min");
                    fpsentry->md->msgcnt++;
                    fpsentry->md->errcnt++;
                    
                    //msglen = strlen(fpsentry->md->message);
                    //sprintf(msgadd, "%s value below min limit\n", fpsentry->parray[pindex].keywordfull);
                    //strncat(fpsentry->md->message, msgadd, FUNCTION_PARAMETER_STRUCT_MSG_SIZE-msglen-1);
                    err = 1;
                }
    }

    if(err == 0)
    {
        // Check max value
        if( fpsentry->parray[pindex].type & FPTYPE_INT64 )
            if( fpsentry->parray[pindex].status & FPFLAG_MAXLIMIT )
                if ( fpsentry->parray[pindex].val.l[0] > fpsentry->parray[pindex].val.l[2] )
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "int64 value above max");
                    fpsentry->md->msgcnt++;
                    fpsentry->md->errcnt++;					
					
//                    msglen = strlen(fpsentry->md->message);
//                    sprintf(msgadd, "%s value above max limit\n", fpsentry->parray[pindex].keywordfull);
//                    strncat(fpsentry->md->message, msgadd, FUNCTION_PARAMETER_STRUCT_MSG_SIZE-msglen-1);
                    err = 1;
                }

        if( fpsentry->parray[pindex].type & FPTYPE_FLOAT64 )
            if( fpsentry->parray[pindex].status & FPFLAG_MAXLIMIT )
                if ( fpsentry->parray[pindex].val.f[0] > fpsentry->parray[pindex].val.f[2] )
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =  FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "float value above max");
                    fpsentry->md->msgcnt++;
                    fpsentry->md->errcnt++;		
					
//                    msglen = strlen(fpsentry->md->message);
//                    sprintf(msgadd, "%s value above max limit\n", fpsentry->parray[pindex].keywordfull);
//                    strncat(fpsentry->md->message, msgadd, FUNCTION_PARAMETER_STRUCT_MSG_SIZE-msglen-1);
                    err = 1;
                }
    }

    if (err == 1)
        fpsentry->parray[pindex].status |= FPFLAG_ERROR;
    else
        fpsentry->parray[pindex].status &= ~FPFLAG_ERROR;

    return err;
}







int functionparameter_CheckParametersAll(FUNCTION_PARAMETER_STRUCT *fpsentry)
{
    int NBparam;
    int pindex;
    int errcnt = 0;


    strcpy(fpsentry->md->message[0], "\0");
    NBparam = fpsentry->md->NBparam;

    // Check if Value is OK
	fpsentry->md->msgcnt = 0;
    fpsentry->md->errcnt = 0;
    //    printf("Checking %d parameter entries\n", NBparam);
    for(pindex=0; pindex<NBparam; pindex++)
        errcnt += functionparameter_CheckParameter(fpsentry, pindex);

    if(errcnt==0)
        fpsentry->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK;
    else
        fpsentry->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK;



    // compute write status

    for(pindex=0; pindex<NBparam; pindex++)
    {
        int writeOK; // do we have write permission ?

        // by default, adopt FPFLAG_WRITE flag
        if ( fpsentry->parray[pindex].status & FPFLAG_WRITE )
            writeOK = 1;
        else
            writeOK = 0;

        // if CONF running
        if( fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF )
        {
            if ( fpsentry->parray[pindex].status & FPFLAG_WRITECONF )
                writeOK = 1;
            else
                writeOK = 0;
        }

        // if RUN running
        if( fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN )
        {
            if ( fpsentry->parray[pindex].status & FPFLAG_WRITERUN )
                writeOK = 1;
            else
                writeOK = 0;
        }
        
        if( writeOK == 0)
			fpsentry->parray[pindex].status &= ~FPFLAG_WRITESTATUS;
		else
			fpsentry->parray[pindex].status |= FPFLAG_WRITESTATUS;
    }


    return 0;
}













int functionparameter_PrintParameterInfo(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex)
{
    printf("%s\n", fpsentry->parray[pindex].description);
    printf("\n");


    printf("------------- FUNCTION PARAMETER STRUCTURE\n");
    printf("FPS name       : %s\n", fpsentry->md->name);
    printf("   %s ", fpsentry->md->pname);
    int i;
    for(i=0; i< fpsentry->md->NBnameindex; i++)
        printf(" [%02d]", fpsentry->md->nameindex[i]);
    printf("\n\n");

    if( fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK )
    {
        printf("[%ld] %sScan OK%s\n", fpsentry->md->msgcnt, BLINKHIGREEN, RESET);
    }
    else
    {
		int msgi;
		
        printf("[%ld] %s%ld ERROR(s)%s\n", fpsentry->md->msgcnt, BLINKHIRED, fpsentry->md->errcnt, RESET);
        for(msgi=0; msgi<fpsentry->md->msgcnt; msgi++)
        {
			printf("[%3d] %s%s%s\n", fpsentry->md->msgpindex[msgi], BOLDHIRED, fpsentry->md->message[msgi], RESET);
		}
    }

    printf("\n");



    printf("------------- FUNCTION PARAMETER \n");
    printf("[%d] Parameter name : %s\n", pindex, fpsentry->parray[pindex].keywordfull);

    if( fpsentry->parray[pindex].type & FPTYPE_UNDEF )
        printf("  TYPE UNDEF\n");
    if( fpsentry->parray[pindex].type & FPTYPE_INT64 )
        printf("  TYPE INT64\n");
    if( fpsentry->parray[pindex].type & FPTYPE_FLOAT64 )
        printf("  TYPE FLOAT64\n");
    if( fpsentry->parray[pindex].type & FPTYPE_PID )
        printf("  TYPE PID\n");
    if( fpsentry->parray[pindex].type & FPTYPE_TIMESPEC )
        printf("  TYPE TIMESPEC\n");
    if( fpsentry->parray[pindex].type & FPTYPE_FILENAME )
        printf("  TYPE FILENAME\n");
    if( fpsentry->parray[pindex].type & FPTYPE_DIRNAME )
        printf("  TYPE DIRNAME\n");
    if( fpsentry->parray[pindex].type & FPTYPE_STREAMNAME )
        printf("  TYPE STREAMNAME\n");
    if( fpsentry->parray[pindex].type & FPTYPE_STRING )
        printf("  TYPE STRING\n");
    if( fpsentry->parray[pindex].type & FPTYPE_ONOFF )
        printf("  TYPE ONOFF\n");


    printf("\n");
    printf("------------- FLAGS \n");

    if( fpsentry->parray[pindex].status & FPFLAG_ACTIVE )
        printf("   FPFLAG_ACTIVE\n");
    if( fpsentry->parray[pindex].status & FPFLAG_VISIBLE )
        printf("   FPFLAG_VISIBLE\n");
    if( fpsentry->parray[pindex].status & FPFLAG_WRITECONF )
        printf("   FPFLAG_WRITECONF\n");
    if( fpsentry->parray[pindex].status & FPFLAG_WRITERUN )
        printf("   FPFLAG_WRITERUN\n");
    if( fpsentry->parray[pindex].status & FPFLAG_LOG )
        printf("   FPFLAG_LOG\n");
    if( fpsentry->parray[pindex].status & FPFLAG_SAVEONCHANGE )
        printf("   FPFLAG_SAVEONCHANGE\n");
    if( fpsentry->parray[pindex].status & FPFLAG_SAVEONCLOSE )
        printf("   FPFLAG_SAVEONCLOSE\n");
    if( fpsentry->parray[pindex].status & FPFLAG_MINLIMIT )
        printf("   FPFLAG_MINLIMIT\n");
    if( fpsentry->parray[pindex].status & FPFLAG_MAXLIMIT )
        printf("   FPFLAG_MAXLIMIT\n");
    if( fpsentry->parray[pindex].status & FPFLAG_CHECKSTREAM )
        printf("   FPFLAG_CHECKSTREAM\n");
    if( fpsentry->parray[pindex].status & FPFLAG_IMPORTED )
        printf("   FPFLAG_IMPORTED\n");
    if( fpsentry->parray[pindex].status & FPFLAG_FEEDBACK )
        printf("   FPFLAG_FEEDBACK\n");

    printf("\n");
    printf("\n");
    printf("cnt0 = %ld\n", fpsentry->parray[pindex].cnt0);

    printf("\n");

    printf("Current value : ");

    if(fpsentry->parray[pindex].type & FPTYPE_UNDEF)
        printf("  %s", "-undef-");

    if(fpsentry->parray[pindex].type & FPTYPE_INT64)
        printf("  %10d", (int) fpsentry->parray[pindex].val.l[0]);

    if(fpsentry->parray[pindex].type & FPTYPE_FLOAT64)
        printf("  %10f", (float) fpsentry->parray[pindex].val.f[0]);

    if(fpsentry->parray[pindex].type & FPTYPE_PID)
        printf("  %10d", (int) fpsentry->parray[pindex].val.pid[0]);

    if(fpsentry->parray[pindex].type & FPTYPE_TIMESPEC)
        printf("  %10s", "-timespec-");

    if(fpsentry->parray[pindex].type & FPTYPE_FILENAME)
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);

    if(fpsentry->parray[pindex].type & FPTYPE_DIRNAME)
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);

    if(fpsentry->parray[pindex].type & FPTYPE_STREAMNAME)
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);

    if(fpsentry->parray[pindex].type & FPTYPE_STRING)
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);

    if(fpsentry->parray[pindex].type & FPTYPE_ONOFF)
    {
		/*if ( fpsentry->parray[pindex].status & FPFLAG_ONOFF )
			printf("    ON  %s\n", fpsentry->parray[pindex].val.string[1]);
		else
			printf("   OFF  %s\n", fpsentry->parray[pindex].val.string[0]);*/
	}

    printf("\n");
    printf("\n");

    return 0;

}





/**
 *
 * ## PURPOSE
 *
 * Enter new value for parameter
 *
 *
 */


int functionparameter_UserInputSetParamValue(FUNCTION_PARAMETER_STRUCT *fpsentry, int pindex)
{
    int inputOK;
    int strlenmax = 20;
    char buff[100];
    char c;

    functionparameter_PrintParameterInfo(fpsentry, pindex);


    if ( fpsentry->parray[pindex].status & FPFLAG_WRITESTATUS )
    {
        inputOK = 0;
        fflush(stdout);

        while(inputOK == 0)
        {
            printf ("\nESC or update value : ");
            fflush(stdout);

            int stringindex = 0;
            c = getchar();
            while( (c != 27) && (c != 10) && (c != 13) && (stringindex<strlenmax-1) )
            {
                buff[stringindex] = c;
                if(c == 127) // delete key
                {
                    putchar (0x8);
                    putchar (' ');
                    putchar (0x8);
                    stringindex --;
                }
                else
                {
                    putchar(c);  // echo on screen
                    // printf("[%d]", (int) c);
                    stringindex++;
                }
                if(stringindex<0)
                    stringindex = 0;
                c = getchar();
            }
            buff[stringindex] = '\0';
            inputOK = 1;
        }

        if(c != 27) // do not update value if escape key
        {

            long lval;
            double fval;
            char *endptr;
            char c;
            int vOK = 1;

            switch (fpsentry->parray[pindex].type) {

            case FPTYPE_INT64:
                errno = 0;    /* To distinguish success/failure after call */
                lval = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if ((errno == ERANGE && (lval == LONG_MAX || lval == LONG_MIN))
                        || (errno != 0 && lval == 0)) {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if (endptr == buff) {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                    fpsentry->parray[pindex].val.l[0] = lval;
                break;

            case FPTYPE_FLOAT64:
                errno = 0;    /* To distinguish success/failure after call */
                fval = strtod(buff, &endptr);

                /* Check for various possible errors */
                if ( (errno == ERANGE)
                        || (errno != 0 && lval == 0)) {
                    perror("strtod");
                    vOK = 0;
                    sleep(1);
                }

                if (endptr == buff) {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                    fpsentry->parray[pindex].val.f[0] = fval;
                break;


            case FPTYPE_PID :
                errno = 0;    /* To distinguish success/failure after call */
                lval = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if ((errno == ERANGE && (lval == LONG_MAX || lval == LONG_MIN))
                        || (errno != 0 && lval == 0)) {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if (endptr == buff) {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                    fpsentry->parray[pindex].val.pid[0] = (pid_t) lval;
                break;


            case FPTYPE_FILENAME :
                snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff);
                break;

            case FPTYPE_DIRNAME :
                snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff);
                break;

            case FPTYPE_STREAMNAME :
                snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff);
                break;

            case FPTYPE_STRING :
                snprintf(fpsentry->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN, "%s", buff);
                break;


            }

            fpsentry->parray[pindex].cnt0++;

            // notify GUI
            fpsentry->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;


            // Save to disk
            if( fpsentry->parray[pindex].status & FPFLAG_SAVEONCHANGE)
            {
                functionparameter_WriteParameterToDisk(fpsentry, pindex, "setval", "UserInputSetParamValue");
            }
        }
    }
    else
    {
        printf("%s Value cannot be modified %s\n", BOLDHIRED, RESET);
        c = getchar();
    }



    return 0;
}






void functionparameter_CTRLscreen_atexit()
{
//	endwin();
}


/**
 * ## Purpose
 *
 * Automatically build simple ASCII GUI from function parameter structure (fps) name mask
 *
 *
 *
 */
errno_t functionparameter_CTRLscreen(uint32_t mode, char *fpsnamemask)
{
    // function parameter structure(s)
    int NBfps;
    int fpsindex;
    FUNCTION_PARAMETER_STRUCT fps[NB_FPS_MAX];
    int fps_symlink[NB_FPS_MAX];

    // display index
    long NBindex;

    // function parameters
    long NBpindex = 0;
    long pindex;
    int *p_fpsindex; // fps index for parameter
    int *p_pindex;   // index within fps

    // keyword tree
    int kwnindex;
    int NBkwn = 0;
    KEYWORD_TREE_NODE keywnode[100];

    int l;

    char  monstring[200];
    int loopOK = 1;
    long long loopcnt = 0;


    int nodechain[MAXNBLEVELS];
    int iSelected[MAXNBLEVELS];

    // current selection
    int fpsindexSelected = 0;
    int pindexSelected = 0;
    int nodeSelected = 0;


	// FPS list file
	FILE *fpfpslist;
	int fpslistcnt = 0;
	char FPSlist[200][100];

    // input command
    FILE *fpinputcmd;






	// request match to file ./fpscomd/fpslist.txt
	if ( mode & 0x0001 )  
	{
		if( (fpfpslist = fopen("fpscmd/fpslist.txt", "r")) != NULL)
		{
			char * FPSlistline = NULL;
            size_t len = 0;
            ssize_t read;
            char FPSlistentry[200];
            
            while ((read = getline(&FPSlistline, &len, fpfpslist)) != -1) {
				if(FPSlistline[0] != '#')
				{
				    char * pch;
                    int nbword = 0;

                    pch = strtok (FPSlistline, " \t\n\r");
                    if(pch != NULL)
                    {
						sprintf( FPSlist[fpslistcnt], "%s", pch);
						fpslistcnt++;
					}
				}
			}
			fclose(fpfpslist);
		}
		else
		{
			printf("Cannot open file fpscmd/fpslist.txt\n");
		}
	
		int fpsi;
		for(fpsi=0; fpsi<fpslistcnt; fpsi++)
			printf("FPSname must match %s\n", FPSlist[fpsi]);
	}

	


    for(l=0; l<MAXNBLEVELS; l++)
    {
        nodechain[l] = 0;
        iSelected[l] = 0;
    }


    //    NBparam = function_parameter_struct_connect(fpsname, &fps[fpsindex]);


    // create ROOT node (invisible)
    keywnode[0].keywordlevel = 0;
    sprintf(keywnode[0].keyword[0], "ROOT");
    keywnode[0].leaf = 0;
    keywnode[0].NBchild = 0;
    NBkwn = 1;


    // scan filesystem for fps entries
    DIR *d;
    struct dirent *dir;


    d = opendir(data.shmdir);
    if(d)
    {
        fpsindex = 0;
        pindex = 0;
        while(((dir = readdir(d)) != NULL))
        {
            char *pch = strstr(dir->d_name, ".fps.shm");

            int matchOK = 0;
            // name filtering
            if(strcmp(fpsnamemask, "_ALL") == 0)
                matchOK = 1;
            else
            {
                if(strncmp(dir->d_name, fpsnamemask, strlen(fpsnamemask)) == 0)
                    matchOK = 1;
            }
            
            
            if(mode & 0x0001) // enforce match to list
            {
				int matchOKlist = 0;
				int fpsi;
				
				for(fpsi=0; fpsi<fpslistcnt; fpsi++)
					if(strncmp(dir->d_name, FPSlist[fpsi], strlen(FPSlist[fpsi])) == 0)
						matchOKlist = 1;

				matchOK *= matchOKlist;
			}




            if((pch) && (matchOK == 1))
            {
                // is file sym link ?
                struct stat buf;
                int retv;
                char fullname[200];

                sprintf(fullname, "%s/%s", data.shmdir, dir->d_name);
                retv = lstat (fullname, &buf);
                if (retv == -1 ) {
                    endwin();
                    printf("File \"%s\"", dir->d_name);
                    perror("Error running lstat on file ");
                    printf("STEP %s %d\n", __FILE__, __LINE__);
                    fflush(stdout);
                    exit( EXIT_FAILURE );
                }

                if (S_ISLNK(buf.st_mode)) // resolve link name
                {
                    char fullname[200];
                    char linknamefull[200];
                    char linkname[200];
                    int nchar;
                    int ret;

                    fps_symlink[fpsindex] = 1;
                    sprintf(fullname, "%s/%s", data.shmdir, dir->d_name);
                    ret = readlink (fullname, linknamefull, 200-1);  // todo: replace with realpath()

                    strcpy(linkname, basename(linknamefull));

                    int lOK = 1;
                    int ii = 0;
                    while((lOK == 1)&&(ii<strlen(linkname)))
                    {
                        if(linkname[ii] == '.')
                        {
                            linkname[ii] = '\0';
                            lOK = 0;
                        }
                        ii++;
                    }

                    //                        strncpy(streaminfo[sindex].linkname, linkname, nameNBchar);
                }
                else
                    fps_symlink[fpsindex] = 0;


                char fpsname[200];
                strncpy(fpsname, dir->d_name, strlen(dir->d_name)-strlen(".fps.shm"));

                int NBparamMAX = function_parameter_struct_connect(fpsname, &fps[fpsindex]);
                int i;
                for(i=0; i<NBparamMAX; i++)
                {
                    if(fps[fpsindex].parray[i].status & FPFLAG_ACTIVE)  // if entry is active
                    {
                        // find or allocate keyword node
                        int level;
                        for(level=1; level < fps[fpsindex].parray[i].keywordlevel+1; level++)
                        {

                            // does node already exist ?
                            int scanOK = 0;
                            for(kwnindex=0; kwnindex<NBkwn; kwnindex++) // scan existing nodes looking for match
                            {
                                if(keywnode[kwnindex].keywordlevel == level) // levels have to match
                                {
                                    int match = 1;
                                    for(l=0; l<level; l++) // keywords at all levels need to match
                                    {
                                        if( strcmp(fps[fpsindex].parray[i].keyword[l], keywnode[kwnindex].keyword[l]) != 0 )
                                            match = 0;
                                    }
                                    if(match == 1) // we have a match
                                        scanOK = 1;
                                }
                            }



                            if(scanOK == 0) // node does not exit -> create it
                            {

                                // look for parent
                                int scanparentOK = 0;
                                int kwnindexp = 0;
                                keywnode[kwnindex].parent_index = 0; // default value, not found -> assigned to ROOT

                                while ((kwnindexp<NBkwn) && (scanparentOK==0))
                                {
                                    if(keywnode[kwnindexp].keywordlevel == level-1) // check parent has level-1
                                    {
                                        int match = 1;

                                        for(l=0; l<level-1; l++) // keywords at all levels need to match
                                        {
                                            if( strcmp(fps[fpsindex].parray[i].keyword[l], keywnode[kwnindexp].keyword[l]) != 0 )
                                                match = 0;
                                        }
                                        if(match == 1) // we have a match
                                            scanparentOK = 1;
                                    }
                                    kwnindexp++;
                                }

                                if(scanparentOK == 1)
                                {
                                    keywnode[kwnindex].parent_index = kwnindexp-1;
                                    int cindex;
                                    cindex = keywnode[keywnode[kwnindex].parent_index].NBchild;
                                    keywnode[keywnode[kwnindex].parent_index].child[cindex] = kwnindex;
                                    keywnode[keywnode[kwnindex].parent_index].NBchild++;
                                }




                                printf("CREATING NODE ");
                                keywnode[kwnindex].keywordlevel = level;



                                for(l=0; l<level; l++) {
                                    char tmpstring[200];
                                    strcpy(keywnode[kwnindex].keyword[l], fps[fpsindex].parray[i].keyword[l]);
                                    printf(" %s", keywnode[kwnindex].keyword[l]);
                                    if(l==0)
                                        strcpy(keywnode[kwnindex].keywordfull, keywnode[kwnindex].keyword[l]);
                                    else
                                    {
                                        sprintf(tmpstring, ".%s", keywnode[kwnindex].keyword[l]);
                                        strcat(keywnode[kwnindex].keywordfull, tmpstring);
                                    }
                                }
                                printf("   %d %d\n", keywnode[kwnindex].keywordlevel, fps[fpsindex].parray[i].keywordlevel);

                                if(keywnode[kwnindex].keywordlevel == fps[fpsindex].parray[i].keywordlevel)
                                {
                                    //									strcpy(keywnode[kwnindex].keywordfull, fps[fpsindex].parray[i].keywordfull);

                                    keywnode[kwnindex].leaf = 1;
                                    keywnode[kwnindex].fpsindex = fpsindex;
                                    keywnode[kwnindex].pindex = i;
                                }
                                else
                                {


                                    keywnode[kwnindex].leaf = 0;
                                }




                                kwnindex ++;
                                NBkwn = kwnindex;
                            }
                        }
                        pindex++;
                    }
                }

                printf("Found fps %-20s %d parameters\n", fpsname, fps[fpsindex].md->NBparam);

                fpsindex ++;
            }
        }
    }
    else
    {
        printf("ERROR: missing %s directory\n", data.shmdir);
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    NBfps = fpsindex;
    NBpindex = pindex;


    // print keywords
    printf("Found %d keyword node(s)\n", NBkwn);
    int level;
    for(level=0; level<FUNCTION_PARAMETER_KEYWORD_MAXLEVEL; level++)
    {
        printf("level %d :\n", level);
        for(kwnindex=0; kwnindex<NBkwn; kwnindex++)
        {
            if(keywnode[kwnindex].keywordlevel == level)
            {
                printf("   %3d->[%3d]->x%d   (%d)", keywnode[kwnindex].parent_index, kwnindex, keywnode[kwnindex].NBchild, keywnode[kwnindex].leaf);
                printf("%s", keywnode[kwnindex].keyword[0]);

                for(l=1; l<level; l++)
                    printf(".%s", keywnode[kwnindex].keyword[l]);
                printf("\n");
            }
        }
    }

    printf("%d function parameter structure(s) imported, %ld parameters\n", NBfps, NBpindex);


    if(NBfps == 0)
    {
        printf("No function parameter structure found\n");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        return 0;
    }



    // INITIALIZE ncurses
    initncurses();
	atexit( functionparameter_CTRLscreen_atexit );
    clear();
	
    int currentnode = 0;
    int currentlevel = 0;
    NBindex = 0;

    while( loopOK == 1 )
    {
        int i;
        int fpsindex;
        int pindex;
        long pcnt;
        char command[500];
        char tmuxname[500];
        int nameindex;
        long icnt = 0;

        usleep(10000); // 100 Hz display



        int ch = getch();

        switch (ch)
        {
        case 'x':     // Exit control screen
            loopOK=0;
            break;

        case 'h':     // help
            endwin();
            system("clear");

            printf("Function Parameter Structure (FPS) Control \n");
            printf("\n");
            printf("\n");
            printf("\n");
            printf("  Arrow keys     NAVIGATE\n");
            printf("  ENTER          Select parameter to read/set\n");
            printf("\n");
            printf("  R/r            start/stop run process\n");
            printf("  C/c            start/stop config process\n");
            printf("  l              list all entries\n");
            printf("\n");
            printf("  (x)            Exit\n");
            printf("\n");
            printf("Press Any Key to Continue\n");
            getchar();
            initncurses();
            break;

        case KEY_UP:
            iSelected[currentlevel] --;
            if(iSelected[currentlevel]<0)
                iSelected[currentlevel] = 0;
            break;

        case KEY_DOWN:
            iSelected[currentlevel] ++;
            if(iSelected[currentlevel] > NBindex-1)
                iSelected[currentlevel] = NBindex-1;
            break;

        case KEY_PPAGE:
            iSelected[currentlevel] -= 10;
            if(iSelected[currentlevel]<0)
                iSelected[currentlevel] = 0;
            break;

        case KEY_NPAGE:
            iSelected[currentlevel] += 10;
            if(iSelected[currentlevel] > NBindex-1)
                iSelected[currentlevel] = NBindex-1;
            break;


        case KEY_LEFT:
            if(currentnode != 0) // ROOT has no parent
            {
                currentnode = keywnode[currentnode].parent_index;
                nodeSelected = currentnode;
            }
            break;


        case KEY_RIGHT :
            if( keywnode[nodeSelected].leaf == 0 ) // this is a directory
            {
                if(keywnode[keywnode[currentnode].child[iSelected[currentlevel]]].leaf == 0)
                {
                    currentnode = keywnode[currentnode].child[iSelected[currentlevel]];
                }
            }
            break;

        case 10 :
            if( keywnode[nodeSelected].leaf == 1 ) // this is a leaf
            {
                endwin();
                if(system("clear") != 0) // clear screen
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
                functionparameter_UserInputSetParamValue(&fps[fpsindexSelected], pindexSelected);
                initncurses();
            }
            break;

        case ' ' : // toggles ON / OFF
            fpsindex = keywnode[nodeSelected].fpsindex;
            pindex = keywnode[nodeSelected].pindex;
            if( fps[fpsindex].parray[pindex].status & FPFLAG_WRITESTATUS )
            {
                if( fps[fpsindex].parray[pindex].type == FPTYPE_ONOFF )
                {
                    if ( fps[fpsindex].parray[pindex].status & FPFLAG_ONOFF ) // ON -> OFF
                        fps[fpsindex].parray[pindex].status &= ~FPFLAG_ONOFF;
                    else // OFF -> ON
                        fps[fpsindex].parray[pindex].status |= FPFLAG_ONOFF;

                    fps[fpsindex].parray[pindex].cnt0 ++;

                    fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
                }
            }
            break;

        case 'R' : // start run process if possible
            if( fps[keywnode[nodeSelected].fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK )
            {
                sprintf(command, "tmux new-session -d -s %s-run &> /dev/null", fps[keywnode[iSelected[currentlevel]].fpsindex].md->name);
                system(command);

                sprintf(command, "tmux send-keys -t %s-run \"./fpscmd/%s-runstart", fps[keywnode[iSelected[currentlevel]].fpsindex].md->name, fps[keywnode[iSelected[currentlevel]].fpsindex].md->pname);
                for(nameindex=0; nameindex<fps[keywnode[iSelected[currentlevel]].fpsindex].md->NBnameindex; nameindex++)
                {
                    char tmpstring[20];

                    sprintf(tmpstring, " %02d", fps[keywnode[iSelected[currentlevel]].fpsindex].md->nameindex[nameindex]);
                    strcat(command, tmpstring);
                }
                strcat(command, "\" C-m");
                system(command);
                fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
                fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            }
            break;

        case 'r' : // stop run process
            sprintf(command, "./fpscmd/%s-runstop", fps[keywnode[iSelected[currentlevel]].fpsindex].md->pname);
            for(nameindex=0; nameindex<fps[keywnode[iSelected[currentlevel]].fpsindex].md->NBnameindex; nameindex++)
            {
                char tmpstring[20];

                sprintf(tmpstring, " %02d", fps[keywnode[iSelected[currentlevel]].fpsindex].md->nameindex[nameindex]);
                strcat(command, tmpstring);
            }          
            system(command);
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            break;


        case 'C' : // start conf process
            //printf("STEP %s %d\n", __FILE__, __LINE__);
            //fflush(stdout);

            sprintf(command, "tmux new-session -d -s %s-conf > /dev/null 2>&1", fps[keywnode[iSelected[currentlevel]].fpsindex].md->name);
            system(command);

            //printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);

            sprintf(command, "tmux send-keys -t %s-conf \"./fpscmd/%s-confstart", fps[keywnode[iSelected[currentlevel]].fpsindex].md->name, fps[keywnode[iSelected[currentlevel]].fpsindex].md->pname);
            for(nameindex=0; nameindex<fps[keywnode[iSelected[currentlevel]].fpsindex].md->NBnameindex; nameindex++)
            {
                char tmpstring[20];

                //printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);

                sprintf(tmpstring, " %02d", fps[keywnode[iSelected[currentlevel]].fpsindex].md->nameindex[nameindex]);
                strcat(command, tmpstring);
            }
            strcat(command, "\" C-m");
            //printf("STEP %s %d\n", __FILE__, __LINE__); fflush(stdout);
            system(command);
            fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update

            break;

        case 'c': // kill conf process
            fps[keywnode[iSelected[currentlevel]].fpsindex].md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
            sprintf(command, "tmux send-keys -t %s-conf C-c &> /dev/null", fps[keywnode[iSelected[currentlevel]].fpsindex].md->name);
            system(command);
            fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
            fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
            break;

        case 'l': // list all parameters
            endwin();
            system("clear");
            printf("FPS entries - Full list \n");
            printf("\n");
            for(kwnindex=0; kwnindex<NBkwn; kwnindex++)
            {
                if(keywnode[kwnindex].leaf==1)
                    printf("%4d  %4d  %s\n", keywnode[kwnindex].fpsindex, keywnode[kwnindex].pindex, keywnode[kwnindex].keywordfull);
            }
            printf("\n");
            printf("Press Any Key to Continue\n");
            getchar();
            initncurses();
            break;


        case 'P': // process input command file
            endwin();
            system("clear");
            fpinputcmd = fopen("confscript", "r");
            if(fpinputcmd != NULL)
            {
                char * FPScmdline = NULL;
                size_t len = 0;
                ssize_t read;
                char FPScommand[50];
                char FPSentryname[500];
                char FPSvaluestring[200];

                while ((read = getline(&FPScmdline, &len, fpinputcmd)) != -1) {
                    // break line in words
                    char * pch;
                    int nbword = 0;

                    pch = strtok (FPScmdline, " \t");
                    
                    sprintf( FPScommand, "%s", pch);

                    while (pch != NULL)
                    {
                        nbword++;
                        pch = strtok (NULL, " \t");
                        if(nbword==1)
                        {
							char *pos;
                            sprintf(FPSentryname, "%s", pch);
                            if ((pos=strchr(FPSvaluestring, '\n')) != NULL)
								*pos = '\0';
                        }
                        if(nbword==2)
                        {
							char *pos;							
                            sprintf(FPSvaluestring, "%s", pch);
							if ((pos=strchr(FPSvaluestring, '\n')) != NULL)
								*pos = '\0';
						}
                    }

                    if( (nbword>2) && (FPScommand[0] != '#') )
                    {
                        // look for entry
                        int kwnindex = -1;
                        int kwnindexscan = 0;
                        while( (kwnindex==-1) && (kwnindexscan<NBkwn))
                        {
                            if(strcmp(keywnode[kwnindexscan].keywordfull, FPSentryname)==0)
                                kwnindex = kwnindexscan;
                            kwnindexscan ++;
                        }

                        printf("[%4d]  ", kwnindex);

                        if(kwnindex!=-1)
                        {
                            fpsindex = keywnode[kwnindex].fpsindex;
                            pindex = keywnode[kwnindex].pindex;

                            if(strcmp(FPScommand,"setval")==0)
                            {
								int updated = 0;
								
                                switch (fps[fpsindex].parray[pindex].type) {
                                case FPTYPE_INT64:
                                    if( functionparameter_SetParamValue_INT64(&fps[fpsindex], FPSentryname, atol(FPSvaluestring)) == EXIT_SUCCESS )
										updated = 1;
                                    printf("setval  INT64       %40s  = %ld", FPSentryname, atol(FPSvaluestring));
                                    break;

                                case FPTYPE_FLOAT64:
                                    if(functionparameter_SetParamValue_FLOAT64(&fps[fpsindex], FPSentryname, atof(FPSvaluestring))==EXIT_SUCCESS)
										updated = 1;
                                    printf("setval  FLOAT64     %40s  = %f", FPSentryname, atof(FPSvaluestring));
                                    break;

                                case FPTYPE_PID:
                                    if(functionparameter_SetParamValue_INT64(&fps[fpsindex], FPSentryname, atol(FPSvaluestring))==EXIT_SUCCESS)
										updated = 1;
                                    printf("setval  PID         %40s  = %ld", FPSentryname, atol(FPSvaluestring));
                                    break;

                                case FPTYPE_TIMESPEC:
                                    //
                                    break;

                                case FPTYPE_FILENAME:
                                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPSvaluestring)==EXIT_SUCCESS)
										updated = 1;
                                    printf("setval  FILENAME    %40s  = %s", FPSentryname, FPSvaluestring);
                                    break;

                                case FPTYPE_DIRNAME:
                                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPSvaluestring)==EXIT_SUCCESS)
										updated = 1;
                                    printf("setval  DIRNAME     %40s  = %s", FPSentryname, FPSvaluestring);
                                    break;

                                case FPTYPE_STREAMNAME:
                                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPSvaluestring)==EXIT_SUCCESS)
										updated = 1;
                                    printf("setval  STREAMNAME  %40s  = %s", FPSentryname, FPSvaluestring);
                                    break;

                                case FPTYPE_STRING:
                                    if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname, FPSvaluestring)==EXIT_SUCCESS)
										updated = 1;
                                    printf("setval  STRING      %40s  = %s", FPSentryname, FPSvaluestring);
                                    break;

                                case FPTYPE_ONOFF:
                                    if( strncmp(FPSvaluestring,"ON", 2) == 0)
                                    {
                                        if(functionparameter_SetParamValue_ONOFF(&fps[fpsindex], FPSentryname, 1)==EXIT_SUCCESS)
											updated = 1;
                                        printf("setval  ONOFF       %40s  = ON", FPSentryname);
                                    }
                                    if( strncmp(FPSvaluestring,"OFF", 3) == 0)
                                    {
                                        if(functionparameter_SetParamValue_ONOFF(&fps[fpsindex], FPSentryname, 0)==EXIT_SUCCESS)
											updated = 1;
                                        printf("setval  ONOFF       %40s  = OFF", FPSentryname);
                                    }
                                    break;
                                }
                                if(updated == 1)
								{
									functionparameter_WriteParameterToDisk(&fps[fpsindex], pindex, "setval", "input command file");
								}
                            }
                            

                        }
                        printf("\n");
                    }

                }
                fclose(fpinputcmd);
            }

            printf("\n");
            printf("Press Any Key to Continue\n");
            getchar();
            initncurses();
            break;

        }

        erase();

        attron(A_BOLD);
        sprintf(monstring, "FUNCTION PARAMETER MONITOR: PRESS (x) TO STOP, (h) FOR HELP");
        print_header(monstring, '-');
        attroff(A_BOLD);
        printw("\n");

        printw("currentlevel = %d   Selected = %d/%d   Current node [%3d]: ", currentlevel, iSelected[currentlevel], NBindex, currentnode);

        if(currentnode==0)
        {
            printw("ROOT");
        }
        else
        {
            for(l=0; l<keywnode[currentnode].keywordlevel; l++)
                printw("%s.", keywnode[currentnode].keyword[l]);
        }
        printw("  NBchild = %d\n", keywnode[currentnode].NBchild);

        printw("tmux sessions :  %s-conf  %s-run\n", fps[keywnode[iSelected[currentlevel]].fpsindex].md->name, fps[keywnode[iSelected[currentlevel]].fpsindex].md->name);
        printw("Selected Node %ld : %s\n", nodeSelected, keywnode[nodeSelected].keywordfull);

        printw("\n");


        currentlevel = keywnode[currentnode].keywordlevel;
        int imax = keywnode[currentnode].NBchild; // number of lines to be displayed

        nodechain[currentlevel] = currentnode;
        l = currentlevel-1;
        while(l>0)
        {
            nodechain[l] = keywnode[nodechain[l+1]].parent_index;
            l--;
        }
        nodechain[0] = 0; // root




        pcnt = 0;


        int i1 = 0;
        for(i=0; i<imax; i++) // i is the line number on GUI display
        {

            for(l=0; l<currentlevel; l++)
            {
                if(keywnode[nodechain[l]].NBchild > imax)
                    imax = keywnode[nodechain[l]].NBchild;

                if(i<keywnode[nodechain[l]].NBchild)
                {
                    int snode = 0; // selected node
                    int ii;

                    if(keywnode[nodechain[l]].child[i] == nodechain[l+1])
                        snode = 1;

                    attron(A_REVERSE);
                    printw(" ");
                    attroff(A_REVERSE);

                    if(snode == 1)
                        attron(A_REVERSE);


                    ii = keywnode[nodechain[l]].child[i];

                    if(keywnode[ii].leaf == 0) // directory
                        attron(COLOR_PAIR(5));

                    printw("%-10s ", keywnode[keywnode[nodechain[l]].child[i]].keyword[l]);

                    if(keywnode[ii].leaf == 0) // directory
                        attroff(COLOR_PAIR(5));

                    if(snode == 1)
                        attroff(A_REVERSE);



                }
                else
                    printw("            ");
            }









            int ii;
            ii = keywnode[currentnode].child[i1];
            fpsindex = keywnode[ii].fpsindex;
            pindex = keywnode[ii].pindex;

            while ( (! ( fps[fpsindex].parray[pindex].status & FPFLAG_VISIBLE )) && (i1 < keywnode[currentnode].NBchild) ) // if not visible, advance to next one
            {
                i1++;
                ii = keywnode[currentnode].child[i1];
                fpsindex = keywnode[ii].fpsindex;
                pindex = keywnode[ii].pindex;
            }

            if(i1 < keywnode[currentnode].NBchild)
            {
                attron(A_REVERSE);
                printw(" ");
                attroff(A_REVERSE);


                if(keywnode[ii].leaf == 0) // If this is a directory
                {
                    if(i == iSelected[currentlevel])
                    {
                        attron(A_REVERSE);
                        nodeSelected = ii;
                    }
                    attron(COLOR_PAIR(5));
                    l = keywnode[ii].keywordlevel;
                    printw("%s", keywnode[ii].keyword[l-1]);
                    attroff(COLOR_PAIR(5));

                    if(i == iSelected[currentlevel])
                        attroff(A_REVERSE);

                }
                else // If this is a parameter
                {
                    fpsindex = keywnode[ii].fpsindex;
                    pindex = keywnode[ii].pindex;


                    if( fps[fpsindex].parray[pindex].status & FPFLAG_VISIBLE )
                    {
                        int kl;

                        if(i == iSelected[currentlevel])
                        {
                            pindexSelected = keywnode[ii].pindex;
                            fpsindexSelected = keywnode[ii].fpsindex;
                            nodeSelected = ii;

                            attron(COLOR_PAIR(10)|A_BOLD);
                        }

                        if ( fps[fpsindex].parray[pindex].status & FPFLAG_WRITESTATUS )
                        {
                            attron(COLOR_PAIR(10)|A_BLINK);
                            printw("W "); // writable
                            attroff(COLOR_PAIR(10)|A_BLINK);
                        }
                        else
                        {
                            attron(COLOR_PAIR(4)|A_BLINK);
                            printw("NW"); // non writable
                            attroff(COLOR_PAIR(4)|A_BLINK);
                        }


                        l = keywnode[ii].keywordlevel;
                        printw(" %-16s", fps[fpsindex].parray[pindex].keyword[l-1]);

                        if(i == iSelected[currentlevel])
                            attroff(COLOR_PAIR(10));

                        printw("   ");

                        pid_t pid;
                        pid = fps[fpsindex].md->confpid;
                        if((getpgid(pid) >= 0)&&(pid>0))
                        {
                            attron(COLOR_PAIR(2));
                            printw("%5d ", (int) pid);
                            attroff(COLOR_PAIR(2));
                        }
                        else
                            printw("----- ");

                        pid = fps[fpsindex].md->runpid;
                        if((getpgid(pid) >= 0)&&(pid>0))
                        {
                            attron(COLOR_PAIR(2));
                            printw("%5d ", (int) pid);
                            attroff(COLOR_PAIR(2));
                        }
                        else
                            printw("----- ");


                        // VALUE

                        int paramsync = 1; // parameter is synchronized

                        if( fps[fpsindex].parray[pindex].status & FPFLAG_ERROR ) // parameter setting error
                            attron(COLOR_PAIR(4));

                        if(fps[fpsindex].parray[pindex].type & FPTYPE_UNDEF)
                            printw("  %s", "-undef-");



                        if(fps[fpsindex].parray[pindex].type & FPTYPE_INT64)
                        {
                            if( fps[fpsindex].parray[pindex].status & FPFLAG_FEEDBACK ) // Check value feedback if available
                                if( ! (fps[fpsindex].parray[pindex].status & FPFLAG_ERROR) )
                                    if( fps[fpsindex].parray[pindex].val.l[0] != fps[fpsindex].parray[pindex].val.l[3] )
                                        paramsync = 0;

                            if(paramsync == 0)
                                attron(COLOR_PAIR(3));

                            printw("  %10d", (int) fps[fpsindex].parray[pindex].val.l[0]);

                            if(paramsync == 0)
                                attroff(COLOR_PAIR(3));
                        }



                        if(fps[fpsindex].parray[pindex].type & FPTYPE_FLOAT64)
                        {
                            if( fps[fpsindex].parray[pindex].status & FPFLAG_FEEDBACK ) // Check value feedback if available
                                if( ! (fps[fpsindex].parray[pindex].status & FPFLAG_ERROR) )
                                {
                                    double absdiff;
                                    double abssum;
                                    double epsrel = 1.0e-6;
                                    double epsabs = 1.0e-10;

                                    absdiff = fabs(fps[fpsindex].parray[pindex].val.f[0] - fps[fpsindex].parray[pindex].val.f[3]);
                                    abssum = fabs(fps[fpsindex].parray[pindex].val.f[0]) + fabs(fps[fpsindex].parray[pindex].val.f[3]);


                                    if( (absdiff < epsrel*abssum) || (absdiff < epsabs) )
                                        paramsync = 1;
                                    else
                                        paramsync = 0;
                                }

                            if(paramsync == 0)
                                attron(COLOR_PAIR(3));

                            printw("  %10f", (float) fps[fpsindex].parray[pindex].val.f[0]);

                            if(paramsync == 0)
                                attroff(COLOR_PAIR(3));
                        }



                        if(fps[fpsindex].parray[pindex].type & FPTYPE_PID)
                        {
                            if( fps[fpsindex].parray[pindex].status & FPFLAG_FEEDBACK ) // Check value feedback if available
                                if( ! (fps[fpsindex].parray[pindex].status & FPFLAG_ERROR) )
                                    if( fps[fpsindex].parray[pindex].val.pid[0] != fps[fpsindex].parray[pindex].val.pid[1] )
                                        paramsync = 0;

                            if(paramsync == 0)
                                attron(COLOR_PAIR(3));

                            printw("  %10d", (float) fps[fpsindex].parray[pindex].val.pid[0]);

                            if(paramsync == 0)
                                attroff(COLOR_PAIR(3));

                            printw("  %10d", (int) fps[fpsindex].parray[pindex].val.pid[0]);
                        }




                        if(fps[fpsindex].parray[pindex].type & FPTYPE_TIMESPEC)
                            printw("  %10s", "-timespec-");


                        if(fps[fpsindex].parray[pindex].type & FPTYPE_FILENAME)
                        {
                            if( fps[fpsindex].parray[pindex].status & FPFLAG_FEEDBACK ) // Check value feedback if available
                                if( ! (fps[fpsindex].parray[pindex].status & FPFLAG_ERROR) )
                                    if( strcmp(fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].parray[pindex].val.string[1] ))
                                        paramsync = 0;

                            if(paramsync == 0)
                                attron(COLOR_PAIR(3));

                            printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                            if(paramsync == 0)
                                attroff(COLOR_PAIR(3));
                        }


                        if(fps[fpsindex].parray[pindex].type & FPTYPE_DIRNAME)
                        {
                            if( fps[fpsindex].parray[pindex].status & FPFLAG_FEEDBACK ) // Check value feedback if available
                                if( ! (fps[fpsindex].parray[pindex].status & FPFLAG_ERROR) )
                                    if( strcmp(fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].parray[pindex].val.string[1] ))
                                        paramsync = 0;

                            if(paramsync == 0)
                                attron(COLOR_PAIR(3));

                            printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                            if(paramsync == 0)
                                attroff(COLOR_PAIR(3));
                        }


                        if(fps[fpsindex].parray[pindex].type & FPTYPE_STREAMNAME)
                        {
                            if( fps[fpsindex].parray[pindex].status & FPFLAG_FEEDBACK ) // Check value feedback if available
                                if( ! (fps[fpsindex].parray[pindex].status & FPFLAG_ERROR) )
                                    if( strcmp(fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].parray[pindex].val.string[1] ))
                                        paramsync = 0;

                            if(paramsync == 0)
                                attron(COLOR_PAIR(3));

                            printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                            if(paramsync == 0)
                                attroff(COLOR_PAIR(3));
                        }


                        if(fps[fpsindex].parray[pindex].type & FPTYPE_STRING)
                        {
                            if( fps[fpsindex].parray[pindex].status & FPFLAG_FEEDBACK ) // Check value feedback if available
                                if( ! (fps[fpsindex].parray[pindex].status & FPFLAG_ERROR) )
                                    if( strcmp(fps[fpsindex].parray[pindex].val.string[0], fps[fpsindex].parray[pindex].val.string[1] ))
                                        paramsync = 0;

                            if(paramsync == 0)
                                attron(COLOR_PAIR(3));

                            printw("  %10s", fps[fpsindex].parray[pindex].val.string[0]);

                            if(paramsync == 0)
                                attroff(COLOR_PAIR(3));
                        }


                        if(fps[fpsindex].parray[pindex].type & FPTYPE_ONOFF)
                        {
                            if ( fps[fpsindex].parray[pindex].status & FPFLAG_ONOFF )
                            {
                                attron(COLOR_PAIR(2));
                                printw("  ON ", fps[fpsindex].parray[pindex].val.string[1]);
                                attroff(COLOR_PAIR(2));
                                printw(" [%15s]", fps[fpsindex].parray[pindex].val.string[1]);
                            }
                            else
                            {
                                attron(COLOR_PAIR(1));
                                printw(" OFF ", fps[fpsindex].parray[pindex].val.string[0]);
                                attroff(COLOR_PAIR(1));
                                printw(" [%15s]", fps[fpsindex].parray[pindex].val.string[0]);
                            }
                        }


                        if( fps[fpsindex].parray[pindex].status & FPFLAG_ERROR ) // parameter setting error
                            attroff(COLOR_PAIR(4));

                        printw("    %s", fps[fpsindex].parray[pindex].description);



                        if(i == iSelected[currentlevel])
                            attroff(A_BOLD);


                        pcnt++;

                    }
                }



                icnt++;

                i1++;

            }



            printw("\n");
        }

        NBindex = icnt;

        if(iSelected[currentlevel] > NBindex-1)
            iSelected[currentlevel] = NBindex-1;

        printw("\n");
        printw("%d parameters\n", pcnt);
        printw("\n");


        printw("------------- FUNCTION PARAMETER STRUCTURE   %s\n", fps[fpsindexSelected].md->name);
        if( fps[fpsindexSelected].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK )
        {
            attron(COLOR_PAIR(2));
            printw("[%ld] PARAMETERS OK - RUN function good to go\n", fps[fpsindexSelected].md->msgcnt);
            attroff(COLOR_PAIR(2));
        }
        else
        {
            int msgi;

            attron(COLOR_PAIR(4));
            printw("[%ld] %d PARAMETER SETTINGS ERROR(s) :\n", fps[fpsindexSelected].md->msgcnt, fps[fpsindexSelected].md->errcnt);
            attroff(COLOR_PAIR(4));


            attron(COLOR_PAIR(8));

            for(msgi=0; msgi < fps->md->msgcnt; msgi++)
                printw("%-50s %s\n", fps->parray[fps->md->msgpindex[msgi]].keywordfull, fps->md->message[msgi]);

            attroff(COLOR_PAIR(8));
        }




        refresh();



        loopcnt++;

        if( (data.signal_TERM == 1) || (data.signal_INT == 1) || (data.signal_ABRT == 1) || (data.signal_BUS == 1) || (data.signal_SEGV == 1) || (data.signal_HUP == 1) || (data.signal_PIPE == 1))
            loopOK = 0;
    }
    endwin();

    for(fpsindex=0; fpsindex<NBfps; fpsindex++)
    {
        function_parameter_struct_disconnect(&fps[fpsindex]);
    }





    return RETURN_SUCESS;
}




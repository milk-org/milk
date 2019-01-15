
/**
 * @file function_parameters.c
 * @brief Tools to help expose and control function parameters
 * 
 * 
 * 
 * # Use Template
 * 
 * @code
 * 
 * #include 
 * 
 * 	// allocate and initialize function parameter array
 * NBparam = FUNCTION_PARAMETER_NBPARAM_DEFAULT;
 * FUNCTION_PARAMETER *funcparam = (FUNCTION_PARAMETER *)malloc(sizeof(FUNCTION_PARAMETER)*NBparam);
 * function_parameter_initarray(funcparam, NBparam);
 * 
 * // add parameter (integer)
 * 
 * 
 * @endcode
 * 
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




int function_parameter_struct_create(
    int NBparam,
    char *name
)
{
    int index;
    char *mapv;
    FUNCTION_PARAMETER_STRUCT  funcparamstruct;
    
  //  FUNCTION_PARAMETER_STRUCT_MD *funcparammd;
  //  FUNCTION_PARAMETER *funcparamarray;
    
    char SM_fname[200];
    size_t sharedsize = 0; // shared memory size in bytes
	int SM_fd; // shared memory file descriptor

    snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", SHAREDMEMDIR, name);
    remove(SM_fname);

	printf("Creating file %s\n", SM_fname);
	fflush(stdout);

    sharedsize = sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    sharedsize += sizeof(FUNCTION_PARAMETER)*NBparam;
    
    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (SM_fd == -1) {
        perror("Error opening file for writing");
        exit(0);
    }

    int result;
    result = lseek(SM_fd, sharedsize-1, SEEK_SET);
    if (result == -1) {
        close(SM_fd);
        printf("ERROR [%s %s %d]: Error calling lseek() to 'stretch' the file\n", __FILE__, __func__, __LINE__);
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if (result != 1) {
        close(SM_fd);
        perror("Error writing last byte of the file");
        exit(0);
    }

    funcparamstruct.md = (FUNCTION_PARAMETER_STRUCT_MD*) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (funcparamstruct.md == MAP_FAILED) {
        close(SM_fd);
        perror("Error mmapping the file");
        exit(0);
    }
	//funcparamstruct->md = funcparammd;

	mapv = (char*) funcparamstruct.md;
	mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
	funcparamstruct.parray = (FUNCTION_PARAMETER*) mapv;
	


    printf("shared memory space = %ld bytes\n", sharedsize); //TEST


	funcparamstruct.md->NBparam = NBparam;
	
    for(index=0; index<NBparam; index++)
    {
        funcparamstruct.parray[index].status = 0; // not active
        funcparamstruct.parray[index].cnt0 = 0;   // update counter
    }
    
    sprintf(funcparamstruct.md->name, name);

    funcparamstruct.md->signal = (uint64_t) FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
	funcparamstruct.md->confwaitus = (uint64_t) 1000; // 1 kHz default

	munmap(funcparamstruct.md, sharedsize);
    

    return 0;
}







long function_parameter_struct_connect(
	char *name,
	FUNCTION_PARAMETER_STRUCT *fps
	)
{
	char SM_fname[200];
	int SM_fd; // shared memory file descriptor
	int NBparam;
    char *mapv;
	
	snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", SHAREDMEMDIR, name);
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
        exit(0);
    }

	mapv = (char*) fps->md;
	mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
	fps->parray = (FUNCTION_PARAMETER*) mapv;

//	NBparam = (int) (file_stat.st_size / sizeof(FUNCTION_PARAMETER));
	NBparam = fps->md->NBparam;
    printf("Connected to %s, %d entries\n", SM_fname, NBparam);
    fflush(stdout);
	
	
	function_parameter_printlist(fps->parray, NBparam);
	
	return(NBparam);
}



int function_parameter_struct_disconnect(FUNCTION_PARAMETER_STRUCT *funcparamstruct, int NBparam)
{
	
	munmap(funcparamstruct, sizeof(FUNCTION_PARAMETER_STRUCT_MD)+sizeof(FUNCTION_PARAMETER)*NBparam);
	
	return(0);
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
        if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_ACTIVE)
        {
            int kl;

            printf("Parameter %4d : %s\n", pindex, funcparamarray[pindex].keywordfull);
            /*for(kl=0; kl< funcparamarray[pindex].keywordlevel; kl++)
            	printf("  %s", funcparamarray[pindex].keyword[kl]);
            printf("\n");*/
            printf("    %s\n", funcparamarray[pindex].description);

            // STATUS FLAGS
            printf("    STATUS FLAGS (0x%02hhx) :", (int) funcparamarray[pindex].status);
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_VISIBLE)
                printf(" VISIBLE");
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_WRITECONF)
                printf(" WRITECONF");
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_WRITERUN)
                printf(" WRITERUN");
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_LOG)
                printf(" LOG");
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_SAVEONCHANGE)
                printf(" SAVEONCHANGE");
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_SAVEONCLOSE)
                printf(" SAVEONCLOSE");
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_MINLIMIT)
                printf(" MINLIMIT");
            if(funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_MAXLIMIT)
                printf(" MAXLIMIT");
            printf("\n");

            // DATA TYPE
            //			printf("    TYPE : 0x%02hhx\n", (int) funcparamarray[pindex].type);
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_UNDEF)
                printf("    TYPE = UNDEF\n");
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_INT64)
            {
                printf("    TYPE  = INT64\n");
                printf("    VALUE = %ld\n", (long) funcparamarray[pindex].val.l[0]);
            }
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_FLOAT64)
                printf("    TYPE = FLOAT64\n");
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_PID)
                printf("    TYPE = PID\n");
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_TIMESPEC)
                printf("    TYPE = TIMESPEC\n");
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_FILENAME)
                printf("    TYPE = FILENAME\n");
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_DIRNAME)
                printf("    TYPE = DIRNAME\n");
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_STREAMNAME)
                printf("    TYPE = STREAMNAME\n");
            if(funcparamarray[pindex].type & FUNCTION_PARAMETER_TYPE_STRING)
                printf("    TYPE = STRING\n");

            pcnt ++;
        }
    }
    printf("\n");
    printf("%d parameters\n", pcnt);
    printf("\n");

    return 0;
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
    FUNCTION_PARAMETER  *funcparamarray,
    char                *keywordstring,
    char                *descriptionstring,
    uint64_t             type,
    int                  NBparam,
    void *               valueptr
)
{
    int pindex = 0;
    char *pch;
    char tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];



    // scan for existing keyword
    int scanOK = 0;
    int pindexscan;
    for(pindexscan=0; pindexscan<NBparam; pindexscan++)
    {
        if(strcmp(keywordstring,funcparamarray[pindexscan].keywordfull)==0)
        {
            pindex = pindexscan;
            scanOK = 1;
        }
    }

    if(scanOK==0) // not found
    {
        // scan for first available entry
        pindex = 0;
        while((funcparamarray[pindex].status & FUNCTION_PARAMETER_STATUS_ACTIVE)&&(pindex<NBparam))
            pindex++;

        if(pindex == NBparam)
        {
            printf("ERROR [%s line %d]: NBparam limit reached\n", __FILE__, __LINE__);
            fflush(stdout);
            exit(0);
        }
    }
    else
    {
        printf("Found matching keyword: applying values to existing entry\n");
    }


    funcparamarray[pindex].status = FUNCTION_PARAMETER_STATUS_ACTIVE | FUNCTION_PARAMETER_STATUS_VISIBLE | FUNCTION_PARAMETER_STATUS_WRITECONF;


    // break full keyword into keywords
    strncpy(funcparamarray[pindex].keywordfull, keywordstring, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL);
    strncpy(tmpstring, keywordstring, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN*FUNCTION_PARAMETER_KEYWORD_MAXLEVEL);
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
    case FUNCTION_PARAMETER_TYPE_INT64 :
        funcparamarray[pindex].val.l[0] = 0;
        break;
    case FUNCTION_PARAMETER_TYPE_FLOAT64 :
        funcparamarray[pindex].val.f[0] = 0.0;
        break;
    case FUNCTION_PARAMETER_TYPE_PID :
        funcparamarray[pindex].val.pid = 0;
        break;
    case FUNCTION_PARAMETER_TYPE_TIMESPEC :
        funcparamarray[pindex].val.ts.tv_sec = 0;
        funcparamarray[pindex].val.ts.tv_nsec = 0;
        break;
    case FUNCTION_PARAMETER_TYPE_FILENAME :
        sprintf(funcparamarray[pindex].val.string, "NULL");
        break;
    case FUNCTION_PARAMETER_TYPE_DIRNAME :
        sprintf(funcparamarray[pindex].val.string, "NULL");
        break;
    case FUNCTION_PARAMETER_TYPE_STREAMNAME :
        sprintf(funcparamarray[pindex].val.string, "NULL");
        break;
    case FUNCTION_PARAMETER_TYPE_STRING :
        sprintf(funcparamarray[pindex].val.string, "NULL");
        break;
    }

    if(valueptr == NULL)
    {
        // attempt to read value for filesystem
        char fname[200];
        FILE *fp;

        sprintf(fname, "FPCONF/%s", funcparamarray[pindex].keywordfull);
        if ( (fp = fopen(fname, "r")) != NULL)
        {
            switch (funcparamarray[pindex].type) {
            case FUNCTION_PARAMETER_TYPE_INT64 :
                if ( fscanf(fp, "%ld", &funcparamarray[pindex].val.l[0]) == 1)
                    funcparamarray[pindex].cnt0++;
                break;
            case FUNCTION_PARAMETER_TYPE_FLOAT64 :
                if ( fscanf(fp, "%lf", &funcparamarray[pindex].val.f[0]) == 1)
                    funcparamarray[pindex].cnt0++;
                break;
            case FUNCTION_PARAMETER_TYPE_PID :
                if ( fscanf(fp, "%d", &funcparamarray[pindex].val.pid) == 1)
                    funcparamarray[pindex].cnt0++;
                break;
            case FUNCTION_PARAMETER_TYPE_TIMESPEC :
                if ( fscanf(fp, "%ld %ld", &funcparamarray[pindex].val.ts.tv_sec, &funcparamarray[pindex].val.ts.tv_nsec) == 2)
                    funcparamarray[pindex].cnt0++;
                break;
            case FUNCTION_PARAMETER_TYPE_FILENAME :
                if ( fscanf(fp, "%s", funcparamarray[pindex].val.string) == 1)
                    funcparamarray[pindex].cnt0++;
                break;
            case FUNCTION_PARAMETER_TYPE_DIRNAME :
                if ( fscanf(fp, "%s", funcparamarray[pindex].val.string) == 1)
                    funcparamarray[pindex].cnt0++;
                break;
            case FUNCTION_PARAMETER_TYPE_STREAMNAME :
                if ( fscanf(fp, "%s", funcparamarray[pindex].val.string) == 1)
                    funcparamarray[pindex].cnt0++;
                break;
            case FUNCTION_PARAMETER_TYPE_STRING :
                if ( fscanf(fp, "%s", funcparamarray[pindex].val.string) == 1)
                    funcparamarray[pindex].cnt0++;
                break;
            }
            fclose(fp);
        }
    }
    else
    {
        int64_t *valueptr_INT64;
        double *valueptr_FLOAT64;

        switch (funcparamarray[pindex].type) {
        case FUNCTION_PARAMETER_TYPE_INT64 :
            valueptr_INT64 = (int64_t *) valueptr;
            funcparamarray[pindex].val.l[0] = *valueptr_INT64;
            funcparamarray[pindex].cnt0++;
            break;
        case FUNCTION_PARAMETER_TYPE_FLOAT64 :
            valueptr_FLOAT64 = (double *) valueptr;
            funcparamarray[pindex].val.f[0] = *valueptr_FLOAT64;
            funcparamarray[pindex].cnt0++;
            break;
        }

    }
    return pindex;
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

   // nonl();


    init_color(COLOR_GREEN, 700, 1000, 700);
    //init_color(COLOR_YELLOW, 1000, 1000, 1000);

    start_color();



    //  colored background
    init_pair(1, COLOR_BLACK, COLOR_WHITE);
    init_pair(2, COLOR_BLACK, COLOR_GREEN);
    init_pair(3, COLOR_BLACK, COLOR_YELLOW);
    init_pair(4, COLOR_WHITE, COLOR_RED);
    init_pair(5, COLOR_WHITE, COLOR_BLUE); // DIRECTORY
	init_pair(10, COLOR_BLACK, COLOR_CYAN);

    init_pair(6, COLOR_GREEN, COLOR_BLACK);
    init_pair(7, COLOR_YELLOW, COLOR_BLACK);
    init_pair(8, COLOR_RED, COLOR_BLACK);
    init_pair(9, COLOR_BLACK, COLOR_RED);



    return 0;
}






int functionparameter_UserInputSetParamValue(FUNCTION_PARAMETER_STRUCT *fpsentry)
{
	int inputOK;
	int strlenmax = 20;
	char buff[100];
	
	printf("name : %s\n", fpsentry->md->name);
	
	
	/*
    int selected = 0;
    long i;
    char *p;

    printf("%d entries in list:\n", NBelem);
    fflush(stdout);
    for(i=0; i<NBelem; i++)
    {
        printf("   %3ld   : %16s   %s\n", i, StringList[i].name, StringList[i].description);
        fflush(stdout);
    }

*/
    inputOK = 0;
	fflush(stdout);
    
    while(inputOK == 0)
    {
        printf ("\nEnter value: ");
        fflush(stdout);

        int stringindex = 0;
        char c;
        while( ((c = getchar()) != 10) && ((c = getchar()) != 13) && (stringindex<strlenmax-1) )
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
                //printf("[%d]", (int) c);
                stringindex++;
            }
            if(stringindex<0)
                stringindex = 0;
        }
        buff[stringindex] = '\0';
        printf("VALUE ENTERED\n");

	
/*
        selected = strtol(buff, &p, strlenmax);

        if((selected<0)||(selected>NBelem-1))
        {
            printf("\nError: number not valid. Must be >= 0 and < %d\n", NBelem);
            inputOK = 0;
        }
        else*/
            inputOK = 1;
    }

    printf("Entered : %s\n", buff);
	sleep(5);

    return 0;
}







/**
 * ## Purpose
 *
 * Automatically build simple ASCII GUI from function parameter structure (fps) name mask
 *
 *
 *
 */
int_fast8_t functionparameter_CTRLscreen(char *fpsnamemask)
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


    d = opendir("/tmp/");
    if(d)
    {
        fpsindex = 0;
        pindex = 0;
        while(((dir = readdir(d)) != NULL))
        {
            char *pch = strstr(dir->d_name, ".fps.shm");

            int matchOK = 0;
            // name filtering
            if(strncmp(dir->d_name, fpsnamemask, strlen(fpsnamemask)) == 0)
                matchOK = 1;


            if((pch) && (matchOK == 1))
            {
                // is file sym link ?
                struct stat buf;
                int retv;
                char fullname[200];

                sprintf(fullname, "/tmp/%s", dir->d_name);
                retv = lstat (fullname, &buf);
                if (retv == -1 ) {
                    endwin();
                    printf("File \"%s\"", dir->d_name);
                    perror("Error running lstat on file ");
                    exit(0);
                }

                if (S_ISLNK(buf.st_mode)) // resolve link name
                {
                    char fullname[200];
                    char linknamefull[200];
                    char linkname[200];
                    int nchar;

                    fps_symlink[fpsindex] = 1;
                    sprintf(fullname, "/tmp/%s", dir->d_name);
                    readlink (fullname, linknamefull, 200-1);

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
                    if(fps[fpsindex].parray[i].status & FUNCTION_PARAMETER_STATUS_ACTIVE)  // if entry is active
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
                                    strcpy(keywnode[kwnindex].keyword[l], fps[fpsindex].parray[i].keyword[l]);
                                    printf(" %s", keywnode[kwnindex].keyword[l]);
                                }
                                printf("   %d %d\n", keywnode[kwnindex].keywordlevel, fps[fpsindex].parray[i].keywordlevel);

                                if(keywnode[kwnindex].keywordlevel == fps[fpsindex].parray[i].keywordlevel)
                                {
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
        printf("ERROR: missing /tmp/ directory\n");
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







    streamCTRL_CatchSignals();
    // INITIALIZE ncurses
    initncurses();
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

        long icnt = 0;


        usleep(10000); // 100 Hz display
        int ch = getch();

        switch (ch)
        {
        case 'x':     // Exit control screen
            loopOK=0;
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


        case KEY_RIGHT:
            if(keywnode[keywnode[currentnode].child[iSelected[currentlevel]]].leaf == 0)
            {
				//iSelected = 0;
                currentnode = keywnode[currentnode].child[iSelected[currentlevel]];
            }
            break;

        case KEY_LEFT:
            if(currentnode != 0) // ROOT has no parent
                currentnode = keywnode[currentnode].parent_index;
            break;


		case 's': // set value (if allowed)
			endwin();
                if(system("clear") != 0) // clear screen
                    printERROR(__FILE__,__func__,__LINE__, "system() returns non-zero value");
				functionparameter_UserInputSetParamValue(&fps[keywnode[iSelected[currentlevel]].fpsindex]);
                initncurses();
			break;
		

        case 'K': // kill conf process
			fps[keywnode[iSelected[currentlevel]].fpsindex].md->signal = 0;
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



        for(i=0; i<imax; i++)
        {
			
			for(l=0;l<currentlevel;l++)
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
					
					printw("%-8s ", keywnode[keywnode[nodechain[l]].child[i]].keyword[l]);
					
					if(keywnode[ii].leaf == 0) // directory
						attroff(COLOR_PAIR(5));
					
					if(snode == 1)
						attroff(A_REVERSE);					



				}
				else
					printw("          ");
			}

			
			
            if(i<keywnode[currentnode].NBchild)
            {
				
                int ii;

                ii = keywnode[currentnode].child[i];

					attron(A_REVERSE);
					printw(" ");
					attroff(A_REVERSE);	


                
	
					
                if(keywnode[ii].leaf == 0) // Directory
                { 	
				if(i == iSelected[currentlevel])
                    attron(A_REVERSE);

					attron(COLOR_PAIR(5));
                    l = keywnode[ii].keywordlevel;
                    printw("%s", keywnode[ii].keyword[l-1]);
                    attroff(COLOR_PAIR(5));
                    
                    if(i == iSelected[currentlevel])
						attroff(A_REVERSE);
                    
                }
                else // parameter
                {
                    fpsindex = keywnode[ii].fpsindex;
                    pindex = keywnode[ii].pindex;


                    if(fps[fpsindex].parray[pindex].status & FUNCTION_PARAMETER_STATUS_ACTIVE)
                    {
                        int kl;
                        
                         if(i == iSelected[currentlevel])
                         {
							attron(COLOR_PAIR(10)|A_BOLD);
						}
						
						
						l = keywnode[ii].keywordlevel;
                        printw("%-16s", fps[fpsindex].parray[pindex].keyword[l-1]);
                        
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
                        
                        if(fps[fpsindex].parray[pindex].cnt0 == 0) // has not been initialized
							attron(COLOR_PAIR(4));

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_UNDEF)
                            printw("  %s", "-undef-");

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_INT64)
                            printw("  %10d", (int) fps[fpsindex].parray[pindex].val.l[0]);

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_FLOAT64)
                            printw("  %10f", (float) fps[fpsindex].parray[pindex].val.f[0]);

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_PID)
                            printw("  %10d", (int) fps[fpsindex].parray[pindex].val.pid);

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_TIMESPEC)
                            printw("  %10s", "-timespec-");

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_FILENAME)
                            printw("  %10s", fps[fpsindex].parray[pindex].val.string);

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_DIRNAME)
                            printw("  %10s", fps[fpsindex].parray[pindex].val.string);

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_STREAMNAME)
                            printw("  %10s", fps[fpsindex].parray[pindex].val.string);

                        if(fps[fpsindex].parray[pindex].type & FUNCTION_PARAMETER_TYPE_STRING)
                            printw("  %10s", fps[fpsindex].parray[pindex].val.string);

                        if(fps[fpsindex].parray[pindex].cnt0 == 0) // has not been initialized
							attroff(COLOR_PAIR(4));

                        printw("    %s", fps[fpsindex].parray[pindex].description);



                    if(i == iSelected[currentlevel])
						attroff(A_BOLD);


                        pcnt++;
                        
                    }
                }

                

                icnt++;
                
                
            }
            printw("\n");
        }

        NBindex = icnt;

        printw("\n");
        printw("%d parameters\n", pcnt);
        printw("\n");






        refresh();


        loopcnt++;

        if( (data.signal_TERM == 1) || (data.signal_INT == 1) || (data.signal_ABRT == 1) || (data.signal_BUS == 1) || (data.signal_SEGV == 1) || (data.signal_HUP == 1) || (data.signal_PIPE == 1))
            loopOK = 0;

    }
    endwin();

    for(fpsindex=0; fpsindex<NBfps; fpsindex++)
    {
        function_parameter_struct_disconnect(&fps[fpsindex], fps[fpsindex].md->NBparam);
    }


    return 0;
}




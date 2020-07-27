/**
 * @file    fps_create.c
 * @brief   create function parameter structure
 */

#include <sys/stat.h>
#include <fcntl.h> // for open
#include <unistd.h> // for close
#include <sys/mman.h>


#include "CommandLineInterface/CLIcore.h"





// ==========================================
// Forward declaration(s)
// ==========================================

errno_t fps_create();



// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t fps_create__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_LONG)
            + CLI_checkarg_noerrmsg(2, CLIARG_STR)
            == 0)
    {
        function_parameter_struct_create(
            data.cmdargtoken[1].val.numl,
            data.cmdargtoken[2].val.string
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

errno_t fps_create_addCLIcmd()
{

    RegisterCLIcommand(
        "fpscreate",
        __FILE__,
        fps_create__cli,
        "create function parameter structure (FPS)",
        "<NBparam> <name>",
        "fpscreate 100 newfps",
        "errno_t function_parameter_struct_create(int NBparamMAX, const char *name");

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

    if(snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", shmdname, name) < 0)
    {
        PRINT_ERROR("snprintf error");
    }
    remove(SM_fname);

    printf("Creating file %s, holding NBparamMAX = %d\n", SM_fname, NBparamMAX);
    fflush(stdout);

    sharedsize = sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    sharedsize += sizeof(FUNCTION_PARAMETER) * NBparamMAX;

    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if(SM_fd == -1)
    {
        perror("Error opening file for writing");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    fps.SMfd = SM_fd;

    int result;
    result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
    if(result == -1)
    {
        close(SM_fd);
        printf("ERROR [%s %s %d]: Error calling lseek() to 'stretch' the file\n",
               __FILE__, __func__, __LINE__);
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if(result != 1)
    {
        close(SM_fd);
        perror("Error writing last byte of the file");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }

    fps.md = (FUNCTION_PARAMETER_STRUCT_MD *) mmap(0, sharedsize,
             PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if(fps.md == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(0);
    }
    //funcparamstruct->md = funcparammd;

    mapv = (char *) fps.md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps.parray = (FUNCTION_PARAMETER *) mapv;



    printf("shared memory space = %ld bytes\n", sharedsize); //TEST


    fps.md->NBparamMAX = NBparamMAX;

    for(index = 0; index < NBparamMAX; index++)
    {
        fps.parray[index].fpflag = 0; // not active
        fps.parray[index].cnt0 = 0;   // update counter
    }


    strncpy(fps.md->name, name, STRINGMAXLEN_FPS_NAME);
	strncpy(fps.md->callprogname, data.package_name, FPS_CALLPROGNAME_STRMAXLEN);
	strncpy(fps.md->callfuncname, data.cmdargtoken[0].val.string, FPS_CALLFUNCNAME_STRMAXLEN);

    char cwd[FPS_CWD_STRLENMAX];
    if(getcwd(cwd, sizeof(cwd)) != NULL)
    {
        strncpy(fps.md->fpsdirectory, cwd, FPS_CWD_STRLENMAX);
    }
    else
    {
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





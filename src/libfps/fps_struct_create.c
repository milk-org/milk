/**
 * @file    fps_struct_create.c
 * @brief   create function parameter structure
 */

#include <fcntl.h> // for open
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h> // for close
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"
#include "fps_shmdirname.h"

#ifdef MILK_MODULE
#include "CLIcore.h"
#endif




errno_t function_parameter_struct_create(
    int NBparamMAX,
    const char *name
)
{
    char                     *mapv = NULL;
    FUNCTION_PARAMETER_STRUCT fps = {0};

    char   SM_fname[200];
    size_t sharedsize = 0; // shared memory size in bytes
    int    SM_fd;          // shared memory file descriptor

    char shmdname[200];
    function_parameter_struct_shmdirname(shmdname);

    if(snprintf(SM_fname, 200, "%s/%s.fps.shm", shmdname, name) < 0)
    {
        PRINT_ERROR("snprintf error");
    }
    remove(SM_fname);

    printf("DEBUG: [%s:%d] Creating file %s, holding NBparamMAX = %d\n", __FILE__, __LINE__, SM_fname, NBparamMAX);
    fflush(stdout);

    sharedsize = sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    sharedsize += sizeof(FUNCTION_PARAMETER) * NBparamMAX;

    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600);
    if(SM_fd == -1)
    {
        perror("Error opening file for writing");
        exit(0);
    }

    fps.SMfd = SM_fd;

    int result;
    result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
    if(result == -1)
    {
        close(SM_fd);
        fprintf(stderr, "Error calling lseek() to 'stretch' the file\n");
        exit(0);
    }

    result = write(SM_fd, "", 1);
    if(result != 1)
    {
        close(SM_fd);
        perror("Error writing last byte of the file");
        exit(0);
    }

    fps.md = (FUNCTION_PARAMETER_STRUCT_MD *)
             mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if(fps.md == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        exit(0);
    }

    mapv = (char *) fps.md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps.parray = (FUNCTION_PARAMETER *) mapv;

    fps.md->NBparamMAX = NBparamMAX;

    memset(fps.parray, 0, NBparamMAX * sizeof(*fps.parray));

    strncpy(fps.md->name, name, STRINGMAXLEN_FPS_NAME - 1);

    // Use global defaults
    strncpy(fps.md->callprogname, FPS_callprogname, FPS_CALLPROGNAME_STRMAXLEN - 1);

    strncpy(fps.md->callfuncname, FPS_callfuncname, FPS_CALLFUNCNAME_STRMAXLEN - 1);

    {
        char path[512];
        ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
        if (len != -1) {
            path[len] = '\0';
            strncpy(fps.md->execfullpath, path, 511);
        } else {
            strncpy(fps.md->execfullpath, "unknown", 511);
        }
    }

    char cwd[FPS_CWD_STRLENMAX];
    if(getcwd(cwd, sizeof(cwd)) != NULL)
    {
        strncpy(fps.md->workdir, cwd, FPS_CWD_STRLENMAX - 1);
    }
    else
    {
        perror("getcwd() error");
        return 1;
    }

    strncpy(fps.md->sourcefname, "NULL", FPS_SRCDIR_STRLENMAX - 1);
    fps.md->sourceline = 0;

    // set default fpsdatadir
    snprintf(fps.md->datadir, FPS_DIR_STRLENMAX, "fps.%s.datadir", fps.md->name);
    // and create the directory
    mkdir(fps.md->datadir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    // set default fpsconfdir
    snprintf(fps.md->confdir, FPS_DIR_STRLENMAX, "fps.%s.confdir", fps.md->name);
    // and create the directory
    mkdir(fps.md->confdir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    // Get keywordarray from environment variable
    char *kwarray = getenv("FPS_KEYWORDARRAY");
    if(kwarray)
    {
        strncpy(fps.md->keywordarray,
                kwarray,
                FPS_KEYWORDARRAY_STRMAXLEN - 1);
    }
    else
    {
        strncpy(fps.md->keywordarray,
                ":",
                FPS_KEYWORDARRAY_STRMAXLEN - 1);
    }

    // write currently loaded modules to fps
    fps.md->NBmodule = 0;
#ifdef MILK_MODULE
    for(int m = 0; m < data.NBmodule; m++)
    {
        if(data.module[m].type != MODULE_TYPE_UNUSED)
        {
            char *mname = data.module[m].name;
            if(data.module[m].type == MODULE_TYPE_CUSTOMLOAD)
            {
                if(strlen(data.module[m].loadname) > 0)
                {
                    mname = data.module[m].loadname;
                }
            }

            if(strlen(mname) > 0)
            {
                strncpy(fps.md->modulename[fps.md->NBmodule],
                        mname,
                        FPS_MODULE_STRMAXLEN - 1);
                fps.md->NBmodule++;
            }
        }
        if(fps.md->NBmodule >= FPS_MAXNB_MODULE)
        {
            break;
        }
    }
#endif

    fps.md->signal     = (uint64_t) FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
    fps.md->confwaitus = (uint64_t) 1000; // 1 kHz default
    fps.md->msgcnt     = 0;

    // initialize pointers
    fps.cmdset.triggermodeptr = NULL;
    fps.cmdset.procinfo_loopcntMax_ptr = NULL;
    fps.cmdset.triggerdelayptr = NULL;
    fps.cmdset.triggertimeoutptr = NULL;

    munmap(fps.md, sharedsize);

    return 0;
}

errno_t function_parameter_struct_realloc(
    FUNCTION_PARAMETER_STRUCT *fps,
    int NBparamMAX_new
)
{
    char shmdname[200];
    char SM_fname[200];
    function_parameter_struct_shmdirname(shmdname);
    snprintf(SM_fname, 200, "%s/%s.fps.shm", shmdname, fps->md->name);

    size_t sharedsize_old = sizeof(FUNCTION_PARAMETER_STRUCT_MD) + sizeof(FUNCTION_PARAMETER) * fps->md->NBparamMAX;
    size_t sharedsize_new = sizeof(FUNCTION_PARAMETER_STRUCT_MD) + sizeof(FUNCTION_PARAMETER) * NBparamMAX_new;

    // 1. Unmap old
    munmap(fps->md, sharedsize_old);

    // 2. Resize file
    if(truncate(SM_fname, sharedsize_new) == -1)
    {
        perror("Error truncating file for realloc");
        return RETURN_FAILURE;
    }

    // 3. Remap
    fps->md = (FUNCTION_PARAMETER_STRUCT_MD *)
              mmap(0, sharedsize_new, PROT_READ | PROT_WRITE, MAP_SHARED, fps->SMfd, 0);
    if(fps->md == MAP_FAILED)
    {
        perror("Error re-mmapping the file");
        return RETURN_FAILURE;
    }

    char *mapv = (char *) fps->md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps->parray = (FUNCTION_PARAMETER *) mapv;

    // 4. Initialize new part
    memset(&fps->parray[fps->md->NBparamMAX], 0, (NBparamMAX_new - fps->md->NBparamMAX) * sizeof(FUNCTION_PARAMETER));

    fps->md->NBparamMAX = NBparamMAX_new;

    // 5. Update pointers in cmdset (if they were set)
    // These pointers point into parray, which changed location
    if (fps->cmdset.procinfo_loopcntMax_ptr != NULL) {
        int pindex = functionparameter_GetParamIndex(fps, ".procinfo.loopcntMax");
        if(pindex > -1) fps->cmdset.procinfo_loopcntMax_ptr = fps->parray[pindex].val.i64;
    }
    if (fps->cmdset.triggermodeptr != NULL) {
        int pindex = functionparameter_GetParamIndex(fps, ".procinfo.triggermode");
        if(pindex > -1) fps->cmdset.triggermodeptr = fps->parray[pindex].val.i64;
    }
    if (fps->cmdset.triggerdelayptr != NULL) {
        int pindex = functionparameter_GetParamIndex(fps, ".procinfo.triggerdelay");
        if(pindex > -1) fps->cmdset.triggerdelayptr = fps->parray[pindex].val.ts;
    }
    if (fps->cmdset.triggertimeoutptr != NULL) {
        int pindex = functionparameter_GetParamIndex(fps, ".procinfo.triggertimeout");
        if(pindex > -1) fps->cmdset.triggertimeoutptr = fps->parray[pindex].val.ts;
    }

    return RETURN_SUCCESS;
}

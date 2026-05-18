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
#include <errno.h>

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
    errno_t                   rv         = RETURN_FAILURE;
    char                     *mapv       = NULL;
    int                       SM_fd      = -1;
    size_t                    sharedsize = 0;
    FPS                       fps        = {0};
    fps.md = MAP_FAILED;

    char shmdname[200];
    function_parameter_struct_shmdirname(shmdname);

    char SM_fname[200];
    if(snprintf(SM_fname, 200, "%s/%s.fps.shm", shmdname, name) < 0)
    {
        PRINT_ERROR("snprintf error");
    }
    remove(SM_fname);

    if (getenv("FPS_DEBUG"))
        printf("DEBUG: [%s:%d] Creating file %s, "
               "NBparamMAX = %d\n",
               __FILE__, __LINE__,
               SM_fname, NBparamMAX);
    fflush(stdout);

    sharedsize = sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    sharedsize += sizeof(FPS_PARAM) * NBparamMAX;

    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600);
    if(SM_fd == -1)
    {
        PRINT_ERROR("open(%s) failed: %s",
                    SM_fname, strerror(errno));
        goto fail;
    }

    fps.SMfd = SM_fd;

    if(lseek(SM_fd, sharedsize - 1, SEEK_SET) == -1)
    {
        PRINT_ERROR("lseek failed: %s", strerror(errno));
        goto fail;
    }

    if(write(SM_fd, "", 1) != 1)
    {
        PRINT_ERROR("write last byte failed: %s",
                    strerror(errno));
        goto fail;
    }

    fps.md = (FUNCTION_PARAMETER_STRUCT_MD *)
             mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if(fps.md == MAP_FAILED)
    {
        PRINT_ERROR("mmap(%s) failed: %s",
                    SM_fname, strerror(errno));
        goto fail;
    }

    mapv = (char *) fps.md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps.parray = (FPS_PARAM *) mapv;

    fps.md->NBparamMAX = NBparamMAX;

    memset(fps.parray, 0, NBparamMAX * sizeof(*fps.parray));

    strncpy(fps.md->name, name, STRINGMAXLEN_FPS_NAME - 1);

    // Use global defaults
    strncpy(fps.md->callprogname,
        FPS_callprogname,
        FPS_CALLPROGNAME_STRMAXLEN - 1);

    strncpy(fps.md->callfuncname,
        FPS_callfuncname,
        FPS_CALLFUNCNAME_STRMAXLEN - 1);

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
        PRINT_ERROR("getcwd failed: %s", strerror(errno));
        goto fail;
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
    for(int mm = 0; mm < data.NBmodule; mm++)
    {
        if(data.module[mm].type != MODULE_TYPE_UNUSED)
        {
            char *mname = data.module[mm].name;
            if(data.module[mm].type == MODULE_TYPE_CUSTOMLOAD)
            {
                if(strlen(data.module[mm].loadname) > 0)
                {
                    mname = data.module[mm].loadname;
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

    rv = RETURN_SUCCESS;

fail:
    if (fps.md != MAP_FAILED)
    {
        munmap(fps.md, sharedsize);
    }
    if (SM_fd != -1)
    {
        close(SM_fd);
    }
    return rv;
}

errno_t function_parameter_struct_realloc(
    FPS *fps,
    int NBparamMAX_new
)
{
    char shmdname[STRINGMAXLEN_DIRNAME];
    char SM_fname[STRINGMAXLEN_FULLFILENAME];
    function_parameter_struct_shmdirname(shmdname);
    snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", shmdname, fps->md->name);

    size_t sharedsize_old = sizeof(FUNCTION_PARAMETER_STRUCT_MD) + sizeof(FPS_PARAM) * fps->md->NBparamMAX;
    size_t sharedsize_new = sizeof(FUNCTION_PARAMETER_STRUCT_MD) + sizeof(FPS_PARAM) * NBparamMAX_new;

    // 1. Unmap old
    munmap(fps->md, sharedsize_old);

    // 2. Resize file
    if(truncate(SM_fname, sharedsize_new) == -1)
    {
        PRINT_ERROR("Error truncating file for realloc: %s", strerror(errno));
        return RETURN_FAILURE;
    }

    // 3. Remap
    fps->md = (FUNCTION_PARAMETER_STRUCT_MD *)
              mmap(0,
                  sharedsize_new,
                  PROT_READ | PROT_WRITE,
                  MAP_SHARED,
                  fps->SMfd,
                  0);
    if(fps->md == MAP_FAILED)
    {
        PRINT_ERROR("Error re-mmapping the file: %s", strerror(errno));
        return RETURN_FAILURE;
    }

    char *mapv = (char *) fps->md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps->parray = (FPS_PARAM *) mapv;

    // 4. Initialize new part
    memset(&fps->parray[fps->md->NBparamMAX],
        0,
        (NBparamMAX_new - fps->md->NBparamMAX) * sizeof(FPS_PARAM));

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

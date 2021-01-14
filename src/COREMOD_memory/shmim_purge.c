/**
 * @file    shmim_purge.c
 * @brief   purge shared memory stream
 */

#include <sys/stat.h>
#include <fcntl.h> // open
#include <unistd.h> // close
#include <sys/mman.h> // mmap

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "read_shmim.h"




// ==========================================
// forward declaration
// ==========================================

errno_t    shmim_purge(
    const char *strfilter
);



// ==========================================
// command line interface wrapper functions
// ==========================================


static errno_t shmim_purge__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR)
            == 0)
    {

        shmim_purge(
            data.cmdargtoken[1].val.string);

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

errno_t shmim_purge_addCLIcmd()
{

    RegisterCLIcommand(
        "shmimpurge",
        __FILE__, 
        shmim_purge__cli,
        "purge orphan streams",
        "<strfilter>",
        "shmimpurge im_",
        "errno_t shmim_purge(const char *strfilter)");    

    return RETURN_SUCCESS;
}




/** @brief purge orphan share memory streams
 * 
 * 
 */
errno_t    shmim_purge(
    const char *strfilter
)
{
    //printf("PURGING ORPHAN STREAMS (matching %s)\n", strfilter);

    int NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    streaminfo = (STREAMINFO *) malloc(sizeof(STREAMINFO) * NBstreamMAX);

    int NBstream = find_streams(streaminfo, 1, strfilter);

    //printf("%d streams found :\n", NBstream);
    for(int sindex = 0; sindex < NBstream; sindex++)
    {
        //printf(" %3d   %s\n", sindex, streaminfo[sindex].sname);
        imageID ID = image_ID(streaminfo[sindex].sname);
        if(ID == -1)
        {
            ID = read_sharedmem_image(streaminfo[sindex].sname);
        }

        pid_t opid; // owner PID
        opid = data.image[ID].md[0].ownerPID;

        if(opid != 0)
        {
			if(getpgid(opid) >= 0)
			{
				//printf("Keeping stream %s\n", streaminfo[sindex].sname);
			}
			else
			{
				printf("Purging stream %s\n", streaminfo[sindex].sname);
				ImageStreamIO_destroyIm(&data.image[ID]);
			}
		}
    }

    free(streaminfo);

    return RETURN_SUCCESS;
}


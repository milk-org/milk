#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "CommandLineInterface/CLIcore.h"
#include "shmimlog_types.h"



// Local variables pointers
static char *logstreamname;
static char *logcmd;



// List of arguments to function
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR, ".in_sname", "input stream name", "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &logstreamname
    },
    {
        CLIARG_STR, ".logcmd", "log command", "logon",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &logcmd
    }
};


// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "shmimlogcmd",
    "log shared memory stream command\n"
    "logon, logoff",
    CLICMD_FIELDS_NOFPS
};








// set the on field in logshim
// IDname is name of image logged
static errno_t logshim_cmd(
    const char *logshimname,
    const char *cmd
)
{
    LOGSHIM_CONF  *map;
    char           SM_fname[STRINGMAXLEN_FILENAME];
    int            SM_fd;
    struct stat    file_stat;

    // read shared mem
    WRITE_FILENAME(SM_fname, "%s/%s.logshimconf.shm", data.shmdir, logshimname);


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

        if(strcmp(cmd, "logon") == 0)
        {
            printf("Setting logging to ON\n");
            map[0].on = 1;
        }
        else if(strcmp(cmd, "logoff") == 0)
        {
            printf("Setting logging to OFF\n");
            map[0].on = 0;
        }
        else if(strcmp(cmd, "logexit") == 0)
        {
            printf("log exit\n");
            map[0].logexit = 1;
        }
        else if(strcmp(cmd, "stat") == 0)
        {
            printf("LOG   on = %d\n", map[0].on);
            printf("    cnt  = %lld\n", map[0].cnt);
            printf(" filecnt = %lld\n", map[0].filecnt);
            printf("interval = %ld\n", map[0].interval);
            printf("logexit  = %d\n", map[0].logexit);
        }



        if(munmap(map, sizeof(LOGSHIM_CONF)) == -1)
        {
            printf("unmapping %s\n", SM_fname);
            perror("Error un-mmapping the file");
        }
        close(SM_fd);
    }
    return RETURN_SUCCESS;
}



// adding INSERT_STD_PROCINFO statements enable processinfo support
static errno_t compute_function()
{

    logshim_cmd(logstreamname, logcmd);

    return RETURN_SUCCESS;
}



INSERT_STD_CLIfunction

// Register function in CLI
errno_t CLIADDCMD_COREMOD_memory__shmimlogcmd()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}






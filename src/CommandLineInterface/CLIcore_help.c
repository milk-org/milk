#include <malloc.h>
#include <libgen.h>

#include <readline/readline.h>
#include <fitsio.h>

#include "CommandLineInterface/CLIcore.h"



errno_t printInfo()
{
    float f1;
    printf("\n");
    printf("  PID = %d\n", CLIPID);


    printf("--------------- GENERAL ----------------------\n");
    printf("%s  %s\n",  data.package_name, data.package_version);
    printf("IMAGESTRUCT_VERSION %s\n", IMAGESTRUCT_VERSION);
    printf("%s BUILT   %s %s\n", __FILE__, __DATE__, __TIME__);
    printf("\n");
    printf("--------------- SETTINGS ---------------------\n");
    printf("procinfo status = %d\n", data.processinfo);

    if(data.precision == 0)
    {
        printf("Default precision upon startup : float\n");
    }
    if(data.precision == 1)
    {
        printf("Default precision upon startup : double\n");
    }
    printf("sizeof(struct timespec)        = %4ld bit\n",
           sizeof(struct timespec) * 8);
    printf("sizeof(pid_t)                  = %4ld bit\n", sizeof(pid_t) * 8);
    printf("sizeof(short int)              = %4ld bit\n", sizeof(short int) * 8);
    printf("sizeof(int)                    = %4ld bit\n", sizeof(int) * 8);
    printf("sizeof(long)                   = %4ld bit\n", sizeof(long) * 8);
    printf("sizeof(long long)              = %4ld bit\n", sizeof(long long) * 8);
    printf("sizeof(int_fast8_t)            = %4ld bit\n", sizeof(int_fast8_t) * 8);
    printf("sizeof(int_fast16_t)           = %4ld bit\n", sizeof(int_fast16_t) * 8);
    printf("sizeof(int_fast32_t)           = %4ld bit\n", sizeof(int_fast32_t) * 8);
    printf("sizeof(int_fast64_t)           = %4ld bit\n", sizeof(int_fast64_t) * 8);
    printf("sizeof(uint_fast8_t)           = %4ld bit\n", sizeof(uint_fast8_t) * 8);
    printf("sizeof(uint_fast16_t)          = %4ld bit\n",
           sizeof(uint_fast16_t) * 8);
    printf("sizeof(uint_fast32_t)          = %4ld bit\n",
           sizeof(uint_fast32_t) * 8);
    printf("sizeof(uint_fast64_t)          = %4ld bit\n",
           sizeof(uint_fast64_t) * 8);
    printf("sizeof(IMAGE_KEYWORD)          = %4ld bit\n",
           sizeof(IMAGE_KEYWORD) * 8);

    size_t offsetval = 0;
    size_t offsetval0 = 0;

    printf("sizeof(IMAGE_METADATA)         = %4ld bit  = %4zu byte ------------------\n",
           sizeof(IMAGE_METADATA) * 8, sizeof(IMAGE_METADATA));

    offsetval = offsetof(IMAGE_METADATA, version);


    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, name);
    printf("   version                     offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, naxis);
    printf("   name                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, size);
    printf("   naxis                       offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, nelement);
    printf("   size                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, datatype);
    printf("   nelement                    offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, imagetype);
    printf("   datatype                    offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, creationtime);
    printf("   imagetype                   offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);


    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, lastaccesstime);
    printf("   creationtime                offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, atime);
    printf("   lastaccesstime              offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, writetime);
    printf("   atime                       offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, location);
    printf("   writetime                   offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, location);
    printf("   shared                      offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, status);
    printf("   location                    offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, flag);
    printf("   status                      offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, sem);
    printf("   flag                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, sem);
    printf("   logflag                     offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cnt0);
    printf("   sem                         offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cnt1);
    printf("   cnt0                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cnt2);
    printf("   cnt1                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, write);
    printf("   cnt2                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, NBkw);
    printf("   write                       offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval, offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval = offsetof(IMAGE_METADATA, cudaMemHandle);
    printf("   NBkw                        offset = %4zu bit  = %4zu byte     [%4zu byte]\n",
           8 * offsetval0, offsetval0, offsetval - offsetval0);

    offsetval0 = offsetval;
    printf("   cudaMemHandle               offset = %4zu bit  = %4zu byte\n",
           8 * offsetval0, offsetval0);



    printf("sizeof(IMAGE)                  offset = %4zu bit  = %4zu byte ------------------\n",
           sizeof(IMAGE) * 8, sizeof(IMAGE));
    printf("   name                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, name),                      offsetof(IMAGE, name));
           
    printf("   used                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, used),                      offsetof(IMAGE, used));
    
    printf("   shmfd                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, shmfd),                     offsetof(IMAGE, shmfd));
    
    printf("   memsize                     offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, memsize),                   offsetof(IMAGE, memsize));
    
    printf("   semlog                      offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, semlog),                    offsetof(IMAGE, semlog));
    
    printf("   md                          offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, md),                        offsetof(IMAGE, md));
    
    printf("   atimearray                  offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, atimearray),                offsetof(IMAGE, atimearray));
    
    printf("   writetimearray              offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, writetimearray),            offsetof(IMAGE,
                   writetimearray));
    
    printf("   flagarray                   offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, flagarray),                 offsetof(IMAGE, flagarray));
    
    printf("   cntarray                    offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, cntarray),                  offsetof(IMAGE, cntarray));
    
    printf("   array                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, array),                     offsetof(IMAGE, array));
    
    printf("   semptr                      offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, semptr),                    offsetof(IMAGE, semptr));
    
    printf("   kw                          offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, kw),                        offsetof(IMAGE, kw));

    
    printf("sizeof(IMAGE_KEYWORD)          offset = %4zu bit  = %4zu byte ------------------\n",
           sizeof(IMAGE_KEYWORD) * 8, sizeof(IMAGE_KEYWORD));
    
    printf("   name                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, name), offsetof(IMAGE_KEYWORD, name));
    
    printf("   type                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, type), offsetof(IMAGE_KEYWORD, type));
    
    printf("   value                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, value), offsetof(IMAGE_KEYWORD, value));
    
    printf("   comment                     offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, comment), offsetof(IMAGE_KEYWORD, comment));

    
    printf("\n");
    printf("--------------- LIBRARIES --------------------\n");
    printf("READLINE : version %x\n", RL_READLINE_VERSION);
# ifdef _OPENMP
    printf("OPENMP   : Compiled by an OpenMP-compliant implementation.\n");
# endif
    printf("CFITSIO  : version %f\n", fits_get_version(&f1));
    printf("\n");

    printf("--------------- DIRECTORIES ------------------\n");
    printf("CONFIGDIR = %s\n", data.configdir);
    printf("SOURCEDIR = %s\n", data.sourcedir);
    printf("\n");

    printf("--------------- MALLOC INFO ------------------\n");
    malloc_stats();

    printf("\n");

    return RETURN_SUCCESS;
}







errno_t list_commands()
{
    char cmdinfoshort[38];

    printf("----------- LIST OF COMMANDS ---------\n");
    for(unsigned int i = 0; i < data.NBcmd; i++)
    {
        strncpy(cmdinfoshort, data.cmd[i].info, 38);
        printf("   %-16s %-20s %-40s %-30s\n", data.cmd[i].key, data.cmd[i].module,
               cmdinfoshort, data.cmd[i].example);
    }

    return RETURN_SUCCESS;
}



errno_t list_commands_module(
    const char *restrict modulename
)
{
    int mOK = 0;
    char cmdinfoshort[38];

    if(strlen(modulename) > 0)
    {
        for(unsigned int i = 0; i < data.NBcmd; i++)
        {
            char cmpstring[200];
//            sprintf(cmpstring, "%s", basename(data.cmd[i].module));
            sprintf(cmpstring, "%s", data.cmd[i].module);

            if(strcmp(modulename, cmpstring) == 0)
            {
                if(mOK == 0)
                {
                    printf("---- MODULE %s: LIST OF COMMANDS ---------\n", modulename);
                }



                strncpy(cmdinfoshort, data.cmd[i].info, 38);
                printf("   %-16s %-20s %-40s %-30s\n", data.cmd[i].key, cmpstring, cmdinfoshort,
                       data.cmd[i].example);
                mOK = 1;
            }
        }
    }

    if(mOK == 0)
    {
        if(strlen(modulename) > 0)
        {
            printf("---- MODULE %s DOES NOT EXIST OR DOES NOT HAVE COMMANDS ---------\n",
                   modulename);
        }

        for(unsigned int i = 0; i < data.NBcmd; i++)
        {
            char cmpstring[200];
            sprintf(cmpstring, "%s", basename(data.cmd[i].module));

            if(strncmp(modulename, cmpstring, strlen(modulename)) == 0)
            {
                if(mOK == 0)
                {
                    printf("---- MODULES %s* commands  ---------\n", modulename);
                }
                strncpy(cmdinfoshort, data.cmd[i].info, 38);
                printf("   %-16s %-20s %-40s %-30s\n", data.cmd[i].key,
                       data.cmd[i].module, cmdinfoshort, data.cmd[i].example);
                mOK = 1;
            }
        }
    }

    return RETURN_SUCCESS;
}



/**
 * @brief command help\n
 *
 * @param[in] cmdkey Commmand name
 */



errno_t help_command(
    const char *restrict cmdkey
)
{
    int cOK = 0;

    for(unsigned int i = 0; i < data.NBcmd; i++)
    {
        if(!strcmp(cmdkey, data.cmd[i].key))
        {
            printf("\n");
            printf("key        :    %s\n", data.cmd[i].key);
            printf("module     :    %ld %s [ \"%s\" ]\n", data.cmd[i].moduleindex, data.cmd[i].module, data.module[data.cmd[i].moduleindex].shortname);
            printf("module src :    %s\n", data.cmd[i].modulesrc);
            printf("info       :    %s\n", data.cmd[i].info);
            printf("syntax     :    %s\n", data.cmd[i].syntax);
            printf("example    :    %s\n", data.cmd[i].example);
            printf("C call     :    %s\n", data.cmd[i].Ccall);
            printf("\n");
            cOK = 1;
        }
    }
    if(cOK == 0)
    {
        printf("\tCommand \"%s\" does not exist\n", cmdkey);
    }

    return RETURN_SUCCESS;
}


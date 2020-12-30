#include <malloc.h>
#include <libgen.h>

#include <readline/readline.h>
#include <fitsio.h>
#include <regex.h>

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

    int moduleindex = -1;
    for(int m = 0; m < data.NBmodule; m++)
    {
        if(strcmp(modulename, data.module[m].name) == 0)
        {
            moduleindex = m;
        }
    }
    if(moduleindex == -1)
    {
        printf("---- MODULE %s DOES NOT EXIST / NOT LOADED ---------\n", modulename);
    }
    else
    {
        printf("   name         %s\n", data.module[moduleindex].name);
        printf("   type         %d\n", data.module[moduleindex].type);
        printf("   short name   %s\n", data.module[moduleindex].shortname);        
        printf("   package      %s\n", data.module[moduleindex].package);
        printf("   loadname     %s\n", data.module[moduleindex].loadname);
        printf("   sofilename   %s\n", data.module[moduleindex].sofilename);
        printf("   version      %d %d %d\n", data.module[moduleindex].versionmajor,
               data.module[moduleindex].versionminor, data.module[moduleindex].versionpatch);
        printf("   date         %s %s\n", data.module[moduleindex].datestring,
               data.module[moduleindex].timestring);
        printf("   info         %s\n", data.module[moduleindex].info);

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



        if(mOK == 0)
        {
            if(strlen(modulename) > 0)
            {
                printf("---- MODULE %s DOES NOT HAVE COMMANDS ---------\n",
                       modulename);
            }
        }
    }
    /*       for(unsigned int i = 0; i < data.NBcmd; i++)
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
    */
    return RETURN_SUCCESS;
}














/** @brief Construct command line (CLI) arguments help string
 *
 */
int CLIhelp_make_argstring(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    char *outargstring
)
{
    char tmpstr[1000];

    for(int arg = 0; arg < nbarg; arg++)
    {
        char typestring[100] = "?";

        switch(fpscliarg[arg].type)
        {
            case CLIARG_FLOAT:
                strcpy(typestring, "float");
                break;

            case CLIARG_LONG:
                strcpy(typestring, "long");
                break;

            case CLIARG_STR_NOT_IMG:
                strcpy(typestring, "string");
                break;

            case CLIARG_IMG:
                strcpy(typestring, "string");
                break;

            case CLIARG_STR:
                strcpy(typestring, "string");
                break;
        }

        if(arg == 0)
        {
            sprintf(tmpstr, "<%s [%s] ->(%s)>", fpscliarg[arg].descr, typestring,
                    fpscliarg[arg].fpstag);
        }
        else
        {
            char tmpstr1[1000];
            sprintf(tmpstr1, " <%s [%s] ->(%s)>", fpscliarg[arg].descr, typestring,
                    fpscliarg[arg].fpstag);
            strcat(tmpstr, tmpstr1);
        }
    }
    strcpy(outargstring, tmpstr);

    return strlen(outargstring);
}




/** @brief Assemble command line (CLI) example command string
 * 
 */
int CLIhelp_make_cmdexamplestring(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    char *shortname,
	char *outcmdexstring
)
{
    char tmpstr[1000];

    sprintf(tmpstr, "%s", shortname);

    for(int arg = 0; arg < nbarg; arg++)
    {
        char tmpstr1[1000];
        sprintf(tmpstr1, " %s", fpscliarg[arg].example);
        strcat(tmpstr, tmpstr1);
    }
    strcpy(outcmdexstring, tmpstr);

    return strlen(outcmdexstring);
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

    for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {
        if(!strcmp(cmdkey, data.cmd[cmdi].key))
        {
            printf("\n");
            printf("key        :    %s\n", data.cmd[cmdi].key);
            printf("module     :    %ld %s [ \"%s\" ]\n", data.cmd[cmdi].moduleindex,
                   data.cmd[cmdi].module, data.module[data.cmd[cmdi].moduleindex].shortname);
            printf("module src :    %s\n", data.cmd[cmdi].modulesrc);
            printf("info       :    %s\n", data.cmd[cmdi].info);
            printf("syntax     :    %s\n", data.cmd[cmdi].syntax);
            printf("example    :    %s\n", data.cmd[cmdi].example);
            printf("C call     :    %s\n", data.cmd[cmdi].Ccall);

            printf("Function arguments and parameters (%d) :\n", data.cmd[cmdi].nbarg);
            printf("  # CLI#       tagname             Value         description\n");

            int CLIargcnt = 0;
            for(int argi = 0; argi < data.cmd[cmdi].nbarg; argi++)
            {
				int colorcode = 34;
                printf("%3d ", argi);
                if(!(data.cmd[cmdi].argdata[argi].flag & CLICMDARG_FLAG_NOCLI))
                {
                    printf("%3d  ", CLIargcnt);
                    CLIargcnt++;
                }
                else
                {
                    printf(" --  ");
                    colorcode = 33;
                }



                char valuestring[100] = "?";

                switch(data.cmd[cmdi].argdata[argi].type)
                {
                    case CLIARG_FLOAT:
                        sprintf(valuestring, "[FLOAT] %f", data.cmd[cmdi].argdata[argi].val.f);
                        break;

                    case CLIARG_LONG:
                        sprintf(valuestring, "[LONG]  %ld", data.cmd[cmdi].argdata[argi].val.l);
                        break;

                    case CLIARG_STR_NOT_IMG:
                        sprintf(valuestring, "[STRnI] %s", data.cmd[cmdi].argdata[argi].val.s);
                        break;

                    case CLIARG_IMG:
                        sprintf(valuestring, "[IMG]   %s", data.cmd[cmdi].argdata[argi].val.s);
                        break;

                    case CLIARG_STR:
                        sprintf(valuestring, "[STR]   %s", data.cmd[cmdi].argdata[argi].val.s);
                        break;
                }


                printf(" %c[%d;%dm%-16s%c[%dm %-24s %s\n",
                       (char) 27, 1, colorcode,
                       data.cmd[cmdi].argdata[argi].fpstag,
                       (char) 27, 0,
                       valuestring, data.cmd[cmdi].argdata[argi].descr);
            }

            printf("\n");
            cOK = 1;
        }
    }

    int foundsubstring = 0;
    int foundregexmatch = 0;
    if(cOK == 0)
    {
        printf("\tCommand \"%s\" does not exist\n", cmdkey);


        regex_t regex;
        int reti;
        /* Compile regular expression */
        reti = regcomp(&regex, cmdkey, REG_EXTENDED);
        if(reti)
        {
            fprintf(stderr, "Could not compile regex : \"%s\"\n", cmdkey);
            exit(1);
        }
        int maxGroups = 8;
        regmatch_t groupArray[maxGroups];



        for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
        {

            int matchsubstring = 0;
            // look for substring match

            if(strstr(data.cmd[cmdi].key, cmdkey) != NULL)
            {
                foundsubstring = 1;
                matchsubstring = 1;
                printf("\t(substring)  %32s in %24s [%s]\n", data.cmd[cmdi].key,
                       data.cmd[cmdi].module,
                       data.module[data.cmd[cmdi].moduleindex].shortname);
            }

            // Regular expression search
            if(matchsubstring == 0)
            {
                // Regular expression search
                reti = regexec(&regex, data.cmd[cmdi].key, maxGroups, groupArray, 0);
                if(!reti)
                {
                    foundregexmatch = 1;
                    printf("\t( regex   )  %32s in %24s [%s]\n", data.cmd[cmdi].key,
                           data.cmd[cmdi].module,
                           data.module[data.cmd[cmdi].moduleindex].shortname);

                    char *cursor = data.cmd[cmdi].key;
                    unsigned int offset = 0;
                    for(int g = 0; g < maxGroups; g++)
                    {
                        if(groupArray[g].rm_so == (regoff_t)((size_t) -1))
                        {
                            break;    // No more groups
                        }

                        if(g == 0)
                        {
                            offset = groupArray[g].rm_eo;
                        }

                        char cursorCopy[strlen(cursor) + 1];
                        strcpy(cursorCopy, cursor);
                        cursorCopy[groupArray[g].rm_eo] = 0;
                        /*printf("\t    Match Group %u: [%2u-%2u]: %s\n",
                               g, groupArray[g].rm_so, groupArray[g].rm_eo,
                               cursorCopy + groupArray[g].rm_so);*/
                    }
                    cursor += offset;
                }
                else if(reti == REG_NOMATCH)
                {
                    //puts("No match");
                }
                else
                {
                    char msgbuf[100];
                    regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                    fprintf(stderr, "Regex match failed: %s\n", msgbuf);
                    exit(1);
                }
            }
        }

        regfree(&regex);

        if(foundsubstring == 0)
        {
            if(foundregexmatch == 0)
            {
                printf("\tNo substring or regex match to \"%s\"\n", cmdkey);
            }
        }
    }

    return RETURN_SUCCESS;
}




errno_t help()
{

    EXECUTE_SYSTEM_COMMAND("more %s/src/CommandLineInterface/doc/help.txt",
                           data.sourcedir);

    return RETURN_SUCCESS;
}


errno_t helpreadline()
{

    EXECUTE_SYSTEM_COMMAND("more %s/src/CommandLineInterface/doc/helpreadline.md",
                           data.sourcedir);

    return RETURN_SUCCESS;
}


errno_t help_cmd()
{
    if((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) || (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE)
            || (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) || (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        help_command(data.cmdargtoken[1].val.string);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}



errno_t help_module()
{

    if(data.cmdargtoken[1].type == 3)
    {
        list_commands_module(data.cmdargtoken[1].val.string);
    }
    else
    {
        long i;
        printf("\n");
        printf("%2s  %10s %32s %10s %7s    %20s %s\n", "#", "shortname", "Name",
               "Package", "Version", "last compiled",
               "description");
        printf("--------------------------------------------------------------------------------------------------------------\n");
        for(i = 0; i < data.NBmodule; i++)
        {
            printf("%2ld %10s \033[1m%32s\033[0m %10s %2d.%02d.%02d    %11s %8s  %s\n",
                   i,
                   data.module[i].shortname,
                   data.module[i].name,
                   data.module[i].package,
                   data.module[i].versionmajor, data.module[i].versionminor,
                   data.module[i].versionpatch,
                   data.module[i].datestring, data.module[i].timestring,
                   data.module[i].info);
        }
        printf("-------------------------------------------------------------------------------------------------------\n");
        printf("\n");
    }

    return RETURN_SUCCESS;
}



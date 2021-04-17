/**
 * @file    image_keyword.c
 */



#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"

















// ==========================================
// forward declarations
// ==========================================

long image_write_keyword_L(
    const char *IDname,
    const char *kname,
    long value, const
    char       *comment
);

long image_list_keywords(
    const char *IDname
);



// ==========================================
// command line interface wrapper functions
// ==========================================

errno_t image_write_keyword_L__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_STR_NOT_IMG)
            == 0)
    {
        image_write_keyword_L(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




errno_t image_list_keywords__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            == 0)
    {
        image_list_keywords(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}





// ==========================================
// Register CLI commands
// ==========================================

errno_t image_keyword_addCLIcmd()
{
    RegisterCLIcommand(
        "imwritekwL",
        __FILE__,
        image_write_keyword_L__cli,
        "write long type keyword",
        "<imname> <kname> <value [long]> <comment>",
        "imwritekwL im1 kw2 34 my_keyword_comment",
        "long image_write_keyword_L(const char *IDname, const char *kname, long value, const char *comment)");

    RegisterCLIcommand(
        "imlistkw",
        __FILE__,
        image_list_keywords__cli,
        "list image keywords",
        "<imname>",
        "imlistkw im1",
        "long image_list_keywords(const char *IDname)");

    return RETURN_SUCCESS;
}











long image_write_keyword_L(
    const char *IDname,
    const char *kname,
    long        value,
    const char *comment
)
{
    imageID  ID;
    long     kw, NBkw, kw0;

    ID = image_ID(IDname);
    NBkw = data.image[ID].md[0].NBkw;

    kw = 0;
    while((data.image[ID].kw[kw].type != 'N') && (kw < NBkw))
    {
        kw++;
    }
    kw0 = kw;

    if(kw0 == NBkw)
    {
        printf("ERROR: no available keyword entry\n");
        exit(0);
    }
    else
    {
        strcpy(data.image[ID].kw[kw].name, kname);
        data.image[ID].kw[kw].type = 'L';
        data.image[ID].kw[kw].value.numl = value;
        strcpy(data.image[ID].kw[kw].comment, comment);
    }

    return kw0;
}



long image_write_keyword_D(
    const char *IDname,
    const char *kname,
    double      value,
    const char *comment
)
{
    imageID  ID;
    long     kw;
    long     NBkw;
    long     kw0;

    ID = image_ID(IDname);
    NBkw = data.image[ID].md[0].NBkw;

    kw = 0;
    while((data.image[ID].kw[kw].type != 'N') && (kw < NBkw))
    {
        kw++;
    }
    kw0 = kw;

    if(kw0 == NBkw)
    {
        printf("ERROR: no available keyword entry\n");
        exit(0);
    }
    else
    {
        strcpy(data.image[ID].kw[kw].name, kname);
        data.image[ID].kw[kw].type = 'D';
        data.image[ID].kw[kw].value.numf = value;
        strcpy(data.image[ID].kw[kw].comment, comment);
    }

    return kw0;
}



long image_write_keyword_S(
    const char *IDname,
    const char *kname,
    const char *value,
    const char *comment
)
{
    imageID ID;
    long    kw;
    long    NBkw;
    long    kw0;

    ID = image_ID(IDname);
    NBkw = data.image[ID].md[0].NBkw;

    kw = 0;
    while((data.image[ID].kw[kw].type != 'N') && (kw < NBkw))
    {
        kw++;
    }
    kw0 = kw;

    if(kw0 == NBkw)
    {
        printf("ERROR: no available keyword entry\n");
        exit(0);
    }
    else
    {
        strcpy(data.image[ID].kw[kw].name, kname);
        data.image[ID].kw[kw].type = 'D';
        strcpy(data.image[ID].kw[kw].value.valstr, value);
        strcpy(data.image[ID].kw[kw].comment, comment);
    }

    return kw0;
}




imageID image_list_keywords(
    const char *IDname
)
{
    imageID ID;
    long    kw;

    ID = image_ID(IDname);

    for(kw = 0; kw < data.image[ID].md[0].NBkw; kw++)
    {
        if(data.image[ID].kw[kw].type == 'L')
        {
            printf("%18s  %20ld %s\n", data.image[ID].kw[kw].name,
                   data.image[ID].kw[kw].value.numl, data.image[ID].kw[kw].comment);
        }
        if(data.image[ID].kw[kw].type == 'D')
        {
            printf("%18s  %20lf %s\n", data.image[ID].kw[kw].name,
                   data.image[ID].kw[kw].value.numf, data.image[ID].kw[kw].comment);
        }
        if(data.image[ID].kw[kw].type == 'S')
        {
            printf("%18s  %20s %s\n", data.image[ID].kw[kw].name,
                   data.image[ID].kw[kw].value.valstr, data.image[ID].kw[kw].comment);
        }
        //	if(data.image[ID].kw[kw].type=='N')
        //	printf("-------------\n");
    }

    return ID;
}




long image_read_keyword_D(
    const char *IDname,
    const char *kname,
    double     *val
)
{
    variableID  ID;
    long        kw;
    long        kw0;

    ID = image_ID(IDname);
    kw0 = -1;
    for(kw = 0; kw < data.image[ID].md[0].NBkw; kw++)
    {
        if((data.image[ID].kw[kw].type == 'D')
                && (strncmp(kname, data.image[ID].kw[kw].name, strlen(kname)) == 0))
        {
            kw0 = kw;
            *val = data.image[ID].kw[kw].value.numf;
        }
    }

    return kw0;
}



long image_read_keyword_L(
    const char *IDname,
    const char *kname,
    long       *val
)
{
    variableID ID;
    long       kw;
    long       kw0;

    ID = image_ID(IDname);
    kw0 = -1;
    for(kw = 0; kw < data.image[ID].md[0].NBkw; kw++)
    {
        if((data.image[ID].kw[kw].type == 'L')
                && (strncmp(kname, data.image[ID].kw[kw].name, strlen(kname)) == 0))
        {
            kw0 = kw;
            *val = data.image[ID].kw[kw].value.numl;
        }
    }

    return kw0;
}


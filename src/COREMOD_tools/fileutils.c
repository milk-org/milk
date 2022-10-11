/**
 * @file fileutils.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#define SBUFFERSIZE 1000

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t write_float_file(const char *fname, float value);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t write_flot_file_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(2, CLIARG_FLOAT) ==
            0)
    {
        write_float_file(data.cmdargtoken[1].val.string,
                         data.cmdargtoken[2].val.numf);

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

errno_t fileutils_addCLIcmd()
{
    RegisterCLIcommand("writef2file",
                       __FILE__,
                       write_flot_file_cli,
                       "write float to file",
                       "<filename> <float variable>",
                       "writef2file val.txt a",
                       "int write_float_file(const char *fname, float value)");

    return RETURN_SUCCESS;
}

int file_exist(char *filename)
{
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

int create_counter_file(const char *fname, unsigned long NBpts)
{
    unsigned long i;
    FILE         *fp;

    if((fp = fopen(fname, "w")) == NULL)
    {
        PRINT_ERROR("cannot create file \"%s\"", fname);
        abort();
    }

    for(i = 0; i < NBpts; i++)
    {
        fprintf(fp, "%ld %f\n", i, (double)(1.0 * i / NBpts));
    }

    fclose(fp);

    return (0);
}

int read_config_parameter_exists(const char *config_file, const char *keyword)
{
    FILE *fp;
    char  line[1000];
    char  keyw[200];
    int   read;

    read = 0;
    if((fp = fopen(config_file, "r")) == NULL)
    {
        PRINT_ERROR("cannot open file \"%s\"", config_file);
        abort();
    }

    while((fgets(line, 1000, fp) != NULL) && (read == 0))
    {
        sscanf(line, " %20s", keyw);
        if(strcmp(keyw, keyword) == 0)
        {
            read = 1;
        }
    }
    if(read == 0)
    {
        PRINT_WARNING("parameter \"%s\" does not exist in file \"%s\"",
                      keyword,
                      config_file);
    }

    fclose(fp);

    return (read);
}

int read_config_parameter(const char *config_file,
                          const char *keyword,
                          char       *content)
{
    FILE *fp;
    char  line[1000];
    char  keyw[200];
    char  cont[200];
    int   read;

    read = 0;
    if((fp = fopen(config_file, "r")) == NULL)
    {
        PRINT_ERROR("cannot open file \"%s\"", config_file);
        abort();
    }

    strcpy(content, "---");
    while(fgets(line, 1000, fp) != NULL)
    {
        sscanf(line, "%100s %100s", keyw, cont);
        if(strcmp(keyw, keyword) == 0)
        {
            strcpy(content, cont);
            read = 1;
        }
        /*      printf("KEYWORD : \"%s\"   CONTENT : \"%s\"\n",keyw,cont);*/
    }
    if(read == 0)
    {
        PRINT_ERROR("parameter \"%s\" does not exist in file \"%s\"",
                    keyword,
                    config_file);
        sprintf(content, "-");
        //  exit(0);
    }

    fclose(fp);

    return (read);
}

float read_config_parameter_float(const char *config_file, const char *keyword)
{
    float value;
    char  content[SBUFFERSIZE];

    read_config_parameter(config_file, keyword, content);
    //printf("content = \"%s\"\n",content);
    value = atof(content);
    //printf("Value = %g\n",value);

    return (value);
}

long read_config_parameter_long(const char *config_file, const char *keyword)
{
    long value;
    char content[SBUFFERSIZE];

    read_config_parameter(config_file, keyword, content);
    value = atol(content);

    return (value);
}

int read_config_parameter_int(const char *config_file, const char *keyword)
{
    int  value;
    char content[SBUFFERSIZE];

    read_config_parameter(config_file, keyword, content);
    value = atoi(content);

    return (value);
}

long file_number_lines(const char *file_name)
{
    long  cnt;
    int   c;
    FILE *fp;

    if((fp = fopen(file_name, "r")) == NULL)
    {
        PRINT_ERROR("cannot open file \"%s\"", file_name);
        abort();
    }

    cnt = 0;
    while((c = fgetc(fp)) != EOF)
        if(c == '\n')
        {
            cnt++;
        }
    fclose(fp);

    return (cnt);
}

FILE *open_file_w(const char *filename)
{
    FILE *fp;

    if((fp = fopen(filename, "w")) == NULL)
    {
        PRINT_ERROR("cannot create file \"%s\"", filename);
        abort();
    }

    return (fp);
}

FILE *open_file_r(const char *filename)
{
    FILE *fp;

    if((fp = fopen(filename, "r")) == NULL)
    {
        PRINT_ERROR("cannot read file \"%s\"", filename);
        abort();
    }

    return (fp);
}

errno_t write_1D_array(double *array, long nbpoints, const char *filename)
{
    FILE *fp;
    long  ii;

    fp = open_file_w(filename);
    for(ii = 0; ii < nbpoints; ii++)
    {
        fprintf(fp, "%ld\t%f\n", ii, array[ii]);
    }
    fclose(fp);

    return RETURN_SUCCESS;
}

errno_t read_1D_array(double *array, long nbpoints, const char *filename)
{
    FILE *fp;
    long  ii;
    long  tmpl;

    fp = open_file_r(filename);
    for(ii = 0; ii < nbpoints; ii++)
    {
        if(fscanf(fp, "%ld\t%lf\n", &tmpl, &array[ii]) != 2)
        {
            PRINT_ERROR("fscanf error");
            exit(0);
        }
    }
    fclose(fp);

    return RETURN_SUCCESS;
}

int read_int_file(const char *fname)
{
    int   value;
    FILE *fp;

    if((fp = fopen(fname, "r")) == NULL)
    {
        value = 0;
    }
    else
    {
        if(fscanf(fp, "%d", &value) != 1)
        {
            PRINT_ERROR("fscanf error");
            exit(0);
        }
        fclose(fp);
    }

    return (value);
}

errno_t write_int_file(const char *fname, int value)
{
    FILE *fp;

    if((fp = fopen(fname, "w")) == NULL)
    {
        PRINT_ERROR("cannot create file \"%s\"\n", fname);
        abort();
    }

    fprintf(fp, "%d\n", value);
    fclose(fp);

    return RETURN_SUCCESS;
}

errno_t write_float_file(const char *fname, float value)
{
    FILE *fp;
    int   mode = 0; // default, create single file

    if(variable_ID("WRITE2FILE_APPEND") != -1)
    {
        mode = 1;
    }

    if(mode == 0)
    {
        if((fp = fopen(fname, "w")) == NULL)
        {
            PRINT_ERROR("cannot create file \"%s\"\n", fname);
            abort();
        }
        fprintf(fp, "%g\n", value);
        fclose(fp);
    }

    if(mode == 1)
    {
        if((fp = fopen(fname, "a")) == NULL)
        {
            PRINT_ERROR("cannot create file \"%s\"\n", fname);
            abort();
        }
        fprintf(fp, " %g", value);
        fclose(fp);
    }

    return RETURN_SUCCESS;
}

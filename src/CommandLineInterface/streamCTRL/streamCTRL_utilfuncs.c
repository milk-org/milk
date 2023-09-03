#define _GNU_SOURCE
#include <string.h>



#include "CLIcore.h"


/**
 * @brief Returns ID number corresponding to a name
 *
 * @param images   pointer to array of images
 * @param name     input image name to be matched
 * @return imageID
 */
imageID image_ID_from_images(
    IMAGE *images,
    const char * __restrict name
)
{
    imageID i;

    i = 0;
    do
    {
        if(images[i].used == 1)
        {
            if((strncmp(name, images[i].name, strlen(name)) == 0) &&
                    (images[i].name[strlen(name)] == '\0'))
            {
                if(images[i].md != NULL)
                {
                    clock_gettime(CLOCK_MILK, &images[i].md->lastaccesstime);
                }
                return i;
            }
        }
        i++;
    }
    while(i != streamNBID_MAX);

    return -1;
}



/**
 * @brief Returns first available ID in image array
 *
 * @param images     pointer to image array
 * @return imageID
 */
imageID image_get_first_ID_available_from_images(
    IMAGE *images
)
{
    imageID i;

    i = 0;
    do
    {
        if(images[i].used == 0)
        {
            images[i].used = 1;
            return i;
        }
        i++;
    }
    while(i != streamNBID_MAX);
    printf("ERROR: ran out of image IDs - cannot allocate new ID\n");
    printf("NB_MAX_IMAGE should be increased above current value (%d)\n",
           streamNBID_MAX);

    return -1;
}



/**
 * @brief Get the process name by pid
 *
 * @param pid
 * @param pname
 * @return error code
 */
errno_t get_process_name_by_pid(
    const int pid,
    char *pname
)
{
    char *fname = (char *) calloc(STRINGMAXLEN_FULLFILENAME, sizeof(char));

    WRITE_FULLFILENAME(fname, "/proc/%d/cmdline", pid);
    FILE *fp = fopen(fname, "r");
    if(fp)
    {
        size_t size;
        size = fread(pname, sizeof(char), 1024, fp);
        if(size > 0)
        {
            if('\n' == pname[size - 1])
            {
                pname[size - 1] = '\0';
            }
        }
        fclose(fp);
    }

    free(fname);

    return RETURN_SUCCESS;
}



/**
 * @brief Get the maximum PID value from system
 *
 * @return int
 */
int get_PIDmax()
{
    FILE *fp;
    int   PIDmax;
    int   fscanfcnt;

    fp = fopen("/proc/sys/kernel/pid_max", "r");

    fscanfcnt = fscanf(fp, "%d", &PIDmax);
    if(fscanfcnt == EOF)
    {
        if(ferror(fp))
        {
            perror("fscanf");
        }
        else
        {
            fprintf(stderr,
                    "Error: fscanf reached end of file, no matching "
                    "characters, no matching failure\n");
        }
        exit(EXIT_FAILURE);
    }
    else if(fscanfcnt != 1)
    {
        fprintf(stderr,
                "Error: fscanf successfully matched and assigned %i input "
                "items, 1 expected\n",
                fscanfcnt);
        exit(EXIT_FAILURE);
    }

    fclose(fp);

    return PIDmax;
}

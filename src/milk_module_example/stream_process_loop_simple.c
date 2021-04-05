/**
 * @file    stream_process_loop_simple.c
 * @brief   template for simple stream processing loop
 *
 *
 */


#include "CommandLineInterface/CLIcore.h"

// required for create_2Dimage_ID()
#include "COREMOD_memory/COREMOD_memory.h"

// required for timespec_diff()
#include "COREMOD_tools/COREMOD_tools.h"

// required for timespec_diff
#include "CommandLineInterface/timeutils.h"




// ==========================================
// Forward declaration(s)
// ==========================================


errno_t milk_module_example__stream_process_loop_simple(
    char *streamA_name,
    char *streamB_name,
    long loopNBiter,
    int semtrig
);




// ==========================================
// Command line interface wrapper function(s)
// ==========================================



static errno_t milk_module_example__stream_process_loop_simple__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            == 0)
    {
        milk_module_example__stream_process_loop_simple(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl);

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


errno_t stream_process_loop_simple_addCLIcmd()
{

    RegisterCLIcommand(
        "streamloop",
        __FILE__,
        milk_module_example__stream_process_loop_simple__cli,
        "simple stream loop",
        "<streamA> <stramB> <NBiter>",
        "streamloop imA imB 10000 2",
        "milk_module_example__stream_process_loop_simple(char *streamA_name, char *streamB_name, long loopNBiter, int semtrig)");

    return RETURN_SUCCESS;
}









/**
 * @brief simple stream processing
 *
 * Requires streamA and streamB in memory\n
 *
 *
 */
errno_t milk_module_example__stream_process_loop_simple(
    char *streamA_name,
    char *streamB_name,
    long loopNBiter,
    int semtrig
)
{
    // collect image identifiers
    // note: imageID type is long
    imageID streamA_ID = image_ID(streamA_name);
    imageID streamB_ID = image_ID(streamB_name);

    uint64_t NBelem;

    NBelem = data.image[streamA_ID].md[0].size[0] *
             data.image[streamA_ID].md[0].size[0];

    struct timespec *tarray;
    tarray = (struct timespec *) malloc(sizeof(struct timespec) * loopNBiter);


    // check that streams are in memory
    //
    if(streamA_ID == -1)
    {
        PRINT_ERROR("Stream \"%s\" not found in memory: cannot proceed", streamA_name);
        exit(0);
    }

    if(streamB_ID == -1)
    {
        PRINT_ERROR("Stream \"%s\" not found in memory: cannot proceed", streamB_name);
        exit(0);
    }



    // run loop
    // In this simple example, loop waits on streamA to update streamB
    //
    for(long iter = 0; iter < loopNBiter; iter++)
    {
        clock_gettime(CLOCK_REALTIME, &tarray[iter]);


        // Wait for semaphore # semtrig of stream A
        // The call will block until semaphore is > 0, and
        // then decrement it and proceed
        //
        sem_wait(data.image[streamA_ID].semptr[semtrig]);


        // set write flag to one to inform other processes that the stream is being written
        data.image[streamB_ID].md[0].write = 1;

        // processing and computations can be inserted here to update stream B
        memcpy(data.image[streamB_ID].array.F, data.image[streamA_ID].array.F,
               sizeof(float)*NBelem);


        // increment image counter
        data.image[streamB_ID].md[0].cnt0++;
        // post all semaphores in output stream
        COREMOD_MEMORY_image_set_sempost_byID(streamB_ID, -1);
        // set write flag to zero
        data.image[streamB_ID].md[0].write = 0;
    }

    // timing analysis
    struct timespec tdiff;

    // average
    tdiff = timespec_diff(tarray[0], tarray[loopNBiter - 1]);
    float loopFrequencyHz = 1.0 * loopNBiter / (1.0 * tdiff.tv_sec + 1.0e-9 *
                            tdiff.tv_nsec);
    printf("loopfrequency   : %.3f Hz\n", loopFrequencyHz);
    printf("Average latency : %9.3f nanosec\n",
           1.0e9 / loopFrequencyHz / 2.0); // divide by two for sinle process latency


    int file_output_mode = 0;
    if(file_output_mode == 1)
    {
        // detailed analysis, print to text
        float *tdiff_nanosec;
        FILE *fp;

        fp = fopen("timinglog.txt", "w");
        tdiff_nanosec = (float *) malloc(sizeof(float) * (loopNBiter - 1));
        for(long iter = 0; iter < loopNBiter - 1; iter++)
        {
            tdiff = timespec_diff(tarray[iter], tarray[iter + 1]);
            tdiff_nanosec[iter] = 1.0e9 * tdiff.tv_sec + 1.0 * tdiff.tv_nsec;
            fprintf(fp, "%6ld  %6.3f \n", iter, tdiff_nanosec[iter]);
        }
        fclose(fp);
    }

    free(tarray);

    return RETURN_SUCCESS;
}






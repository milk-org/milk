/**
 * @file    image_set_couters.c
 * @brief   SET IMAGE FLAGS / COUNTERS
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t COREMOD_MEMORY_image_set_status(const char *IDname, int status);

errno_t COREMOD_MEMORY_image_set_cnt0(const char *IDname, int cnt0);

errno_t COREMOD_MEMORY_image_set_cnt1(const char *IDname, int cnt1);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t COREMOD_MEMORY_image_set_status__cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_LONG) == 0)
        {
            COREMOD_MEMORY_image_set_status(data.cmdargtoken[1].val.string,
                                            (int) data.cmdargtoken[2].val.numl);
            return CLICMD_SUCCESS;
        }
    else
        {
            return CLICMD_INVALID_ARG;
        }
}

static errno_t COREMOD_MEMORY_image_set_cnt0__cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_LONG) == 0)
        {
            COREMOD_MEMORY_image_set_cnt0(data.cmdargtoken[1].val.string,
                                          (int) data.cmdargtoken[2].val.numl);
            return CLICMD_SUCCESS;
        }
    else
        {
            return CLICMD_INVALID_ARG;
        }
}

static errno_t COREMOD_MEMORY_image_set_cnt1__cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_LONG) == 0)
        {
            COREMOD_MEMORY_image_set_cnt1(data.cmdargtoken[1].val.string,
                                          (int) data.cmdargtoken[2].val.numl);
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

errno_t image_set_counters_addCLIcmd()
{
    RegisterCLIcommand(
        "imsetstatus",
        __FILE__,
        COREMOD_MEMORY_image_set_status__cli,
        "set image status variable",
        "<image> <value [long]>",
        "imsetstatus im1 2",
        "long COREMOD_MEMORY_image_set_status(const char *IDname, int status)");

    RegisterCLIcommand(
        "imsetcnt0",
        __FILE__,
        COREMOD_MEMORY_image_set_cnt0__cli,
        "set image cnt0 variable",
        "<image> <value [long]>",
        "imsetcnt0 im1 2",
        "long COREMOD_MEMORY_image_set_cnt0(const char *IDname, int status)");

    RegisterCLIcommand(
        "imsetcnt1",
        __FILE__,
        COREMOD_MEMORY_image_set_cnt1__cli,
        "set image cnt1 variable",
        "<image> <value [long]>",
        "imsetcnt1 im1 2",
        "long COREMOD_MEMORY_image_set_cnt1(const char *IDname, int status)");

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_image_set_status(const char *IDname, int status)
{
    imageID ID;

    ID                          = image_ID(IDname);
    data.image[ID].md[0].status = status;

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_image_set_cnt0(const char *IDname, int cnt0)
{
    imageID ID;

    ID                        = image_ID(IDname);
    data.image[ID].md[0].cnt0 = cnt0;

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_image_set_cnt1(const char *IDname, int cnt1)
{
    imageID ID;

    ID                        = image_ID(IDname);
    data.image[ID].md[0].cnt1 = cnt1;

    return RETURN_SUCCESS;
}

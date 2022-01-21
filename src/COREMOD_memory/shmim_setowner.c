/**
 * @file    shmim_setowner.c
 * @brief   set stream owner PID
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"

// ==========================================
// forward declaration
// ==========================================

imageID shmim_setowner_creator(const char *name);

imageID shmim_setowner_current(const char *name);

imageID shmim_setowner_init(const char *name);

// ==========================================
// command line interface wrapper functions
// ==========================================

static errno_t shmim_setowner_creator__cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) == 0)
    {

        shmim_setowner_creator(data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t shmim_setowner_current__cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) == 0)
    {

        shmim_setowner_current(data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t shmim_setowner_init__cli()
{
    if (0 + CLI_checkarg(1, CLIARG_IMG) == 0)
    {

        shmim_setowner_init(data.cmdargtoken[1].val.string);

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

errno_t shmim_setowner_addCLIcmd()
{

    RegisterCLIcommand("shmimsetowncreator",
                       __FILE__,
                       shmim_setowner_creator__cli,
                       "set owner to creator PID",
                       "<sname>",
                       "shmimsetowncreator im3",
                       "imageID shmim_setowner_creator(const char *name)");

    RegisterCLIcommand("shmimsetowncurrent",
                       __FILE__,
                       shmim_setowner_current__cli,
                       "set owner to current PID",
                       "<sname>",
                       "shmimsetowncurrent im3",
                       "imageID shmim_setowner_current(const char *name)");

    RegisterCLIcommand("shmimsetowninit",
                       __FILE__,
                       shmim_setowner_init__cli,
                       "set owner to init PID",
                       "<sname>",
                       "shmimsetowninit im3",
                       "imageID shmim_setowner_init(const char *name)");

    return RETURN_SUCCESS;
}

/** @brief set owner to creator */
imageID shmim_setowner_creator(const char *name)
{
    imageID ID;

    ID = image_ID(name);
    if (ID != -1)
    {
        data.image[ID].md[0].ownerPID = data.image[ID].md[0].creatorPID;
    }

    return ID;
}

/** @brief set owner to current PID */
imageID shmim_setowner_current(const char *name)
{
    imageID ID;

    ID = image_ID(name);
    if (ID != -1)
    {
        data.image[ID].md[0].ownerPID = getpid();
    }

    return ID;
}

/** @brief set owner to init process
 *
 * This makes the stream immune to orphan purging
 */
imageID shmim_setowner_init(const char *name)
{
    imageID ID;

    ID = image_ID(name);
    if (ID != -1)
    {
        data.image[ID].md[0].ownerPID = 1;
    }

    return ID;
}

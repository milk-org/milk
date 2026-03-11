/**
 * @file list_variable.h
 * @brief List variables in memory
 */

errno_t
CLIADDCMD_COREMOD_memory__list_variable();

errno_t list_variable_ID();

errno_t list_variable_ID_file(
    const char *fname);

/**
 * @file list_variable.h
 * @brief List variables in memory
 */

errno_t
CLIADDCMD_COREMOD_memory__list_variable();

/** @brief List variables matching optional regex */
errno_t list_variable_ID(
    const char *regexstr);

errno_t list_variable_ID_file(
    const char *fname);

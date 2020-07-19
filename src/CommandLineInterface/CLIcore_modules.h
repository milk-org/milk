/**
 * @file CLIcore_modules.h
 * 
 * @brief Modules functions
 *
 */


#ifndef CLICORE_MODULES_H

#define CLICORE_MODULES_H


errno_t load_sharedobj(const char *restrict libname);

errno_t load_module_shared(const char *restrict modulename);

errno_t load_module_shared_ALL();


#endif

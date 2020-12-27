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

errno_t RegisterModule(
    const char *restrict FileName,
    const char *restrict PackageName,
    const char *restrict InfoString,
    int versionmajor,
    int versionminor,
    int versionpatch
);


uint32_t RegisterCLIcommand(
    const char *restrict CLIkey,
    const char *restrict CLImodulesrc,
    errno_t (*CLIfptr)(),
    const char *restrict CLIinfo,
    const char *restrict CLIsyntax,
    const char *restrict CLIexample,
    const char *restrict CLICcall
);


uint32_t RegisterCLIcmd(
    CLICMDDATA CLIcmddata,
    errno_t (*CLIfptr)()
);

#endif

/**
 * @file logfunc.h
 */

void CORE_logFunctionCall(const int   funclevel,
                          const int   loglevel,
                          const int   logfuncMODE,
                          const char *FileName,
                          const char *FunctionName,
                          const long  line,
                          char       *comments);

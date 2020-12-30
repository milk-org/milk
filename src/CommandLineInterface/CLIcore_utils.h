/**
 * @file    CLIcore_utils.h
 * @brief   Util functions for coding convenience
 * 
 */

#ifndef CLICORE_UTILS_H
#define CLICORE_UTILS_H

#include "CommandLineInterface/IMGID.h"

IMGID makeIMGID(const char * restrict name);

imageID resolveIMGID(IMGID * img);

#endif

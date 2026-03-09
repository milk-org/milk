/**
 * @file fileutils.h
 */

#ifndef _COREMOD_TOOLS_FILEUTILS_H
#define _COREMOD_TOOLS_FILEUTILS_H

errno_t fileutils_addCLIcmd();

int create_counter_file(const char *fname, unsigned long NBpts);

int read_config_parameter_exists(const char *config_file, const char *keyword);

int read_config_parameter(const char *config_file,
                          const char *keyword,
                          char       *content);

float read_config_parameter_float(const char *config_file, const char *keyword);

long read_config_parameter_long(const char *config_file, const char *keyword);

int read_config_parameter_int(const char *config_file, const char *keyword);

long file_number_lines(const char *file_name);

FILE *open_file_w(const char *filename);

FILE *open_file_r(const char *filename);

errno_t write_1D_array(double *array, long nbpoints, const char *filename);

errno_t read_1D_array(double *array, long nbpoints, const char *filename);

int read_int_file(const char *fname);

errno_t write_int_file(const char *fname, int value);

errno_t write_float_file(const char *fname, float value);

#endif

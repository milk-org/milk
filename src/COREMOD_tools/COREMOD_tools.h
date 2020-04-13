#ifndef _TOOLS_H
#define _TOOLS_H


void __attribute__((constructor)) libinit_COREMOD_tools();

/** @brief Log function call to file */
void CORE_logFunctionCall(
    const int funclevel,
    const int loglevel,
    const int logfuncMODE,
    const char *FileName,
    const char *FunctionName,
    const long line,
    char *comments
);



struct timespec timespec_diff(
    struct timespec start,
    struct timespec end
);

double timespec_diff_double(
    struct timespec start,
    struct timespec end
);



int COREMOD_TOOLS_mvProcCPUset(const char *csetname);

int create_counter_file(const char *fname, unsigned long NBpts);

int bubble_sort(double *array, unsigned long count);

void qs_float(float *array, unsigned long left, unsigned long right);
void qs_long(long *array, unsigned long left, unsigned long right);
void qs_double(double *array, unsigned long left, unsigned long right);
void qs_ushort(unsigned short *array, unsigned long left, unsigned long right);

void quick_sort_float(float *array, unsigned long count);
void quick_sort_long(long *array, unsigned long count);
void quick_sort_double(double *array, unsigned long count);
void quick_sort_ushort(unsigned short *array, unsigned long count);

void qs3(double *array, double *array1, double *array2, unsigned long left,
         unsigned long right);

void qs3_double(double *array, double *array1, double *array2,
                unsigned long left, unsigned long right);

void quick_sort3(double *array, double *array1, double *array2,
                 unsigned long count);
void quick_sort3_float(float *array, float *array1, float *array2,
                       unsigned long count);
void quick_sort3_double(double *array, double *array1, double *array2,
                        unsigned long count);

void qs2l(double *array, long *array1, unsigned long left, unsigned long right);

void quick_sort2l(double *array, long *array1, unsigned long count);

void quick_sort2l_double(double *array, long *array1, unsigned long count);
void quick_sort2ul_double(double *array, unsigned long *array1,
                          unsigned long count);

void quick_sort3ll_double(double *array, long *array1, long *array2,
                          unsigned long count);
void quick_sort3ulul_double(double *array, unsigned long *array1,
                            unsigned long *array2, unsigned long count);


errno_t lin_regress(
    double *a,
    double *b,
    double *Xi2,
    double *x,
    double *y,
    double *sig,
    unsigned int nb_points
);


int replace_char(char *content, char cin, char cout);

int read_config_parameter_exists(const char *config_file, const char *keyword);

int read_config_parameter(const char *config_file, const char *keyword,
                          char *content);

float read_config_parameter_float(const char *config_file, const char *keyword);

long read_config_parameter_long(const char *config_file, const char *keyword);

int read_config_parameter_int(const char *config_file, const char *keyword);

long file_number_lines(const char *file_name);

FILE *open_file_w(const char *filename);

FILE *open_file_r(const char *filename);


errno_t write_1D_array(
    double *array,
    long nbpoints,
    const char *filename
);


errno_t read_1D_array(
    double *array,
    long nbpoints,
    const char *filename
);


errno_t tp(
    const char *word
);


int read_int_file(
    const char *fname
);

errno_t write_int_file(
    const char *fname,
    int         value
);


errno_t write_float_file(
    const char *fname,
    float       value
);

errno_t COREMOD_TOOLS_imgdisplay3D(
    const char *IDname,
    long        step
);

imageID COREMOD_TOOLS_statusStat(
    const char *IDstat_name,
    long        indexmax
);

#endif









































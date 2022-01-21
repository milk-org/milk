/**
 * @file quicksort.h
 */

int bubble_sort(double *array, unsigned long count);

void qs_float(float *array, unsigned long left, unsigned long right);
void qs_long(long *array, unsigned long left, unsigned long right);
void qs_double(double *array, unsigned long left, unsigned long right);
void qs_ushort(unsigned short *array, unsigned long left, unsigned long right);

void quick_sort_float(float *array, unsigned long count);
void quick_sort_long(long *array, unsigned long count);
void quick_sort_double(double *array, unsigned long count);
void quick_sort_ushort(unsigned short *array, unsigned long count);

void qs3(double       *array,
         double       *array1,
         double       *array2,
         unsigned long left,
         unsigned long right);

void qs3_double(double       *array,
                double       *array1,
                double       *array2,
                unsigned long left,
                unsigned long right);

void quick_sort3(double       *array,
                 double       *array1,
                 double       *array2,
                 unsigned long count);
void quick_sort3_float(float        *array,
                       float        *array1,
                       float        *array2,
                       unsigned long count);
void quick_sort3_double(double       *array,
                        double       *array1,
                        double       *array2,
                        unsigned long count);

void qs2l(double *array, long *array1, unsigned long left, unsigned long right);

void quick_sort2l(double *array, long *array1, unsigned long count);

void quick_sort2l_double(double *array, long *array1, unsigned long count);
void quick_sort2ul_double(double        *array,
                          unsigned long *array1,
                          unsigned long  count);

void quick_sort3ll_double(double       *array,
                          long         *array1,
                          long         *array2,
                          unsigned long count);
void quick_sort3ulul_double(double        *array,
                            unsigned long *array1,
                            unsigned long *array2,
                            unsigned long  count);

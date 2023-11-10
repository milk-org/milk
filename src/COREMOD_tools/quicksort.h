/**
 * @file quicksort.h
 */

int bubble_sort(
    double * __restrict array,
    unsigned long count
);

void qs_float(
    float * __restrict array,
    unsigned long left,
    unsigned long right
);

void qs_long(
    long * __restrict array,
    unsigned long left,
    unsigned long right
);

void qs_double(
    double * __restrict array,
    unsigned long left,
    unsigned long right
);

void qs_ushort(
    unsigned short * __restrict array,
    unsigned long left,
    unsigned long right
);

void quick_sort_float(
    float * __restrict array,
    unsigned long count
);

void quick_sort_long(
    long * __restrict array,
    unsigned long count
);

void quick_sort_double(
    double * __restrict array,
    unsigned long count
);

void quick_sort_ushort(
    unsigned short * __restrict array,
    unsigned long count
);

void qs2(
    double       * __restrict array,
    double       * __restrict array1,
    unsigned long left,
    unsigned long right
);


void qs3(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long left,
    unsigned long right
);

void qs3_double(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long left,
    unsigned long right
);

void quick_sort2(
    double       * __restrict array,
    double       * __restrict array1,
    unsigned long count
);

void quick_sort3(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long count
);

void quick_sort3_float(
    float        * __restrict array,
    float        * __restrict array1,
    float        * __restrict array2,
    unsigned long count
);

void quick_sort3_double(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long count
);

void qs2l(
    double * __restrict array,
    long   * __restrict array1,
    unsigned long left,
    unsigned long right
);

void quick_sort2l(
    double * __restrict array,
    long   * __restrict array1,
    unsigned long count
);



void quick_sort2l_double(
    double * __restrict array,
    long   * __restrict array1,
    unsigned long count
);


void quick_sort2ul_double(
    double        * __restrict array,
    unsigned long * __restrict array1,
    unsigned long  count
);

void quick_sort3ll_double(
    double       * __restrict array,
    long         * __restrict array1,
    long         * __restrict array2,
    unsigned long count
);

void quick_sort3ulul_double(
    double        * __restrict array,
    unsigned long * __restrict array1,
    unsigned long * __restrict array2,
    unsigned long  count
);

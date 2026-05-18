/**
 * @file    quicksort_multiarray.c
 * @brief   Multi-array quicksort with satellite arrays
 *
 * Quicksort variants that co-permute one or more
 * satellite arrays alongside the key array.  When two
 * key elements swap, the corresponding elements in
 * every satellite array are swapped as well.
 *
 * Naming convention:
 *  - qs<N>[l|ul]_*()   -- recursive Hoare-partition
 *                        quicksort with N satellite
 *                        arrays.  'l' = long, 'ul' =
 *                        unsigned long satellites.
 *  - quick_sort<N>*()  -- public entry points.
 *
 * All sort functions sort in ascending order.
 *
 * @see quicksort.c for single-array sorts.
 */

/**
 * @brief Quicksort with one double satellite array
 *
 * Sorts @p array ascending, co-permuting @p array1.
 *
 * @param array   Key array (double)
 * @param array1  Satellite array (double)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs2(
    double       * __restrict array,
    double       * __restrict array1,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    double                 x, y;
    double                 y1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            y1        = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2(array, array1, left, j);
    }
    if(i < right)
    {
        qs2(array, array1, i, right);
    }
}

/**
 * @brief Quicksort with two double satellite arrays
 *
 * Sorts @p array ascending, co-permuting both
 * @p array1 and @p array2.
 *
 * @param array   Key array (double)
 * @param array1  First satellite array (double)
 * @param array2  Second satellite array (double)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs3(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    double                 x, y;
    double                 y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            y1        = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2        = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3(array, array1, array2, i, right);
    }
}

/**
 * @brief Quicksort with two float satellite arrays
 *
 * Float variant of qs3().
 *
 * @param array   Key array (float)
 * @param array1  First satellite array (float)
 * @param array2  Second satellite array (float)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs3_float(
    float        * __restrict array,
    float        * __restrict array1,
    float        * __restrict array2,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    float                  x, y;
    float                  y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            y1        = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2        = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3_float(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3_float(array, array1, array2, i, right);
    }
}

/**
 * @brief Quicksort with two double satellite arrays
 *
 * Explicit-double variant of qs3().
 *
 * @param array   Key array (double)
 * @param array1  First satellite array (double)
 * @param array2  Second satellite array (double)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs3_double(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    double                 x, y;
    double                 y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            y1        = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2        = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3_double(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3_double(array, array1, array2, i, right);
    }
}

/**
 * @brief Quicksort double key with long satellite
 *
 * Sorts @p array ascending, co-permuting the
 * long satellite array @p array1.
 *
 * @param array   Key array (double)
 * @param array1  Satellite array (long)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs2l(
    double * __restrict array,
    long   * __restrict array1,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    double                 x, y;
    long                   l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            l1        = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2l(array, array1, left, j);
    }
    if(i < right)
    {
        qs2l(array, array1, i, right);
    }
}

/**
 * @brief Quicksort double key with ulong satellite
 *
 * @param array   Key array (double)
 * @param array1  Satellite array (unsigned long)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs2ul(
    double        * __restrict array,
    unsigned long * __restrict array1,
    unsigned long  left,
    unsigned long  right)
{
    unsigned long i, j;
    double                 x, y;
    unsigned long          l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            l1        = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2ul(array, array1, left, j);
    }
    if(i < right)
    {
        qs2ul(array, array1, i, right);
    }
}

/**
 * @brief Quicksort double key with long satellite
 *
 * Explicit-double variant of qs2l().
 *
 * @param array   Key array (double)
 * @param array1  Satellite array (long)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs2l_double(
    double       * __restrict array,
    long         * __restrict array1,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    double                 x, y;
    long                   l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            l1        = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2l_double(array, array1, left, j);
    }
    if(i < right)
    {
        qs2l_double(array, array1, i, right);
    }
}

/**
 * @brief Quicksort double key with ulong satellite
 *
 * Explicit-double variant of qs2ul().
 *
 * @param array   Key array (double)
 * @param array1  Satellite (unsigned long)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs2ul_double(
    double        * __restrict array,
    unsigned long * __restrict array1,
    unsigned long  left,
    unsigned long  right
)
{
    unsigned long i, j;
    double                 x, y;
    unsigned long          l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            l1        = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs2ul_double(array, array1, left, j);
    }
    if(i < right)
    {
        qs2ul_double(array, array1, i, right);
    }
}

/**
 * @brief Quicksort double key with two long satellites
 *
 * @param array   Key array (double)
 * @param array1  First satellite (long)
 * @param array2  Second satellite (long)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs3ll_double(
    double       * __restrict array,
    long         * __restrict array1,
    long         * __restrict array2,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    double                 x, y;
    long                   l1, l2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            l1        = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            l2        = array2[i];
            array2[i] = array2[j];
            array2[j] = l2;

            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3ll_double(array, array1, array2, left, j);
    }
    if(i < right)
    {
        qs3ll_double(array, array1, array2, i, right);
    }
}

/**
 * @brief Quicksort double key with two ulong satellites
 *
 * @param array   Key array (double)
 * @param array1  First satellite (unsigned long)
 * @param array2  Second satellite (unsigned long)
 * @param left    Left index (inclusive)
 * @param right   Right index (inclusive)
 */
void qs3ulul_double(
    double        * __restrict array,
    unsigned long * __restrict array1,
    unsigned long * __restrict array2,
    unsigned long  left,
    unsigned long  right
)
{
    unsigned long i, j;
    double                 x, y;
    unsigned long          l1, l2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while(array[i] < x && i < right)
        {
            i++;
        }
        while(x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if(i <= j)
        {
            y        = array[i];
            array[i] = array[j];
            array[j] = y;

            l1        = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            l2        = array2[i];
            array2[i] = array2[j];
            array2[j] = l2;

            i++;
            if(j > 0)
            {
                j--;
            }
        }
    }
    while(i <= j);

    if(left < j)
    {
        qs3ulul_double(array, array1, array2, left, j);
    }

    if(i < right)
    {
        qs3ulul_double(array, array1, array2, i, right);
    }
}

/* ============================================================
 * Public entry points -- multi-array variants
 *
 * Convert (array, count) to (array, 0, count-1) and
 * delegate to the recursive qs_ variant.
 * ========================================================== */

/**
 * @brief Sort double key + one double satellite
 *
 * @param array   Key array (sorted ascending)
 * @param array1  Satellite co-permuted with key
 * @param count   Number of elements
 */
void quick_sort2(
    double       * __restrict array,
    double       * __restrict array1,
    unsigned long count
)
{
    qs2(array, array1, 0, count - 1);
}

/**
 * @brief Sort double key + two double satellites
 *
 * @param array   Key array (sorted ascending)
 * @param array1  First satellite
 * @param array2  Second satellite
 * @param count   Number of elements
 */
void quick_sort3(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long count
)
{
    qs3(array, array1, array2, 0, count - 1);
}

/**
 * @brief Sort float key + two float satellites
 *
 * @param array   Key array (sorted ascending)
 * @param array1  First satellite
 * @param array2  Second satellite
 * @param count   Number of elements
 */
void quick_sort3_float(
    float        * __restrict array,
    float        * __restrict array1,
    float        * __restrict array2,
    unsigned long count
)
{
    qs3_float(array, array1, array2, 0, count - 1);
}

/**
 * @brief Sort double key + two double satellites
 *
 * @param array   Key array (sorted ascending)
 * @param array1  First satellite
 * @param array2  Second satellite
 * @param count   Number of elements
 */
void quick_sort3_double(
    double       * __restrict array,
    double       * __restrict array1,
    double       * __restrict array2,
    unsigned long count
)
{
    qs3_double(array, array1, array2, 0, count - 1);
}

/**
 * @brief Sort double key + one long satellite
 *
 * @param array   Key array (sorted ascending)
 * @param array1  Satellite (long)
 * @param count   Number of elements
 */
void quick_sort2l(
    double * __restrict array,
    long * __restrict array1,
    unsigned long count
)
{
    qs2l(array, array1, 0, count - 1);
}

/**
 * @brief Sort double key + one ulong satellite
 *
 * @param array   Key array (sorted ascending)
 * @param array1  Satellite (unsigned long)
 * @param count   Number of elements
 */
void quick_sort2ul(
    double * __restrict array,
    unsigned long * __restrict array1,
    unsigned long count
)
{
    qs2ul(array, array1, 0, count - 1);
}

/**
 * @brief Sort double key + one long satellite
 *
 * Explicit-double variant of quick_sort2l().
 *
 * @param array   Key array (sorted ascending)
 * @param array1  Satellite (long)
 * @param count   Number of elements
 */
void quick_sort2l_double(
    double * __restrict array,
    long * __restrict array1,
    unsigned long count
)
{
    qs2l_double(array, array1, 0, count - 1);
}

/**
 * @brief Sort double key + one ulong satellite
 *
 * Explicit-double variant of quick_sort2ul().
 *
 * @param array   Key array (sorted ascending)
 * @param array1  Satellite (unsigned long)
 * @param count   Number of elements
 */
void quick_sort2ul_double(
    double        * __restrict array,
    unsigned long * __restrict array1,
    unsigned long  count
)
{
    qs2ul_double(array, array1, 0, count - 1);
}

/**
 * @brief Sort double key + two long satellites
 *
 * @param array   Key array (sorted ascending)
 * @param array1  First satellite (long)
 * @param array2  Second satellite (long)
 * @param count   Number of elements
 */
void quick_sort3ll_double(
    double       * __restrict array,
    long         * __restrict array1,
    long         * __restrict array2,
    unsigned long count
)
{
    qs3ll_double(array, array1, array2, 0, count - 1);
}

/**
 * @brief Sort double key + two ulong satellites
 *
 * @param array   Key array (sorted ascending)
 * @param array1  First satellite (unsigned long)
 * @param array2  Second satellite (unsigned long)
 * @param count   Number of elements
 */
void quick_sort3ulul_double(
    double        * __restrict array,
    unsigned long * __restrict array1,
    unsigned long * __restrict array2,
    unsigned long  count
)
{
    qs3ulul_double(array, array1, array2, 0, count - 1);
}

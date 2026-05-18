/**
 * @file    quicksort.c
 * @brief   Single-array in-place sorting routines
 *
 * Provides quicksort and bubble sort implementations
 * for a single key array, used by processinfo statistics
 * (median, percentile) and by general-purpose array
 * reordering throughout the framework.
 *
 * Naming convention:
 *  - qs_<type>()       -- recursive Hoare-partition
 *                        quicksort on a single array.
 *  - quick_sort_*()    -- public entry points that
 *                        convert (array, count) to
 *                        (array, 0, count-1) and call
 *                        the recursive qs_ variant.
 *
 * All sort functions sort in ascending order.
 *
 * @see quicksort_multiarray.c for co-sort variants.
 */

/**
 * @brief Sort array in ascending order via bubble sort
 *
 * Simple O(n^2) sort used for very small arrays
 * where quicksort overhead is not justified.
 *
 * @param array  Array of doubles to sort in-place
 * @param count  Number of elements
 * @return 0 on success
 */
int bubble_sort(
    double * __restrict array,
    unsigned long count
)
{
    unsigned long a, b;
    double        t;

    for(a = 1; a < count; a++)
        for(b = count - 1; b >= a; b--)
            if(array[b - 1] > array[b])
            {
                t            = array[b - 1];
                array[b - 1] = array[b];
                array[b]     = t;
            }

    return (0);
}

/**
 * @brief Recursive Hoare-partition quicksort on float
 *
 * Partitions around the midpoint pivot and recurses.
 *
 * @param array  Float array to sort in-place
 * @param left   Left index of partition (inclusive)
 * @param right  Right index of partition (inclusive)
 */
void qs_float(
    float * __restrict array,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    float         x, y;

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
        qs_float(array, left, j);
    }
    if(i < right)
    {
        qs_float(array, i, right);
    }
}

/**
 * @brief Recursive Hoare-partition quicksort on long
 *
 * @param array  Long array to sort in-place
 * @param left   Left index of partition (inclusive)
 * @param right  Right index of partition (inclusive)
 */
void qs_long(
    long * __restrict array,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    long                   x, y;

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
        qs_long(array, left, j);
    }
    if(i < right)
    {
        qs_long(array, i, right);
    }
}

/**
 * @brief Recursive Hoare-partition quicksort on double
 *
 * @param array  Double array to sort in-place
 * @param left   Left index of partition (inclusive)
 * @param right  Right index of partition (inclusive)
 */
void qs_double(
    double * __restrict array,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    double                 x, y;

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
        qs_double(array, left, j);
    }
    if(i < right)
    {
        qs_double(array, i, right);
    }
}

/**
 * @brief Recursive quicksort on unsigned short
 *
 * @param array  Unsigned short array to sort
 * @param left   Left index (inclusive)
 * @param right  Right index (inclusive)
 */
void qs_ushort(
    unsigned short * __restrict array,
    unsigned long left,
    unsigned long right
)
{
    unsigned long i, j;
    unsigned short         x, y;

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
        qs_ushort(array, left, j);
    }
    if(i < right)
    {
        qs_ushort(array, i, right);
    }
}

/* ============================================================
 * Public entry points -- single-array variants
 *
 * Convert (array, count) to (array, 0, count-1) and
 * delegate to the recursive qs_ variant.
 * ========================================================== */

/**
 * @brief Sort float array in ascending order
 *
 * @param array  Float array to sort in-place
 * @param count  Number of elements
 */
void quick_sort_float(
    float * __restrict array,
    unsigned long count
)
{
    qs_float(array, 0, count - 1);
}

/**
 * @brief Sort long array in ascending order
 *
 * @param array  Long array to sort in-place
 * @param count  Number of elements
 */
void quick_sort_long(
    long * __restrict array,
    unsigned long count
)
{
    qs_long(array, 0, count - 1);
}

/**
 * @brief Sort double array in ascending order
 *
 * @param array  Double array to sort in-place
 * @param count  Number of elements
 */
void quick_sort_double(
    double * __restrict array,
    unsigned long count
)
{
    qs_double(array, 0, count - 1);
}

/**
 * @brief Sort unsigned short array ascending
 *
 * @param array  Unsigned short array to sort
 * @param count  Number of elements
 */
void quick_sort_ushort(
    unsigned short * __restrict array,
    unsigned long count
)
{
    qs_ushort(array, 0, count - 1);
}

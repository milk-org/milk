/**
 * @file quicksort.c
 */

int bubble_sort(double *array, unsigned long count)
{
    unsigned long a, b;
    double t;

    for (a = 1; a < count; a++)
        for (b = count - 1; b >= a; b--)
            if (array[b - 1] > array[b])
            {
                t = array[b - 1];
                array[b - 1] = array[b];
                array[b] = t;
            }

    return (0);
}

void qs_float(float *array, unsigned long left, unsigned long right)
{
    unsigned long i, j;
    float x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;
            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs_float(array, left, j);
    }
    if (i < right)
    {
        qs_float(array, i, right);
    }
}

void qs_long(long *array, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    long x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;
            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs_long(array, left, j);
    }
    if (i < right)
    {
        qs_long(array, i, right);
    }
}

void qs_double(double *array, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs_double(array, left, j);
    }
    if (i < right)
    {
        qs_double(array, i, right);
    }
}

void qs_ushort(unsigned short *array, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    unsigned short x, y;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs_ushort(array, left, j);
    }
    if (i < right)
    {
        qs_ushort(array, i, right);
    }
}

void qs3(double *array, double *array1, double *array2, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    double y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            y1 = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2 = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs3(array, array1, array2, left, j);
    }
    if (i < right)
    {
        qs3(array, array1, array2, i, right);
    }
}

void qs3_float(float *array, float *array1, float *array2, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    float x, y;
    float y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            y1 = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2 = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs3_float(array, array1, array2, left, j);
    }
    if (i < right)
    {
        qs3_float(array, array1, array2, i, right);
    }
}

void qs3_double(double *array, double *array1, double *array2, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    double y1, y2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            y1 = array1[i];
            array1[i] = array1[j];
            array1[j] = y1;

            y2 = array2[i];
            array2[i] = array2[j];
            array2[j] = y2;

            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs3_double(array, array1, array2, left, j);
    }
    if (i < right)
    {
        qs3_double(array, array1, array2, i, right);
    }
}

void qs2l(double *array, long *array1, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs2l(array, array1, left, j);
    }
    if (i < right)
    {
        qs2l(array, array1, i, right);
    }
}

void qs2ul(double *array, unsigned long *array1, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    unsigned long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;
            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs2ul(array, array1, left, j);
    }
    if (i < right)
    {
        qs2ul(array, array1, i, right);
    }
}

void qs2l_double(double *array, long *array1, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs2l_double(array, array1, left, j);
    }
    if (i < right)
    {
        qs2l_double(array, array1, i, right);
    }
}

void qs2ul_double(double *array, unsigned long *array1, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    unsigned long l1;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            i++;

            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs2ul_double(array, array1, left, j);
    }
    if (i < right)
    {
        qs2ul_double(array, array1, i, right);
    }
}

void qs3ll_double(double *array, long *array1, long *array2, unsigned long left, unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    long l1, l2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            l2 = array2[i];
            array2[i] = array2[j];
            array2[j] = l2;

            i++;
            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs3ll_double(array, array1, array2, left, j);
    }
    if (i < right)
    {
        qs3ll_double(array, array1, array2, i, right);
    }
}

void qs3ulul_double(double *array, unsigned long *array1, unsigned long *array2, unsigned long left,
                    unsigned long right)
{
    register unsigned long i, j;
    double x, y;
    unsigned long l1, l2;

    i = left;
    j = right;
    x = array[(left + right) / 2];

    do
    {
        while (array[i] < x && i < right)
        {
            i++;
        }
        while (x < array[j] && j > left && j > 0)
        {
            j--;
        }

        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;

            l1 = array1[i];
            array1[i] = array1[j];
            array1[j] = l1;

            l2 = array2[i];
            array2[i] = array2[j];
            array2[j] = l2;

            i++;
            if (j > 0)
            {
                j--;
            }
        }
    } while (i <= j);

    if (left < j)
    {
        qs3ulul_double(array, array1, array2, left, j);
    }

    if (i < right)
    {
        qs3ulul_double(array, array1, array2, i, right);
    }
}

void quick_sort_float(float *array, unsigned long count) { qs_float(array, 0, count - 1); }

void quick_sort_long(long *array, unsigned long count) { qs_long(array, 0, count - 1); }

void quick_sort_double(double *array, unsigned long count) { qs_double(array, 0, count - 1); }

void quick_sort_ushort(unsigned short *array, unsigned long count) { qs_ushort(array, 0, count - 1); }

void quick_sort3(double *array, double *array1, double *array2, unsigned long count)
{
    qs3(array, array1, array2, 0, count - 1);
}

void quick_sort3_float(float *array, float *array1, float *array2, unsigned long count)
{
    qs3_float(array, array1, array2, 0, count - 1);
}

void quick_sort3_double(double *array, double *array1, double *array2, unsigned long count)
{
    qs3_double(array, array1, array2, 0, count - 1);
}

void quick_sort2l(double *array, long *array1, unsigned long count) { qs2l(array, array1, 0, count - 1); }

void quick_sort2ul(double *array, unsigned long *array1, unsigned long count) { qs2ul(array, array1, 0, count - 1); }

void quick_sort2l_double(double *array, long *array1, unsigned long count) { qs2l_double(array, array1, 0, count - 1); }

void quick_sort2ul_double(double *array, unsigned long *array1, unsigned long count)
{
    qs2ul_double(array, array1, 0, count - 1);
}

void quick_sort3ll_double(double *array, long *array1, long *array2, unsigned long count)
{
    qs3ll_double(array, array1, array2, 0, count - 1);
}

void quick_sort3ulul_double(double *array, unsigned long *array1, unsigned long *array2, unsigned long count)
{
    qs3ulul_double(array, array1, array2, 0, count - 1);
}

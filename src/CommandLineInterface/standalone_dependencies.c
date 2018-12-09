
#define _GNU_SOURCE

/* ===============================================================================================
 */
/* ===============================================================================================
 */
/*                                        HEADER FILES */
/* ===============================================================================================
 */
/* ===============================================================================================
 */

#include "standalone_dependencies.h"
#include <ncurses.h>
#include <string.h>
#include <time.h>

static int wrow, wcol;
int C_ERRNO=0;

/* ===============================================================================================
 */
/*                                         DUPLICATED CODE */
/* ===============================================================================================
 */
struct timespec info_time_diff(struct timespec start, struct timespec end) {
  struct timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return temp;
}

int print_header(const char *str, char c) {
  long n;
  long i;

  attron(A_BOLD);
  n = strlen(str);
  for (i = 0; i < (wcol - n) / 2; i++) printw("%c", c);
  printw("%s", str);
  for (i = 0; i < (wcol - n) / 2 - 1; i++) printw("%c", c);
  printw("\n");
  attroff(A_BOLD);

  return (0);
}

void qs2l_double(double *array, long *array1, long left, long right) {
  register long i, j;
  double x, y;
  long l1;

  i = left;
  j = right;
  x = array[(left + right) / 2];

  do {
    while (array[i] < x && i < right) i++;
    while (x < array[j] && j > left) j--;

    if (i <= j) {
      y = array[i];
      array[i] = array[j];
      array[j] = y;

      l1 = array1[i];
      array1[i] = array1[j];
      array1[j] = l1;

      i++;
      j--;
    }
  } while (i <= j);

  if (left < j) qs2l_double(array, array1, left, j);
  if (i < right) qs2l_double(array, array1, i, right);
}

void quick_sort2l_double(double *array, long *array1, long count) {
  qs2l_double(array, array1, 0, count - 1);
}

void qs_long(long *array, long left, long right) {
  register long i, j;
  long x, y;

  i = left;
  j = right;
  x = array[(left + right) / 2];

  do {
    while (array[i] < x && i < right) i++;
    while (x < array[j] && j > left) j--;

    if (i <= j) {
      y = array[i];
      array[i] = array[j];
      array[j] = y;
      i++;
      j--;
    }
  } while (i <= j);

  if (left < j) qs_long(array, left, j);
  if (i < right) qs_long(array, i, right);
}

void quick_sort_long(long *array, long count) { qs_long(array, 0, count - 1); }

/**
 * ## Purpose
 * 
 * Print error string
 * 
 * ## Arguments
 * 
 * @param[in]
 * file		CHAR*
 * 			file name from which error is issued
 * 
 * @param[in]
 * func		CHAR*
 * 			function name from which error is issued
 * 
 * @param[in]
 * line		int
 * 			line number from which error is issued
 * 
 * @param[in]
 * warnmessage		CHAR*
 * 			error message to be printed
 *  
 */ 
int printERROR(const char *file, const char *func, int line, char *errmessage)
{
    fprintf(stderr, "%c[%d;%dm ERROR [ %s:%d: %s ]  %c[%d;m\n", (char) 27, 1, 31, file, line, func, (char) 27, 0);
    if(C_ERRNO != 0)
    {
        char buff[256];
        if( strerror_r( errno, buff, 256 ) == 0 ) {
            fprintf(stderr,"C Error: %s\n", buff );
        }
        else
            fprintf(stderr,"Unknown C Error\n");
    }
    else
        fprintf(stderr,"No C error (errno = 0)\n");

    fprintf(stderr,"%c[%d;%dm %s  %c[%d;m\n", (char) 27, 1, 31, errmessage, (char) 27, 0);

    C_ERRNO = 0;

    return(0);
}

/* ===============================================================================================
 */
/*                                 END OF DUPLICATED CODE */
/* ===============================================================================================
 */

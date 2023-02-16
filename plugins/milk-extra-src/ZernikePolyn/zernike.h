/**
 * @file zernike.h
 */


#ifndef _ZERNIKE_H
#define _ZERNIKE_H

// structure to store Zernike coefficients

typedef struct
{
    long    ZERMAX;

    long   *Zer_n;
    long   *Zer_m;

    // Noll index. Starts at 1 for piston
    long   *Zer_Nollindex;

    // Reverse Noll index, obtained by sorting with Noll index
    // This points to the index in this structure for Noll index
    //
    long   *Zer_reverseNollindex;

    double *R_array;
} ZERNIKE;


#endif

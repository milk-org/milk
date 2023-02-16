/** @file MVM_CPU.c
 */

#include <stdio.h>

void matrixMulCPU(float *cMat, float *wfsVec, float *dmVec, int M, int N)
{
    printf("Conventional mat mult %d %d\n", M, N);
    for(int m = 0; m < M; m++)
    {
        dmVec[m] = 0.0;
        for(int n = 0; n < N; n++)
        {
            int index = m * N + n;
            dmVec[m] += cMat[index] * wfsVec[n];
        }
        //cMat[n*M+m]*wfsVec[n];
    }
    /*
        printf("cMat  : ");
        for(int i = 0; i < 5; i++)
        {
            printf("%f ", cMat[i]);
        }
        printf(" ... ");
        for(int i = N * M - 5; i < N * M; i++)
        {
            printf("%f ", cMat[i]);
        }
        printf("\n");

        printf("wfsVec: ");
        for(int n = 0; n < 5; n++)
        {
            printf("%f ", wfsVec[n]);
        }
        printf(" ... ");
        for(int n = N - 5; n < N; n++)
        {
            printf("%f ", wfsVec[n]);
        }
        printf("\n");
        */
}

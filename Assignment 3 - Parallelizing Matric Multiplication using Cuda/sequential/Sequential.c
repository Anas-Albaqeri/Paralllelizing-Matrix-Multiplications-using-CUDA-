#include<stdio.h>
#include<stdlib.h>
#include<time.h>
// Function to perform matrix multiplication
void matrixMultiplication(int M, int N, int P, const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}



int main(){

    // Define matrices
    int M = 10000;
    int N = 30000;
    int K = 20000;

    // Allocate memory on the host for input and output matrices
    float *A, *B, *C;
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));

    // Initialize input matrices with random values
    for (int i = 0; i < M * K; i++) A[i] = rand();
    for (int i = 0; i < K * N; i++) B[i] = rand();


    // Start the clock
    clock_t start, end;
    start = clock();

    // Multiply matrices
    matrixMultiplication(M,K,N, A, B, C);

    // Stop the clock
    end = clock();
    double time_taken = ((double)end - start) / CLOCKS_PER_SEC; // in seconds

    // Print the resultant matrix
    // printf("Resultant Matrix:\n");
    // printMatrix(3, 3, C);

    printf("Elapsed time: %f\n", time_taken);

    return 0;
}

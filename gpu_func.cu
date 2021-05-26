#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define GLOBAL 1
#define SMEM 2
#define BLOCK_SIZE 32

/* Function signatures */

__global__
void globalMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, int K, bool isVec=false);

__global__
void smeMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, int K, bool isVec=false);


/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C

m, n, k
    Integers indicating the size of the matrices:
    A: m rows by k columns
    B: k rows by n columns
    C: m rows by n columns
*/
int myGEMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta,
           int M, int N, int K, bool isVec) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int algorithm = GLOBAL;
    switch (algorithm) {
        case GLOBAL:
            globalMM<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K, isVec);
            break;
        case SMEM:
            smeMM<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K, isVec);
            break;
      }
  return 0;
}

__global__
void globalMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, int K, bool isVec) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        real accumulator = 0;  
        for (int i = 0; i < K; i++) {
            real a = A[row + i * M];
            real b = B[i + col * K];
            accumulator += a * b;
        }
        real C_term = (isVec) ? C[row] : C[row + col * M];
        C[row + col * M] = alpha * accumulator + beta * C_term;
    }
}

__global__
void smeMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, const int K, bool isVec) {

    int blockRow = blockIdx.y * blockDim.y;
    int row = threadIdx.y;
    int gRow = blockRow + row;

    int blockCol = blockIdx.x * blockDim.x;
    int col = threadIdx.x;
    int gCol = blockCol + col;

    int blockSize = blockDim.y;
    int steps = (K + blockSize - 1) / blockSize;

    real accumulator = 0;
    for (int k = 0; k < steps; ++k) {
        __shared__ real A_block[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ real B_block[BLOCK_SIZE][BLOCK_SIZE];

        real* A_submat = A + (blockRow + k * blockSize * M);
        real* B_submat = B + (k * blockSize + blockCol * K);

        if (gRow < M && i + col < K) {
            A_block[row][col] = A_submat[row + col * M];
        }
        if (gCol < N && i + row < K) {
            B_block[row][col] = B_submat[row + col * K];
        }

        __syncthreads();
        
        if (gRow < M && gCol < N) {
            for (int j = 0; j < min(blockSize, K - i * blockSize); j++) {
                accumulator += A_block[row][j] * B_block[j][col];
            }
        }
    }

    if (gRow < M && gCol < N) {
        real C_term = (isVec) ? C[gRow] : C[gRow + gCol * M];
        C[gRow + gCol * M] = alpha * accumulator + beta * C_term;
    }
}
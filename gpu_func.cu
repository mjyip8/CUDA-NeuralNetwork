#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#include "mpi.h"

#define GLOBAL 1
#define SMEM 2
#define STRIDED_SMEM 3
#define BLOCK_SIZE 32

#define SMAX_STRIDE 16
#define N_CLASSES 10
#define SUM_STRIDE 32

#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

/***************************************************************
 *                           KERNELS
 ***************************************************************/

/* MatMul Kernels */

__global__
void globalMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, int K, 
           bool isVec, bool transposeA, bool transposeB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        real accumulator = 0;  
        for (int i = 0; i < K; i++) {
            int A_index = (transposeA)? i + row*K : row + i*M;
            int B_index = (transposeB)? col + i*N : i + col*K;
            if (A_index < M * K && B_index < N * K) {
                real a = A[A_index];
                real b = B[B_index];
                accumulator += a * b;
            }
        }
        real add_term = 0.;
        if (beta) {
            real C_term = (isVec) ? C[row + N*M] : C[row + col * M];
            add_term = beta * C_term;
        }
        C[row + col * M] = alpha * accumulator + add_term;
    }
}

__global__
void smeMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, const int K, 
           bool isVec, bool transposeA, bool transposeB) {

    int blockRow = blockIdx.y * blockDim.y;
    int blockCol = blockIdx.x * blockDim.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int gRow = blockRow + row;
    int gCol = blockCol + col;

    real accumulator = 0;
    for (int i = 0; i < K; i += BLOCK_SIZE) {
        __syncthreads();
        __shared__ real A_block[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ real B_block[BLOCK_SIZE * BLOCK_SIZE];
        if (gRow < M && i + col < K) {
            int gA_index = (transposeA)? (i + col) + gRow * K : gRow + (i + col) * M;
            A_block[row + col * BLOCK_SIZE] = A[gA_index];
        }
        if (gCol < N && i + row < K) {
            int gB_index = (transposeB)? gCol + (i + row) * N : (i + row) + gCol * K;
            B_block[row + col * BLOCK_SIZE] = B[gB_index];
        }

        __syncthreads();
        
        if (gRow < M && gCol < N) {
            for (int j = 0; j < min(BLOCK_SIZE, K - i); j++) {
                accumulator += A_block[row + j * BLOCK_SIZE] * B_block[j + col * BLOCK_SIZE];
            }
        }
    }

    if (gRow < M && gCol < N) {
        real add_term = 0.;
        if (beta) {
            real C_term = (isVec) ? C[gRow + N*M] : C[gRow + gCol * M];
            add_term = beta * C_term;
        }
        C[gRow + gCol * M] = alpha * accumulator + add_term;
    }
}

// TODO: do transposeA and transposeB tests later
__global__
void stridedSMEMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, int K, 
           bool isVec, bool transposeA, bool transposeB) {
    int blockRow = blockIdx.y * 64;
    int blockCol = blockIdx.x * 16;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tr = threadIdx.y + blockDim.y * threadIdx.x;
    int gRow = blockRow + tr;

    if (blockRow >= M || blockCol >= N) return;

    real accumulator[16];
    memset(&accumulator[0], 0, sizeof(real) * 16);               
    real A_miniblock[4];

    __shared__ real B_block[16][4];

    for (int ptr = 0; ptr < K; ptr += 4) {
        int br = ptr + col;
        int bc = blockCol + row;
        if (br < K && bc < N)
            B_block[row][col] = B[br + bc * K];

        __syncthreads();

        memset(&A_miniblock[0], 0, sizeof(real) * 4);

        # pragma unroll 
        for (int i = 0; i < 4 && ptr + i < K; ++i) {
            A_miniblock[i] = A[gRow + (ptr + i) * M];
        }

        # pragma unroll 
        for (int c = 0; c < 16; ++c) {
            for (int r = 0; r < 4; ++r) {
                accumulator[c] += A_miniblock[r] * B_block[c][r];
            }
        }

        __syncthreads();

    }

    if (gRow >= M) return;

    # pragma unroll
    for (int c = 0; c < 16; ++c) {
        int idx = gRow + (blockCol + c) * M;
        int add_idx = idx;
        if (blockCol + c >= N) break;
        C[idx] = accumulator[c] * alpha + C[add_idx] * beta;
    }
}

/*------------------ FORWARD PASS KERNELS ---------------------*/

__global__
void sigmoid(real* Z, int M, int N) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N) return;
    Z[row + col * M] = 1. / (1. + exp(-Z[row + col * M]));
}

__global__
void softmax(real* Z, int M, int N) {
    int gCol = blockDim.x * blockIdx.x + threadIdx.x;
    int gRow = threadIdx.y;
    int col = threadIdx.x;
    int row = threadIdx.y;
    int psum_stride = N_CLASSES/2;

    __shared__ real smem_exp[SMAX_STRIDE][N_CLASSES];
    __shared__ real smem_sum[SMAX_STRIDE][2];

    if (gCol < N) {
        smem_exp[col][row] = exp(Z[gRow + gCol * M]);
    
        __syncthreads();

        if (row % psum_stride == 0) {
            int psum_idx = row / psum_stride;
            smem_sum[col][psum_idx] = 0;
            for (int i = 0; i < psum_stride; ++i) {
                smem_sum[col][psum_idx] += smem_exp[col][row + i];
            }
        }
        __syncthreads();

        if (row == 0) {
            smem_sum[col][0] += smem_sum[col][1];
        }
        __syncthreads();

        Z[gRow + gCol * M] = smem_exp[col][row] / smem_sum[col][0];
    }
}

/*------------------ BACKWARD PASS KERNELS ---------------------*/

__global__ 
void subtract(real* A, real*B, real k, int M, int N) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N) return;
    B[row + col * M] = k * (A[row + col * M] - B[row + col * M]);
}

__global__ 
void updateParam(real* A, real*B, real lr, int M, int N) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N) return;
    B[row + col * M] = B[row + col * M] - (lr * A[row + col * M]);
}

__global__ 
void sigmoidBackward(real* S, real* A, int M, int N) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N) return;
    real s = S[row + col * M];
    real a = A[row + col * M];
    A[row + col * M] = a * s * (1. - s);
}

template <unsigned int blockSize>
__device__ void reduce(volatile real *sdata, const int col) {
    if (blockSize >=  64) { sdata[col] += sdata[col + 32]; }
    if (blockSize >=  32) { sdata[col] += sdata[col + 16]; }
    if (blockSize >=  16) { sdata[col] += sdata[col +  8]; }
    if (blockSize >=   8) { sdata[col] += sdata[col +  4]; }
    if (blockSize >=   4) { sdata[col] += sdata[col +  2]; }
    if (blockSize >=   2) { sdata[col] += sdata[col +  1]; }
}

template <unsigned int blockSize>
__global__ void sum(real* __restrict__ A, real* __restrict__ out, real k, int K, int N) {
    extern __shared__ real sdata[];

    const int col = threadIdx.x;
    const int row = blockIdx.x;

    // load into shared memory
    unsigned int i = col;
    sdata[col] = 0;
    while (i < N) { 
        sdata[col] += A[row + i * K] + 
                    ((i + blockSize < N)? A[row + (i + blockSize) * K] : 0); 
        i += (2 * blockSize); 
    }

    __syncthreads();

    if (blockSize >= 512) { if (col < 256) { sdata[col] += sdata[col + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (col < 128) { sdata[col] += sdata[col + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (col <  64) { sdata[col] += sdata[col +  64]; } __syncthreads(); }

    if (col < 32) reduce<blockSize>(sdata, col);

    if (col == 0) out[row] = k * sdata[0];
}

/***************************************************************
 *                           WRAPPERS
 ***************************************************************/

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
           int M, int N, int K, 
           bool isVec, bool transposeA, bool transposeB) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int algorithm = STRIDED_SMEM;
    switch (algorithm) {
        case GLOBAL:
            globalMM<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K, isVec, transposeA, transposeB);
            break;
        case SMEM:
            smeMM<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K, isVec, transposeA, transposeB);
            break;
        case STRIDED_SMEM:
            dim3 stridedThreads(4, 16);
            dim3 stridedBlocks((N + 15) / 16, (M + 63) / 64);
            stridedSMEMM<<<stridedBlocks, stridedThreads>>>(A, B, C, *alpha, *beta, M, N, K, isVec, transposeA, transposeB);
            break;
      }
  return 0;
}

// without pointers to weights
int myGEMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, int K, 
           bool isVec, bool transposeA, bool transposeB) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int algorithm = GLOBAL;
    switch (algorithm) {
        case GLOBAL:
            globalMM<<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K, isVec, transposeA, transposeB);
            break;
        case SMEM:
            smeMM<<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K, isVec, transposeA, transposeB);
            break;
      }
  return 0;
}

int myGEMMAsync(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta,
           int M, int N, int K, CudaStreams& streams, int i,
           bool isVec, bool transposeA, bool transposeB) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int algorithm = GLOBAL;
    switch (algorithm) {
        case GLOBAL:
            globalMM<<<blocks, threads, 0, streams.stream[i]>>>(A, B, C, alpha, beta, M, N, K, isVec, transposeA, transposeB);
            break;
        case SMEM:
            smeMM<<<blocks, threads, 0, streams.stream[i]>>>(A, B, C, alpha, beta, M, N, K, isVec, transposeA, transposeB);
            break;
      }
  return 0;
}

/*------------------ FORWARD PASS WRAPPERS ---------------------*/

void deviceSigmoid(real* Z, int M, int N) {
    real* result;
    checkCudaErrors(cudaMalloc(&result, sizeof(real) * M * N));

    dim3 threadDims(32, 32);
    dim3 blockDims((M + threadDims.x - 1) / threadDims.x, (N + threadDims.y - 1) / threadDims.y);
    sigmoid<<<blockDims, threadDims>>>(Z, M, N);
}

void deviceSoftmax(real* Z, int M, int N) {
    dim3 threadDims(SMAX_STRIDE, M);
    dim3 blockDims((N + threadDims.x - 1) / threadDims.x, 1);
    softmax<<<blockDims, threadDims>>>(Z, M, N);
}

/*------------------ BACKWARD PASS WRAPPERS ---------------------*/

void deviceSubtract(real* A, real*B, real k, int M, int N) {
    dim3 threadDims(32, 32);
    dim3 blockDims((M + threadDims.x - 1) / threadDims.x, (N + threadDims.y) / threadDims.y);
    subtract<<<blockDims, threadDims>>>(A, B, k, M, N);
}

void deviceSum(real* A, real* result, real k, int K, int N, CudaStreams& streams, int i) {

    // batch_size cases: 800, 400, 267, 266, 200, 100
    dim3 gridDim(K);
    dim3 blockDim(K);
    int threads;
    size_t sharedMemSize;

    int floor_N = (N / 2) * 2;
    if (floor_N > 512) {
        threads = 512;
        blockDim.x = threads;
        sharedMemSize = threads * sizeof(real);
        sum<512><<<gridDim, blockDim, sharedMemSize, streams.stream[i]>>>(A, result, k, K, N);
    } else if (floor_N <= 512 && floor_N > 256) { 
        threads = 256;
        blockDim.x = threads;
        sharedMemSize = threads * sizeof(real);
        sum<256><<<gridDim, blockDim, sharedMemSize, streams.stream[i]>>>(A, result, k, K, N);
    } else if (floor_N <= 256 && floor_N > 128) {
        threads = 128;
        blockDim.x = threads;
        sharedMemSize = threads * sizeof(real);
        sum<128><<<gridDim, blockDim, sharedMemSize, streams.stream[i]>>>(A, result, k, K, N);
    } else if (floor_N <= 128 && floor_N > 64) {
        threads = 64;
        blockDim.x = threads;
        sharedMemSize = threads * sizeof(real);
        sum<64><<<gridDim, blockDim, sharedMemSize, streams.stream[i]>>>(A, result, k, K, N);
    } else {
        threads = 32;
        blockDim.x = threads;
        sharedMemSize = threads * sizeof(real);
        sum<32><<<gridDim, blockDim, sharedMemSize, streams.stream[i]>>>(A, result, k, K, N);
    }
}

void deviceSigmoidBackward(real* S, real* A, int M, int N) {
    dim3 threadDims(32, 32);
    dim3 blockDims((M + threadDims.x - 1) / threadDims.x, (N + threadDims.y) / threadDims.y);
    sigmoidBackward<<<blockDims, threadDims>>>(S, A, M, N);
}

void deviceUpdateParam(real* A, real*B, real lr, int M, int N, cudaStream_t& s) {
    dim3 threadDims(32, (N > 1)? 32 : 1);
    dim3 blockDims((M + threadDims.x - 1) / threadDims.x, 
                    (N > 1)? ((N + threadDims.y) / threadDims.y) : 1);
    updateParam<<<blockDims, threadDims, 0, s>>>(A, B, lr, M, N);
}

void deviceUpdateStep(HostData& host, DeviceGrads& grads, DeviceNeuralNetwork& dnn, 
                        real learning_rate, CudaStreams& streams) {

    streams.synchronizeAll();
    // Copying from device to host
    checkCudaErrors(cudaMemcpyAsync(host.local_dW[0], grads.dW[0], sizeof(real) * grads.H[1] * grads.H[0], 
                  cudaMemcpyDeviceToHost, streams.stream[0]));
    checkCudaErrors(cudaMemcpyAsync(host.local_db[0], grads.db[0], sizeof(real) * grads.H[1], 
                  cudaMemcpyDeviceToHost, streams.stream[2]));
    checkCudaErrors(cudaMemcpyAsync(host.local_dW[1], grads.dW[1], sizeof(real) * grads.H[2] * grads.H[1], 
                  cudaMemcpyDeviceToHost, streams.stream[1]));
    checkCudaErrors(cudaMemcpyAsync(host.local_db[1], grads.db[1], sizeof(real) * grads.H[2], 
                  cudaMemcpyDeviceToHost, streams.stream[3]));

    // Performing the MPI Call Here
    MPI_Request reqs[grads.num_layers * 2];
    cudaStreamSynchronize(streams.stream[0]);
    MPI_SAFE_CALL(MPI_Iallreduce(host.local_dW[0], host.sum_dW[0], 
                dnn.H[0] * dnn.H[1], MPI_FP, MPI_SUM, MPI_COMM_WORLD, &reqs[0]));

    cudaStreamSynchronize(streams.stream[2]);
    MPI_SAFE_CALL(MPI_Iallreduce(host.local_db[0], host.sum_db[0], 
                dnn.H[1], MPI_FP, MPI_SUM, MPI_COMM_WORLD, &reqs[2]));

    cudaStreamSynchronize(streams.stream[1]);
    MPI_SAFE_CALL(MPI_Iallreduce(host.local_dW[1], host.sum_dW[1],  
                dnn.H[1] * dnn.H[2], MPI_FP, MPI_SUM, MPI_COMM_WORLD, &reqs[1]));

    cudaStreamSynchronize(streams.stream[3]);
    MPI_SAFE_CALL(MPI_Iallreduce(host.local_db[1], host.sum_db[1],
                dnn.H[2], MPI_FP, MPI_SUM, MPI_COMM_WORLD, &reqs[3])); 

    // Coping back to device
    MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    checkCudaErrors(cudaMemcpyAsync(grads.dW[0], host.sum_dW[0], sizeof(real) * dnn.H[1] * dnn.H[0], 
                      cudaMemcpyHostToDevice, streams.stream[0]));
    MPI_Wait(&reqs[2], MPI_STATUS_IGNORE);
    checkCudaErrors(cudaMemcpyAsync(grads.db[0], host.sum_db[0], sizeof(real) * dnn.H[1], 
                      cudaMemcpyHostToDevice, streams.stream[2]));
    MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
    checkCudaErrors(cudaMemcpyAsync(grads.dW[1], host.sum_dW[1], sizeof(real) * dnn.H[2] * dnn.H[1], 
                      cudaMemcpyHostToDevice, streams.stream[1]));
    MPI_Wait(&reqs[3], MPI_STATUS_IGNORE);
    checkCudaErrors(cudaMemcpyAsync(grads.db[1], host.sum_db[1], sizeof(real) * dnn.H[2], 
                      cudaMemcpyHostToDevice, streams.stream[3]));

    // Performing the actual update
    deviceUpdateParam(grads.dW[0], dnn.W[0], learning_rate, dnn.H[1], dnn.H[0], streams.stream[0]);
    deviceUpdateParam(grads.db[0], dnn.b[0], learning_rate, dnn.H[1], 1, streams.stream[2]); 
    deviceUpdateParam(grads.dW[1], dnn.W[1], learning_rate, dnn.H[2], dnn.H[1], streams.stream[1]);
    deviceUpdateParam(grads.db[1], dnn.b[1], learning_rate, dnn.H[2], 1, streams.stream[3]); 
}

/*------------------ HELPER FUNCTIONS ---------------------*/

void setToZero(real*& ptr, int size) {
    checkCudaErrors(cudaMemset(ptr, 0, sizeof(real) * size));
}

void deviceCleanUp(real* ptr) { 
    checkCudaErrors(cudaFree(ptr)); 
}

void deviceMalloc(real*& ptr, int size) {
  checkCudaErrors(cudaMalloc(&ptr, sizeof(real) * size));
}

real* deviceToDeviceCopy(real* orig, int size) {
  real* ptr = nullptr;
  checkCudaErrors(cudaMalloc(&ptr, sizeof(real) * size));
  checkCudaErrors(cudaMemcpy(ptr, orig, sizeof(real) * size, cudaMemcpyDeviceToDevice));
  return ptr;
}
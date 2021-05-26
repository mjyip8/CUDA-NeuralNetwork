#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real* alpha, real* beta,
           int M, int N, int K) {
  // TODO
  return 0;
}

/* Helper functions for neural networks */

// TODO
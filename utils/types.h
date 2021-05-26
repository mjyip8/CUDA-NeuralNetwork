#ifndef TYPES_H_
#define TYPES_H_

// #define USE_DOUBLE

#ifndef USE_DOUBLE

#define real float
#define MPI_FP MPI_FLOAT
#define cublas_gemm cublasSgemm

#else

#define real double
#define MPI_FP MPI_DOUBLE
#define cublas_gemm cublasDgemm

#endif
#endif
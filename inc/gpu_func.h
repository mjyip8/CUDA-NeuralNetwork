#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <armadillo>

#include "../utils/types.h"

/***************************************************************
 *                     DEVICE CLASSES
 ***************************************************************/
class DeviceNeuralNetwork {
 public:
  const int num_layers = 2;
  // H[i] is the number of neurons in layer i (where i=0 implies input layer)
  std::vector<int> H;
  // Weights of the neural network
  // W[i] are the weights of the i^th layer
  std::vector<real *> W;
  // Biases of the neural network
  // b[i] is the row vector biases of the i^th layer
  std::vector<real *> b;

  DeviceNeuralNetwork(std::vector<int> _H) {
    W.resize(num_layers);
    b.resize(num_layers);
    H = _H;

    for (int i = 0; i < num_layers; i++) {
      checkCudaErrors(cudaMalloc(&W[i], sizeof(real) * H[i + 1] * H[i]));
      checkCudaErrors(cudaMalloc(&b[i], sizeof(real) * H[i + 1]));
    }
  }

  ~DeviceNeuralNetwork() {
    for (int i = 0; i < num_layers; ++i) {
      checkCudaErrors(cudaFree(W[i]));
      checkCudaErrors(cudaFree(b[i]));
    }
  }

  void CopyToDevice(std::vector<arma::Mat<real>>& hW, std::vector<arma::Col<real>>& hb) {
    for (int i = 0; i < num_layers; ++i) {
      checkCudaErrors(cudaMemcpy(W[i], hW[i].memptr(), sizeof(real) * H[i + 1] * H[i], cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(b[i], hb[i].memptr(), sizeof(real) * H[i + 1], cudaMemcpyHostToDevice));
    }
  }

  void CopyToHost(std::vector<arma::Mat<real>>& hW, std::vector<arma::Col<real>>& hb) {
    for (int i = 0; i < num_layers; ++i) {
      checkCudaErrors(cudaMemcpy(hW[i].memptr(), W[i], sizeof(real) * H[i + 1] * H[i], cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hb[i].memptr(), b[i], sizeof(real) * H[i + 1], cudaMemcpyDeviceToHost));
    }
  }

  void CopyToDevice(real* hW, real* hb, int i) {
    checkCudaErrors(cudaMemcpy(W[i], hW, sizeof(real) * H[i + 1] * H[i], cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(b[i], hb, sizeof(real) * H[i + 1], cudaMemcpyHostToDevice));
  }

  void CopyToHost(real* hW, real* hb, int i) {
    checkCudaErrors(cudaMemcpy(hW, W[i], sizeof(real) * H[i + 1] * H[i], cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hb, b[i], sizeof(real) * H[i + 1], cudaMemcpyDeviceToHost));
  }
};

class DeviceGrads {
  public:
    const int num_layers = 2;
    std::vector<real *> dW;
    std::vector<real *> db;
    std::vector<int> H;

    DeviceGrads(std::vector<int> _H) {
      H = _H;

      dW.resize((size_t) num_layers);
      db.resize((size_t) num_layers);

      for (int i = 0; i < num_layers; i++) {
        checkCudaErrors(cudaMalloc(&dW[i], sizeof(real) * H[i + 1] * H[i]));
        checkCudaErrors(cudaMalloc(&db[i], sizeof(real) * H[i + 1]));
      }
    }

    ~DeviceGrads() {
      for (uint i = 0; i < dW.size(); ++i) {
        checkCudaErrors(cudaFree(dW[i]));
        checkCudaErrors(cudaFree(db[i]));
      }
    }
};

class DeviceCache {
  public: 
    real* X;
    std::vector<real *> z;
    std::vector<real *> a;
    real* yc;
    std::vector<int> H;

    DeviceCache(std::vector<int> _H, int N, real* _X) {
      H = _H;
      z.resize(2);
      a.resize(2);

      // copying over X
      checkCudaErrors(cudaMalloc(&X, sizeof(real) * H[0] * N));
      checkCudaErrors(cudaMemcpy(X, _X, sizeof(real) * H[0] * N, cudaMemcpyHostToDevice));

      checkCudaErrors(cudaMalloc(&z[0], sizeof(real) * H[1] * N));
      checkCudaErrors(cudaMalloc(&a[0], sizeof(real) * H[1] * N));
      checkCudaErrors(cudaMalloc(&z[1], sizeof(real) * H[2] * N));
      checkCudaErrors(cudaMalloc(&a[1], sizeof(real) * H[2] * N));
      checkCudaErrors(cudaMalloc(&yc,   sizeof(real) * H[2] * N));
    } 

    ~DeviceCache() {
      checkCudaErrors(cudaFree(X));
      checkCudaErrors(cudaFree(z[0]));
      checkCudaErrors(cudaFree(z[1]));
      checkCudaErrors(cudaFree(a[0]));
      checkCudaErrors(cudaFree(a[1]));
      checkCudaErrors(cudaFree(yc));
    }
};

class DeviceData {
  public: 
    real* X;
    real* y;
    int N;
    int K;

    DeviceData(real* hX, real* hy, int _N, int _K) {
      N = _N;
      K = _K;
      checkCudaErrors(cudaMalloc(&X, sizeof(real) * N * K));
      checkCudaErrors(cudaMalloc(&y, sizeof(real) * N));
      checkCudaErrors(cudaMemcpy(X, hX, sizeof(real) * N * K, cudaMemcpyHostToDevice));   
      checkCudaErrors(cudaMemcpy(y, hy, sizeof(real) * N, cudaMemcpyHostToDevice));
    }

    ~DeviceData() {
      checkCudaErrors(cudaFree(X));
      checkCudaErrors(cudaFree(y));
    }
};

/***************************************************************/

struct event_pair {
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair* p) {
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}

inline double stop_timer(event_pair* p) {
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

/***************************************************************
 *                           WRAPPERS
 ***************************************************************/

int myGEMM(real* A, real* B, real* C, real* alpha, real* beta, int M, int N,
           int K, bool isVec=false, bool transposeB=false);

/*------------------ FORWARD PASS WRAPPERS ---------------------*/

real* deviceLinear(real* A, real* B, real* C, real* alpha, real* beta, int M, int N,
           int K, bool isVec=false, bool transposeB=false);

real* deviceSigmoid(real* Z, int M, int N);

real* deviceSoftmax(real* Z, int M, int N);

/*------------------ BACKWARD PASS WRAPPERS ---------------------*/

real* deviceSubtract(real* A, real*B, real k, int M, int N);

void deviceSum(real* A, real* result, int K, int N);

real* deviceSigmoidBackward(real* S, real* A, int M, int N);

#endif

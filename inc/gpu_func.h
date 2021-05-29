#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <armadillo>

#include "../utils/types.h"

void deviceCleanUp(real* ptr);

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
      checkCudaErrors(cudaMalloc(&b[i], sizeof(real) * H[i + 1] * H[i]));
    }
  }

  ~DeviceNeuralNetwork() {
    for (int i = 0; i < num_layers; ++i) {
      checkCudaErrors(cudaFree(W[i]));
      checkCudaErrors(cudaFree(b[i]));
    }
  }

  void CopyToDevice(std::vector<arma::Mat<real>>& hdW, 
                  std::vector<arma::Col<real>>& hdb) {
    for (int i = 0; i < num_layers; ++i) {
      checkCudaErrors(cudaMemcpy(W[i], hdW[i].memptr(), sizeof(real) * H[i + 1] * H[i], cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(b[i], hdb[i].memptr(), sizeof(real) * H[i + 1] * H[i], cudaMemcpyHostToDevice));
    }
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
      checkCudaErrors(cudaMalloc(&dW[1], sizeof(real) * H[2] * H[1]));

      for (int i = 0; i < num_layers; i++) {
        checkCudaErrors(cudaMalloc(&db[i], sizeof(real) * H[i + 1]));
      }
    }

    ~DeviceGrads() {
      for (int i = 0; i < num_layers; ++i) {
        checkCudaErrors(cudaFree(db[i]));
      }
      cudaFree(dW[1]);
    }

    void LoadWeightMatrices(std::vector<real *> W) {

      dW[0] = W[0];
      checkCudaErrors(cudaMemcpy(dW[1], W[1], sizeof(real) * H[2] * H[1], cudaMemcpyDeviceToDevice));
    }

    void CopyToHost(std::vector<arma::Mat<real>>& hdW, 
                    std::vector<arma::Col<real>>& hdb) {
      for (int i = 0; i < num_layers; ++i) {
        checkCudaErrors(cudaMemcpy(hdW[i].memptr(), dW[i], sizeof(real) * H[i + 1] * H[i], cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hdb[i].memptr(), db[i], sizeof(real) * H[i + 1], cudaMemcpyDeviceToHost));
      }
    }
};

class DeviceCache {
  public: 
    real* X;
    int N;
    std::vector<real *> z;
    std::vector<real *> a;
    real* yc;
    std::vector<int> H;

    DeviceCache(std::vector<int> _H, int _N, real* _X) {
      H = _H;
      N = _N;
      X = _X; //copy of pointer to device X batch
      z.resize(2);
      a.resize(1);

      checkCudaErrors( cudaMalloc(&z[0], (N + 1) * H[1] * sizeof(real)) );
      checkCudaErrors( cudaMalloc(&z[1], (N + 1) * H[2] * sizeof(real)) );

      checkCudaErrors(cudaMalloc(&a[0], sizeof(real) * H[1] * N));
    } 

    ~DeviceCache() {
      checkCudaErrors(cudaFree(a[0]));
      checkCudaErrors(cudaFree(z[0]));
      checkCudaErrors(cudaFree(z[1]));
    }

    void recordAFromZ() {
      checkCudaErrors(cudaMemcpy(a[0], z[0], sizeof(real) * H[1] * N, cudaMemcpyDeviceToDevice));
    }

    void LoadBias(std::vector<real *>& b) {
      for (int i = 0; i < (int) z.size(); ++i) {
        checkCudaErrors(cudaMemcpy(z[i] + (N * H[i+1]), b[i], sizeof(real) * H[i + 1], cudaMemcpyDeviceToDevice));
      }
    }
};

class DeviceData {
  public: 
    real* X;
    real* y; // y is one hot
    int N;
    int K;

    DeviceData(real* hX, real* hy, int _N, int _K, int numclasses) {
      N = _N;
      K = _K;
      checkCudaErrors(cudaMalloc(&X, sizeof(real) * N * K));
      checkCudaErrors(cudaMalloc(&y, sizeof(real) * N * numclasses));
      checkCudaErrors(cudaMemcpy(X, hX, sizeof(real) * N * K, cudaMemcpyHostToDevice));  
      checkCudaErrors(cudaMemcpy(y, hy, sizeof(real) * N * numclasses, cudaMemcpyHostToDevice));
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
           int K, bool isVec=false, bool transposeA=false, bool transposeB=false);
int myGEMM(real* A, real* B, real* C, real alpha, real beta, int M, int N,
           int K, bool isVec=false, bool transposeA=false, bool transposeB=false);

int myGEMMAlloc(real* A, real* B, real*& C, real alpha, real beta, int M, int N,
           int K, bool isVec=false, bool transposeA=false, bool transposeB=false);
void transpose(real* A, real*& AT, int M, int N);

/*------------------ FORWARD PASS WRAPPERS ---------------------*/

real* deviceLinear(real* A, real* B, real* C, real* alpha, real* beta, int M, int N,
           int K, bool isVec=false, bool transposeB=false);

void deviceSigmoid(real* Z, int M, int N);

void deviceSoftmax(real* Z, int M, int N);

/*------------------ BACKWARD PASS WRAPPERS ---------------------*/

void deviceSubtract(real* A, real*B, real k, int M, int N);

void deviceSum(real* A, real* result, real k, int K, int N);

void deviceSigmoidBackward(real* S, real* A, int M, int N);

#endif

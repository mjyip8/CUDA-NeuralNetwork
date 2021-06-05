#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <armadillo>

#include "../utils/types.h"

# define NUM_LAYERS 2

void deviceCleanUp(real* ptr);

real* deviceToDeviceCopy(real* orig, int size); 

void setToZero(real*& ptr, int size);

void deviceMalloc(real*& ptr, int size);

/***************************************************************
 *                     DEVICE CLASSES
 ***************************************************************/

class CudaStreams {
  public:
    const int n_streams = 4;
    std::vector<cudaStream_t> stream;
    // Create Streams

    CudaStreams() {
      stream.resize(n_streams);
      for (auto i = 0; i < n_streams; ++i) 
        checkCudaErrors( cudaStreamCreate(&stream[i]) );
    }

    ~CudaStreams() {
      for (int i = 0; i < n_streams; ++i) 
          checkCudaErrors( cudaStreamDestroy(stream[i]) );
    }

    void synchronize(int i) {
      cudaStreamSynchronize(stream[i]);
    }

    void synchronizeAll() {
      for (int i = 0; i < n_streams; ++i)
        cudaStreamSynchronize(stream[i]);
    }

};

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
    W.resize(NUM_LAYERS);
    b.resize(NUM_LAYERS);
    H = _H;

    checkCudaErrors(cudaMalloc(&W[0], sizeof(real) * H[1] * H[0]));
    checkCudaErrors(cudaMalloc(&b[0], sizeof(real) * H[1]));
    checkCudaErrors(cudaMalloc(&W[1], sizeof(real) * H[2] * H[1]));
    checkCudaErrors(cudaMalloc(&b[1], sizeof(real) * H[2]));
  }

  ~DeviceNeuralNetwork() {
    checkCudaErrors(cudaFree(W[0]));
    checkCudaErrors(cudaFree(W[1]));
    checkCudaErrors(cudaFree(b[0]));
    checkCudaErrors(cudaFree(b[1]));
  }

  void CopyToDevice(std::vector<arma::Mat<real>>& hW, 
                  std::vector<arma::Col<real>>& hb) {
    checkCudaErrors(cudaMemcpy(W[0], hW[0].memptr(), sizeof(real) * H[1] * H[0], 
                      cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(b[0], hb[0].memptr(), sizeof(real) * H[1], 
                      cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(W[1], hW[1].memptr(), sizeof(real) * H[2] * H[1], 
                      cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(b[1], hb[1].memptr(), sizeof(real) * H[2], 
                      cudaMemcpyHostToDevice));
  }

  void CopyToHost(std::vector<arma::Mat<real>>& hW, 
                  std::vector<arma::Col<real>>& hb) {
    checkCudaErrors(cudaMemcpy(hW[0].memptr(), W[0], sizeof(real) * H[1] * H[0], 
                    cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hb[0].memptr(), b[0], sizeof(real) * H[1],
                    cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hW[1].memptr(), W[1], sizeof(real) * H[2] * H[1], 
                    cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hb[1].memptr(), b[1], sizeof(real) * H[2],
                    cudaMemcpyDeviceToHost));
  }

};

class DeviceGrads {
  public:
    const int num_layers = 2;
    std::vector<real *> dW;
    std::vector<real *> db;
    std::vector<int> H;
    real* dz;
    int N;

    DeviceGrads(std::vector<int> _H, const int _N) : H(_H), N(_N) {
      dW.resize((size_t) NUM_LAYERS);
      db.resize((size_t) NUM_LAYERS);

      checkCudaErrors(cudaMalloc(&dW[0], sizeof(real) * H[1] * H[0]));
      checkCudaErrors(cudaMalloc(&db[0], sizeof(real) * H[1]));
      checkCudaErrors(cudaMalloc(&dW[1], sizeof(real) * H[2] * H[1]));
      checkCudaErrors(cudaMalloc(&db[1], sizeof(real) * H[2]));
      checkCudaErrors(cudaMalloc(&dz, sizeof(real) * H[1] * N));
    }

    ~DeviceGrads() {
      checkCudaErrors(cudaFree(db[0]));
      checkCudaErrors(cudaFree(dW[0]));
      checkCudaErrors(cudaFree(db[1]));
      checkCudaErrors(cudaFree(dW[1]));
      checkCudaErrors(cudaFree(dz));
    }

    void LoadWeightMatrices(std::vector<real *> W, CudaStreams& streams) {
      checkCudaErrors(cudaMemcpyAsync(dW[0], W[0], sizeof(real) * H[1] * H[0], 
                      cudaMemcpyDeviceToDevice, streams.stream[0]));
      checkCudaErrors(cudaMemcpyAsync(dW[1], W[1], sizeof(real) * H[2] * H[1], 
                      cudaMemcpyDeviceToDevice, streams.stream[1]));
    }

    void CopyToHost(std::vector<real*>& hdW, 
                    std::vector<real*>& hdb) {
      checkCudaErrors(cudaMemcpy(hdW[0], dW[0], sizeof(real) * H[1] * H[0], 
                      cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hdb[0], db[0], sizeof(real) * H[1], 
                      cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hdW[1], dW[1], sizeof(real) * H[2] * H[1], 
                      cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hdb[1], db[1], sizeof(real) * H[2], 
                      cudaMemcpyDeviceToHost));
    }

    void CopyToDevice(std::vector<real*>& hdW, 
                    std::vector<real*>& hdb) {
      checkCudaErrors(cudaMemcpy(dW[0], hdW[0], sizeof(real) * H[1] * H[0], 
                      cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(db[0], hdb[0], sizeof(real) * H[1], 
                      cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dW[1], hdW[1], sizeof(real) * H[2] * H[1], 
                      cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(db[1], hdb[1], sizeof(real) * H[2], 
                      cudaMemcpyHostToDevice));
    }
};

class SendData {
  public:
    real* X = nullptr;
    real* y = nullptr;
    int N;
    std::vector<int> H;
    const int num_layers = 2;
    bool isAllocated = false;

    SendData(std::vector<int> _H, int _N) : H(_H), N(_N) {
    }

    ~SendData() {
      if (isAllocated) {
        cudaFreeHost(X);
        cudaFreeHost(y);
      }
    }

    void copyDataToHost(const real* hX, const real* hy) {
      isAllocated = true;
      checkCudaErrors( cudaMallocHost(&X, sizeof(real) * N * H[0]) );
      checkCudaErrors( cudaMallocHost(&y, sizeof(real) * N * H[2]) );
      checkCudaErrors( cudaMemcpy(X, hX, sizeof(real) * N * H[0], cudaMemcpyHostToHost) );
      checkCudaErrors( cudaMemcpy(y, hy, sizeof(real) * N * H[2], cudaMemcpyHostToHost) );
    }
};

class HostData {
  public:
    real* X;
    real* y;

    std::vector<real*> local_dW;
    std::vector<real*> local_db;
    std::vector<real*> sum_dW;
    std::vector<real*> sum_db;
    int N;
    std::vector<int> H;
    const int num_layers = 2;

    HostData(std::vector<int> _H, int _N) : H(_H), N(_N) {
      checkCudaErrors( cudaMallocHost(&X, sizeof(real) * N * H[0]) );
      checkCudaErrors( cudaMallocHost(&y, sizeof(real) * N * H[2]) );
      initialize(local_dW, local_db);
      initialize(sum_dW, sum_db);
    }

    ~HostData() {
      for (size_t i = 0; i < local_dW.size(); ++i) {
        cudaFreeHost(local_dW[i]);
        cudaFreeHost(sum_dW[i]);
        cudaFreeHost(local_db[i]);
        cudaFreeHost(sum_db[i]);
      }
      cudaFreeHost(X);
      cudaFreeHost(y);
    }

    void copyDataToHost(const real* hX, const real* hy) {
      checkCudaErrors( cudaMemcpy(X, hX, sizeof(real) * N * H[0], cudaMemcpyHostToHost) );
      checkCudaErrors( cudaMemcpy(y, hy, sizeof(real) * N * H[2], cudaMemcpyHostToHost) );
    }

  private:
    void initialize(std::vector<real*>& W, std::vector<real*>& b) {
      W.resize((size_t) num_layers);
      b.resize((size_t) num_layers);
      checkCudaErrors( cudaMallocHost(&W[0], sizeof(real) * H[0] * H[1]) );
      checkCudaErrors( cudaMallocHost(&b[0], sizeof(real) * H[1] ) );
      checkCudaErrors( cudaMallocHost(&W[1], sizeof(real) * H[1] * H[2]) );
      checkCudaErrors( cudaMallocHost(&b[1], sizeof(real) * H[2] ) );
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

    DeviceCache(std::vector<int> _H, int _N) {
      H = _H;
      N = _N; // upper estimate of minibatch size 
      z.resize(2);
      a.resize(1);

      checkCudaErrors(cudaMalloc(&z[0], (N + 1) * H[1] * sizeof(real)));
      checkCudaErrors(cudaMalloc(&z[1], (N + 1) * H[2] * sizeof(real)));

      checkCudaErrors(cudaMalloc(&a[0], sizeof(real) * H[1] * N));
    } 

    ~DeviceCache() {
      checkCudaErrors(cudaFree(a[0]));
      checkCudaErrors(cudaFree(z[0]));
      checkCudaErrors(cudaFree(z[1]));
    }

    void recordAFromZ(CudaStreams& streams) {
      checkCudaErrors(cudaMemcpyAsync(a[0], z[0], sizeof(real) * H[1] * N, 
                      cudaMemcpyDeviceToDevice, streams.stream[1]));
    }

    void LoadBias(std::vector<real *>& b, CudaStreams& streams) {
      checkCudaErrors(cudaMemcpyAsync(z[0] + (N * H[1]), b[0], sizeof(real) * H[1], 
                      cudaMemcpyDeviceToDevice, streams.stream[2]));
      checkCudaErrors(cudaMemcpyAsync(z[1] + (N * H[2]), b[1], sizeof(real) * H[2], 
                      cudaMemcpyDeviceToDevice, streams.stream[3]));
    }
};

class DeviceData {
  public: 
    real* X;
    real* y; // y is one hot
    int img_size;
    int n_classes;
    int N;

    DeviceData(int _N, int _img_size, int _n_classes) :
              N(_N), img_size(_img_size), n_classes(_n_classes) {
      checkCudaErrors(cudaMalloc(&X, sizeof(real) * N * img_size));
      checkCudaErrors(cudaMalloc(&y, sizeof(real) * N * n_classes));
    }

    ~DeviceData() {
      checkCudaErrors(cudaFree(X));
      checkCudaErrors(cudaFree(y));
    }

    void CopyToDevice(real* hX, real* hy, int X_offset, int y_offset, int n) {
      checkCudaErrors(cudaMemcpy(X + X_offset, hX + X_offset, sizeof(real) * n * img_size, cudaMemcpyHostToDevice));  
      checkCudaErrors(cudaMemcpy(y + y_offset, hy + y_offset, sizeof(real) * n * n_classes, cudaMemcpyHostToDevice));
    }
};

class DeviceDataVec {
  public: 
    std::vector<real*> X;
    std::vector<real*> y; // y is one hot
    int N;
    int K;
    int num_classes;

    DeviceDataVec(int _K, int _num_classes, const int num_batches) {
      K = _K;
      num_classes = _num_classes;
      X.resize(num_batches);
      y.resize(num_batches);
    }

    ~DeviceDataVec() {
      for (size_t i = 0; i < X.size(); ++i) {
        checkCudaErrors(cudaFree(X[i]));
        checkCudaErrors(cudaFree(y[i]));
      }
    }

    void CopyToDevice(arma::Mat<real>& hX, arma::Mat<real>& hy, int batch, 
                      int minibatch_size, int img_size, int n_classes) {
      checkCudaErrors(cudaMemcpy(X[batch], hX.memptr(), sizeof(real) * minibatch_size * img_size, 
                      cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(y[batch], hy.memptr(), sizeof(real) * minibatch_size * n_classes, 
                      cudaMemcpyHostToDevice));
      
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
int myGEMMAsync(real* A, real* B, real* C, real alpha, real beta, int M, int N,
           int K, CudaStreams& streams, int i, 
           bool isVec=false, bool transposeA=false, bool transposeB=false);
void transpose(real* A, real*& AT, int M, int N);

/*------------------ FORWARD PASS WRAPPERS ---------------------*/

real* deviceLinear(real* A, real* B, real* C, real* alpha, real* beta, int M, int N,
           int K, bool isVec=false, bool transposeB=false);

void deviceSigmoid(real* Z, int M, int N);

void deviceSoftmax(real* Z, int M, int N);

/*------------------ BACKWARD PASS WRAPPERS ---------------------*/

void deviceSubtract(real* A, real*B, real k, int M, int N);

void deviceSum(real* A, real* result, real k, int K, int N, CudaStreams& streams, int i);

void deviceSigmoidBackward(real* S, real* A, int M, int N);

void deviceUpdateParam(real* A, real*B, real lr, int M, int N, cudaStream_t& s);

void deviceUpdateStep(HostData& host, DeviceGrads& grads, DeviceNeuralNetwork& dnn, 
                      real learning_rate, CudaStreams& streams);

#endif

#include "tests.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"
#include "mnist.h"
#include "types.h"
#include "common.h"
using namespace std;

#define SCALE 1       // Factor to SCALE the GEMM problem size by
#define NUM_ITERS 10  // Number of GEMMs run for timing purposes

#define FILE_TRAIN_IMAGES "/data/train-images-idx3-ubyte"
#define FILE_TRAIN_LABELS "/data/train-labels-idx1-ubyte"
#define FILE_TEST_IMAGES "/data/t10k-images-idx3-ubyte"
#define FILE_TEST_OUTPUT "Outputs/Pred_testset.txt"
#define NUM_TRAIN 60000
#define IMAGE_SIZE 784  // 28 x 28
#define NUM_CLASSES 10
#define NUM_TEST 10000

#ifdef USE_DOUBLE
#define TOL 1e-12  // Tolerance for tests
#else
#define TOL 1e-6  // Tolerance for tests
#endif

// check whether the matrix from Seq is the same as from Par.
// write out mismatches to a file.
int checkErrors(const arma::Mat<real>& Seq, const arma::Mat<real>& Par,
                std::ofstream& ofs, std::vector<real>& errors) {
  int error = 0;

  for (int i = 0; i < Seq.n_rows; ++i) {
    for (int j = 0; j < Seq.n_cols; ++j) {
      if (abs(Seq(i, j) - Par(i, j)) > TOL) {
        ofs << "Mismatch at pos (" << i << ", " << j
            << ") diff: " << Seq(i, j) - Par(i, j) << " seq: " << Seq(i, j)
            << " par: " << Par(i, j) << endl;
        ++error;
      }
    }
  }

  if (error) {
    ofs << "There were " << error
        << " total locations where there was a difference between the seq and "
           "par"
        << endl;
  } else {
    ofs << "No errors were found" << endl;
  }

  real err_max = arma::norm(Seq - Par, "inf") / arma::norm(Seq, "inf");
  real err_l2 = arma::norm(Seq - Par, 2) / arma::norm(Seq, 2);

  if (err_max > TOL * 1e2) {
    cout << "Correctness test failed" << endl;
  }

  errors.push_back(err_max);
  errors.push_back(err_l2);

  return error;
}

int checkNNErrors(NeuralNetwork& seq_nn, NeuralNetwork& par_nn,
                  std::string filename) {
  std::vector<real> errors_w, errors_b;
  int error = 0;
  std::ofstream ofs(filename.c_str());

  cout << endl;

  for (int i = 0; i < seq_nn.num_layers; i++) {
    ofs << "Mismatches for W[" << i << "]" << endl;
    error += checkErrors(seq_nn.W[i], par_nn.W[i], ofs, errors_w);
    ofs << "Mismatches for b[" << i << "]" << endl;
    error += checkErrors(seq_nn.b[i], par_nn.b[i], ofs, errors_b);
    cout << "Max norm of diff b/w seq and par: W[" << i
         << "]: " << setprecision(6) << errors_w[2 * i] << ", b[" << i
         << "]: " << errors_b[2 * i] << endl;
    cout << "l2  norm of diff b/w seq and par: W[" << i
         << "]: " << setprecision(6) << errors_w[2 * i + 1] << ", b[" << i
         << "]: " << errors_b[2 * i + 1] << endl;
  }

  ofs.close();
  return error;
}

void createMATS(real* A, real* B, real* C1, real* C2, int NI, int NJ, int NK) {
  int i, j;

  for (j = 0; j < NK; j++) {
    for (i = 0; i < NI; i++) {
      A[i + j * NI] = ((real)i * j) / NI;
    }
  }

  for (j = 0; j < NJ; j++) {
    for (i = 0; i < NK; i++) {
      B[i + j * NK] = ((real)i * j + 1) / NJ;
    }
  }

  for (j = 0; j < NJ; j++) {
    for (i = 0; i < NI; i++) {
      C1[i + j * NI] = 0;
      C2[i + j * NI] = ((real)i * j + 2) / NJ;
    }
  }
}

int compareGEMMResults(real* myC, real* refC, int NI, int NJ, bool print=false) {
  int i, j;
  int fail = 0;

  arma::Mat<real> mysol = arma::Mat<real>(myC, NI, NJ, false);
  arma::Mat<real> refsol = arma::Mat<real>(refC, NI, NJ, false);

  real reldiff = arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL) {
    fail = 1;
  }

  // Print results
  if (fail) {
    std::cout << "My GEMM output not matching with reference. Rel diff = "
              << reldiff << std::endl;
    if (print) {
      mysol.save("Tests/mySol.mat", arma::raw_ascii);
      refsol.save("Tests/refSol.mat", arma::raw_ascii);
    }
  } else {
    std::cout << "GEMM matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }

  return fail;
}

int compareHostDeviceResults(real* device, real* host, int NI, int NJ, std::string name = "", bool print=false) {
  if (!name.empty()) {
      std::cout << "\n************"<< name << "************" << std::endl;
  }
  arma::Mat<real> hostCopy = arma::zeros<arma::Mat<real>>(NI, NJ);
  cudaMemcpy(hostCopy.memptr(), device, sizeof(real) * NI * NJ, cudaMemcpyDeviceToHost);
  return compareGEMMResults(hostCopy.memptr(), host, NI, NJ, print);
}

void TestGEMM(int M, int N, int K) {
  real* A;
  real* B;
  real* C1;
  real* C2;

  real* dA;
  real* dB;
  real* dC1;
  real* dC2;
  real* dummy;

  real alpha = 2.0;
  real beta = 5.0;

  int num_iters = 100;

  A = (real*)malloc(M * K * sizeof(real));
  B = (real*)malloc(K * N * sizeof(real));
  C1 = (real*)malloc(M * N * sizeof(real));
  C2 = (real*)malloc(M * N * sizeof(real));

  cudaMalloc((void**)&dA, sizeof(real) * M * K);
  cudaMalloc((void**)&dB, sizeof(real) * K * N);
  cudaMalloc((void**)&dC1, sizeof(real) * M * N);
  cudaMalloc((void**)&dC2, sizeof(real) * M * N);
  cudaMalloc((void**)&dummy, sizeof(real) * M * N);

  // C1 and C2 are same. We just have two copies to compare results
  createMATS(A, B, C1, C2, M, N, K);

  cudaMemcpy(dA, A, sizeof(real) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(real) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC1, C2, sizeof(real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC2, C2, sizeof(real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dummy, C2, sizeof(real) * M * N, cudaMemcpyHostToDevice);

  /* Warm up GPU before we run. We run one extra CuBlas */
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS initialization failed!" << std::endl;
    return;
  }

  stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M,
                     dB, K, &beta, dummy, M);

  /* Compute reference solution and time the CuBlas */
  using namespace std::chrono;
  high_resolution_clock::time_point ref_t1 = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERS; i++) {
    stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M,
                       dB, K, &beta, dC2, M);
  }

  check_launch("Reference GEMM");
  high_resolution_clock::time_point ref_t2 = high_resolution_clock::now();
  duration<double> ref_time_span =
      duration_cast<duration<double>>(ref_t2 - ref_t1);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS gemm error at " << __FILE__ << ":" << __LINE__
              << std::endl;
  }

  cudaMemcpy(C2, dC2, sizeof(real) * M * N, cudaMemcpyDeviceToHost);

  /* We are calling your GEMM function here */
  /* We will make one dummy call and check_launch here */
  int err;
  err = myGEMM(dA, dB, dummy, &alpha, &beta, M, N, K);
  check_launch("myGEMM dummy");

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERS; i++) {
    err = myGEMM(dA, dB, dC1, &alpha, &beta, M, N, K);
  }

  check_launch("myGEMM");
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> my_time_span = duration_cast<duration<double>>(t2 - t1);

  /* This error code is for your own debugging, it does not catch
     illegal memory accesses or bad kernel launches */
  if (err != 0) {
    std::cout << "Error in my GEMM. Error code: " << err << std::endl;
  }

  cudaMemcpy(C1, dC1, sizeof(real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareGEMMResults(C1, C2, M, N);

  if (fail == 0) {
    std::cout << "Time for reference GEMM implementation: "
              << ref_time_span.count() << " seconds" << std::endl;
    std::cout << "Time for my GEMM implementation: " << my_time_span.count()
              << " seconds" << std::endl;
  }

  free(A);
  free(B);
  free(C1);
  free(C2);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC1);
  cudaFree(dC2);
  cudaFree(dummy);
}

void BenchmarkGEMM() {
  std::cout << std::endl
            << "Entering GEMM Benchmarking mode! Stand by." << std::endl;

  /* First GEMM problem size */
  int M = 800 * SCALE, N = 1000 * SCALE, K = 784 * SCALE;

  std::cout << std::endl
            << "Starting GEMM 1: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 1" << std::endl;

  /* Second GEMM problem size */
  M = 800 * SCALE, N = 10 * SCALE, K = 1000 * SCALE;
  std::cout << std::endl
            << "Starting GEMM 2: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 2" << std::endl;
}

void print_mat(arma::Mat<real> my_matrix) {
    
    uint cols = my_matrix.n_cols;
    uint rows = my_matrix.n_rows;
    
    std::cout << "--------\n";
    for(uint rX = 0; rX < rows; rX++) {
        std::cout << " " << rX << ": ";
        for(uint cX = 0; cX < cols; cX++) {
            std::cout << my_matrix(rX, cX) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "--------\n";
}

void TestKernels() {
  using namespace std::chrono;
  int test_batch_size = 267;
  /* softmax test */
  std::cout << "\n************Softmax Kernel************" << std::endl;
  arma::Mat<real> A(10, test_batch_size);
  arma::Mat<real> A_result(10, test_batch_size);
  A.randu();
  A *= 10.;
  real* result;

  high_resolution_clock::time_point t1_cpu = high_resolution_clock::now();
  softmax(A, A_result);
  high_resolution_clock::time_point t2_cpu = high_resolution_clock::now();
  duration<double> time_span_cpu = duration_cast<duration<double>>(t2_cpu - t1_cpu);

  real* dA;
  arma::Mat<real> dA_result(10, test_batch_size);
  cudaMalloc(&dA, sizeof(real) * A.n_elem);
  cudaMemcpy(dA, A.memptr(), sizeof(real) * A.n_elem, cudaMemcpyHostToDevice);
  high_resolution_clock::time_point t1_gpu = high_resolution_clock::now();
  deviceSoftmax(dA, A.n_rows, A.n_cols);
  high_resolution_clock::time_point t2_gpu = high_resolution_clock::now();
  duration<double> time_span_gpu = duration_cast<duration<double>>(t2_gpu - t1_gpu); 

  cudaMemcpy(dA_result.memptr(), result, sizeof(real) * A.n_elem, cudaMemcpyDeviceToHost);
  cudaFree(dA);

  int fail = compareGEMMResults(dA_result.memptr(), A_result.memptr(), A.n_rows, A.n_cols);
  if (fail == 0) {
    std::cout << "Time for CPU Softmax implementation: "
              << time_span_cpu.count() << " seconds" << std::endl;
    std::cout << "Time for GPU Softmax implementation: " << time_span_gpu.count()
              << " seconds" << std::endl;
  }

  /* sigmoid test */
  std::cout << "\n************Sigmoid Kernel************" << std::endl;
  A.randu();
  A *= 10.;
  t1_cpu = high_resolution_clock::now();
  sigmoid(A, A_result);
  t2_cpu = high_resolution_clock::now();
  time_span_cpu = duration_cast<duration<double>>(t2_cpu - t1_cpu);

  cudaMalloc(&dA, sizeof(real) * A.n_elem);
  cudaMemcpy(dA, A.memptr(), sizeof(real) * A.n_elem, cudaMemcpyHostToDevice);

  t1_gpu = high_resolution_clock::now();
  deviceSigmoid(dA, A.n_rows, A.n_cols);
  t2_gpu = high_resolution_clock::now();
  time_span_gpu = duration_cast<duration<double>>(t2_gpu - t1_gpu); 

  cudaMemcpy(dA_result.memptr(), dA, sizeof(real) * A.n_elem, cudaMemcpyDeviceToHost);
  cudaFree(dA);

  fail = compareGEMMResults(dA_result.memptr(), A_result.memptr(), A.n_rows, A.n_cols);
  if (fail == 0) {
    std::cout << "Time for CPU Sigmoid implementation: "
              << time_span_cpu.count() << " seconds" << std::endl;
    std::cout << "Time for GPU Sigmoid implementation: " << time_span_gpu.count()
              << " seconds" << std::endl;
  } else {
    A_result.save("Outputs/CPUmats/CPUsigmoid.mat", arma::raw_ascii);
    dA_result.save("Outputs/GPUmats/GPUsigmoid.mat", arma::raw_ascii);
  }

  /* sum reduction test */
  std::cout << "\n************Sum Reduction Kernel************" << std::endl;
  A.randu();
  A *= 10.;
  t1_cpu = high_resolution_clock::now();
  arma::Mat<real> A_sum = arma::sum(A, 1);
  t2_cpu = high_resolution_clock::now();
  time_span_cpu = duration_cast<duration<double>>(t2_cpu - t1_cpu);

  cudaMalloc(&dA, sizeof(real) * A.n_elem);
  cudaMemcpy(dA, A.memptr(), sizeof(real) * A.n_elem, cudaMemcpyHostToDevice);
  cudaMalloc(&result, sizeof(real) * 10);

  t1_gpu = high_resolution_clock::now();
  deviceSum(dA, result, 1., A.n_rows, A.n_cols);
  t2_gpu = high_resolution_clock::now();
  time_span_gpu = duration_cast<duration<double>>(t2_gpu - t1_gpu); 

  arma::Mat<real> dA_sum(10, 1);
  cudaMemcpy(dA_sum.memptr(), result, sizeof(real) * dA_sum.n_elem, cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(result);

  fail = compareGEMMResults(dA_sum.memptr(), A_sum.memptr(), A_sum.n_rows, A_sum.n_cols);
  if (fail == 0) {
    std::cout << "Time for CPU Sum implementation: "
              << time_span_cpu.count() << " seconds" << std::endl;
    std::cout << "Time for GPU deviceSum implementation: " << time_span_gpu.count()
              << " seconds" << std::endl;
  } else {
    A_sum.save("Outputs/CPUmats/CPUsum.mat", arma::raw_ascii);
    dA_sum.save("Outputs/GPUmats/GPUsum.mat", arma::raw_ascii);
  }
}

void TestForwardBackProp() {
  /* SETUP */
  std::vector<int> H(3);
  real reg = 1e-4;
  real learning_rate = 0.001;
  int num_epochs = 20;
  int batch_size = 100;
  int num_neuron = 1000;
  int run_seq = 0;
  int debug = 0;
  int grade = 0;
  int print_every = 0;

  H[0] = IMAGE_SIZE;
  H[1] = num_neuron;
  H[2] = NUM_CLASSES;

  arma::Mat<real> x_train, y_train, label_train, x_dev, y_dev, label_dev,
      x_test;
  NeuralNetwork nn(H);
  // Read MNIST images into Armadillo mat vector
  arma::Mat<real> x(IMAGE_SIZE, NUM_TRAIN);
  // label contains the prediction for each
  arma::Row<real> label = arma::zeros<arma::Row<real>>(NUM_TRAIN);
  // y is the matrix of one-hot label vectors where only y[c] = 1,
  // where c is the right class.
  arma::Mat<real> y = arma::zeros<arma::Mat<real>>(NUM_CLASSES, NUM_TRAIN);

  std::cout << "Loading training data..." << std::endl;
  read_mnist(FILE_TRAIN_IMAGES, x);
  read_mnist_label(FILE_TRAIN_LABELS, label);
  label_to_y(label, NUM_CLASSES, y);

  /* FORWARD PROP */
  // set up for forward prop
  arma::Mat<real> c_X_batch = x.cols(0, batch_size - 1);
  arma::Mat<real> c_y_batch = y.cols(0, batch_size - 1);
  struct cache c_cache;
  c_cache.z.resize(2);
  c_cache.a.resize(2);
  c_cache.X = c_X_batch;

  arma::Mat<real> g_X_batch = x.cols(0, batch_size - 1);
  arma::Mat<real> g_y_batch = y.cols(0, batch_size - 1);
  DeviceNeuralNetwork dnn(nn.H);
  dnn.CopyToDevice(nn.W, nn.b);
  DeviceGrads grads(nn.H);
  DeviceData data(g_X_batch.memptr(), g_y_batch.memptr(), g_X_batch.n_cols, g_X_batch.n_rows);
  DeviceCache g_cache(nn.H, g_X_batch.n_cols, data.X);
  real contrib = 1.;

  // official forward prop
  int N = c_X_batch.n_cols;
  g_cache.LoadBias(dnn.b);

  arma::Mat<real> z1 = nn.W[0] * c_X_batch+ arma::repmat(nn.b[0], 1, N);
  c_cache.z[0] = z1;

  myGEMM(dnn.W[0], data.X, g_cache.z[0], 1., 1., dnn.H[1], N, dnn.H[0], true);
  compareHostDeviceResults(g_cache.z[0], z1.memptr(), nn.H[1], N, "Linear Layer 1");

  arma::Mat<real> a1;
  sigmoid(z1, a1);
  c_cache.a[0] = a1;
  deviceSigmoid(g_cache.z[0], dnn.H[1], N);
  g_cache.recordAFromZ();
  compareHostDeviceResults(g_cache.a[0], a1.memptr(), nn.H[1], N, "Sigmoid Activation");

  arma::Mat<real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  c_cache.z[1] = z2;
  myGEMM(dnn.W[1], g_cache.a[0], g_cache.z[1], 1., 1., dnn.H[2], N, dnn.H[1], true);
  compareHostDeviceResults(g_cache.z[1], z2.memptr(), nn.H[2], N, "Linear Layer 2");

  arma::Mat<real> a2;
  softmax(z2, a2);
  c_cache.a[1] = c_cache.yc = a2;
  deviceSoftmax(g_cache.z[1], dnn.H[2], N);
  g_cache.yc = g_cache.z[1];
  compareHostDeviceResults(g_cache.yc, a2.memptr(), nn.H[2], N, "Softmax Layer");

  std::cout << "______________________BACKWARD PROP______________________\n";
  struct grads c_bpgrads;
  c_bpgrads.dW.resize(nn.num_layers);
  c_bpgrads.db.resize(nn.num_layers);
  DeviceGrads bpgrads(nn.H);
  bpgrads.LoadWeightMatrices(dnn.W);

  arma::Mat<real> c_diff = (1.0 / N) * (c_cache.yc - c_y_batch);
  real* diff = data.y;   // reuse
  deviceSubtract(g_cache.yc, diff, 1.0/N, nn.H[2], N);
  compareHostDeviceResults(diff, c_diff.memptr(), nn.H[2], N, "Reverse Loss + Softmax");

  c_bpgrads.dW[1] = c_diff * c_cache.a[0].t() + reg * nn.W[1];

  myGEMM(diff, g_cache.a[0], bpgrads.dW[1], 
          contrib, contrib * reg, nn.H[2], nn.H[1], N, 
          false, false, true);
  compareHostDeviceResults(bpgrads.dW[1], c_bpgrads.dW[1].memptr(), nn.H[2], nn.H[1], "dW2 result");

  c_bpgrads.db[1] = arma::sum(c_diff, 1);
  deviceSum(diff, bpgrads.db[1], contrib, nn.H[2], N);
  compareHostDeviceResults(bpgrads.db[1], c_bpgrads.db[1].memptr(), nn.H[2], 1, "db2 result");

  arma::Mat<real> c_da1 = nn.W[1].t() * c_diff;
  real* da1; 
  myGEMMAlloc(dnn.W[1], diff, da1, 1., 0., nn.H[1], N, nn.H[2], 
          false, true, false);
  compareHostDeviceResults(da1, c_da1.memptr(), nn.H[1], N, "DA1 result");

  arma::Mat<real> c_dz1 = c_da1 % c_cache.a[0] % (1 - c_cache.a[0]);
  real* dz1 = da1; // reuse
  deviceSigmoidBackward(g_cache.a[0], dz1, nn.H[1], N);
  compareHostDeviceResults(dz1, c_dz1.memptr(), nn.H[1], N, "DZ1 result", true);

  c_bpgrads.dW[0] = c_dz1 * c_cache.X.t() + reg * nn.W[0];
  myGEMM(dz1, g_cache.X, bpgrads.dW[0], 
          contrib, contrib * reg, nn.H[1], nn.H[0], N, 
          false, false, true);
  compareHostDeviceResults(bpgrads.dW[0], c_bpgrads.dW[0].memptr(), nn.H[1], nn.H[0], "dW2 result");

  c_bpgrads.db[0] = arma::sum(c_dz1, 1);
  deviceSum(dz1, bpgrads.db[0], contrib, nn.H[1], N);
  compareHostDeviceResults(bpgrads.db[0], c_bpgrads.db[0].memptr(), nn.H[1], 1, "db2 result");

  deviceCleanUp(dz1);
}

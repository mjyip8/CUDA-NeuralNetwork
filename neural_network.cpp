#include "neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "iomanip"
#include "mpi.h"
#include "utils/common.h"

#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

real norms(NeuralNetwork& nn) {
  real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

void initialize(std::vector<arma::Mat<real>>& W, 
                      std::vector<arma::Col<real>>& b, NeuralNetwork& nn) {
  W.resize((size_t) nn.num_layers);
  b.resize((size_t) nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    W[i] = arma::zeros<arma::Mat<real>>(arma::size(nn.W[i]));
    b[i] = arma::zeros<arma::Mat<real>>(arma::size(nn.b[i]));
  }
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_gpudata_tofile(NeuralNetwork& nn, int iter) {
  std::stringstream s;
  s << "Outputs/GPUmats/ParallelW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/GPUmats/ParallelW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/GPUmats/Parallelb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/GPUmats/Parallelb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
  arma::Mat<real> A, B, C, D;

  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  A.load(s.str(), arma::raw_ascii);
  real max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
  real L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  B.load(t.str(), arma::raw_ascii);
  real max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
  real L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  C.load(u.str(), arma::raw_ascii);
  real max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
  real L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  D.load(v.str(), arma::raw_ascii);
  real max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
  real L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

  int ow = 15;

  if (iter == 0) {
    error_file << std::left << std::setw(ow) << "Iteration" << std::left
               << std::setw(ow) << "Max Err W0" << std::left << std::setw(ow)
               << "Max Err W1" << std::left << std::setw(ow) << "Max Err b0"
               << std::left << std::setw(ow) << "Max Err b1" << std::left
               << std::setw(ow) << "L2 Err W0" << std::left << std::setw(ow)
               << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0"
               << std::left << std::setw(ow) << "L2 Err b1"
               << "\n";
  }

  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow)
             << max_errW0 << std::left << std::setw(ow) << max_errW1
             << std::left << std::setw(ow) << max_errb0 << std::left
             << std::setw(ow) << max_errb1 << std::left << std::setw(ow)
             << L2_errW0 << std::left << std::setw(ow) << L2_errW1 << std::left
             << std::setw(ow) << L2_errb0 << std::left << std::setw(ow)
             << L2_errb1 << "\n";
}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

void parallel_feedforward(DeviceNeuralNetwork& nn, real* X,
                          DeviceCache& cache) {
  int N = cache.N;
  cache.LoadBias(nn.b);

  myGEMM(nn.W[0], X, cache.z[0], 1., 1., nn.H[1], N, nn.H[0], true);

  deviceSigmoid(cache.z[0], nn.H[1], N);
  cache.recordAFromZ();

  myGEMM(nn.W[1], cache.z[0], cache.z[1], 1., 1., nn.H[2], N, nn.H[1], true);

  deviceSoftmax(cache.z[1], nn.H[2], N);

  cache.yc = cache.z[1];
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<real>& y, real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  arma::Mat<real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<real> da1 = nn.W[1].t() * diff;

  arma::Mat<real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

void parallel_backprop(DeviceNeuralNetwork& nn, real* y, real reg,
              DeviceCache& bpcache, DeviceGrads& bpgrads, real contrib, 
              real lr) {

  int N = bpcache.N;
  bpgrads.LoadWeightMatrices(nn.W);

  real* diff = deviceToDeviceCopy(y, nn.H[2]* N);   // reuse
  deviceSubtract(bpcache.yc, diff, 1.0/N, nn.H[2], N);

  myGEMM(diff, bpcache.a[0], bpgrads.dW[1], 
          1., reg, nn.H[2], nn.H[1], N, 
          false, false, true);

  deviceSum(diff, bpgrads.db[1], 1., nn.H[2], N);

  setToZero(bpgrads.dz, nn.H[1]*N);
  myGEMM(nn.W[1], diff, bpgrads.dz, 1., 0., nn.H[1], N, nn.H[2], 
          false, true, false);

  deviceSigmoidBackward(bpcache.a[0], bpgrads.dz, nn.H[1], N);

  myGEMM(bpgrads.dz, bpcache.X, bpgrads.dW[0], 
          1., reg, nn.H[1], nn.H[0], N, 
          false, false, true);

  deviceSum(bpgrads.dz, bpgrads.db[0], 1., nn.H[1], N);

  # pragma unroll
  for (int i=0; i < nn.num_layers; i++) {
    deviceUpdateParam(bpgrads.dW[i], nn.W[i], lr, contrib, nn.H[i+1], nn.H[i]);
    deviceUpdateParam(bpgrads.db[i], nn.b[i], lr, contrib, nn.H[i+1], 1);
  }
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
real loss(NeuralNetwork& nn, const arma::Mat<real>& yc,
          const arma::Mat<real>& y, real reg) {
  int N = yc.n_cols;
  real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  real data_loss = ce_sum / N;
  real reg_loss = 0.5 * reg * norms(nn);
  real loss = data_loss + reg_loss;
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<real>& X,
             arma::Row<real>& label) {
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i) {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::Mat<real>& X,
             const arma::Mat<real>& y, real reg, struct grads& numgrads) {
  real h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

    for (int j = 0; j < nn.W[i].n_rows; ++j) {
      for (int k = 0; k < nn.W[i].n_cols; ++k) {
        real oldval = nn.W[i](j, k);
        nn.W[i](j, k) = oldval + h;
        feedforward(nn, X, numcache);
        real fxph = loss(nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward(nn, X, numcache);
        real fxnh = loss(nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

    for (int j = 0; j < nn.b[i].size(); ++j) {
      real oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward(nn, X, numcache);
      real fxph = loss(nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward(nn, X, numcache);
      real fxnh = loss(nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network nn
 */
void train(NeuralNetwork& nn, const arma::Mat<real>& X,
           const arma::Mat<real>& y, real learning_rate, real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
        if (grad_check) {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag) {
        write_cpudata_tofile(nn, iter);
      }

      iter++;
    }
  }
  std::cout << "Exited train" << std::endl;
}

/*********************************************************************************
 *                          PARALLEL IMPLEMENTATION
 *********************************************************************************/

/*
 * TODO
 * Train the neural network nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::Mat<real>& X,
                    const arma::Mat<real>& y, real learning_rate, real reg,
                    const int epochs, const int batch_size, bool grad_check,
                    int print_every, int debug) {
  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int N = (rank == 0) ? X.n_cols : 0;
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  int img_size = nn.H[0];
  int n_classes = nn.H[2];

  MPIBatchInfo bi(num_procs, batch_size, N);
  DeviceNeuralNetwork dnn(nn.H);
  DeviceGrads grads(nn.H, N);

  int total_batch_size = std::min(batch_size - 1, N - 1) + 1;
  int ceil_minibatch_size = (total_batch_size + num_procs - 1) / num_procs;
  ceil_minibatch_size = (total_batch_size % ceil_minibatch_size)? total_batch_size % ceil_minibatch_size : ceil_minibatch_size;
  DeviceCache cache(nn.H, ceil_minibatch_size);

  // std::ofstream error_file;
  // error_file.open("Outputs/CpuGpuDiff.txt");
  int print_flag = 0;
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int first_col = batch_size * batch;
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      total_batch_size = last_col - first_col + 1;

      if (rank == 0) {
        bi.batchUpdate(X, y, total_batch_size);
      }

      int minibatch_size = (total_batch_size + num_procs - 1) / num_procs;
      if (rank == num_procs - 1) {
        minibatch_size = (total_batch_size % minibatch_size)? 
                                    total_batch_size % minibatch_size : minibatch_size;
      }

      const real* batch_X = (rank == 0) ? X.colptr(first_col) : nullptr;
      const real* batch_y = (rank == 0) ? y.colptr(first_col) : nullptr;

      arma::Mat<real> minibatch_X(minibatch_size, img_size);
      arma::Mat<real> minibatch_y(minibatch_size, n_classes);
      MPI_SAFE_CALL(MPI_Scatterv(batch_X, bi.minibatch_sizes_X, bi.displacements_X, 
                    MPI_FP, minibatch_X.memptr(), minibatch_size * img_size, MPI_FP, 0, MPI_COMM_WORLD));

      MPI_SAFE_CALL(MPI_Scatterv(batch_y, bi.minibatch_sizes_y, bi.displacements_y, 
                    MPI_FP, minibatch_y.memptr(), minibatch_size * n_classes, MPI_FP, 0, MPI_COMM_WORLD));

      // Device variables 
      dnn.CopyToDevice(nn.W, nn.b);
      DeviceData data(minibatch_X.memptr(), minibatch_y.memptr(), minibatch_size, 
                      img_size, n_classes);
      real contrib = ((real) minibatch_size) / ((real) total_batch_size);
      cache.X = data.X;
      cache.N = minibatch_size;

      // 2. compute each sub-batch of images' contribution to network coefficient updates
      parallel_feedforward(dnn, data.X, cache);
      parallel_backprop(dnn, data.y, reg, cache, grads, contrib, learning_rate);

      // Copy gradients from device to host 
      // just using the bpgrads struct to save nn W videos
      std::vector<arma::Mat<real>> sendW;
      std::vector<arma::Col<real>> sendb;
      initialize(sendW, sendb, nn);
      dnn.CopyToHost(sendW, sendb);

      // 3. reduce the coefficient updates and broadcast to all nodes with`MPI_Allreduce()

      MPI_Request reqs[nn.num_layers * 2];
      # pragma unroll
      for (int i = 0; i < nn.num_layers; ++i) {
        MPI_SAFE_CALL(MPI_Iallreduce(sendW[i].memptr(), nn.W[i].memptr(), 
                      sendW[i].n_elem, MPI_FP, MPI_SUM, MPI_COMM_WORLD, &reqs[i]));
        MPI_SAFE_CALL(MPI_Iallreduce(sendb[i].memptr(), nn.b[i].memptr(), 
                      sendb[i].n_elem, MPI_FP, MPI_SUM, MPI_COMM_WORLD, &reqs[i + nn.num_layers])); 
      }
      MPI_Waitall(num_procs, reqs, MPI_STATUSES_IGNORE);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      // if (print_every <= 0) {
      //   print_flag = batch == 0;
      // } else {
      //   print_flag = iter % print_every == 0;
      // }

      // if (debug && rank == 0 && print_flag) {
      //   write_diff_gpu_cpu(nn, iter, error_file);
      //   write_gpudata_tofile(nn, iter);
      // }

      iter++;
    }
  }
  // error_file.close();
}

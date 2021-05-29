#ifndef UTILS_TWO_LAYER_NET_H_
#define UTILS_TWO_LAYER_NET_H_

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>

#include "../utils/types.h"

class NeuralNetwork {
 public:
  const int num_layers = 2;
  // H[i] is the number of neurons in layer i (where i=0 implies input layer)
  std::vector<int> H;
  // Weights of the neural network
  // W[i] are the weights of the i^th layer
  std::vector<arma::Mat<real>> W;
  // Biases of the neural network
  // b[i] is the row vector biases of the i^th layer
  std::vector<arma::Col<real>> b;

  NeuralNetwork(std::vector<int> _H) {
    W.resize(num_layers);
    b.resize(num_layers);
    H = _H;

    for (int i = 0; i < num_layers; i++) {
      arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
      W[i] = 0.01 * arma::randn<arma::Mat<real>>(H[i + 1], H[i]);
      b[i].zeros(H[i + 1]);
    }
  }
};

class MPIBatchInfo {
  public:
    int num_procs;
    int batch_size;
    int N;

    // whole batch info
    int total_batch_size;
    int first_col;
    int last_col;
    int* minibatch_sizes;
    int* minibatch_sizes_X;
    int* displacements_X;
    int* minibatch_sizes_y;
    int* displacements_y;

    MPIBatchInfo(int _num_procs, int _batch_size, int _N) : 
            num_procs(_num_procs), batch_size(_batch_size), N(_N) {
      minibatch_sizes = (int*) std::malloc(sizeof(int) * num_procs);
      minibatch_sizes_X = (int*) std::malloc(sizeof(int) * num_procs);
      displacements_X = (int*) std::malloc(sizeof(int) * num_procs);
      minibatch_sizes_y = (int*) std::malloc(sizeof(int) * num_procs);
      displacements_y = (int*) std::malloc(sizeof(int) * num_procs);
    }

    ~MPIBatchInfo() {
      std::free(minibatch_sizes);
      std::free(minibatch_sizes_X);
      std::free(displacements_X);
      std::free(minibatch_sizes_y);
      std::free(displacements_y);
    }

    void batchUpdate(const arma::Mat<real>& X, const arma::Mat<real>& y, const int batch) {
      first_col = batch_size * batch;
      last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      total_batch_size = last_col - first_col + 1;
      int minibatch_size = (total_batch_size + num_procs - 1) / num_procs;
      int last_batch_size = (total_batch_size % minibatch_size)? total_batch_size % minibatch_size : minibatch_size;
      assert(total_batch_size == minibatch_size * (num_procs - 1) + last_batch_size);

      # pragma unroll
      for (int i = 0; i < num_procs; ++i) {
        minibatch_sizes[i] = (i < num_procs - 1) ? minibatch_size : last_batch_size;
        displacements_X[i] = i * minibatch_sizes[i] * X.n_rows;
        displacements_y[i] = i * minibatch_sizes[i] * y.n_rows;
        minibatch_sizes_X[i] = minibatch_sizes[i] * X.n_rows;
        minibatch_sizes_y[i] = minibatch_sizes[i] * y.n_rows;
      }
    }
};

void feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& bpcache);
real loss(NeuralNetwork& nn, const arma::Mat<real>& yc,
          const arma::Mat<real>& y, real reg);
void backprop(NeuralNetwork& nn, const arma::Mat<real>& y, real reg,
              const struct cache& bpcache, struct grads& bpgrads);
void numgrad(NeuralNetwork& nn, const arma::Mat<real>& X,
             const arma::Mat<real>& y, real reg, struct grads& numgrads);
void train(NeuralNetwork& nn, const arma::Mat<real>& X,
           const arma::Mat<real>& y, real learning_rate, real reg = 0.0,
           const int epochs = 15, const int batch_size = 800,
           bool grad_check = false, int print_every = -1, int debug = 0);
void predict(NeuralNetwork& nn, const arma::Mat<real>& X,
             arma::Row<real>& label);

void parallel_train(NeuralNetwork& nn, const arma::Mat<real>& X,
                    const arma::Mat<real>& y, real learning_rate,
                    real reg = 0.0, const int epochs = 15,
                    const int batch_size = 800, bool grad_check = false,
                    int print_every = -1, int debug = 0);

void initialize_grads(struct grads& grad, NeuralNetwork& nn);

#endif
